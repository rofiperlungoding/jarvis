"""Unit tests for ``jarvis.voice.stt.faster_whisper.FasterWhisperSTT``.

The faster-whisper engine relies on two external systems we do not
exercise from unit tests:

* The ``faster_whisper`` Python package, which loads CTranslate2 model
  weights and performs neural inference. The engine lazy-imports it so
  we can install a fake ``faster_whisper`` module on ``sys.modules`` for
  the duration of a test.
* numpy, which we *do* have available — the engine converts int16 PCM
  bytes to ``float32`` arrays before handing them to Whisper. Tests
  assert on the conversion.

The tests below cover the behaviours the design and Requirements 1.3
and 13.2 demand:

* :class:`FasterWhisperSTT` conforms to the runtime-checkable
  :class:`~jarvis.voice.stt.base.STTEngine` Protocol.
* Inference is offloaded to the executor (``run_in_executor`` semantics)
  rather than running on the event loop.
* Confidence is computed as ``mean(exp(segment.avg_logprob))`` across
  the decoded segments.
* :attr:`Transcript.duration_ms` is derived from the input PCM size at
  16 kHz / 16-bit / mono.
* :attr:`Transcript.started_at` is stamped from an injected
  :class:`~jarvis.utils.time_source.TimeSource`.
* :meth:`aclose` shuts down an engine-owned executor and is idempotent;
  caller-supplied executors are left untouched.

Validates: Requirements 1.3, 13.2
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import math
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import STTEngine, Transcript
from jarvis.voice.stt.faster_whisper import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_SIZE,
    FasterWhisperSTT,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSegment:
    """Mimics ``faster_whisper.transcribe.Segment`` (text + avg_logprob)."""

    def __init__(self, text: str, avg_logprob: float) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeInfo:
    """Mimics ``faster_whisper.transcribe.TranscriptionInfo``."""

    def __init__(self, language: str = "en") -> None:
        self.language = language


class _FakeWhisperModel:
    """In-memory stand-in for :class:`faster_whisper.WhisperModel`.

    Each call to :meth:`transcribe` records the audio array passed in
    (as numpy ``float32``) and returns a configurable list of segments
    plus an info object. Tests inspect :attr:`calls` to assert order,
    audio shape, and language passthrough; :attr:`thread_ids` records
    the thread on which each transcribe call ran so we can verify
    inference is offloaded off the event loop thread.
    """

    def __init__(
        self,
        segments: list[_FakeSegment] | None = None,
        info: _FakeInfo | None = None,
    ) -> None:
        self.segments: list[_FakeSegment] = segments or []
        self.info: _FakeInfo = info or _FakeInfo()
        self.calls: list[dict[str, Any]] = []
        self.thread_ids: list[int] = []

    def transcribe(self, audio: Any, **kwargs: Any):
        self.calls.append({"audio": audio, **kwargs})
        self.thread_ids.append(threading.get_ident())
        # Return a tuple matching faster-whisper's public contract:
        # (segments_iterable, info). We return a list rather than a
        # generator so the engine's ``list(...)`` materialisation is a
        # no-op.
        return list(self.segments), self.info


def _install_fake_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
    model: _FakeWhisperModel,
    *,
    load_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Register a fake ``faster_whisper`` module for the test's lifetime."""
    fake_module = types.ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, model_size: str, **kwargs: Any) -> None:
            if load_calls is not None:
                load_calls.append({"model_size": model_size, **kwargs})

        def transcribe(self, audio: Any, **kwargs: Any):
            return model.transcribe(audio, **kwargs)

    fake_module.WhisperModel = _WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def _aware_dt(year: int = 2024) -> datetime:
    return datetime(year, 5, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Construction / Protocol conformance
# ---------------------------------------------------------------------------


def test_faster_whisper_stt_conforms_to_stt_engine_protocol() -> None:
    engine = FasterWhisperSTT()
    assert isinstance(engine, STTEngine)


def test_default_constants_match_design() -> None:
    """Design's ``[voice.stt]`` defaults from ``default.toml``."""
    assert DEFAULT_MODEL_SIZE == "small.en"
    assert DEFAULT_DEVICE == "cpu"
    assert DEFAULT_COMPUTE_TYPE == "int8"


def test_constructor_default_properties() -> None:
    engine = FasterWhisperSTT()
    assert engine.model_size == DEFAULT_MODEL_SIZE
    assert engine.device == DEFAULT_DEVICE
    assert engine.compute_type == DEFAULT_COMPUTE_TYPE


def test_constructor_rejects_non_positive_beam_size() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        FasterWhisperSTT(beam_size=0)
    with pytest.raises(ValueError, match="beam_size"):
        FasterWhisperSTT(beam_size=-1)


def test_constructor_does_not_load_model_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction must not import ``faster_whisper`` or load weights."""
    load_calls: list[dict[str, Any]] = []
    _install_fake_faster_whisper(
        monkeypatch, _FakeWhisperModel(), load_calls=load_calls
    )

    FasterWhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
    # Construction alone must not have hit the fake loader.
    assert load_calls == []


# ---------------------------------------------------------------------------
# transcribe — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_returns_concatenated_text_and_design_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """confidence == mean(exp(avg_logprob)) across segments (Design / Req 1.3)."""
    segments = [
        _FakeSegment(text="Hello, ", avg_logprob=-0.1),
        _FakeSegment(text="sir.", avg_logprob=-0.2),
    ]
    model = _FakeWhisperModel(segments=segments, info=_FakeInfo("en"))
    _install_fake_faster_whisper(monkeypatch, model)

    fake_clock = FakeTimeSource(now=_aware_dt())
    engine = FasterWhisperSTT(time_source=fake_clock)
    try:
        # 1 second of silence at 16 kHz / 16-bit mono = 32000 bytes.
        pcm = b"\x00\x00" * 16000
        result = await engine.transcribe(pcm, "en")
    finally:
        await engine.aclose()

    assert isinstance(result, Transcript)
    # Each segment text is stripped before joining with a single space,
    # so "Hello, " + "sir." → "Hello, sir." (no double-space).
    assert result.text == "Hello, sir."
    expected_conf = (math.exp(-0.1) + math.exp(-0.2)) / 2
    assert result.confidence == pytest.approx(expected_conf)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_transcribe_passes_language_to_whisper_and_echoes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-supplied language is forwarded to Whisper and echoed back."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ciao", -0.05)],
        info=_FakeInfo("it"),  # would auto-detect Italian
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        # Caller forces "en"; Transcript must echo the request, not the
        # auto-detected "it".
        result = await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    assert model.calls[0]["language"] == "en"
    assert result.language == "en"


@pytest.mark.asyncio
async def test_transcribe_falls_back_to_detected_language_when_argument_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``language`` argument means auto-detect; the engine must echo
    Whisper's detection."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("bonjour", -0.05)],
        info=_FakeInfo("fr"),
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        result = await engine.transcribe(b"\x00\x00" * 8000, "")
    finally:
        await engine.aclose()

    # Whisper is called with ``language=None`` for auto-detect.
    assert model.calls[0]["language"] is None
    assert result.language == "fr"


@pytest.mark.asyncio
async def test_transcribe_returns_zero_confidence_for_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure silence / empty buffer → no segments → confidence 0.0.

    The Dialog_Manager gates at ``< 0.4`` (Requirement 1.8), so reporting
    0.0 here ensures the gate trips and the user is re-prompted.
    """
    model = _FakeWhisperModel(segments=[], info=_FakeInfo("en"))
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        result = await engine.transcribe(b"", "en")
    finally:
        await engine.aclose()

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.duration_ms == 0


@pytest.mark.asyncio
async def test_transcribe_uses_injected_time_source_for_started_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 1.3: started_at reflects the moment capture finished."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    fixed_now = _aware_dt(2030)
    fake_clock = FakeTimeSource(now=fixed_now)
    engine = FasterWhisperSTT(time_source=fake_clock)
    try:
        result = await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    assert result.started_at == fixed_now
    assert result.started_at.tzinfo is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n_samples,expected_ms",
    [
        (16000, 1000),  # 1 s
        (8000, 500),  # 0.5 s
        (160, 10),  # 10 ms
        (0, 0),
    ],
)
async def test_transcribe_duration_ms_derived_from_pcm_size(
    monkeypatch: pytest.MonkeyPatch,
    n_samples: int,
    expected_ms: int,
) -> None:
    model = _FakeWhisperModel(segments=[], info=_FakeInfo("en"))
    _install_fake_faster_whisper(monkeypatch, model)
    engine = FasterWhisperSTT()
    try:
        pcm = b"\x00\x00" * n_samples  # 16-bit mono
        result = await engine.transcribe(pcm, "en")
    finally:
        await engine.aclose()
    assert result.duration_ms == expected_ms


@pytest.mark.asyncio
async def test_transcribe_converts_int16_pcm_to_float32_in_unit_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whisper expects float32 in [-1.0, 1.0); engine must convert from int16."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("hi", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        # Construct PCM containing the extreme positive int16 value 32767.
        # After the engine's /32768 normalisation the result must fit in
        # the target dtype and stay within [-1.0, 1.0).
        pcm = np.array([32767, -32768, 0, 16384], dtype=np.int16).tobytes()
        await engine.transcribe(pcm, "en")
    finally:
        await engine.aclose()

    audio = model.calls[0]["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (4,)
    # Values fall within the documented range.
    assert audio.min() >= -1.0
    assert audio.max() < 1.0
    # 32767 / 32768 ≈ 0.99997
    assert audio[0] == pytest.approx(32767 / 32768, rel=1e-6)
    assert audio[1] == pytest.approx(-1.0)
    assert audio[2] == 0.0


@pytest.mark.asyncio
async def test_transcribe_rejects_non_bytes_input() -> None:
    engine = FasterWhisperSTT()
    try:
        with pytest.raises(TypeError, match="raw PCM bytes"):
            await engine.transcribe("not bytes", "en")  # type: ignore[arg-type]
    finally:
        await engine.aclose()


# ---------------------------------------------------------------------------
# Executor offload (ThreadPoolExecutor / run_in_executor)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_runs_inference_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model.transcribe`` must run on a worker thread, not the event loop."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    main_thread_id = threading.get_ident()
    assert model.thread_ids, "transcribe was never invoked"
    # Each transcribe call must run on a different thread than the
    # caller (the event loop thread).
    for tid in model.thread_ids:
        assert tid != main_thread_id


@pytest.mark.asyncio
async def test_transcribe_uses_caller_supplied_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-provided executor is used and must NOT be shut down by aclose."""
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="caller-owned"
    )
    try:
        engine = FasterWhisperSTT(executor=executor)
        await engine.transcribe(b"\x00\x00" * 8000, "en")
        await engine.aclose()
        # Caller-supplied executor must still be usable after aclose.
        future = executor.submit(lambda: 42)
        assert future.result(timeout=1.0) == 42
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_aclose_shuts_down_engine_owned_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    await engine.transcribe(b"\x00\x00" * 8000, "en")
    owned = engine._executor  # capture before aclose nulls the reference
    assert owned is not None

    await engine.aclose()
    # The owned executor must reject new work.
    with pytest.raises(RuntimeError):
        owned.submit(lambda: None)


@pytest.mark.asyncio
async def test_aclose_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    await engine.transcribe(b"\x00\x00" * 8000, "en")
    await engine.aclose()
    await engine.aclose()  # second call must be a no-op

    with pytest.raises(RuntimeError, match="closed"):
        await engine.transcribe(b"\x00\x00" * 8000, "en")


@pytest.mark.asyncio
async def test_aclose_without_transcribe_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing an engine that never transcribed must not import faster_whisper."""
    load_calls: list[dict[str, Any]] = []
    _install_fake_faster_whisper(
        monkeypatch, _FakeWhisperModel(), load_calls=load_calls
    )

    engine = FasterWhisperSTT()
    await engine.aclose()
    assert load_calls == []


# ---------------------------------------------------------------------------
# Model loading kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_load_forwards_device_and_compute_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom device / compute_type / cpu_threads reach WhisperModel."""
    load_calls: list[dict[str, Any]] = []
    _install_fake_faster_whisper(
        monkeypatch, _FakeWhisperModel(), load_calls=load_calls
    )

    engine = FasterWhisperSTT(
        model_size="medium",
        device="cuda",
        compute_type="float16",
        cpu_threads=4,
        num_workers=2,
        download_root="/tmp/models",
        local_files_only=True,
    )
    try:
        await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    assert len(load_calls) == 1
    call = load_calls[0]
    assert call["model_size"] == "medium"
    assert call["device"] == "cuda"
    assert call["compute_type"] == "float16"
    assert call["cpu_threads"] == 4
    assert call["num_workers"] == 2
    assert call["download_root"] == "/tmp/models"
    assert call["local_files_only"] is True


@pytest.mark.asyncio
async def test_model_load_omits_optional_kwargs_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional knobs default to ``None`` and must not be forwarded."""
    load_calls: list[dict[str, Any]] = []
    _install_fake_faster_whisper(
        monkeypatch, _FakeWhisperModel(), load_calls=load_calls
    )

    engine = FasterWhisperSTT()
    try:
        await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    call = load_calls[0]
    assert "cpu_threads" not in call
    assert "num_workers" not in call
    assert "download_root" not in call
    # Defaults that are *always* sent.
    assert call["device"] == DEFAULT_DEVICE
    assert call["compute_type"] == DEFAULT_COMPUTE_TYPE
    assert call["local_files_only"] is False


@pytest.mark.asyncio
async def test_transcribe_forwards_beam_size_and_vad_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT(beam_size=3, vad_filter=True)
    try:
        await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    call = model.calls[0]
    assert call["beam_size"] == 3
    assert call["vad_filter"] is True


# ---------------------------------------------------------------------------
# Confidence edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confidence_handles_segment_without_avg_logprob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment lacking ``avg_logprob`` is treated as zero probability."""

    class _BareSegment:
        def __init__(self, text: str) -> None:
            self.text = text
            # Intentionally no ``avg_logprob`` attribute.

    model = _FakeWhisperModel(
        segments=[
            _FakeSegment("good", -0.1),
            _BareSegment("bare"),  # type: ignore[list-item]
        ],
        info=_FakeInfo("en"),
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        result = await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    expected = (math.exp(-0.1) + 0.0) / 2
    assert result.confidence == pytest.approx(expected)


@pytest.mark.asyncio
async def test_confidence_clamped_into_validator_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extreme avg_logprob values must not violate Transcript validators."""
    # avg_logprob == 0 → exp(0) == 1.0 (the validator's upper bound).
    # A tiny floating-point drift over 1.0 must be clamped down.
    model = _FakeWhisperModel(
        segments=[_FakeSegment("perfect", 0.0)],
        info=_FakeInfo("en"),
    )
    _install_fake_faster_whisper(monkeypatch, model)

    engine = FasterWhisperSTT()
    try:
        result = await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Concurrent / repeated calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_transcribe_loads_model_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[dict[str, Any]] = []
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model, load_calls=load_calls)

    engine = FasterWhisperSTT()
    try:
        for _ in range(3):
            await engine.transcribe(b"\x00\x00" * 8000, "en")
    finally:
        await engine.aclose()

    assert len(load_calls) == 1
    assert len(model.calls) == 3


@pytest.mark.asyncio
async def test_concurrent_transcribe_races_load_under_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first calls must still load the model exactly once."""
    load_calls: list[dict[str, Any]] = []
    model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", -0.05)], info=_FakeInfo("en")
    )
    _install_fake_faster_whisper(monkeypatch, model, load_calls=load_calls)

    engine = FasterWhisperSTT()
    try:
        await asyncio.gather(
            engine.transcribe(b"\x00\x00" * 8000, "en"),
            engine.transcribe(b"\x00\x00" * 8000, "en"),
            engine.transcribe(b"\x00\x00" * 8000, "en"),
        )
    finally:
        await engine.aclose()

    assert len(load_calls) == 1
