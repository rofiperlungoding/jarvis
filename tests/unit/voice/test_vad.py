"""Unit tests for :class:`jarvis.voice.vad.SileroVAD`.

These tests exercise the VAD state machine with a deterministic injected
probability function so they do not require ``silero-vad`` or ``torch``
to be installed. The default silero-backed probability path is covered
indirectly via the construction of an instance with a non-default
``probability_fn``.

The tests verify the contract relevant to:

* Requirement 1.3 — 700 ms trailing-silence threshold gates ``SPEECH_END``.
* Requirement 1.7 — ``SPEECH_START`` fires on the first frame above the
  start threshold so the audio capture loop can call
  :meth:`AudioPlayer.stop` for barge-in.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.audio_io import VAD_FRAME_SAMPLES
from jarvis.voice.vad import (
    SileroVAD,
    VADEvent,
    VADEventKind,
    load_default_silero_probability_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Default frame at 30 ms / 480 samples / 16-bit signed PCM mono = 960 bytes.
_FRAME_BYTES = VAD_FRAME_SAMPLES * 2  # sample_width=2, channels=1


def _frame() -> bytes:
    """A single zero-filled frame of the right size."""
    return bytes(_FRAME_BYTES)


def _make_vad(
    *,
    probabilities: list[float],
    trailing_silence_ms: int = 700,
    speech_start_threshold: float = 0.5,
    speech_end_threshold: float | None = None,
    time_source: FakeTimeSource | None = None,
) -> SileroVAD:
    """Build a VAD whose probability backend replays a fixed list."""
    sequence = iter(probabilities)

    def fn(_frame: bytes) -> float:
        try:
            return next(sequence)
        except StopIteration:
            return 0.0

    return SileroVAD(
        trailing_silence_ms=trailing_silence_ms,
        speech_start_threshold=speech_start_threshold,
        speech_end_threshold=speech_end_threshold,
        probability_fn=fn,  # type: ignore[arg-type]
        time_source=time_source or FakeTimeSource(),
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_defaults_match_requirements() -> None:
    vad = SileroVAD(probability_fn=lambda _f: 0.0)
    # Requirement 1.3 default trailing silence.
    assert vad.trailing_silence_ms == 700
    # Documented default for the start threshold.
    assert vad.speech_start_threshold == 0.5
    # End threshold defaults to start when not specified.
    assert vad.speech_end_threshold == 0.5
    # 30 ms frames at 16 kHz = exactly 30.0 ms cadence.
    assert vad.frame_duration_ms == pytest.approx(30.0)
    assert vad.frame_size_bytes == _FRAME_BYTES
    assert not vad.is_speaking
    assert vad.silent_run_ms == 0.0
    assert vad.elapsed_ms == 0.0
    assert vad.frame_count == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"trailing_silence_ms": 0}, "trailing_silence_ms"),
        ({"trailing_silence_ms": -1}, "trailing_silence_ms"),
        ({"speech_start_threshold": 1.5}, "speech_start_threshold"),
        ({"speech_start_threshold": -0.1}, "speech_start_threshold"),
        ({"speech_end_threshold": 1.1}, "speech_end_threshold"),
        # End threshold must be <= start threshold (asymmetric hysteresis).
        (
            {"speech_start_threshold": 0.4, "speech_end_threshold": 0.5},
            "speech_end_threshold",
        ),
        ({"sample_rate_hz": 0}, "sample_rate_hz"),
        ({"frame_samples": 0}, "frame_samples"),
        ({"sample_width": 0}, "sample_width"),
        ({"channels": 0}, "channels"),
    ],
)
def test_construction_rejects_invalid_arguments(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SileroVAD(probability_fn=lambda _f: 0.0, **kwargs)


# ---------------------------------------------------------------------------
# process() — frame validation
# ---------------------------------------------------------------------------


def test_process_rejects_wrong_frame_size() -> None:
    vad = _make_vad(probabilities=[0.0])
    with pytest.raises(ValueError, match="frame size mismatch"):
        vad.process(b"\x00" * (_FRAME_BYTES - 1))
    with pytest.raises(ValueError, match="frame size mismatch"):
        vad.process(b"\x00" * (_FRAME_BYTES + 1))


def test_process_rejects_non_bytes() -> None:
    vad = _make_vad(probabilities=[0.0])
    with pytest.raises(TypeError):
        vad.process("not bytes")  # type: ignore[arg-type]


def test_process_accepts_bytearray_and_memoryview() -> None:
    vad = _make_vad(probabilities=[0.0, 0.0])
    # ``process`` is annotated ``frame: bytes`` but accepts any bytes-like
    # object via the runtime check; the type-ignores below pin that
    # behaviour for tests.
    assert vad.process(bytearray(_FRAME_BYTES)) == []  # type: ignore[arg-type]
    assert (
        vad.process(memoryview(bytes(_FRAME_BYTES))) == []  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Speech start (Requirement 1.7)
# ---------------------------------------------------------------------------


def test_silence_does_not_trigger_speech_start() -> None:
    vad = _make_vad(probabilities=[0.0, 0.1, 0.2, 0.49])
    for _ in range(4):
        assert vad.process(_frame()) == []
    assert not vad.is_speaking


def test_first_above_threshold_emits_speech_start() -> None:
    vad = _make_vad(probabilities=[0.49, 0.5])
    assert vad.process(_frame()) == []
    events = vad.process(_frame())
    assert len(events) == 1
    event = events[0]
    assert event.kind is VADEventKind.SPEECH_START
    assert event.probability == pytest.approx(0.5)
    assert event.frame_index == 1  # second frame, zero-indexed
    assert event.elapsed_ms == pytest.approx(60.0)  # two 30 ms frames
    assert vad.is_speaking


def test_speech_start_threshold_inclusive_boundary() -> None:
    # Exactly at threshold should fire ``SPEECH_START``.
    vad = _make_vad(probabilities=[0.5], speech_start_threshold=0.5)
    events = vad.process(_frame())
    assert [e.kind for e in events] == [VADEventKind.SPEECH_START]


def test_repeated_speech_does_not_re_emit_start() -> None:
    vad = _make_vad(probabilities=[0.9, 0.9, 0.9])
    starts = sum(
        1
        for ev in (e for _ in range(3) for e in vad.process(_frame()))
        if ev.kind is VADEventKind.SPEECH_START
    )
    assert starts == 1


# ---------------------------------------------------------------------------
# Speech end (Requirement 1.3 — 700 ms trailing silence)
# ---------------------------------------------------------------------------


def test_speech_end_requires_full_trailing_silence() -> None:
    # 30 ms per frame, 700 ms trailing silence -> needs ceil(700/30) = 24
    # silent frames after the speech run before SPEECH_END fires.
    speech_frames = 5
    silent_frames_needed = 24  # 24 * 30 ms = 720 ms >= 700 ms
    probs = [0.9] * speech_frames + [0.0] * silent_frames_needed
    vad = _make_vad(probabilities=probs)

    # Speech run.
    events: list[VADEvent] = []
    for _ in range(speech_frames):
        events.extend(vad.process(_frame()))
    assert [e.kind for e in events] == [VADEventKind.SPEECH_START]
    assert vad.is_speaking

    # Silence: the threshold is crossed on the 24th silent frame
    # (cumulative 720 ms >= 700 ms). Frames 1..23 produce no events.
    silent_events: list[VADEvent] = []
    for i in range(silent_frames_needed):
        new = vad.process(_frame())
        if i < silent_frames_needed - 1:
            assert new == []
            assert vad.is_speaking
        silent_events.extend(new)
    assert [e.kind for e in silent_events] == [VADEventKind.SPEECH_END]
    assert not vad.is_speaking
    end_event = silent_events[0]  # type: ignore[unreachable]
    assert end_event.elapsed_ms == pytest.approx(
        (speech_frames + silent_frames_needed) * 30.0
    )


def test_speech_end_does_not_fire_below_threshold_silence() -> None:
    # 23 silent frames = 690 ms < 700 ms. SPEECH_END must NOT fire.
    probs = [0.9] + [0.0] * 23
    vad = _make_vad(probabilities=probs)
    vad.process(_frame())  # consume start
    silent_events: list[VADEvent] = []
    for _ in range(23):
        silent_events.extend(vad.process(_frame()))
    assert silent_events == []
    assert vad.is_speaking
    assert vad.silent_run_ms == pytest.approx(23 * 30.0)


def test_brief_silence_inside_utterance_resets_accumulator() -> None:
    # Speak, fall silent for 600 ms (< 700 ms threshold), speak again,
    # then fall silent long enough. SPEECH_END should fire ONCE at the
    # very end, not during the brief mid-utterance gap.
    probs = (
        [0.9]
        + [0.0] * 20  # 600 ms silence
        + [0.9] * 3  # speech resumes — accumulator must reset
        + [0.0] * 24  # 720 ms silence — now SPEECH_END fires
    )
    vad = _make_vad(probabilities=probs)
    all_events: list[VADEvent] = []
    for _ in range(len(probs)):
        all_events.extend(vad.process(_frame()))
    kinds = [e.kind for e in all_events]
    assert kinds == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    # Verify the accumulator did indeed reset by checking the elapsed
    # time at SPEECH_END equals the entire feed.
    assert all_events[-1].elapsed_ms == pytest.approx(len(probs) * 30.0)


def test_custom_trailing_silence_ms_is_honoured() -> None:
    # Three 30 ms silent frames = 90 ms, threshold 80 ms.
    probs = [0.9, 0.0, 0.0, 0.0]
    vad = _make_vad(probabilities=probs, trailing_silence_ms=80)
    events: list[VADEvent] = []
    for _ in range(len(probs)):
        events.extend(vad.process(_frame()))
    assert [e.kind for e in events] == [
        VADEventKind.SPEECH_START,
        VADEventKind.SPEECH_END,
    ]


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


def test_hysteresis_keeps_speech_alive_between_thresholds() -> None:
    # Speech starts at >= 0.6, only ends below 0.3.
    # A run at 0.4 (between thresholds) should KEEP speech alive (silence
    # accumulator does not advance) because 0.4 >= speech_end_threshold.
    probs = [0.7] + [0.4] * 50 + [0.0] * 24
    vad = _make_vad(
        probabilities=probs,
        speech_start_threshold=0.6,
        speech_end_threshold=0.3,
    )
    events: list[VADEvent] = []
    for _ in range(len(probs)):
        events.extend(vad.process(_frame()))
    kinds = [e.kind for e in events]
    # Exactly one start, exactly one end, and the end is at the very tail.
    assert kinds == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    assert events[-1].elapsed_ms == pytest.approx(len(probs) * 30.0)


# ---------------------------------------------------------------------------
# Probability clamping (defensive)
# ---------------------------------------------------------------------------


def test_out_of_range_probability_is_clamped() -> None:
    # An ill-behaved backend returning 1.5 must NOT corrupt the state
    # machine; it should be treated as 1.0 (firmly speech).
    vad = _make_vad(probabilities=[1.5])
    events = vad.process(_frame())
    assert [e.kind for e in events] == [VADEventKind.SPEECH_START]
    assert events[0].probability == 1.0


# ---------------------------------------------------------------------------
# Event metadata
# ---------------------------------------------------------------------------


def test_events_carry_timestamps_from_time_source() -> None:
    fixed = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
    clock = FakeTimeSource(now=fixed)
    vad = _make_vad(probabilities=[0.9], time_source=clock)
    events = vad.process(_frame())
    assert events[0].occurred_at == fixed


# ---------------------------------------------------------------------------
# flush() and reset()
# ---------------------------------------------------------------------------


def test_flush_emits_synthetic_speech_end_when_speaking() -> None:
    vad = _make_vad(probabilities=[0.9, 0.9])
    vad.process(_frame())
    vad.process(_frame())
    assert vad.is_speaking
    events = vad.flush()
    assert [e.kind for e in events] == [VADEventKind.SPEECH_END]
    assert events[0].probability == 0.0
    assert not vad.is_speaking


def test_flush_when_silent_returns_empty() -> None:
    vad = _make_vad(probabilities=[0.0, 0.0])
    vad.process(_frame())
    vad.process(_frame())
    assert vad.flush() == []


def test_reset_clears_counters_and_state() -> None:
    vad = _make_vad(probabilities=[0.9, 0.9, 0.9])
    for _ in range(3):
        vad.process(_frame())
    assert vad.frame_count == 3
    assert vad.elapsed_ms == pytest.approx(90.0)
    assert vad.is_speaking
    vad.reset()
    assert vad.frame_count == 0
    assert vad.elapsed_ms == 0.0
    assert not vad.is_speaking
    assert vad.silent_run_ms == 0.0  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# iter_events()
# ---------------------------------------------------------------------------


async def _frames(probs: list[float]) -> AsyncIterator[bytes]:
    for _ in probs:
        yield _frame()


@pytest.mark.asyncio
async def test_iter_events_yields_transitions_in_order() -> None:
    probs = [0.0, 0.9, 0.9] + [0.0] * 24
    vad = _make_vad(probabilities=probs)
    seen: list[VADEventKind] = []
    async for event in vad.iter_events(_frames(probs)):
        seen.append(event.kind)
    assert seen == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]


@pytest.mark.asyncio
async def test_iter_events_invokes_callbacks() -> None:
    probs = [0.9] + [0.0] * 24
    vad = _make_vad(probabilities=probs)

    starts: list[VADEvent] = []
    ends: list[VADEvent] = []

    async def on_start(ev: VADEvent) -> None:
        starts.append(ev)

    async def on_end(ev: VADEvent) -> None:
        ends.append(ev)

    yielded = [
        ev
        async for ev in vad.iter_events(
            _frames(probs),
            on_speech_start=on_start,
            on_speech_end=on_end,
        )
    ]
    assert [e.kind for e in starts] == [VADEventKind.SPEECH_START]
    assert [e.kind for e in ends] == [VADEventKind.SPEECH_END]
    assert [e.kind for e in yielded] == [
        VADEventKind.SPEECH_START,
        VADEventKind.SPEECH_END,
    ]


@pytest.mark.asyncio
async def test_iter_events_flushes_on_close_when_still_speaking() -> None:
    # Source ends mid-utterance: VAD must still emit SPEECH_END.
    probs = [0.9, 0.9]
    vad = _make_vad(probabilities=probs)
    seen: list[VADEventKind] = []
    async for event in vad.iter_events(_frames(probs)):
        seen.append(event.kind)
    assert seen == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]


@pytest.mark.asyncio
async def test_iter_events_can_disable_flush_on_close() -> None:
    probs = [0.9, 0.9]
    vad = _make_vad(probabilities=probs)
    seen: list[VADEventKind] = []
    async for event in vad.iter_events(_frames(probs), flush_on_close=False):
        seen.append(event.kind)
    # Only the start fired; the trailing utterance was deliberately dropped.
    assert seen == [VADEventKind.SPEECH_START]
    assert vad.is_speaking


@pytest.mark.asyncio
async def test_iter_events_propagates_cancellation() -> None:
    probs = [0.9] * 100  # plenty
    vad = _make_vad(probabilities=probs)

    async def consumer() -> None:
        async for _event in vad.iter_events(_frames(probs)):
            await asyncio.sleep(0)  # yield to let outer task cancel us

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let consumer enter the loop
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Default silero loader — error path
# ---------------------------------------------------------------------------


def test_default_loader_rejects_non_16k() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        load_default_silero_probability_fn(sample_rate_hz=8000)
