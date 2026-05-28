"""Unit tests for ``jarvis.voice.tts.piper.PiperTTS``.

The Piper engine relies on two external systems we do not exercise in
unit tests:

* The ``piper`` Python package, which loads ONNX weights from disk and
  performs neural inference. The engine lazy-imports it so we can install
  a fake ``piper`` module on ``sys.modules`` for the duration of a test.
* The ``sounddevice``/PortAudio audio device, owned by
  :class:`~jarvis.voice.audio_io.AudioPlayer`. We monkey-patch the
  player attribute on the engine after construction so tests never open
  a real device.

The tests below verify the behaviours the design and Requirements 1.7,
11.2, and 12.2 demand:

* :class:`PiperTTS` conforms to the runtime-checkable
  :class:`~jarvis.voice.tts.base.TTSEngine` Protocol.
* Default voice id is ``en_GB-alan-medium`` (Requirement 11.2).
* :meth:`speak` is non-blocking and enqueues sentences in order.
* The worker drains queued text, calls piper, and forwards PCM bytes to
  the audio player (the streaming contract from Requirement 12.2).
* :meth:`stop` aborts active playback and discards pending sentences
  (Requirement 1.7).
* :meth:`aclose` is idempotent and shuts the worker down promptly.
* :meth:`is_playing` reports ``True`` for queued / synthesising / playing
  states.

Validates: Requirements 1.7, 11.2, 12.2
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import types
from typing import Any

import pytest

from jarvis.voice.audio_io import AudioFormat
from jarvis.voice.tts.base import TTSEngine
from jarvis.voice.tts.piper import DEFAULT_VOICE_ID, PiperTTS

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAudioChunk:
    """Mimics ``piper.AudioChunk.audio_int16_bytes`` only."""

    def __init__(self, payload: bytes) -> None:
        self.audio_int16_bytes = payload


class _FakeConfig:
    sample_rate = 22050


class _FakePiperVoice:
    """In-memory stand-in for :class:`piper.PiperVoice`.

    Each call to :meth:`synthesize` returns a sequence of fake audio
    chunks whose byte payloads encode the input text. Tests can inspect
    :attr:`calls` to assert ordering and :attr:`chunks_per_call` to
    drive multi-chunk responses.
    """

    config = _FakeConfig()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.chunks_per_call = 1
        self.synthesize_event = asyncio.Event()
        self.release_synthesize: asyncio.Event | None = None

    def synthesize(self, text: str, syn_config: Any = None):
        self.calls.append(text)
        # Each chunk contains `text.encode()` so callers can inspect it.
        for _ in range(self.chunks_per_call):
            yield _FakeAudioChunk(text.encode("utf-8"))


class _FakeAudioPlayer:
    """Minimal stand-in for :class:`AudioPlayer`.

    Records every call to :meth:`aplay` and supports cancellation via
    :meth:`stop`. Acts as a simple, observable replacement so the
    PiperTTS worker can be verified without opening PortAudio.
    """

    def __init__(self) -> None:
        self.played: list[list[bytes]] = []
        self._task: asyncio.Task[None] | None = None
        self._closed: bool = False
        self.stop_called: int = 0
        self.aclose_called: int = 0
        # When set, ``aplay`` will block on this event before completing —
        # tests use it to observe ``is_playing == True`` mid-playback.
        self.gate: asyncio.Event | None = None

    async def aplay(self, chunks):
        if self._closed:
            raise RuntimeError("FakeAudioPlayer is closed")
        # Materialise chunks so we capture order even if the input is a
        # generator that PiperTTS would otherwise consume lazily.
        materialised = [bytes(c) for c in chunks]
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._aplay_inner(materialised))
        self._task = task
        try:
            await task
        finally:
            if self._task is task:
                self._task = None

    async def _aplay_inner(self, chunks: list[bytes]) -> None:
        if self.gate is not None:
            await self.gate.wait()
        self.played.append(chunks)

    def is_playing(self) -> bool:
        task = self._task
        return task is not None and not task.done()

    async def stop(self) -> None:
        self.stop_called += 1
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    async def aclose(self) -> None:
        self.aclose_called += 1
        await self.stop()
        self._closed = True


def _install_fake_piper(monkeypatch: pytest.MonkeyPatch, voice: _FakePiperVoice) -> None:
    """Register a fake ``piper`` module for the test's lifetime."""
    fake_module = types.ModuleType("piper")

    class _PiperVoice:
        @staticmethod
        def load(*args: Any, **kwargs: Any) -> _FakePiperVoice:
            return voice

    class _SynthesisConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_module.PiperVoice = _PiperVoice  # type: ignore[attr-defined]
    fake_module.SynthesisConfig = _SynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake_module)


async def _wait_for(predicate, *, timeout: float = 1.0, interval: float = 0.005) -> None:  # noqa: ASYNC109 - poll helper, not a true timeout primitive
    """Poll ``predicate`` until truthy or ``timeout`` elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Predicate did not become truthy within {timeout}s")


# ---------------------------------------------------------------------------
# Construction / Protocol conformance
# ---------------------------------------------------------------------------


def test_piper_tts_conforms_to_tts_engine_protocol() -> None:
    engine = PiperTTS("/nonexistent/voice.onnx")
    assert isinstance(engine, TTSEngine)


def test_default_voice_id_is_jarvis_persona() -> None:
    """Requirement 11.2: default voice is mature, calm, British-accented."""
    assert DEFAULT_VOICE_ID == "en_GB-alan-medium"
    engine = PiperTTS("/nonexistent/voice.onnx")
    assert engine.voice_id == DEFAULT_VOICE_ID


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, -0.5])
def test_constructor_rejects_non_positive_speaking_rate(bad_rate: float) -> None:
    with pytest.raises(ValueError):
        PiperTTS("/nonexistent/voice.onnx", speaking_rate=bad_rate)


@pytest.mark.parametrize("bad_chunk", [0, -1])
def test_constructor_rejects_non_positive_chunk_frame_samples(bad_chunk: int) -> None:
    with pytest.raises(ValueError):
        PiperTTS("/nonexistent/voice.onnx", chunk_frame_samples=bad_chunk)


def test_initial_is_playing_returns_false() -> None:
    engine = PiperTTS("/nonexistent/voice.onnx")
    assert engine.is_playing() is False


# ---------------------------------------------------------------------------
# Behaviour with fakes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speak_drives_synthesis_and_playback_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentences enqueued via ``speak`` are synthesised and played in FIFO order.

    Verifies the streaming contract from Requirement 12.2: the engine
    accepts text without blocking and the worker emits PCM through the
    audio player in order.
    """
    voice = _FakePiperVoice()
    _install_fake_piper(monkeypatch, voice)

    engine = PiperTTS("/nonexistent/voice.onnx")
    fake_player = _FakeAudioPlayer()

    # Patch the player factory by waiting until the engine creates one,
    # then swap it out. We do this by overriding ``_ensure_started`` to
    # install our fake player after the standard initialisation.
    original_ensure = engine._ensure_started

    async def _ensure_with_fake_player() -> None:
        await original_ensure()
        engine._player = fake_player  # type: ignore[assignment]

    engine._ensure_started = _ensure_with_fake_player  # type: ignore[method-assign]

    await engine.speak("Hello, sir.")
    await engine.speak("Shall we proceed?")

    await _wait_for(lambda: len(fake_player.played) == 2)

    assert voice.calls == ["Hello, sir.", "Shall we proceed?"]
    assert fake_player.played == [[b"Hello, sir."], [b"Shall we proceed?"]]
    assert engine.is_playing() is False

    await engine.aclose()


@pytest.mark.asyncio
async def test_speak_drops_empty_and_whitespace_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice = _FakePiperVoice()
    _install_fake_piper(monkeypatch, voice)

    engine = PiperTTS("/nonexistent/voice.onnx")
    try:
        await engine.speak("")
        await engine.speak("   \n\t  ")
        # Neither call should have started anything.
        assert engine.is_playing() is False
        assert voice.calls == []
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_stop_aborts_playback_and_drains_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 1.7: barge-in stops playback and drops pending text."""
    voice = _FakePiperVoice()
    _install_fake_piper(monkeypatch, voice)

    engine = PiperTTS("/nonexistent/voice.onnx")
    fake_player = _FakeAudioPlayer()
    gate = asyncio.Event()
    fake_player.gate = gate

    original_ensure = engine._ensure_started

    async def _ensure_with_fake_player() -> None:
        await original_ensure()
        engine._player = fake_player  # type: ignore[assignment]

    engine._ensure_started = _ensure_with_fake_player  # type: ignore[method-assign]

    # First sentence will block in the (gated) fake player.
    await engine.speak("First sentence.")
    # Queue several more sentences while the first is held.
    await engine.speak("Second sentence.")
    await engine.speak("Third sentence.")

    # Wait until the first sentence reached the player.
    await _wait_for(fake_player.is_playing)
    assert engine.is_playing() is True

    # Barge-in.
    await engine.stop()

    assert fake_player.stop_called >= 1
    # After stop, queued sentences are dropped — the worker observed the
    # cancellation and returned to the queue, but our drain ensures
    # the queued items are gone.
    assert engine._text_queue.qsize() == 0

    # Release the gate so any in-flight fake task can finish cleanly.
    gate.set()

    # Subsequent ``speak`` after barge-in resumes normal operation.
    fake_player.gate = None
    await engine.speak("Resumed sentence.")
    await _wait_for(lambda: any(b"Resumed sentence." in b"".join(p) for p in fake_player.played))

    await engine.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _FakePiperVoice()
    _install_fake_piper(monkeypatch, voice)

    engine = PiperTTS("/nonexistent/voice.onnx")
    fake_player = _FakeAudioPlayer()

    original_ensure = engine._ensure_started

    async def _ensure_with_fake_player() -> None:
        await original_ensure()
        engine._player = fake_player  # type: ignore[assignment]

    engine._ensure_started = _ensure_with_fake_player  # type: ignore[method-assign]

    await engine.speak("Hello.")
    await _wait_for(lambda: len(fake_player.played) == 1)

    await engine.aclose()
    await engine.aclose()  # second call must be a no-op

    assert fake_player.aclose_called >= 1

    with pytest.raises(RuntimeError):
        await engine.speak("After close.")


@pytest.mark.asyncio
async def test_aclose_without_speak_does_not_open_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing an engine that never spoke must not load piper or open audio.

    Lazy-loading is what allows the module to be imported on hosts with
    neither piper nor PortAudio. ``aclose`` must respect that.
    """
    # Sentinel that flips if anybody tries to load the fake voice.
    loaded = {"value": False}

    fake_module = types.ModuleType("piper")

    class _PiperVoice:
        @staticmethod
        def load(*args: Any, **kwargs: Any) -> Any:
            loaded["value"] = True
            return _FakePiperVoice()

    class _SynthesisConfig:
        def __init__(self, **kwargs: Any) -> None: ...

    fake_module.PiperVoice = _PiperVoice  # type: ignore[attr-defined]
    fake_module.SynthesisConfig = _SynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake_module)

    engine = PiperTTS("/nonexistent/voice.onnx")
    await engine.aclose()
    assert loaded["value"] is False


@pytest.mark.asyncio
async def test_audio_format_uses_voice_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AudioPlayer is configured at the voice's native sample rate."""
    voice = _FakePiperVoice()
    voice.config = _FakeConfig()
    voice.config.sample_rate = 16000  # type: ignore[assignment]
    _install_fake_piper(monkeypatch, voice)

    engine = PiperTTS("/nonexistent/voice.onnx", chunk_frame_samples=512)
    # Replace the real player with a fake immediately *after* construction
    # so the engine's private ``_format`` is computed from the fake voice
    # we control.
    await engine._ensure_started()
    fmt = engine._format
    assert isinstance(fmt, AudioFormat)
    assert fmt.sample_rate_hz == 16000
    assert fmt.frame_samples == 512
    assert fmt.channels == 1
    assert fmt.sample_width == 2

    await engine.aclose()
