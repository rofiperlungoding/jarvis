"""Silero-backed voice activity detection for the JARVIS voice pipeline.

This module owns the VAD layer described in ``design.md §STT_Engine`` and
the second item of the wake-to-response sequence diagram in ``design.md
§Concurrency Model``. Its sole responsibility is to convert a stream of
fixed-size 16 kHz / 16-bit / mono PCM frames produced by
:class:`jarvis.voice.audio_io.AudioReframer` into ``speech_start`` /
``speech_end`` events that the audio capture loop uses to bracket
utterances and to trigger barge-in.

Two requirements drive the design here:

* **Requirement 1.3** — *"WHEN the user finishes speaking, as determined
  by a voice activity detector with a 700 millisecond trailing-silence
  threshold, THE STT_Engine SHALL produce a Transcript of the captured
  audio."* :class:`SileroVAD` enforces this by emitting
  :pyattr:`VADEventKind.SPEECH_END` only after a configurable run of
  silent frames whose cumulative duration is at least
  :pyattr:`SileroVAD.trailing_silence_ms` (default 700 ms).
* **Requirement 1.7** — *"WHERE the user has enabled barge-in, WHEN the
  user speaks while TTS_Engine is playing, THE Voice_Pipeline SHALL stop
  playback within 150 milliseconds and capture the new utterance."*
  :pyattr:`VADEventKind.SPEECH_START` is fired on the first frame whose
  speech probability crosses :pyattr:`SileroVAD.speech_start_threshold`,
  giving the capture loop the earliest possible signal to call
  :meth:`jarvis.voice.audio_io.AudioPlayer.stop`.

Design notes:

* The class is **frame-rate agnostic**: it derives event timing from the
  configured ``frame_samples`` / ``sample_rate_hz`` so callers running at
  the project default of 30 ms / 480 samples and callers running at the
  silero-native 32 ms / 512 samples both get correct trailing-silence
  arithmetic without code changes.
* The silero-vad model itself is **loaded lazily** on first use so this
  module imports cleanly on CI runners where ``silero-vad`` and ``torch``
  are not installed (every other voice module that depends on this one,
  e.g. ``faster_whisper.py``, can therefore be exercised without the
  audio extras). For unit tests, a deterministic ``probability_fn`` can
  be injected directly.
* Hysteresis is supported via an optional ``speech_end_threshold`` lower
  than the start threshold, matching the silero documentation's
  recommendation against rapid start/stop oscillation on borderline
  audio. The default is the same threshold for both transitions, which
  is what the requirements call for.

Validates: Requirements 1.3, 1.7
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
import contextlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Final, Protocol, runtime_checkable

from jarvis.utils.time_source import SystemTimeSource, TimeSource
from jarvis.voice.audio_io import (
    PORCUPINE_SAMPLE_RATE_HZ,
    VAD_FRAME_SAMPLES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SileroVAD",
    "SpeechProbabilityFn",
    "VADEvent",
    "VADEventKind",
    "load_default_silero_probability_fn",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 16-bit signed PCM is the project-wide capture format. Two bytes per
#: sample, mono. Defined here as a private constant rather than re-imported
#: from ``audio_io`` (which keeps it private too) to keep this module's
#: public surface focused on VAD concerns.
_DEFAULT_SAMPLE_WIDTH: Final[int] = 2
_DEFAULT_CHANNELS: Final[int] = 1

#: Silero v5 at 16 kHz expects exactly 512-sample chunks per inference
#: call. The default probability function therefore re-buffers incoming
#: frames into 512-sample windows before invoking the model. This is a
#: silero-specific detail; the :class:`SileroVAD` state machine itself
#: is agnostic to chunk size and runs at whatever cadence the caller
#: feeds it.
_SILERO_NATIVE_CHUNK_SAMPLES: Final[int] = 512


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class VADEventKind(Enum):
    """Distinguishable VAD state transitions.

    The audio capture loop branches on this value — ``SPEECH_START``
    triggers utterance buffering and barge-in; ``SPEECH_END`` triggers
    handoff to the STT engine.
    """

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADEvent:
    """A single ``speech_start`` or ``speech_end`` transition.

    Attributes:
        kind: Which transition occurred.
        frame_index: Zero-based index of the frame that produced the
            transition. For ``SPEECH_START`` this is the frame whose
            speech probability first crossed
            :pyattr:`SileroVAD.speech_start_threshold`. For
            ``SPEECH_END`` this is the frame at which the running
            silence accumulator reached :pyattr:`SileroVAD.trailing_silence_ms`.
        elapsed_ms: Cumulative milliseconds of audio processed since
            this VAD instance was constructed (or since the last
            :meth:`SileroVAD.reset`). Computed from the frame cadence so
            it does not drift when the host event loop is slow.
        occurred_at: Timezone-aware UTC timestamp from the configured
            :class:`~jarvis.utils.time_source.TimeSource`. Useful for
            audit logs and for the latency budget instrumentation
            (Requirement 12.1) that lives downstream.
        probability: The speech probability returned by the model on the
            frame that produced this transition. ``0.0 <= p <= 1.0``.
    """

    kind: VADEventKind
    frame_index: int
    elapsed_ms: float
    occurred_at: datetime
    probability: float


# ---------------------------------------------------------------------------
# Probability function protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SpeechProbabilityFn(Protocol):
    """Maps a single PCM frame to a speech probability in ``[0.0, 1.0]``.

    Concrete implementations may be backed by silero-vad (the default,
    via :func:`load_default_silero_probability_fn`), webrtcvad, a
    handcrafted threshold for testing, or any other model. The function
    is called once per frame fed into :meth:`SileroVAD.process`; it is
    expected to be cheap (silero v5 is sub-millisecond on CPU for a
    32 ms chunk) so the realtime audio loop is not stalled.

    Implementations MUST be safe to call from a single thread; they are
    NOT required to be reentrant or thread-safe.
    """

    def __call__(self, frame: bytes) -> float:
        ...


# ---------------------------------------------------------------------------
# Default silero-vad backed probability function
# ---------------------------------------------------------------------------


def load_default_silero_probability_fn(
    *,
    sample_rate_hz: int = PORCUPINE_SAMPLE_RATE_HZ,
) -> SpeechProbabilityFn:
    """Build a :class:`SpeechProbabilityFn` backed by the silero-vad model.

    The actual ``silero-vad`` and ``torch`` imports happen here, on
    first call, so that importing :mod:`jarvis.voice.vad` itself does
    not pull in the audio extras. This keeps CI runners that omit the
    optional voice dependencies importable.

    The returned callable accepts arbitrary-size 16-bit signed PCM
    frames and internally re-buffers them into the 512-sample chunks
    required by silero v5 at 16 kHz. Until the first 512-sample chunk
    is filled, the callable returns ``0.0`` (treated as silence by the
    state machine), which is the conservative default — silence cannot
    cause spurious wake-up.
    """
    if sample_rate_hz != PORCUPINE_SAMPLE_RATE_HZ:
        # Silero v5 supports 8 kHz with a 256-sample chunk and 16 kHz
        # with a 512-sample chunk; the project standardises on 16 kHz
        # everywhere so anything else is almost certainly a config bug.
        raise ValueError(
            "Default silero probability fn supports 16 kHz only; "
            f"got sample_rate_hz={sample_rate_hz}. Inject a custom "
            "SpeechProbabilityFn for non-standard rates."
        )

    try:
        # Lazy imports — these are heavy and optional. Both ``torch`` and
        # ``silero_vad`` ship as transitive deps of the ``silero-vad``
        # package; ``torch`` is needed to build the input tensor.
        from silero_vad import load_silero_vad  # noqa: PLC0415
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "silero-vad / torch are not installed. Install the audio "
            "extras (pip install silero-vad torch) or inject a custom "
            "SpeechProbabilityFn into SileroVAD for tests."
        ) from exc

    model = load_silero_vad()

    # Mutable closure state — a small accumulator that holds the trailing
    # samples not yet fed to the model. ``last_prob`` carries the most
    # recent inference result so callers receive a stable probability on
    # every frame, even when the current frame did not finish a 512-sample
    # window.
    accumulator = bytearray()
    chunk_bytes = _SILERO_NATIVE_CHUNK_SAMPLES * _DEFAULT_SAMPLE_WIDTH
    last_prob = 0.0

    def _probability(frame: bytes) -> float:
        nonlocal last_prob
        if not frame:
            return last_prob
        accumulator.extend(frame)
        # Drain as many full 512-sample chunks as possible. ``while`` covers
        # the (rare) case where a single fed frame is larger than one
        # silero chunk — for example if the caller decides to use 1024
        # samples per frame for batching.
        while len(accumulator) >= chunk_bytes:
            chunk = bytes(accumulator[:chunk_bytes])
            del accumulator[:chunk_bytes]
            tensor = torch.frombuffer(chunk, dtype=torch.int16).to(torch.float32)
            tensor = tensor / 32768.0  # int16 -> [-1.0, 1.0]
            with torch.no_grad():
                prob_tensor: Any = model(tensor, sample_rate_hz)
            try:
                last_prob = float(prob_tensor.item())
            except AttributeError:  # pragma: no cover - torch shape variants
                last_prob = float(prob_tensor)
        return last_prob

    return _probability


# ---------------------------------------------------------------------------
# SileroVAD state machine
# ---------------------------------------------------------------------------


class SileroVAD:
    """Voice activity state machine driven by a speech probability function.

    Frames are pushed into :meth:`process` one at a time; each call
    returns the list of state transitions (zero, one, or in pathological
    cases two) that the frame produced. The cadence of incoming frames
    is governed by the caller (typically :class:`AudioStream`'s 30 ms
    reframer); this class never blocks on I/O.

    State machine:

    .. code-block:: text

                        prob >= speech_start_threshold
                ┌────────────────────────────────────────┐
                │                                        ▼
        ┌──────────────┐                       ┌─────────────────┐
        │   SILENT     │                       │   SPEAKING      │
        │ (idle / wait │                       │ (capturing)     │
        │  for speech) │ ◀───────────────────  │                 │
        └──────────────┘   silence run >=      └─────────────────┘
                ▲           trailing_silence_ms       │
                │                                     │
                │   prob < speech_end_threshold       │
                │   (silent_run_ms increments)        │
                │                                     │
                └─────────────────────────────────────┘

    On each frame:

    * If the state is ``SILENT`` and the probability is at or above
      :pyattr:`speech_start_threshold`, the machine transitions to
      ``SPEAKING`` and emits :pyattr:`VADEventKind.SPEECH_START`.
    * If the state is ``SPEAKING`` and the probability is at or above
      :pyattr:`speech_end_threshold`, the silence accumulator is reset
      to zero (the speaker is still talking).
    * If the state is ``SPEAKING`` and the probability is below
      :pyattr:`speech_end_threshold`, the silence accumulator advances
      by ``frame_duration_ms``. When it crosses
      :pyattr:`trailing_silence_ms`, the machine transitions to
      ``SILENT`` and emits :pyattr:`VADEventKind.SPEECH_END`.

    Note that the threshold comparisons use ``>=`` for start and ``<``
    for end so a probability exactly equal to the start threshold begins
    speech, and a probability exactly equal to the end threshold keeps
    speech going. This matches the silero reference implementation.

    The state machine is not thread-safe. The audio capture loop owns
    the instance for the lifetime of a stream.
    """

    __slots__ = (
        "_elapsed_ms",
        "_frame_count",
        "_frame_duration_ms",
        "_frame_size_bytes",
        "_probability_fn",
        "_silent_run_ms",
        "_speaking",
        "_speech_end_threshold",
        "_speech_start_threshold",
        "_time_source",
        "_trailing_silence_ms",
    )

    def __init__(
        self,
        *,
        trailing_silence_ms: int = 700,
        speech_start_threshold: float = 0.5,
        speech_end_threshold: float | None = None,
        sample_rate_hz: int = PORCUPINE_SAMPLE_RATE_HZ,
        frame_samples: int = VAD_FRAME_SAMPLES,
        sample_width: int = _DEFAULT_SAMPLE_WIDTH,
        channels: int = _DEFAULT_CHANNELS,
        probability_fn: SpeechProbabilityFn | None = None,
        time_source: TimeSource | None = None,
    ) -> None:
        """Construct a VAD state machine.

        Parameters:
            trailing_silence_ms: Minimum cumulative silence required to
                emit ``SPEECH_END`` after a ``SPEECH_START``. Defaults
                to 700 ms per Requirement 1.3. Must be positive and an
                integer multiple-or-greater of one frame duration; the
                check is implicit (the accumulator advances by frame
                duration each silent frame, so any positive value is
                accepted, but values less than one frame duration will
                effectively round up to one frame).
            speech_start_threshold: Probability at or above which the
                machine transitions ``SILENT -> SPEAKING``. Defaults to
                ``0.5``. Must be in ``[0.0, 1.0]``.
            speech_end_threshold: Probability below which silent frames
                accumulate toward the trailing-silence budget. Defaults
                to ``speech_start_threshold`` (no hysteresis); set lower
                (e.g. ``0.35``) to reduce oscillation on borderline
                audio.
            sample_rate_hz: Sample rate of incoming frames. Defaults to
                16 kHz to match the project's capture format.
            frame_samples: Number of samples per frame. Defaults to 480
                (30 ms at 16 kHz) to match
                :data:`jarvis.voice.audio_io.VAD_FRAME_SAMPLES`.
            sample_width: Bytes per sample. Defaults to 2 (16-bit).
            channels: Number of audio channels. Defaults to 1 (mono).
            probability_fn: Speech probability backend. Defaults to
                :func:`load_default_silero_probability_fn` which lazy-
                loads silero-vad. Tests inject a deterministic callable.
            time_source: Clock used for ``occurred_at`` timestamps on
                emitted events. Defaults to :class:`SystemTimeSource`;
                tests inject :class:`FakeTimeSource` for determinism.
        """
        if trailing_silence_ms <= 0:
            raise ValueError(
                "trailing_silence_ms must be positive; "
                f"got {trailing_silence_ms!r}"
            )
        if not 0.0 <= speech_start_threshold <= 1.0:
            raise ValueError(
                "speech_start_threshold must be in [0.0, 1.0]; "
                f"got {speech_start_threshold!r}"
            )
        if speech_end_threshold is None:
            speech_end_threshold = speech_start_threshold
        if not 0.0 <= speech_end_threshold <= 1.0:
            raise ValueError(
                "speech_end_threshold must be in [0.0, 1.0]; "
                f"got {speech_end_threshold!r}"
            )
        if speech_end_threshold > speech_start_threshold:
            raise ValueError(
                "speech_end_threshold must be <= speech_start_threshold "
                f"({speech_end_threshold!r} > {speech_start_threshold!r}); "
                "hysteresis is asymmetric, the end threshold should be "
                "lower (or equal) to avoid spurious oscillation."
            )
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if sample_width <= 0:
            raise ValueError("sample_width must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._trailing_silence_ms = trailing_silence_ms
        self._speech_start_threshold = speech_start_threshold
        self._speech_end_threshold = speech_end_threshold
        self._frame_duration_ms = 1000.0 * frame_samples / sample_rate_hz
        self._frame_size_bytes = frame_samples * sample_width * channels

        # Defer the heavy default until first use; caller-supplied
        # probability functions are stored as-is.
        self._probability_fn: SpeechProbabilityFn | None = probability_fn
        self._time_source = time_source or SystemTimeSource()

        # Mutable runtime state.
        self._speaking = False
        self._silent_run_ms = 0.0
        self._frame_count = 0
        self._elapsed_ms = 0.0

    # -- properties -----------------------------------------------------------

    @property
    def trailing_silence_ms(self) -> int:
        """Configured trailing-silence threshold in milliseconds."""
        return self._trailing_silence_ms

    @property
    def speech_start_threshold(self) -> float:
        """Probability threshold for the ``SILENT -> SPEAKING`` transition."""
        return self._speech_start_threshold

    @property
    def speech_end_threshold(self) -> float:
        """Probability threshold below which silence accumulates."""
        return self._speech_end_threshold

    @property
    def frame_duration_ms(self) -> float:
        """Duration of one input frame in milliseconds."""
        return self._frame_duration_ms

    @property
    def frame_size_bytes(self) -> int:
        """Expected size, in bytes, of each frame fed into :meth:`process`."""
        return self._frame_size_bytes

    @property
    def is_speaking(self) -> bool:
        """``True`` while the machine is in the ``SPEAKING`` state."""
        return self._speaking

    @property
    def silent_run_ms(self) -> float:
        """Cumulative ms of silence observed in the current ``SPEAKING`` run.

        Resets to zero on every frame whose probability is at or above
        :pyattr:`speech_end_threshold`, and on every state transition.
        Exposed primarily for diagnostics and unit tests.
        """
        return self._silent_run_ms

    @property
    def elapsed_ms(self) -> float:
        """Total ms of audio processed since construction or last :meth:`reset`."""
        return self._elapsed_ms

    @property
    def frame_count(self) -> int:
        """Total number of frames processed since construction or last :meth:`reset`."""
        return self._frame_count

    # -- core API -------------------------------------------------------------

    def process(self, frame: bytes) -> list[VADEvent]:
        """Advance the state machine by one frame and return any transitions.

        ``frame`` must be a bytes-like object whose length equals
        :pyattr:`frame_size_bytes`. Empty frames are rejected because
        they would advance state with no audio content. Oversized or
        undersized frames are rejected because they would corrupt the
        per-frame timing arithmetic; callers should drive an
        :class:`~jarvis.voice.audio_io.AudioReframer` upstream so frames
        always arrive at the configured size.

        The list of returned events is in the order the transitions
        occurred. In normal operation a frame produces zero or one
        event; the list shape exists so future extensions (e.g.
        rapid speech-then-silence within a single low-rate frame) can
        return more without breaking callers.
        """
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise TypeError("frame must be a bytes-like object")
        frame_bytes_view = bytes(frame)
        if len(frame_bytes_view) != self._frame_size_bytes:
            raise ValueError(
                f"VAD frame size mismatch: expected {self._frame_size_bytes} "
                f"bytes, got {len(frame_bytes_view)}. Reframe upstream so "
                "the VAD always sees fixed-size frames."
            )

        probability = self._invoke_probability(frame_bytes_view)
        if not 0.0 <= probability <= 1.0:
            # Defensive — a misbehaving probability backend must not be
            # able to drive the state machine into incoherent states.
            logger.warning(
                "Speech probability out of range: %r (clamping to [0,1])",
                probability,
            )
            probability = max(0.0, min(1.0, probability))

        # Advance the cadence counters BEFORE state evaluation so the
        # event's ``frame_index`` and ``elapsed_ms`` correctly identify
        # the frame that produced the transition.
        self._frame_count += 1
        self._elapsed_ms += self._frame_duration_ms

        events: list[VADEvent] = []

        if not self._speaking:
            if probability >= self._speech_start_threshold:
                self._speaking = True
                self._silent_run_ms = 0.0
                events.append(self._make_event(VADEventKind.SPEECH_START, probability))
        elif probability >= self._speech_end_threshold:
            # Voice is still active; reset the trailing-silence
            # accumulator so brief intra-utterance pauses do not
            # prematurely terminate the capture.
            self._silent_run_ms = 0.0
        else:
            self._silent_run_ms += self._frame_duration_ms
            if self._silent_run_ms >= self._trailing_silence_ms:
                self._speaking = False
                self._silent_run_ms = 0.0
                events.append(self._make_event(VADEventKind.SPEECH_END, probability))

        return events

    def flush(self) -> list[VADEvent]:
        """Force a ``SPEECH_END`` if currently in the ``SPEAKING`` state.

        Called by the audio capture loop on stream shutdown so an
        in-flight utterance is delivered to the STT engine instead of
        silently dropped. Returns an empty list when the machine is
        already in ``SILENT``.

        The synthetic ``SPEECH_END`` carries ``probability=0.0`` and the
        last advanced frame index / elapsed ms; no new frame is consumed.
        """
        if not self._speaking:
            return []
        self._speaking = False
        self._silent_run_ms = 0.0
        return [self._make_event(VADEventKind.SPEECH_END, probability=0.0)]

    def reset(self) -> None:
        """Discard all state and counters.

        The probability backend's own internal state (e.g., silero's
        residual sample buffer) is NOT reset by this call — that is the
        backend's concern. Tests using a stateless probability function
        can rely on :meth:`reset` alone to start from a clean slate.
        """
        self._speaking = False
        self._silent_run_ms = 0.0
        self._frame_count = 0
        self._elapsed_ms = 0.0

    # -- async iterator helper -----------------------------------------------

    async def iter_events(
        self,
        frames: AsyncIterable[bytes],
        *,
        on_speech_start: Callable[[VADEvent], Awaitable[None]] | None = None,
        on_speech_end: Callable[[VADEvent], Awaitable[None]] | None = None,
        flush_on_close: bool = True,
    ) -> AsyncIterator[VADEvent]:
        """Drive the state machine from an async stream of frames.

        Designed to be composed with :class:`jarvis.voice.audio_io.AudioStream`::

            vad = SileroVAD()
            async with AudioStream(format=fmt) as stream:
                async for event in vad.iter_events(stream):
                    if event.kind is VADEventKind.SPEECH_START:
                        await player.stop()  # barge-in (Requirement 1.7)
                    elif event.kind is VADEventKind.SPEECH_END:
                        await stt.transcribe(captured_buffer, language="en")

        Optional ``on_speech_start`` / ``on_speech_end`` async callbacks
        are awaited inline before the corresponding event is yielded to
        the iterator. They satisfy the "callbacks or async iterator"
        wording in the design without forcing callers to choose one
        style — both work, and the callback runs first so it can do
        latency-critical work (e.g. ``AudioPlayer.stop()`` for barge-in)
        without waiting on the consumer to drain the iterator.

        Cancellation propagates: if the consumer's ``async for`` is
        cancelled, the underlying frame iterator is closed via
        ``aclose`` (when supported) on the way out. When ``flush_on_close``
        is true (the default) and the machine is still ``SPEAKING`` at
        end-of-stream, a synthetic ``SPEECH_END`` is yielded last so the
        in-flight utterance is not dropped.
        """
        try:
            async for frame in frames:
                events = self.process(frame)
                for event in events:
                    if (
                        event.kind is VADEventKind.SPEECH_START
                        and on_speech_start is not None
                    ):
                        await on_speech_start(event)
                    elif (
                        event.kind is VADEventKind.SPEECH_END
                        and on_speech_end is not None
                    ):
                        await on_speech_end(event)
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            # Surface cancellation / generator close, but make sure the
            # upstream frame iterator is closed first so the producer
            # side does not leak. We deliberately do NOT yield a
            # synthetic SPEECH_END here: yielding from an exception
            # path would raise ``RuntimeError("async generator ignored
            # GeneratorExit")`` in Python 3.11+.
            aclose = getattr(frames, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            raise

        # Normal completion of the source iterator: drain any in-flight
        # utterance so the caller sees a final SPEECH_END instead of a
        # silently truncated capture.
        if flush_on_close:
            for event in self.flush():
                if (
                    event.kind is VADEventKind.SPEECH_END
                    and on_speech_end is not None
                ):
                    with contextlib.suppress(Exception):
                        await on_speech_end(event)
                yield event

    # -- internals ------------------------------------------------------------

    def _invoke_probability(self, frame: bytes) -> float:
        """Call the probability backend, lazy-loading the default on first use."""
        fn = self._probability_fn
        if fn is None:
            fn = load_default_silero_probability_fn()
            self._probability_fn = fn
        return float(fn(frame))

    def _make_event(self, kind: VADEventKind, probability: float) -> VADEvent:
        return VADEvent(
            kind=kind,
            # ``frame_count`` was incremented before this method runs, so
            # the *index* of the producing frame is one less than the count.
            frame_index=self._frame_count - 1,
            elapsed_ms=self._elapsed_ms,
            occurred_at=self._time_source.now(),
            probability=probability,
        )
