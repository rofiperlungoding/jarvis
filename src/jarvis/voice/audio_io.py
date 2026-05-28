"""Audio I/O primitives for the JARVIS voice pipeline.

This module owns the raw audio capture and playback paths described in
``design.md §Concurrency Model`` and the Wake_Word_Detector / TTS_Engine
sections:

* :class:`AudioReframer` — pure-Python ring buffer that adapts arbitrary
  PortAudio callback chunk sizes to the *fixed* frame sizes required by
  downstream consumers. Porcupine wants 512-sample / 16 kHz / 16-bit mono
  frames; silero-vad wants 30 ms (480-sample) frames at the same sample
  rate. The reframer is intentionally free of any ``sounddevice`` import
  so it can be exercised on CI runners without ``libportaudio2`` and so
  unit tests (task 4.2) can drive it directly with synthetic byte
  streams.

* :class:`AudioStream` — async iterator wrapping a
  ``sounddevice.RawInputStream``. The PortAudio callback (which runs on
  a non-asyncio thread) reframes the raw chunks and posts the
  fixed-size frames into a *bounded* :class:`asyncio.Queue`; on overflow
  the oldest pending frame is dropped so realtime capture keeps up
  rather than building unbounded backpressure (Requirement 1.2).

* :class:`AudioPlayer` — cancellable PCM player. ``aplay`` streams the
  configured PCM into a ``RawOutputStream`` in fixed-size chunks; ``stop``
  cancels the in-flight playback task and aborts the stream so the
  device buffer is flushed within the 150 ms barge-in budget
  (Requirement 1.7).

The module is designed so that **importing it never touches** the
``sounddevice``/PortAudio stack. ``sounddevice`` is loaded lazily inside
:func:`_import_sounddevice` only when an actual stream is opened. CI
runners and developer machines without audio drivers therefore can
import every other voice module that depends on this one (e.g.
``wake_word.py``, ``tts/piper.py``) and run pure-logic tests against
:class:`AudioReframer` without any system dependency.

Validates: Requirements 1.2, 1.7
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable
import contextlib
import logging
from types import TracebackType
from typing import Any, Final

logger = logging.getLogger(__name__)

__all__ = [
    "PORCUPINE_FRAME_SAMPLES",
    "PORCUPINE_SAMPLE_RATE_HZ",
    "VAD_FRAME_SAMPLES",
    "AudioFormat",
    "AudioPlayer",
    "AudioReframer",
    "AudioStream",
]


# ---------------------------------------------------------------------------
# Constants — pinned by the wake-word and VAD model contracts.
# ---------------------------------------------------------------------------

#: Porcupine consumes 16 kHz / 16-bit signed PCM mono frames of exactly 512
#: samples (Wake_Word_Detector section of design.md). The reframer is
#: parameterised, but these constants are exported for convenience and to
#: make the shape of the contract explicit at call sites.
PORCUPINE_SAMPLE_RATE_HZ: Final[int] = 16000
PORCUPINE_FRAME_SAMPLES: Final[int] = 512

#: silero-vad operates on 30 ms hops which, at 16 kHz mono, is 480 samples.
VAD_FRAME_SAMPLES: Final[int] = 480

#: 16-bit signed PCM is the lingua franca across Porcupine, silero-vad,
#: faster-whisper, and Piper TTS. Two bytes per sample, mono by default.
_DEFAULT_SAMPLE_WIDTH: Final[int] = 2
_DEFAULT_CHANNELS: Final[int] = 1


# ---------------------------------------------------------------------------
# Lazy ``sounddevice`` import
# ---------------------------------------------------------------------------


def _import_sounddevice() -> Any:
    """Import :mod:`sounddevice` on demand.

    Raised lazily so the module remains importable on hosts where
    ``libportaudio2`` is not installed (CI Linux runners with the audio
    extras stripped, or developer environments that only use the
    pure-logic :class:`AudioReframer`).
    """
    try:
        import sounddevice  # noqa: PLC0415 - lazy import is intentional
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
        # ``OSError`` covers the case where the library is installed but the
        # underlying PortAudio shared object cannot be located at runtime.
        raise RuntimeError(
            "sounddevice/PortAudio is not available; install the audio extras "
            "to enable real microphone/speaker I/O. AudioReframer remains "
            "usable without sounddevice."
        ) from exc
    return sounddevice


# ---------------------------------------------------------------------------
# Audio format dataclass
# ---------------------------------------------------------------------------


class AudioFormat:
    """Describes a PCM byte stream's framing properties.

    The voice pipeline standardises on signed 16-bit little-endian PCM.
    Sample-rate, channel count and frame size are tunable so the same
    primitives serve both 16 kHz capture (wake word / VAD / STT) and the
    higher rates produced by the TTS engine when it streams its synthesis
    output to playback.
    """

    __slots__ = ("channels", "frame_samples", "sample_rate_hz", "sample_width")

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        frame_samples: int,
        channels: int = _DEFAULT_CHANNELS,
        sample_width: int = _DEFAULT_SAMPLE_WIDTH,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if sample_width not in (1, 2, 3, 4):
            raise ValueError("sample_width must be one of {1, 2, 3, 4}")
        self.sample_rate_hz = sample_rate_hz
        self.frame_samples = frame_samples
        self.channels = channels
        self.sample_width = sample_width

    @property
    def frame_bytes(self) -> int:
        """Return the size of a single fixed-size frame in bytes."""
        return self.frame_samples * self.channels * self.sample_width

    @property
    def frame_duration_ms(self) -> float:
        """Return the duration of one fixed-size frame in milliseconds."""
        return 1000.0 * self.frame_samples / self.sample_rate_hz

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AudioFormat(sample_rate_hz={self.sample_rate_hz}, "
            f"frame_samples={self.frame_samples}, channels={self.channels}, "
            f"sample_width={self.sample_width})"
        )


# ---------------------------------------------------------------------------
# AudioReframer — pure logic, no sounddevice imports
# ---------------------------------------------------------------------------


class AudioReframer:
    """Adapt arbitrary callback chunk sizes to fixed-size output frames.

    PortAudio (via ``sounddevice``) delivers PCM bytes in chunks whose
    size is determined by the host audio backend, not by the consumer's
    requirements. Porcupine demands frames of exactly
    :data:`PORCUPINE_FRAME_SAMPLES` samples; silero-vad demands 30 ms
    hops. :class:`AudioReframer` stitches the irregular input together
    into a stream of byte-equal, fixed-size frames.

    Properties (verified by tests/property/test_audio_reframer.py):

    * **Byte preservation** — concatenating every frame yielded by
      :meth:`feed` plus :meth:`flush` reproduces a prefix of every byte
      ever pushed in, in order, with no insertions or substitutions.
      Whatever bytes are not yet a complete frame remain in the internal
      buffer until enough data accumulates.
    * **Frame size invariant** — every frame returned by :meth:`feed` is
      exactly :pyattr:`AudioFormat.frame_bytes` bytes long. :meth:`flush`
      returns at most one (possibly short) tail frame, optionally
      zero-padded on request.
    * **No global state** — each instance is independent. The reframer
      is *not* thread-safe; the audio callback owns it for the lifetime
      of a stream.

    The implementation uses a single :class:`bytearray` as a sliding
    buffer; ``del buf[:n]`` is O(n) but ``n`` is bounded by one frame so
    the per-call work is constant. No allocation occurs on the
    fast-path of equal-sized chunks.
    """

    __slots__ = ("_buf", "_frame_bytes")

    def __init__(self, *, frame_bytes: int) -> None:
        if frame_bytes <= 0:
            raise ValueError("frame_bytes must be positive")
        self._frame_bytes = frame_bytes
        self._buf = bytearray()

    @classmethod
    def for_format(cls, fmt: AudioFormat) -> AudioReframer:
        """Build a reframer matching ``fmt``'s frame byte size."""
        return cls(frame_bytes=fmt.frame_bytes)

    @classmethod
    def for_porcupine(cls) -> AudioReframer:
        """Build a reframer producing 512-sample / 16-bit / mono frames."""
        return cls(frame_bytes=PORCUPINE_FRAME_SAMPLES * _DEFAULT_SAMPLE_WIDTH)

    @classmethod
    def for_vad(cls) -> AudioReframer:
        """Build a reframer producing 30 ms (480-sample) mono frames."""
        return cls(frame_bytes=VAD_FRAME_SAMPLES * _DEFAULT_SAMPLE_WIDTH)

    @property
    def frame_bytes(self) -> int:
        """Return the configured output frame size in bytes."""
        return self._frame_bytes

    @property
    def buffered_bytes(self) -> int:
        """Return the number of input bytes not yet emitted as a frame."""
        return len(self._buf)

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[bytes]:
        """Append ``chunk`` to the buffer and return any complete frames.

        Returns frames as freshly-allocated :class:`bytes` so callers may
        retain or hand them off to other threads / asyncio queues without
        worrying about the reframer mutating them later. The order of the
        returned list matches the order of bytes in the underlying
        stream; callers should consume the list in order.
        """
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("chunk must be a bytes-like object")
        if len(chunk) == 0:
            return []

        # Cheaper than ``self._buf += bytes(chunk)`` for memoryview inputs.
        self._buf.extend(chunk)
        frame_bytes = self._frame_bytes

        # Hot path: exactly one frame fits. Avoid the loop overhead.
        if len(self._buf) < frame_bytes:
            return []
        if len(self._buf) == frame_bytes:
            frame = bytes(self._buf)
            self._buf.clear()
            return [frame]

        frames: list[bytes] = []
        # ``memoryview`` over the buffer lets us slice without copying until
        # the final ``bytes(...)`` materialisation.
        view = memoryview(self._buf)
        offset = 0
        total = len(self._buf)
        while total - offset >= frame_bytes:
            frames.append(bytes(view[offset : offset + frame_bytes]))
            offset += frame_bytes
        # Release the memoryview before mutating the bytearray (CPython
        # raises ``BufferError`` if a view is still active).
        view.release()
        # Drop the consumed prefix; the remainder (< frame_bytes) stays
        # for the next ``feed`` call.
        del self._buf[:offset]
        return frames

    def flush(self, *, pad: bool = False, pad_value: int = 0) -> bytes | None:
        """Drain any leftover bytes.

        By default returns ``None`` when the trailing partial frame is
        shorter than :attr:`frame_bytes` to preserve the frame-size
        invariant. With ``pad=True`` the tail is zero-padded (or padded
        with ``pad_value``) up to a full frame so callers (e.g. the
        VAD's ``speech_end`` boundary handling) can still emit a final
        frame at end-of-utterance.

        After ``flush``, the internal buffer is empty regardless of
        whether a frame was returned.
        """
        if not self._buf:
            return None
        if not 0 <= pad_value <= 255:
            raise ValueError("pad_value must be in [0, 255]")

        tail = bytes(self._buf)
        self._buf.clear()
        if len(tail) == self._frame_bytes:
            return tail
        if not pad:
            return None
        missing = self._frame_bytes - len(tail)
        return tail + bytes([pad_value]) * missing

    def reset(self) -> None:
        """Discard any buffered bytes without emitting them."""
        self._buf.clear()


# ---------------------------------------------------------------------------
# AudioStream — async iterator over reframed microphone frames
# ---------------------------------------------------------------------------


# Default queue depth: enough to absorb a brief scheduling stall (200 ms at
# 30 ms frames ≈ 7 frames) without growing unboundedly. Tuned conservatively;
# the overflow policy drops *oldest* frames so realtime semantics are
# preserved.
_DEFAULT_QUEUE_FRAMES: Final[int] = 16


class AudioStream(AsyncIterable[bytes]):
    """Async iterator yielding reframed PCM frames from a microphone.

    Owns a :class:`sounddevice.RawInputStream` whose callback fires on
    a non-asyncio thread. The callback feeds incoming PortAudio chunks
    into an :class:`AudioReframer` and posts the resulting fixed-size
    frames into a *bounded* :class:`asyncio.Queue` via
    :func:`asyncio.AbstractEventLoop.call_soon_threadsafe`. When the
    queue is full, the oldest pending frame is dropped so realtime
    capture keeps up rather than allowing the producer thread to block
    or the queue to grow without bound (Requirement 1.2 — the wake-word
    pipeline must begin capture within 200 ms of detection, which is
    impossible if the queue is starving the consumer with stale data).

    Use as an async context manager so the underlying PortAudio stream
    is always stopped and closed::

        async with AudioStream(format=fmt) as stream:
            async for frame in stream:
                await wake_word.process(frame)

    The class is reentrant in the sense that ``__aenter__`` may be
    called only once per instance; create a fresh instance to restart
    capture after :meth:`aclose`.
    """

    def __init__(
        self,
        *,
        format: AudioFormat,
        device: int | str | None = None,
        queue_size: int = _DEFAULT_QUEUE_FRAMES,
        loop: asyncio.AbstractEventLoop | None = None,
        on_overflow: object = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._format = format
        self._device = device
        self._queue_size = queue_size
        self._loop_override = loop
        self._reframer = AudioReframer.for_format(format)
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_size)
        self._stream: Any | None = None
        self._opened = False
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_frames = 0
        self._on_overflow = on_overflow  # optional callable(int_dropped_total)

    # -- lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> AudioStream:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Open the underlying PortAudio stream."""
        if self._opened:
            raise RuntimeError("AudioStream already started")
        if self._closed:
            raise RuntimeError("AudioStream is closed; create a new instance")
        self._loop = self._loop_override or asyncio.get_running_loop()
        sd = _import_sounddevice()
        # 16-bit signed PCM is the project-wide default. ``RawInputStream``
        # delivers raw bytes (vs ``InputStream`` which delivers numpy arrays),
        # which matches the byte-oriented contract of :class:`AudioReframer`.
        dtype = self._dtype_for_sample_width(self._format.sample_width)
        self._stream = sd.RawInputStream(
            samplerate=self._format.sample_rate_hz,
            channels=self._format.channels,
            dtype=dtype,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        self._opened = True

    async def aclose(self) -> None:
        """Stop and close the PortAudio stream and unblock any pending consumer."""
        if self._closed:
            return
        self._closed = True
        stream = self._stream
        self._stream = None
        if stream is not None:
            # ``stop`` and ``close`` are blocking I/O calls; offload to a
            # thread so we do not stall the event loop when devices are slow
            # to release buffers.
            await asyncio.to_thread(self._safe_stop_close, stream)
        # Sentinel to wake any waiting ``__anext__``.
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    @staticmethod
    def _safe_stop_close(stream: Any) -> None:
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    # -- properties -----------------------------------------------------------

    @property
    def format(self) -> AudioFormat:
        return self._format

    @property
    def dropped_frames(self) -> int:
        """Total frames discarded due to consumer backpressure."""
        return self._dropped_frames

    # -- iteration ------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if not self._opened and not self._closed:
            # Permit ``async for`` without an explicit ``async with``.
            await self.start()
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    # -- callback hot path ----------------------------------------------------

    def _callback(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """PortAudio callback. Runs on a non-asyncio thread."""
        if status:  # pragma: no cover - hardware-dependent
            # ``status`` is non-empty for over/underflows. Log but do not
            # raise — raising would tear down the stream.
            logger.warning("AudioStream input status: %s", status)
        loop = self._loop
        if loop is None or self._closed:
            return
        # ``indata`` is a CFFI buffer for ``RawInputStream``; ``bytes(...)``
        # snapshots the contents so the reframer can mutate freely.
        try:
            chunk = bytes(indata)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to snapshot input chunk: %s", exc)
            return
        for frame in self._reframer.feed(chunk):
            loop.call_soon_threadsafe(self._enqueue, frame)

    def _enqueue(self, frame: bytes) -> None:
        """Post a frame to the asyncio queue from the loop thread.

        Runs on the event loop thread (scheduled via
        ``call_soon_threadsafe``) so it can interact with the queue
        safely. On overflow, the oldest queued frame is discarded
        before inserting the new one, preserving realtime semantics
        and bounding memory use.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest frame to make room. ``get_nowait`` cannot fail
            # because the queue is full by definition.
            try:
                discarded = self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race-impossible
                discarded = None
            if discarded is not None:
                self._dropped_frames += 1
                if callable(self._on_overflow):
                    with contextlib.suppress(Exception):
                        self._on_overflow(self._dropped_frames)
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(frame)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _dtype_for_sample_width(sample_width: int) -> str:
        # ``sounddevice`` accepts numpy-style dtype strings.
        return {
            1: "int8",
            2: "int16",
            3: "int24",
            4: "int32",
        }[sample_width]


# ---------------------------------------------------------------------------
# AudioPlayer — cancellable PCM playback for barge-in
# ---------------------------------------------------------------------------


# How long ``stop`` waits for the playback task to acknowledge cancellation
# before giving up. Must be < the 150 ms barge-in budget while leaving
# headroom for the actual ``stream.abort()`` syscall on Windows / WASAPI.
_STOP_CANCEL_TIMEOUT_S: Final[float] = 0.10
_STOP_TOTAL_BUDGET_S: Final[float] = 0.15


class AudioPlayer:
    """Cancellable PCM player used for TTS playback and barge-in.

    :meth:`aplay` opens an output stream (lazily, on first call) and
    writes the supplied PCM stream into it in fixed-size chunks. The
    write loop runs as an :class:`asyncio.Task` so :meth:`stop` can
    cancel it cooperatively. Cancellation also calls ``abort()`` on the
    PortAudio stream, which discards any audio still buffered by the
    host, satisfying the 150 ms barge-in budget (Requirement 1.7) far
    more aggressively than ``stop()`` would (the latter waits for the
    buffer to drain).

    The player is single-track: only one playback task may be active at
    a time. A second :meth:`aplay` call is rejected with
    :class:`RuntimeError`. Callers compose multiple utterances by
    awaiting ``aplay`` to completion before starting the next one (the
    Dialog_Manager's TTS sentence accumulator does exactly this).
    """

    def __init__(
        self,
        *,
        format: AudioFormat,
        device: int | str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._format = format
        self._device = device
        self._loop_override = loop
        self._stream: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    # -- lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> AudioPlayer:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop any in-flight playback and release the underlying stream."""
        if self._closed:
            return
        self._closed = True
        await self.stop()
        stream = self._stream
        self._stream = None
        if stream is not None:
            await asyncio.to_thread(self._safe_stop_close, stream)

    @staticmethod
    def _safe_stop_close(stream: Any) -> None:
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    # -- properties -----------------------------------------------------------

    @property
    def format(self) -> AudioFormat:
        return self._format

    def is_playing(self) -> bool:
        """Return ``True`` if a playback task is currently active."""
        task = self._task
        return task is not None and not task.done()

    # -- playback -------------------------------------------------------------

    async def aplay(
        self, pcm: bytes | bytearray | memoryview | Iterable[bytes]
    ) -> None:
        """Play ``pcm`` synchronously (from the caller's perspective).

        ``pcm`` may be a single bytes-like object or any iterable of
        bytes-like chunks (e.g. a generator that yields TTS synthesis
        chunks as they are produced). The coroutine returns when the
        last byte has been handed to PortAudio *or* when :meth:`stop`
        cancels playback. In the cancellation case, the coroutine
        re-raises :class:`asyncio.CancelledError` so callers see the
        same control flow they would expect from any other cancellable
        coroutine.
        """
        if self._closed:
            raise RuntimeError("AudioPlayer is closed")
        # Reject re-entrant playback rather than silently overlapping streams.
        if self.is_playing():
            raise RuntimeError(
                "AudioPlayer.aplay called while another playback is active"
            )

        loop = self._loop_override or asyncio.get_running_loop()
        await self._ensure_stream_open()

        chunks_iter = self._normalise_chunks(pcm)
        task = loop.create_task(self._playback_loop(chunks_iter))
        self._task = task
        try:
            await task
        except asyncio.CancelledError:
            # ``stop`` raised this; surface to the caller so they know
            # playback was interrupted (e.g. barge-in).
            raise
        finally:
            if self._task is task:
                self._task = None

    async def stop(self) -> None:
        """Cancel any in-flight playback within the 150 ms barge-in budget.

        Calls ``abort()`` on the underlying PortAudio stream (discarding
        the device buffer) and cancels the playback task. The combined
        wait for cancellation acknowledgement is bounded by
        :data:`_STOP_TOTAL_BUDGET_S` so the caller's code path (the audio
        capture loop's barge-in handler) does not stall.
        """
        task = self._task
        stream = self._stream

        # Abort the device first; ``abort`` is non-blocking on most hosts and
        # discards the buffered samples, killing audible playback within a
        # few milliseconds even if our write loop is paused inside ``write``.
        if stream is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._safe_abort, stream),
                    timeout=_STOP_TOTAL_BUDGET_S,
                )
            except TimeoutError:  # pragma: no cover - hardware-dependent
                logger.warning(
                    "AudioPlayer.stop: stream abort exceeded barge-in budget"
                )

        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._await_done(task)),
                    timeout=_STOP_CANCEL_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning(
                    "AudioPlayer.stop: playback task did not cancel within %.0f ms",
                    _STOP_CANCEL_TIMEOUT_S * 1000,
                )

    @staticmethod
    def _safe_abort(stream: Any) -> None:
        with contextlib.suppress(Exception):
            stream.abort()

    @staticmethod
    async def _await_done(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(BaseException):
            await task

    # -- internals ------------------------------------------------------------

    async def _ensure_stream_open(self) -> None:
        if self._stream is not None:
            return
        async with self._lock:
            # Double-checked locking: another caller may have opened the
            # stream while we were awaiting the lock.
            if self._stream is not None:
                return  # type: ignore[unreachable]
            sd = _import_sounddevice()
            dtype = AudioStream._dtype_for_sample_width(self._format.sample_width)
            stream = sd.RawOutputStream(
                samplerate=self._format.sample_rate_hz,
                channels=self._format.channels,
                dtype=dtype,
                device=self._device,
            )
            stream.start()
            self._stream = stream

    async def _playback_loop(self, chunks: Iterable[bytes]) -> None:
        """Write ``chunks`` into the output stream until exhausted or cancelled."""
        stream = self._stream
        if stream is None:  # pragma: no cover - defensive
            return
        frame_bytes = self._format.frame_bytes
        try:
            for chunk in chunks:
                if not chunk:
                    continue
                # Write in frame-aligned slices. ``RawOutputStream.write``
                # blocks when the device buffer is full, so we offload to a
                # thread and yield to the loop between slices to keep ``stop``
                # responsive.
                offset = 0
                view = memoryview(
                    chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk)
                )
                length = len(view)
                while offset < length:
                    end = min(offset + frame_bytes, length)
                    await asyncio.to_thread(stream.write, bytes(view[offset:end]))
                    offset = end
                    # Give the loop a chance to deliver a cancellation.
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Re-raise so the task transitions to the cancelled state.
            raise

    @staticmethod
    def _normalise_chunks(
        pcm: bytes | bytearray | memoryview | Iterable[bytes],
    ) -> Iterable[bytes]:
        if isinstance(pcm, (bytes, bytearray)):
            return [bytes(pcm)]
        if isinstance(pcm, memoryview):
            return [bytes(pcm)]
        # Iterable of chunks; coerce each to ``bytes`` so the playback loop
        # can pass them straight to ``stream.write``.
        return (bytes(c) for c in pcm)
