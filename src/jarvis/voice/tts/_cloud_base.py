"""Shared scaffolding for cloud TTS adapters.

This module is an *internal implementation detail* of the cloud TTS
adapters under :mod:`jarvis.voice.tts`. The two public engines
(:class:`~jarvis.voice.tts.elevenlabs.ElevenLabsTTS` and
:class:`~jarvis.voice.tts.openai_tts.OpenAITTS`) extend
:class:`_CloudTTSEngine` here so they can share the queue + worker loop
that turns the project's enqueue-and-stream :class:`TTSEngine` Protocol
into a sequence of provider-specific HTTP calls.

The base class implements the engine lifecycle and barge-in semantics
documented on :class:`~jarvis.voice.tts.base.TTSEngine`:

* :meth:`speak` is non-blocking — it appends the sentence to an internal
  :class:`asyncio.Queue` and (re)spawns a worker task on demand. This
  matches Requirement 12.2 / 19.5 where the Dialog_Manager pushes
  finished sentences as soon as the LLM stream emits them, expecting
  the call to return promptly.
* :meth:`stop` cancels the worker, aborts the underlying audio device
  via :class:`~jarvis.voice.audio_io.AudioPlayer.stop`, and drops every
  queued sentence. This is the barge-in path (Requirement 1.7) and is
  bounded by the player's 150 ms abort budget.
* :meth:`is_playing` is a synchronous, side-effect-free probe.
* :meth:`aclose` is idempotent.

Concrete subclasses implement :meth:`_synthesize` to produce raw PCM
matching their declared :class:`~jarvis.voice.audio_io.AudioFormat`.
The base class then routes that PCM through a single per-engine
:class:`~jarvis.voice.audio_io.AudioPlayer` so playback control
(stop, is_playing, aclose) is uniform across providers.

Validates: Requirements 1.7, 11.2, 12.2, 19.5
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Final

from jarvis.voice.audio_io import AudioFormat, AudioPlayer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.voice.tts.base import TTSEngine

logger = logging.getLogger(__name__)


# Maximum number of unspoken sentences to retain. Cloud TTS providers are
# typically faster than wall-clock playback, so the queue rarely grows
# beyond a single utterance; the cap exists to bound memory if the user
# barges in repeatedly without the worker getting a chance to drain.
_DEFAULT_QUEUE_DEPTH: Final[int] = 64


class _CloudTTSEngine(abc.ABC):
    """Abstract base for cloud TTS adapters using HTTP synthesis.

    The class is *not* exported from ``jarvis.voice.tts``. It exists so
    the public adapters can share queue, worker, and playback management
    while keeping their wire-format code (request bodies, response
    parsing, error mapping) self-contained.

    Subclasses must:

    1. Pick a fixed :class:`AudioFormat` matching the PCM bytes that
       :meth:`_synthesize` will return. Using the provider's *native*
       PCM rate avoids resampling, so ElevenLabs uses 16 kHz and OpenAI
       uses 24 kHz by default.
    2. Implement :meth:`_synthesize` as an ``async`` method returning a
       single ``bytes`` blob of PCM. Streaming chunk-by-chunk into the
       player is possible but adds complexity for negligible
       latency benefit on per-sentence utterances; we buffer one
       sentence at a time and rely on sentence-boundary streaming
       (Requirement 12.2) for end-to-end responsiveness.
    3. Optionally override :meth:`_close_provider` to release any
       provider-specific resources (e.g. an :mod:`openai` async client).
    """

    # Subclasses set this so error messages and logs identify the
    # provider without leaking secrets.
    _provider_name: str = "cloud-tts"

    def __init__(
        self,
        *,
        audio_format: AudioFormat,
        device: int | str | None = None,
        queue_depth: int = _DEFAULT_QUEUE_DEPTH,
    ) -> None:
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self._audio_format = audio_format
        self._player = AudioPlayer(format=audio_format, device=device)
        # Accepts ``str`` for utterances and ``None`` as a shutdown sentinel.
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=queue_depth)
        self._worker_task: asyncio.Task[None] | None = None
        # Tracks whether a ``_synthesize`` call is currently in flight, so
        # ``is_playing`` reports True even during the brief window between
        # dequeue and first PCM hand-off to the player.
        self._synthesizing: bool = False
        self._closed: bool = False

    # -- async context manager ------------------------------------------------

    async def __aenter__(self) -> _CloudTTSEngine:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- public engine API ----------------------------------------------------

    async def speak(self, text: str) -> None:
        """Enqueue ``text`` for synthesis and playback.

        The call returns once the sentence has been accepted onto the
        internal queue; it does *not* await synthesis or playback. This
        matches the streaming sentence-boundary contract from Requirement
        12.2 — each sentence emitted by the Dialog_Manager's
        :class:`~jarvis.voice.tts.base.SentenceAccumulator` is forwarded
        immediately and rendered in order by the worker.

        Empty / whitespace-only ``text`` is silently ignored. Callers
        relying on speak-as-fence semantics should pass a non-empty
        sentinel; the project's sentence accumulator never produces
        whitespace-only strings.
        """
        if self._closed:
            raise RuntimeError(
                f"{self._provider_name}: speak() after aclose() is not supported"
            )
        if not text or not text.strip():
            return
        await self._queue.put(text)
        self._ensure_worker_running()

    async def stop(self) -> None:
        """Cancel any in-flight synthesis / playback and drop queued sentences.

        Implements barge-in (Requirement 1.7): aborts the audio device
        immediately via :meth:`AudioPlayer.stop`, drains every queued
        sentence (so a single ``stop`` cancels the *whole* assistant
        turn, not just the currently-playing one), and cancels the
        worker task so an in-flight HTTP synthesis is torn down rather
        than allowed to complete and play after the user has spoken.

        After ``stop`` returns, :meth:`is_playing` reports ``False``
        within the player's 150 ms abort budget (modulo any racing
        playback start that has not yet placed bytes on the device).
        Subsequent :meth:`speak` calls re-spawn a fresh worker.
        """
        # 1. Drop pending sentences so they are *not* spoken once the
        #    worker is restarted by a future ``speak``.
        self._drain_queue()

        # 2. Abort the audio device so audible playback ceases ASAP.
        with contextlib.suppress(Exception):
            await self._player.stop()

        # 3. Cancel the worker. Any in-flight HTTP request inside
        #    ``_synthesize`` is interrupted by the cancellation
        #    propagating through ``await``.
        task = self._worker_task
        self._worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        # In-flight synth flag is reset by the worker's ``finally`` block;
        # if cancellation interrupted that, force a reset so subsequent
        # ``is_playing`` calls reflect reality.
        self._synthesizing = False

    def is_playing(self) -> bool:
        """Return ``True`` while the engine is actively rendering audio.

        Reports ``True`` whenever any of the following holds:

        * The :class:`AudioPlayer` is currently writing PCM to the
          device.
        * A ``_synthesize`` call is in flight (HTTP request to the
          provider has been issued and not yet returned a complete
          buffer).
        * Sentences are queued and waiting for the worker.

        The probe is synchronous and non-blocking so the audio capture
        loop and Reminder_Service can poll it without scheduling
        overhead.
        """
        if self._player.is_playing():
            return True
        if self._synthesizing:
            return True
        return not self._queue.empty()

    async def aclose(self) -> None:
        """Release resources. Idempotent.

        Cancels the worker, closes the audio player, and gives subclasses
        a hook (:meth:`_close_provider`) to release their HTTP client.
        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        # Drain the queue so the worker (if it gets one more iteration)
        # exits promptly rather than synthesizing a sentence that will
        # never play.
        self._drain_queue()

        # Tear down playback first; the player's own aclose handles its
        # internal task cancellation.
        with contextlib.suppress(Exception):
            await self._player.aclose()

        task = self._worker_task
        self._worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

        # Provider-specific cleanup (HTTP client, vendor SDK).
        with contextlib.suppress(Exception):
            await self._close_provider()

    # -- subclass hooks -------------------------------------------------------

    @abc.abstractmethod
    async def _synthesize(self, text: str) -> bytes:
        """Produce the raw PCM bytes for ``text``.

        Subclasses MUST return PCM whose framing matches the
        :class:`AudioFormat` passed into ``__init__`` — i.e. the same
        sample rate, channel count, and sample width. The base class
        does not perform any resampling.

        Implementations should not catch :class:`asyncio.CancelledError`
        — it indicates a barge-in or shutdown and must propagate so the
        worker exits cleanly.
        """
        raise NotImplementedError

    async def _close_provider(self) -> None:
        """Release provider-specific resources. Default: no-op.

        Override to close vendor SDK clients (e.g. ``openai.AsyncOpenAI``)
        or shared :class:`httpx.AsyncClient` instances.
        """
        return None

    # -- worker loop ----------------------------------------------------------

    def _ensure_worker_running(self) -> None:
        """Spawn a worker task if one is not already active.

        The worker is single-shot per ``stop()`` cycle: ``stop`` cancels
        the worker, and the next :meth:`speak` re-spawns a fresh one
        here. This makes the cancellation semantics straightforward —
        we never have to thread a "should I keep going?" boolean through
        the synth/playback path; cancellation is the explicit signal.
        """
        if self._closed:
            raise RuntimeError(
                f"{self._provider_name}: cannot start worker after aclose()"
            )
        task = self._worker_task
        if task is None or task.done():
            self._worker_task = asyncio.create_task(
                self._run_worker(),
                name=f"{self._provider_name}-tts-worker",
            )

    async def _run_worker(self) -> None:
        """Process queued sentences until cancelled or shutdown sentinel."""
        try:
            while True:
                text = await self._queue.get()
                if text is None:
                    # Shutdown sentinel; cleanly exit.
                    return
                await self._render_one(text)
        except asyncio.CancelledError:
            # Either ``stop`` or ``aclose`` cancelled us. Re-raise so the
            # task transitions to ``cancelled`` rather than ``done`` —
            # callers awaiting on ``stop`` propagate the cancellation
            # cleanly through their suppress block.
            raise

    async def _render_one(self, text: str) -> None:
        """Synthesize and play a single sentence.

        Errors raised by :meth:`_synthesize` or playback are logged but
        do *not* take the worker down; the next sentence in the queue
        proceeds. This matches the design's principle that a single
        provider hiccup should not silence the assistant for the rest
        of the turn — the Dialog_Manager will surface a separate
        diagnostic when persistent failures warrant it.
        """
        self._synthesizing = True
        try:
            try:
                pcm = await self._synthesize(text)
            except asyncio.CancelledError:
                # Barge-in / shutdown — propagate so the worker exits.
                raise
            except Exception as exc:
                logger.warning(
                    "%s: synthesis failed for sentence (len=%d): %s",
                    self._provider_name,
                    len(text),
                    exc,
                )
                return
        finally:
            self._synthesizing = False

        if not pcm:
            return
        try:
            await self._player.aplay(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "%s: playback failed for sentence (pcm_bytes=%d): %s",
                self._provider_name,
                len(pcm),
                exc,
            )

    # -- helpers --------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Discard every queued sentence without emitting it."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    @property
    def audio_format(self) -> AudioFormat:
        """The PCM format produced by :meth:`_synthesize`."""
        return self._audio_format


# Static structural conformance check: at type-check time, mypy verifies that
# :class:`_CloudTTSEngine` (and therefore every concrete subclass) satisfies
# the :class:`~jarvis.voice.tts.base.TTSEngine` Protocol declared in
# ``base.py``. The assignment is guarded by :data:`typing.TYPE_CHECKING` so
# it has zero runtime cost; if the public engine API drifts, mypy will fail
# the build here rather than at the call site.
if TYPE_CHECKING:  # pragma: no cover - type-checker only

    def _check_protocol(engine: _CloudTTSEngine) -> TTSEngine:
        return engine
