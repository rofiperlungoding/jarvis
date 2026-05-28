"""Piper-based local Text-to-Speech engine.

This module implements :class:`PiperTTS`, the default
:class:`~jarvis.voice.tts.base.TTSEngine` for the JARVIS voice pipeline.
It wraps the local `piper-tts <https://github.com/OHF-Voice/piper1-gpl>`_
neural TTS engine — an ONNX-backed model that runs near real-time on
CPU and ships a curated set of voices (``en_GB-alan-medium`` is the
JARVIS persona default per Requirement 11.2).

Architecture
------------

The engine is structured as a *single-track, queued playback worker*:

* :meth:`speak` is non-blocking: it enqueues a sentence onto an
  :class:`asyncio.Queue` and returns. This satisfies the streaming
  contract from Requirement 12.2 / 19.5 — the
  :class:`~jarvis.dialog.manager.DialogManager` feeds finalised
  sentences from the :class:`~jarvis.voice.tts.base.SentenceAccumulator`
  one by one without waiting for synthesis or playback to finish.
* A background ``_worker`` task drains the queue. For each utterance it
  runs synthesis on a worker thread (ONNX inference is CPU-bound and
  would otherwise stall the event loop), then hands the resulting PCM
  to the shared :class:`~jarvis.voice.audio_io.AudioPlayer`. The player
  writes the PCM into ``sounddevice`` in fixed-size frames so cancellation
  remains responsive.
* :meth:`stop` is the barge-in entry point. It marks any in-flight
  synthesis as aborted, drains any pending queued text, and calls
  :meth:`AudioPlayer.stop` which aborts the audio device buffer and
  cancels the playback task. The combined budget is bounded by the
  player's 150 ms barge-in window — the same window required by
  Requirement 1.7.
* :meth:`is_playing` is a synchronous probe used by the audio capture
  loop to decide whether to invoke barge-in. It returns ``True`` while
  any of the three pipeline stages — queued, synthesizing, playing —
  is still busy.
* :meth:`aclose` shuts the worker down and releases the audio device.
  Idempotent.

Lazy imports
------------

Both ``piper`` and ``sounddevice`` are imported lazily on first
:meth:`speak` so the module remains importable on hosts that have neither
installed (CI runners, environments that only run the pure-logic
:class:`~jarvis.voice.tts.base.SentenceAccumulator` tests, machines
without PortAudio). Construction therefore has no system-level side
effects beyond storing arguments.

Voice model files
-----------------

Piper voices are distributed as ``<voice_id>.onnx`` plus a sibling JSON
config file. Users typically download them via
``python -m piper.download_voices en_GB-alan-medium`` to a local
directory. :class:`PiperTTS` accepts an explicit ``model_path`` so the
application bootstrap (``app.py``, task 19.1) can resolve it from
``${app.data_dir}/voices/`` or any other location the user configures.

Validates: Requirements 1.7, 11.2, 12.2
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from jarvis.voice.audio_io import AudioFormat, AudioPlayer

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from concurrent.futures import Executor

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_VOICE_ID", "PiperTTS"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: JARVIS persona-default voice identifier (Requirement 11.2 — mature, calm,
#: British-accented). The voice ONNX file is expected to live alongside its
#: ``.json`` config at the configured ``model_path``.
DEFAULT_VOICE_ID: Final[str] = "en_GB-alan-medium"

#: PCM playback chunk size, in samples per frame. Drives how :class:`AudioPlayer`
#: slices the synthesised PCM into ``sounddevice.RawOutputStream.write`` calls.
#: 1024 samples ≈ 46 ms at 22 050 Hz (Piper's typical voice rate), which is
#: short enough to keep the playback loop responsive to cancellation while
#: avoiding excessive syscall overhead.
_DEFAULT_CHUNK_FRAME_SAMPLES: Final[int] = 1024

#: Total budget for :meth:`aclose` to await the worker task before forcibly
#: cancelling it. Generous enough for a graceful in-flight write to drain,
#: tight enough that ``aclose`` never stalls application shutdown.
_ACLOSE_WORKER_TIMEOUT_S: Final[float] = 0.5


# ---------------------------------------------------------------------------
# PiperTTS
# ---------------------------------------------------------------------------


class PiperTTS:
    """Local neural TTS engine backed by piper-tts ONNX.

    Implements the :class:`~jarvis.voice.tts.base.TTSEngine` Protocol.

    Parameters
    ----------
    model_path:
        Filesystem path to the voice's ``.onnx`` model. The sibling JSON
        config is auto-discovered as ``<model_path>.json`` unless an
        explicit ``config_path`` is supplied. The path is *not* checked
        at construction time — the file is opened lazily on first
        :meth:`speak` so that boot ordering is decoupled from the
        existence of the voice file.
    config_path:
        Optional path to the voice's JSON configuration. Defaults to
        ``str(model_path) + ".json"`` per piper convention.
    voice_id:
        Human-readable voice identifier used in log lines and diagnostic
        output. Defaults to :data:`DEFAULT_VOICE_ID`.
    speaking_rate:
        Multiplier applied to the synthesised speed. ``1.0`` is the
        voice's natural pace. Larger values speak faster, smaller
        values slower. Maps to piper's ``length_scale`` via the inverse
        relationship (``length_scale = 1.0 / speaking_rate``).
    use_cuda:
        Pass through to :meth:`piper.PiperVoice.load`. Requires the
        ``onnxruntime-gpu`` package when ``True``.
    espeak_data_dir, download_dir:
        Optional overrides forwarded to :meth:`piper.PiperVoice.load`.
        ``None`` lets piper use its own defaults.
    output_device:
        ``sounddevice`` device index or substring used by the underlying
        :class:`~jarvis.voice.audio_io.AudioPlayer`. ``None`` selects the
        host's default output device.
    chunk_frame_samples:
        Number of samples per PCM playback frame; controls how the
        synthesised buffer is sliced into ``RawOutputStream.write``
        calls. Defaults to :data:`_DEFAULT_CHUNK_FRAME_SAMPLES`.
    executor:
        Optional :class:`concurrent.futures.Executor` for synthesis
        offload. ``None`` lets ``asyncio.to_thread`` use the default
        thread pool, which is the recommended choice and keeps the
        engine pluggable.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        config_path: str | Path | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        speaking_rate: float = 1.0,
        use_cuda: bool = False,
        espeak_data_dir: str | Path | None = None,
        download_dir: str | Path | None = None,
        output_device: int | str | None = None,
        chunk_frame_samples: int = _DEFAULT_CHUNK_FRAME_SAMPLES,
        executor: Executor | None = None,
    ) -> None:
        if speaking_rate <= 0:
            raise ValueError("speaking_rate must be positive")
        if chunk_frame_samples <= 0:
            raise ValueError("chunk_frame_samples must be positive")

        self._model_path: Path = Path(model_path)
        self._config_path: Path | None = Path(config_path) if config_path is not None else None
        self._voice_id: str = voice_id
        self._speaking_rate: float = speaking_rate
        self._use_cuda: bool = use_cuda
        self._espeak_data_dir: Path | None = (
            Path(espeak_data_dir) if espeak_data_dir is not None else None
        )
        self._download_dir: Path | None = Path(download_dir) if download_dir is not None else None
        self._output_device: int | str | None = output_device
        self._chunk_frame_samples: int = chunk_frame_samples
        self._executor: Executor | None = executor

        # State populated lazily on first :meth:`speak`. ``_voice`` is
        # ``Any``-typed to avoid coupling import-time to the ``piper``
        # package; the lazy import inside :meth:`_load_voice` ensures
        # the module remains importable without piper installed.
        self._voice: Any | None = None
        self._format: AudioFormat | None = None
        self._player: AudioPlayer | None = None

        # Sentence text queue and worker. Queue depth is unbounded —
        # the producer (DialogManager) only enqueues finalised sentences
        # at a rate the LLM emits them, which is far slower than playback.
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._init_lock: asyncio.Lock = asyncio.Lock()

        # Liveness flags. ``_currently_synthesizing`` is set while
        # :meth:`_synthesize` is running on a worker thread; it lets
        # :meth:`is_playing` report ``True`` even before the player has
        # received the first byte of PCM, so the audio capture loop's
        # barge-in check covers the synthesis window too. ``_aborted``
        # is consulted both inside the synthesis loop (to short-circuit
        # the per-chunk loop) and by the worker (to discard a fully
        # synthesised buffer when :meth:`stop` was called between
        # synthesis and playback).
        self._currently_synthesizing: bool = False
        self._aborted: bool = False
        self._closed: bool = False

    # -- properties ---------------------------------------------------------

    @property
    def voice_id(self) -> str:
        """Configured voice identifier (e.g., ``"en_GB-alan-medium"``)."""
        return self._voice_id

    @property
    def model_path(self) -> Path:
        """Filesystem path to the ONNX voice model."""
        return self._model_path

    # -- TTSEngine Protocol -------------------------------------------------

    async def speak(self, text: str) -> None:
        """Enqueue ``text`` for synthesis and playback.

        Returns as soon as the text is on the internal queue, without
        waiting for synthesis or playback. This satisfies the streaming
        contract: the :class:`~jarvis.dialog.manager.DialogManager` can
        push sentences in as fast as the LLM emits them and the worker
        plays them out sequentially.

        Whitespace-only / empty strings are silently dropped — they
        produce no audible output and would otherwise spend a worker
        cycle initialising piper for nothing.
        """
        if self._closed:
            raise RuntimeError("PiperTTS is closed")
        if not text or not text.strip():
            return
        await self._ensure_started()
        await self._text_queue.put(text)

    async def stop(self) -> None:
        """Abort any in-flight playback and drop pending queued text.

        This is the barge-in path (Requirement 1.7). The combined work
        budget is bounded by :class:`AudioPlayer`'s 150 ms barge-in
        window: the device buffer is aborted via PortAudio's ``abort``
        call (which discards already-queued samples) and the playback
        task is cancelled.

        Pending queued text is also discarded so the next user turn
        does not have to wait for a backlog of sentences from the
        interrupted assistant response.

        After ``stop``, the engine remains usable: subsequent
        :meth:`speak` calls will resume normal operation.
        """
        # Drain pending text so the worker, after seeing playback
        # cancellation, returns immediately to the blocking
        # ``queue.get`` instead of consuming and synthesising stale
        # sentences.
        self._drain_text_queue()

        # Mark current synthesis as aborted. The synthesis thread checks
        # this between piper chunks and exits its loop early; if the
        # buffer was already complete and is sitting between synthesis
        # and playback, the worker's post-synthesis check rejects it.
        self._aborted = True

        # Stop active playback. ``AudioPlayer.stop`` is idempotent and
        # bounded by the 150 ms barge-in budget.
        player = self._player
        if player is not None:
            await player.stop()

    def is_playing(self) -> bool:
        """Return ``True`` while any pipeline stage is still busy.

        The audio capture loop polls this to decide whether a fresh
        ``speech_start`` from the VAD should trigger barge-in. We
        report ``True`` for the entire span between :meth:`speak`
        accepting a sentence and the worker finishing its playback,
        including the synthesis window — that way the user can
        interrupt JARVIS *before* audio reaches the speakers, not just
        after.
        """
        if self._closed:
            return False
        player = self._player
        if player is not None and player.is_playing():
            return True
        if self._currently_synthesizing:
            return True
        # ``qsize`` is approximate but good enough for a liveness probe;
        # on CPython for an ``asyncio.Queue`` it is exact.
        return self._text_queue.qsize() > 0

    async def aclose(self) -> None:
        """Stop the worker, release the audio device, and forbid further use.

        Idempotent: a second call returns immediately.
        """
        if self._closed:
            return
        self._closed = True

        # Drain queued text and post a sentinel so the worker's
        # ``queue.get`` unblocks promptly.
        self._drain_text_queue()
        with contextlib.suppress(Exception):
            self._text_queue.put_nowait(None)

        # Abort active playback so the worker's ``aplay`` returns now
        # rather than after the in-flight audio finishes.
        player = self._player
        if player is not None:
            with contextlib.suppress(Exception):
                await player.stop()

        # Wait for the worker to drain. If it overruns the budget,
        # cancel and absorb. A misbehaving worker should not block
        # application shutdown.
        worker = self._worker_task
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._await_worker(worker)),
                    timeout=_ACLOSE_WORKER_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning(
                    "PiperTTS.aclose: worker did not exit within %.1fs; cancelling.",
                    _ACLOSE_WORKER_TIMEOUT_S,
                )
                worker.cancel()
                with contextlib.suppress(BaseException):
                    await worker

        # Finally, release the audio device.
        if player is not None:
            with contextlib.suppress(Exception):
                await player.aclose()

    # -- Internal: lifecycle ------------------------------------------------

    async def _ensure_started(self) -> None:
        """Lazily load the voice and start the playback worker."""
        if self._voice is not None:
            return
        async with self._init_lock:
            # Double-checked under the lock: another caller may have raced
            # us through the fast-path check above.
            if self._voice is not None:
                return  # type: ignore[unreachable]

            # ONNX session construction is CPU- and disk-bound; offload
            # to a thread so the event loop remains responsive while the
            # voice model loads.
            voice = await asyncio.to_thread(self._load_voice)

            sample_rate = int(voice.config.sample_rate)
            self._format = AudioFormat(
                sample_rate_hz=sample_rate,
                frame_samples=self._chunk_frame_samples,
                channels=1,
                sample_width=2,
            )
            self._player = AudioPlayer(
                format=self._format,
                device=self._output_device,
            )
            self._voice = voice

            # Start the worker only after both ``_voice`` and
            # ``_player`` are populated; the worker assumes both.
            self._worker_task = asyncio.create_task(
                self._worker(), name=f"piper-tts-{self._voice_id}"
            )

    def _load_voice(self) -> Any:
        """Lazy-import ``piper`` and load the configured voice.

        Runs on a worker thread; raises :class:`RuntimeError` with a
        clear remediation message if the ``piper-tts`` package is not
        installed. Other failures (missing model file, invalid JSON
        config) propagate as their native exceptions so the bootstrap
        can surface them to the user verbatim.
        """
        try:
            from piper import PiperVoice  # noqa: PLC0415 - lazy import is intentional
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "piper-tts is not installed; install the voice extras to "
                "enable local TTS synthesis. "
                "See https://github.com/OHF-Voice/piper1-gpl for setup."
            ) from exc

        load_kwargs: dict[str, Any] = {"use_cuda": self._use_cuda}
        if self._config_path is not None:
            load_kwargs["config_path"] = str(self._config_path)
        if self._espeak_data_dir is not None:
            load_kwargs["espeak_data_dir"] = str(self._espeak_data_dir)
        if self._download_dir is not None:
            load_kwargs["download_dir"] = str(self._download_dir)

        logger.info(
            "Loading Piper voice %s from %s (cuda=%s)",
            self._voice_id,
            self._model_path,
            self._use_cuda,
        )
        return PiperVoice.load(str(self._model_path), **load_kwargs)

    # -- Internal: worker ---------------------------------------------------

    async def _worker(self) -> None:
        """Drain the text queue, synthesising and playing each item.

        The loop runs until the engine is closed (``_closed=True``) or a
        sentinel ``None`` is observed on the queue. Each utterance is:

        1. Synthesised on a worker thread via :meth:`_synthesize`.
        2. Checked against the abort flag (set by :meth:`stop`) — if
           the user interrupted between synthesis start and synthesis
           end, the buffered PCM is discarded.
        3. Played through :class:`AudioPlayer.aplay`, which raises
           :class:`asyncio.CancelledError` when :meth:`stop` aborts it.
           That exception is caught locally so the worker stays alive
           for subsequent utterances.
        """
        assert self._player is not None  # established by _ensure_started
        while not self._closed:
            text = await self._text_queue.get()
            try:
                if text is None or self._closed:
                    return
                self._aborted = False

                self._currently_synthesizing = True
                try:
                    chunks = await asyncio.to_thread(self._synthesize, text)
                finally:
                    self._currently_synthesizing = False

                if not chunks or self._aborted or self._closed:
                    # ``stop`` (or shutdown) won the race against synthesis.
                    continue

                try:
                    await self._player.aplay(chunks)
                except asyncio.CancelledError:
                    # ``stop`` cancelled playback. The worker task
                    # itself has not been cancelled (only the inner
                    # playback task was), so we swallow the exception
                    # and resume on the next utterance.
                    continue
            except Exception:
                # Synthesis errors must not kill the worker — a single
                # malformed sentence should not silence the assistant.
                # Log with traceback for diagnostics and move on.
                logger.exception(
                    "PiperTTS worker error processing utterance for voice %s",
                    self._voice_id,
                )
            finally:
                self._text_queue.task_done()

    def _synthesize(self, text: str) -> list[bytes]:
        """Run piper synthesis and collect PCM int16 byte chunks.

        Runs on a worker thread (driven by :func:`asyncio.to_thread`).
        Returns a list of ``bytes`` — typically a single element since
        :class:`~jarvis.voice.tts.base.SentenceAccumulator` feeds whole
        sentences and piper emits one chunk per sentence — that
        :class:`AudioPlayer` can iterate over.

        The :pyattr:`_aborted` flag is consulted between chunks so a
        late-arriving :meth:`stop` short-circuits the synthesis loop
        before piper finishes processing a long utterance.
        """
        # Lazy import here as well: the worker thread can outlive the
        # event loop tear-down on Windows in some shutdown paths, and a
        # second import is cheap (Python caches the module).
        from piper import SynthesisConfig  # noqa: PLC0415 - lazy import is intentional

        # Map JARVIS speaking_rate (multiplier) onto piper's length_scale
        # (inverse: 2.0 = twice as slow). When ``speaking_rate == 1.0``
        # we leave ``length_scale`` unset so piper picks the voice's
        # native value, which is the most faithful reproduction.
        if self._speaking_rate == 1.0:
            syn_config: Any | None = None
        else:
            syn_config = SynthesisConfig(length_scale=1.0 / self._speaking_rate)

        chunks: list[bytes] = []
        assert self._voice is not None  # invariant: caller awaited _ensure_started
        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            if self._aborted or self._closed:
                # Drop any partially-built buffer; the worker will
                # discard the empty list anyway.
                return []
            # ``audio_int16_bytes`` is the canonical 16-bit signed PCM
            # representation piper computes from its float output.
            chunks.append(chunk.audio_int16_bytes)
        return chunks

    # -- Internal: helpers --------------------------------------------------

    def _drain_text_queue(self) -> None:
        """Discard every pending item in the text queue.

        ``task_done`` is called for each drained item to keep the
        queue's outstanding-task counter consistent; otherwise a future
        ``join`` would hang.
        """
        while True:
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._text_queue.task_done()

    @staticmethod
    async def _await_worker(task: asyncio.Task[None]) -> None:
        """Await the worker task, swallowing any exception it raises."""
        with contextlib.suppress(BaseException):
            await task
