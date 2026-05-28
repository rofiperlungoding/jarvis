"""Local faster-whisper Speech-to-Text engine.

This module implements :class:`FasterWhisperSTT`, the JARVIS voice
pipeline's default :class:`~jarvis.voice.stt.base.STTEngine`. It wraps
the `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_
package, which re-implements OpenAI Whisper on top of CTranslate2 — the
combination the design selects because it runs ~5x faster than the
reference implementation and is local-only by construction
(Requirement 13.2: privacy mode forbids cloud STT).

Inference is **CPU-bound** (or GPU-bound when CUDA is configured), so
:meth:`FasterWhisperSTT.transcribe` offloads the entire ``model.transcribe``
call to a :class:`concurrent.futures.ThreadPoolExecutor` via
:meth:`asyncio.AbstractEventLoop.run_in_executor`. That keeps the
asyncio event loop responsive — wake-word detection, VAD, and TTS
playback continue to make progress while Whisper decodes the captured
utterance.

Confidence is computed as the design dictates::

    confidence = mean(exp(segment.avg_logprob) for segment in segments)

faster-whisper exposes a per-segment ``avg_logprob`` field, which is the
average per-token log probability for that segment. Exponentiating
recovers the per-token probability; averaging across segments yields the
utterance-level confidence the :class:`~jarvis.dialog.manager.DialogManager`
gates at ``< 0.4`` (Requirement 1.8). When the engine produces no
segments — typical for trimmed silence or a sub-token VAD cut — the
confidence is reported as ``0.0`` so the gate trips on re-prompt.

Lazy imports
------------

Both ``faster_whisper`` and ``numpy`` are imported lazily on first
:meth:`transcribe` so this module remains importable on hosts that have
neither installed (CI runners that only exercise the
:class:`~jarvis.voice.stt.base.Transcript` validators, hosts without
CTranslate2 wheels, etc.). Construction therefore performs **no**
filesystem or model-load work; the heavy lifting happens on the first
real utterance.

Validates: Requirements 1.3, 13.2
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
import contextlib
import logging
import math
from typing import TYPE_CHECKING, Any, Final, Literal

from jarvis.utils.time_source import SystemTimeSource, TimeSource
from jarvis.voice.stt.base import AudioBuffer, STTEngine, Transcript

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_SIZE",
    "FasterWhisperSTT",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default Whisper checkpoint. Matches ``[voice.stt] model`` in
#: ``default.toml`` ("small.en"); kept here as a Python-level fallback so
#: callers that bypass the config (tests, ad-hoc scripts) still get the
#: design-recommended model.
DEFAULT_MODEL_SIZE: Final[str] = "small.en"

#: Default inference device. CPU keeps the engine portable across
#: developer laptops that lack a CUDA toolkit; users can override to
#: ``"cuda"`` via config. ``"auto"`` is also forwarded to
#: ``WhisperModel`` for callers that want CTranslate2 to pick.
DEFAULT_DEVICE: Final[Literal["cpu", "cuda", "auto"]] = "cpu"

#: Default CTranslate2 compute type. ``int8`` quantization is the design
#: default — small.en at int8 fits comfortably under the latency budget
#: from Requirement 12 on commodity CPUs.
DEFAULT_COMPUTE_TYPE: Final[str] = "int8"

#: Sample rate (Hz) the audio capture loop produces and Whisper expects.
#: 16 kHz is the project-wide voice-pipeline rate (see
#: ``jarvis.voice.audio_io``).
_SAMPLE_RATE_HZ: Final[int] = 16000

#: Bytes per sample of the captured PCM (16-bit signed mono).
_PCM_SAMPLE_WIDTH: Final[int] = 2

#: Scale factor from int16 to float32 in the range ``[-1.0, 1.0)`` —
#: 32768 = ``2 ** 15``. Whisper expects floats in that range.
_INT16_SCALE: Final[float] = 32768.0

#: Default ``ThreadPoolExecutor`` worker count when no executor is
#: supplied. Whisper transcription is sequential at the per-utterance
#: level — only one request is in flight at a time on the wake-to-
#: response path — but a small pool size lets the engine overlap a
#: background language-detection call with a future utterance without
#: starving the loop's default executor.
_DEFAULT_EXECUTOR_WORKERS: Final[int] = 1


# ---------------------------------------------------------------------------
# FasterWhisperSTT
# ---------------------------------------------------------------------------


class FasterWhisperSTT(STTEngine):
    """Local Whisper engine backed by CTranslate2.

    Implements the :class:`~jarvis.voice.stt.base.STTEngine` Protocol.

    Parameters
    ----------
    model_size:
        Whisper checkpoint identifier passed to
        :class:`faster_whisper.WhisperModel`. Accepts the standard
        OpenAI sizes (``"tiny"``, ``"base"``, ``"small"``, ``"medium"``,
        ``"large-v3"``), their English-only variants (``"small.en"`` …),
        a HuggingFace repo id (``"Systran/faster-whisper-small.en"``),
        or a local directory containing CTranslate2-converted weights.
        Defaults to :data:`DEFAULT_MODEL_SIZE`.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"``. CTranslate2 picks the
        backend; ``"cuda"`` requires a CUDA-enabled CTranslate2 build.
    compute_type:
        CTranslate2 compute precision. ``"int8"`` (the default) is the
        recommended quantized mode for CPU; ``"float16"`` is typical for
        GPU. Forwarded verbatim to ``WhisperModel``.
    cpu_threads, num_workers, download_root, local_files_only:
        Optional overrides forwarded to ``WhisperModel``. ``None`` lets
        faster-whisper apply its own defaults.
    beam_size:
        Decoder beam width. Higher values trade latency for accuracy.
    vad_filter:
        Whether faster-whisper should apply *its own* VAD before
        decoding. Defaults to ``False`` because the JARVIS voice
        pipeline already runs Silero VAD upstream
        (``jarvis.voice.vad.SileroVAD``); double-VADing only wastes
        cycles. Exposed so callers wiring a non-VAD audio source can
        opt back in.
    executor:
        Optional :class:`concurrent.futures.Executor` used to run the
        blocking ``model.transcribe`` call. ``None`` (the default)
        causes the engine to construct its own
        :class:`concurrent.futures.ThreadPoolExecutor` with
        :data:`_DEFAULT_EXECUTOR_WORKERS` workers, which is the design's
        recommended choice and is shut down by :meth:`aclose`.
    time_source:
        Injectable :class:`~jarvis.utils.time_source.TimeSource` used to
        stamp :attr:`Transcript.started_at`. Defaults to
        :class:`~jarvis.utils.time_source.SystemTimeSource`. A
        :class:`~jarvis.utils.time_source.FakeTimeSource` makes the
        engine deterministic in tests.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        cpu_threads: int | None = None,
        num_workers: int | None = None,
        download_root: str | None = None,
        local_files_only: bool = False,
        beam_size: int = 5,
        vad_filter: bool = False,
        executor: Executor | None = None,
        time_source: TimeSource | None = None,
    ) -> None:
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")

        self._model_size: str = model_size
        self._device: str = device
        self._compute_type: str = compute_type
        self._cpu_threads: int | None = cpu_threads
        self._num_workers: int | None = num_workers
        self._download_root: str | None = download_root
        self._local_files_only: bool = local_files_only
        self._beam_size: int = beam_size
        self._vad_filter: bool = vad_filter
        self._time_source: TimeSource = time_source or SystemTimeSource()

        # Distinguish a caller-supplied executor (we must NOT shut it
        # down) from one we constructed ourselves (we own its lifecycle).
        self._owns_executor: bool = executor is None
        self._executor: Executor | None = executor

        # Lazy state. ``_model`` is ``Any``-typed to keep the runtime
        # import graph free of ``faster_whisper`` until actually needed.
        self._model: Any | None = None
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._closed: bool = False


    # -- properties ---------------------------------------------------------

    @property
    def model_size(self) -> str:
        """Configured Whisper checkpoint identifier."""
        return self._model_size

    @property
    def device(self) -> str:
        """Configured CTranslate2 device (``"cpu"`` / ``"cuda"`` / ``"auto"``)."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Configured CTranslate2 compute precision (e.g., ``"int8"``)."""
        return self._compute_type

    # -- STTEngine Protocol -------------------------------------------------

    async def transcribe(
        self,
        audio: AudioBuffer,
        language: str,
    ) -> Transcript:
        """Transcribe a captured utterance.

        ``audio`` is the raw 16 kHz / 16-bit / mono PCM buffer the audio
        capture loop produced after the VAD signaled ``speech_end``
        (Requirement 1.3). The buffer is converted in-process to the
        ``float32`` numpy array faster-whisper expects, then handed to
        ``model.transcribe`` on the engine's executor.

        The returned :class:`Transcript` has:

        * ``text`` — concatenated segment text, stripped.
        * ``confidence`` — ``mean(exp(segment.avg_logprob))`` across the
          decoded segments (``0.0`` if no segments).
        * ``started_at`` — captured via the injected :class:`TimeSource`
          immediately before inference begins.
        * ``duration_ms`` — derived from the input buffer size (``len /
          sample_rate / sample_width``); the on-device wall-clock
          duration of the *audio*, not the decode latency.
        * ``language`` — echoed from the call argument when supplied
          non-empty; otherwise the language detected by Whisper.
        """
        if self._closed:
            raise RuntimeError("FasterWhisperSTT is closed")
        if not isinstance(audio, (bytes, bytearray, memoryview)):
            raise TypeError(
                "FasterWhisperSTT.transcribe expects raw PCM bytes; got "
                f"{type(audio).__name__}"
            )

        await self._ensure_started()

        # Stamp the start *before* we hand off to the executor so the
        # timestamp reflects when capture finished, not when decode
        # finished. The audit log treats ``started_at`` as the
        # "utterance moment".
        started_at: datetime = self._time_source.now()
        duration_ms: int = self._duration_ms(audio)

        # Hand the raw bytes to a worker thread. The worker is the only
        # place that touches numpy / faster-whisper APIs, so a host
        # without those packages can still import this module.
        loop = asyncio.get_running_loop()
        executor = self._executor  # snapshot under aclose race
        text, confidence, detected_language = await loop.run_in_executor(
            executor,
            self._run_transcribe,
            bytes(audio),  # ensure immutable bytes-like for thread safety
            language,
        )

        # Echo the requested language when the caller specified one; fall
        # back to Whisper's detection only when caller passes the empty
        # sentinel (matches the contract documented on
        # :meth:`STTEngine.transcribe`).
        resolved_language = language if language else detected_language
        if not resolved_language:
            # Whisper *should* always return a language; if it doesn't
            # (corrupt audio, model bug), fall back to "en" rather than
            # constructing an invalid Transcript that violates the
            # base-class validator.
            resolved_language = "en"

        return Transcript(
            text=text,
            confidence=confidence,
            started_at=started_at,
            duration_ms=duration_ms,
            language=resolved_language,
        )

    async def aclose(self) -> None:
        """Release the model and the owned executor.

        Idempotent. If the executor was supplied by the caller it is
        left untouched — the caller owns its lifecycle.
        """
        if self._closed:
            return
        self._closed = True

        # Drop the model reference so CTranslate2 can free its weights.
        # ``WhisperModel`` does not expose an explicit ``close`` in the
        # public API; relying on garbage collection is fine.
        self._model = None

        executor = self._executor
        if executor is not None and self._owns_executor:
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None


    # -- Internal: lifecycle ------------------------------------------------

    async def _ensure_started(self) -> None:
        """Lazily load the Whisper model and, if needed, the executor."""
        if self._model is not None:
            return
        async with self._init_lock:
            # Double-checked under the lock: another caller may have raced
            # us through the fast-path check above.
            if self._model is not None:
                return  # type: ignore[unreachable]

            # Build our own executor only if the caller did not supply
            # one. ``ThreadPoolExecutor`` is created lazily so test
            # environments that never call ``transcribe`` do not spawn
            # a worker thread.
            if self._executor is None and self._owns_executor:
                self._executor = ThreadPoolExecutor(
                    max_workers=_DEFAULT_EXECUTOR_WORKERS,
                    thread_name_prefix="faster-whisper-stt",
                )

            # Model construction is disk- and CPU-bound (CTranslate2
            # mmap, weight load, JIT). Offload to the executor so the
            # event loop stays responsive while a many-hundred-MB model
            # is brought up.
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                self._executor,
                self._load_model,
            )

    def _load_model(self) -> Any:
        """Lazy-import ``faster_whisper`` and instantiate the model.

        Runs on a worker thread; raises :class:`RuntimeError` with a
        clear remediation message if the package is not installed.
        Other failures (missing weights, incompatible CUDA toolkit)
        propagate as their native exceptions so the bootstrap can
        surface them verbatim.
        """
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415 - lazy import is intentional
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "faster-whisper is not installed; install the voice "
                "extras to enable local Speech-to-Text. "
                "See https://github.com/SYSTRAN/faster-whisper for setup."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "device": self._device,
            "compute_type": self._compute_type,
            "local_files_only": self._local_files_only,
        }
        if self._cpu_threads is not None:
            load_kwargs["cpu_threads"] = self._cpu_threads
        if self._num_workers is not None:
            load_kwargs["num_workers"] = self._num_workers
        if self._download_root is not None:
            load_kwargs["download_root"] = self._download_root

        logger.info(
            "Loading faster-whisper model %s (device=%s, compute_type=%s)",
            self._model_size,
            self._device,
            self._compute_type,
        )
        return WhisperModel(self._model_size, **load_kwargs)


    # -- Internal: inference (worker-thread side) --------------------------

    def _run_transcribe(
        self,
        audio: bytes,
        language: str,
    ) -> tuple[str, float, str]:
        """Blocking inference body. Runs on the executor.

        Returns a ``(text, confidence, detected_language)`` tuple. Text
        is the concatenated, stripped segment text. Confidence is
        ``mean(exp(segment.avg_logprob))`` — the design's exact
        formula — clamped into ``[0.0, 1.0]`` so the
        :class:`Transcript` validator never rejects a slightly
        out-of-range value caused by floating-point drift.
        ``detected_language`` is the ISO tag faster-whisper inferred,
        used as a fallback when the caller passes an empty ``language``.
        """
        assert self._model is not None  # invariant: caller awaited _ensure_started

        # Convert int16 PCM bytes to float32 numpy in [-1.0, 1.0).
        # Lazy-import numpy here so this module can be imported on hosts
        # without numpy installed (e.g., tests that only exercise the
        # Transcript validators on the base module).
        import numpy as np  # noqa: PLC0415 - lazy import is intentional

        if len(audio) == 0:
            audio_array = np.zeros(0, dtype=np.float32)
        else:
            int16 = np.frombuffer(audio, dtype=np.int16)
            audio_array = int16.astype(np.float32) / _INT16_SCALE

        # ``language=None`` triggers Whisper's auto-detect; otherwise we
        # pass the caller's explicit tag. faster-whisper rejects empty
        # strings, so we normalize "" to None here.
        whisper_language: str | None = language if language else None

        # Quality tuning notes:
        # * ``condition_on_previous_text=False`` — disables the prior-
        #   context bias that frequently regenerates the previous reply
        #   verbatim ("Thank you for watching", "Bye!" etc.) on a fresh
        #   utterance, which was the dominant hallucination mode in
        #   earlier builds.
        # * ``vad_filter=True`` with ``min_silence_duration_ms=500`` —
        #   strips Whisper-internal silence that otherwise gets glued
        #   into a phantom segment with random text.
        # * ``temperature=(0.0, 0.2, 0.4)`` — Whisper falls back through
        #   higher temperatures on each repetition / log-prob failure.
        #   Starting at 0 gives deterministic decoding for the common
        #   case; the fallback rungs catch noisy-mic edge cases.
        # * ``no_speech_threshold=0.6`` — Whisper's per-segment "this
        #   is silence" gate. Raising from the 0.45 default suppresses
        #   the "you" / "Thank you." silence false-positives.
        # * ``log_prob_threshold=-1.0`` — accept lower-confidence
        #   tokens; combined with the temperature ladder this avoids
        #   the engine refusing perfectly fine speech because a single
        #   token dipped below the default cutoff.
        segments_iter, info = self._model.transcribe(
            audio_array,
            language=whisper_language,
            beam_size=self._beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )

        # ``segments_iter`` is a generator; materialise it once so we can
        # both compute confidence and concatenate text without a second
        # decoding pass.
        segments = list(segments_iter)

        text = " ".join(
            (segment.text or "").strip()
            for segment in segments
            if (segment.text or "").strip()
        ).strip()

        confidence = self._segment_confidence(segments)

        # ``info.language`` is the auto-detected ISO tag. Fall back to
        # the empty string if the model failed to set it (defensive;
        # the public API always populates it on success).
        detected_language = getattr(info, "language", "") or ""

        return text, confidence, detected_language

    @staticmethod
    def _segment_confidence(segments: list[Any]) -> float:
        """Compute ``mean(exp(segment.avg_logprob))`` per the design.

        Returns ``0.0`` when there are no segments (empty buffer or pure
        silence) so the Dialog_Manager's confidence gate
        (``< 0.4``) trips on re-prompt rather than letting an unknown
        confidence through.
        """
        if not segments:
            return 0.0

        probs: list[float] = []
        for segment in segments:
            logprob = getattr(segment, "avg_logprob", None)
            if logprob is None:
                # A segment without an ``avg_logprob`` is treated as a
                # zero-probability sample; this keeps a malformed
                # backend from inflating confidence by silently dropping
                # uncertain segments.
                probs.append(0.0)
                continue
            try:
                probs.append(math.exp(float(logprob)))
            except (OverflowError, ValueError):
                # ``exp`` of a NaN or extremely small logprob can blow
                # up; treat as zero so the gate trips defensively.
                probs.append(0.0)

        mean = sum(probs) / len(probs)
        # Clamp to the validator-accepted range. ``avg_logprob`` is
        # always <= 0 in practice (it is a log probability), so
        # ``exp(...)`` is in ``(0, 1]``; the clamp guards against
        # floating-point drift only.
        if mean < 0.0:
            return 0.0
        if mean > 1.0:
            return 1.0
        return mean

    @staticmethod
    def _duration_ms(audio: AudioBuffer) -> int:
        """Compute the audio duration in milliseconds from raw PCM size.

        Assumes the project-wide capture format (16 kHz, 16-bit, mono).
        Truncates rather than rounds so a buffer one byte short of the
        next millisecond does not over-report duration. Returns ``0``
        for an empty buffer (legal per :class:`Transcript`'s validator).
        """
        if not audio:
            return 0
        n_samples = len(audio) // _PCM_SAMPLE_WIDTH
        return (n_samples * 1000) // _SAMPLE_RATE_HZ
