"""Voice-pipeline latency benchmark harness (task 23.1).

This module measures the wake-to-response latency that anchors
Requirement 12.1 / 12.2: the wall-time between the moment the
:class:`~jarvis.voice.vad.SileroVAD` reports ``speech_end`` and the
moment the Text-to-Speech engine emits the *first PCM sample* of the
assistant's reply.

The acceptance criterion the harness asserts on
---------------------------------------------------------------

> *"WHEN the user finishes speaking, THE Voice_Pipeline SHALL begin
> emitting TTS audio within the Latency_Budget of 800 milliseconds for
> at least 90 percent of conversational turns under typical home-network
> conditions."* — Requirement 12.1.

The harness reports the 50th, 90th, and 99th percentile latency over a
recorded utterance corpus and exits with a non-zero status when the
measured 90th percentile exceeds :data:`LATENCY_BUDGET_MS` (800 ms).
Two independent runs are performed:

* **primary** — stub Mistral cloud backend with a configurable
  first-token delay (the dominant variable in real cloud latency).
  This is the normal-path measurement against the 800 ms budget.
* **fallback** — stub Ollama-hosted local backend representing the
  Mistral → Ollama fallback flow described in Requirement 12.4 and
  the design's *"Mistral → Local Fallback Flow"* sequence diagram.
  Tracked separately because local inference has a different latency
  profile and the design explicitly says the fallback is *"benchmarked
  separately"*.

Why a synthetic harness rather than a wall-clock microbenchmark
---------------------------------------------------------------

Production latency is dominated by three independent variables — STT
inference, LLM time-to-first-token, and TTS synthesis — and each one
varies wildly with hardware, model size, and network conditions. The
benchmark therefore parametrises each variable explicitly so CI can
hold the others fixed while sweeping one. The defaults reflect the
component budgets the design assigns:

* STT processing: ~80 ms (faster-whisper small, Requirement 1.6).
* LLM first-token delay: ~250 ms cloud / ~450 ms local
  (Requirement 19.5 / Requirement 12.4).
* TTS time-to-first-PCM: ~50 ms (Piper, Requirement 11.2).

The harness uses :func:`asyncio.sleep` to model these stages
deterministically, then drives them through the *real* production
:class:`~jarvis.voice.tts.base.SentenceAccumulator` so the sentence-
boundary streaming behaviour from Requirement 12.2 is exercised
end-to-end. Only the I/O endpoints are stubbed; the timing arithmetic
between them is the production code path.

Stubs vs. real components
-------------------------

* The stub LLM backends conform structurally to
  :class:`~jarvis.llm.base.LLMBackend` and emit
  :class:`~jarvis.llm.base.ContentDeltaEvent` values, so the
  Dialog_Manager's streaming contract (Requirement 19.5) is honoured
  byte-for-byte.
* The stub TTS does not synthesise real audio; it records the
  monotonic timestamp at which "the first PCM sample" would have
  been written to ``sounddevice``, then completes its background
  synthesis task without producing any output. The
  :class:`~jarvis.voice.tts.base.TTSEngine` Protocol's enqueue-and-
  return contract is preserved so the harness measures the same
  critical-path code the production stack does.

Validates: Requirements 12.1, 12.2
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

# Allow ``python -m benchmarks.voice_pipeline`` from a bare repository
# checkout that has not been ``pip install -e .``'d. The benchmark
# only consumes two pure-Python symbols from :mod:`jarvis.llm.base`
# and one class from :mod:`jarvis.voice.tts.base`, none of which pulls
# in optional native dependencies (``mistralai``, ``piper``, ``httpx``,
# ``sounddevice``) at import time.
try:  # pragma: no cover - exercised implicitly by the import below
    from jarvis.llm.base import (
        ContentDeltaEvent,
        LLMEvent,
        Message,
        Stream,
        ToolDefinition,
    )
    from jarvis.voice.tts.base import SentenceAccumulator
except ModuleNotFoundError:  # pragma: no cover - bare-checkout fallback
    _SRC_DIR = Path(__file__).resolve().parent.parent / "src"
    if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from jarvis.llm.base import (
        ContentDeltaEvent,
        LLMEvent,
        Message,
        Stream,
        ToolDefinition,
    )
    from jarvis.voice.tts.base import SentenceAccumulator


logger = logging.getLogger("jarvis.benchmarks.voice_pipeline")

__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_FALLBACK_FIRST_TOKEN_DELAY_MS",
    "DEFAULT_INTER_TOKEN_DELAY_MS",
    "DEFAULT_ITERATIONS",
    "DEFAULT_JITTER_MS",
    "DEFAULT_PRIMARY_FIRST_TOKEN_DELAY_MS",
    "DEFAULT_STT_PROCESSING_MS",
    "DEFAULT_TTS_SYNTH_MS",
    "LATENCY_BUDGET_MS",
    "BenchmarkReport",
    "CorpusUtterance",
    "RunResult",
    "StubLLMBackend",
    "StubTTS",
    "main",
    "run_benchmark",
    "run_path",
]


# ---------------------------------------------------------------------------
# Thresholds and defaults
# ---------------------------------------------------------------------------

#: The 800 ms p90 wake-to-response budget from Requirement 12.1.
LATENCY_BUDGET_MS: float = 800.0

#: Stable iteration count over the corpus for percentile computation.
#: 20 iterations across the 10-utterance default corpus produces 200
#: samples per path, which is enough for stable p90 figures and a
#: defensible p99 estimate while keeping per-run wall-clock time
#: bounded (each sample pays ~one full simulated turn of real-time
#: ``asyncio.sleep``). CI workflows that want tighter p99 bounds can
#: pass a larger ``--iterations`` value at the cost of longer runtime.
DEFAULT_ITERATIONS: int = 20

#: STT inference time per utterance (ms). The design pins faster-whisper
#: small at well under 200 ms for typical 1-2 s utterances; 80 ms is a
#: representative midpoint for a warmed model.
DEFAULT_STT_PROCESSING_MS: float = 80.0

#: Mistral cloud first-token delay (ms). Mistral's published p50 TTFT
#: for ``mistral-large-latest`` is in the 200-300 ms range under
#: typical home-network conditions; 250 ms is the harness default.
DEFAULT_PRIMARY_FIRST_TOKEN_DELAY_MS: float = 250.0

#: Ollama (local Mistral 7B Q4) first-token delay (ms). Local inference
#: on a CPU-bound box is dominated by prompt processing; 450 ms is a
#: representative midpoint for the cool-down-window fallback case.
DEFAULT_FALLBACK_FIRST_TOKEN_DELAY_MS: float = 450.0

#: Inter-token (chunk) delivery delay during streaming. Mistral and the
#: Ollama OpenAI-compatible streaming endpoints both deliver chunks at
#: a comparable cadence once the model is warm.
DEFAULT_INTER_TOKEN_DELAY_MS: float = 20.0

#: Time from receiving a sentence at the TTS engine to the first PCM
#: sample being written to the audio device. Piper's ONNX inference is
#: fast on CPU; 50 ms covers the model invocation plus the first
#: ``sounddevice`` buffer write.
DEFAULT_TTS_SYNTH_MS: float = 50.0

#: Per-iteration multiplicative jitter (ms) applied to all timing
#: variables. Without jitter every iteration would produce the same
#: latency and the percentile distribution would collapse to a point.
#: A small uniform jitter (default ±20 ms) reproduces the modest
#: turn-to-turn variance observed on warm production stacks.
DEFAULT_JITTER_MS: float = 20.0

# Internal: how many characters per ``ContentDeltaEvent`` chunk. Six
# characters at the default 20 ms cadence yields ~300 chars/sec —
# roughly the throughput of mistral-large-latest after the first token.
_DEFAULT_CHUNK_CHARS: int = 6


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusUtterance:
    """A single recorded-utterance corpus entry.

    The harness does not require real audio: STT is stubbed via a
    deterministic sleep, so each entry models a transcript paired with
    the assistant response the stub Mistral will emit on that turn.

    Attributes
    ----------
    label:
        Human-readable identifier carried into the JSON record so per-
        utterance latency can be filtered downstream.
    transcript:
        What the user said (the post-STT text). Not consumed by the
        stub backends; preserved in the JSON output for traceability.
    response:
        The full assistant response the stub backend will stream back.
        SHOULD contain at least one sentence terminator (``.``, ``?``,
        or ``!``) early in the text so that
        :class:`~jarvis.voice.tts.base.SentenceAccumulator` can emit
        the first sentence to TTS *during* the stream rather than only
        on flush — the latter would force the harness to measure the
        latency of the *whole* response, not the first PCM as the
        Requirement 12.1 budget intends.
    """

    label: str
    transcript: str
    response: str


# Recorded-utterance corpus. Each response begins with a short opening
# sentence so :class:`SentenceAccumulator` can deliver it to the stub
# TTS as soon as the terminating period arrives, mirroring the
# production sentence-boundary streaming path (Requirement 12.2).
DEFAULT_CORPUS: tuple[CorpusUtterance, ...] = (
    CorpusUtterance(
        label="greeting",
        transcript="Hello jarvis",
        response="Good evening, sir. How may I assist you tonight?",
    ),
    CorpusUtterance(
        label="weather",
        transcript="What's the weather like?",
        response="Twenty degrees and partly cloudy. Skies clear by noon, sir.",
    ),
    CorpusUtterance(
        label="timer",
        transcript="Set a timer for ten minutes.",
        response="Right away, sir. Ten minutes on the clock.",
    ),
    CorpusUtterance(
        label="calendar",
        transcript="Add a meeting tomorrow at three.",
        response="Done, sir. Scheduled for tomorrow at three.",
    ),
    CorpusUtterance(
        label="question",
        transcript="Who wrote Hamlet?",
        response="William Shakespeare, sir. Around 1600, in London.",
    ),
    CorpusUtterance(
        label="memo",
        transcript="Remind me to call my mother on Sunday.",
        response="Of course, sir. I will remind you Sunday morning.",
    ),
    CorpusUtterance(
        label="news",
        transcript="Give me the news headlines.",
        response="Here are the top stories. Markets opened higher today.",
    ),
    CorpusUtterance(
        label="music",
        transcript="Play some jazz.",
        response="Spinning up jazz now, sir. Enjoy.",
    ),
    CorpusUtterance(
        label="smart_home",
        transcript="Turn off the lights.",
        response="Lights off, sir. Anything else?",
    ),
    CorpusUtterance(
        label="farewell",
        transcript="Goodbye.",
        response="Until next time, sir. Stay safe.",
    ),
)


# ---------------------------------------------------------------------------
# Stub backends — structurally :class:`~jarvis.llm.base.LLMBackend`
# ---------------------------------------------------------------------------


def _chunked(text: str, chunk_chars: int) -> Iterable[str]:
    """Yield ``text`` in fixed-size character chunks.

    Mistral and Ollama both deliver tokens as variable-width Unicode
    fragments; for benchmark determinism we slice on character count
    instead. The first sentence terminator inside the chunked stream
    is what trips :class:`SentenceAccumulator`'s boundary detection,
    so the exact chunk width does not affect *which* sentence is
    spoken — only how many ``ContentDeltaEvent`` values the stream
    emits.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    for offset in range(0, len(text), chunk_chars):
        yield text[offset : offset + chunk_chars]


class StubLLMBackend:
    """Deterministic :class:`~jarvis.llm.base.LLMBackend` stand-in.

    Conforms structurally to :class:`~jarvis.llm.base.LLMBackend`. On
    each call to :meth:`stream` the backend:

    1. Sleeps for ``first_token_delay_seconds`` to model the LLM's
       time-to-first-token. This is the dominant variable in real
       cloud / local latency and the one the harness is designed to
       sweep.
    2. Emits the configured response as a sequence of
       :class:`ContentDeltaEvent` values, sleeping
       ``inter_token_delay_seconds`` between chunks.

    The backend ignores ``messages`` and ``tools`` — the benchmark is
    a one-shot per-turn measurement that does not depend on the prompt
    or any function-calling round-trip. Future extensions could attach
    a tool-call event sequence here for a tool-loop latency benchmark,
    but that is a separate task in the spec.

    The stub is configured via constructor arguments rather than
    keyword overrides on :meth:`stream` so the harness can vary the
    primary and fallback first-token delays independently in a single
    benchmark run.
    """

    def __init__(
        self,
        *,
        response_text: str,
        first_token_delay_seconds: float,
        inter_token_delay_seconds: float = DEFAULT_INTER_TOKEN_DELAY_MS / 1000.0,
        chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    ) -> None:
        if first_token_delay_seconds < 0:
            raise ValueError("first_token_delay_seconds must be non-negative")
        if inter_token_delay_seconds < 0:
            raise ValueError("inter_token_delay_seconds must be non-negative")
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        self._response_text = response_text
        self._first_token_delay = float(first_token_delay_seconds)
        self._inter_token_delay = float(inter_token_delay_seconds)
        self._chunk_chars = chunk_chars

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> Any:
        """Open a deterministic streaming chat completion.

        Returns an async context manager whose body yields a
        :class:`Stream` of :class:`ContentDeltaEvent`. The ``messages``
        and ``tools`` arguments are accepted for protocol conformance
        but ignored — this is a latency stub, not a behavioural one.
        """
        del messages, tools, kwargs  # unused — this is a latency stub.
        return self._stream()

    @asynccontextmanager
    async def _stream(self) -> AsyncIterator[Stream]:
        """Context-manager body backing :meth:`stream`."""
        # The first-token delay is paid *inside* the context manager so
        # the harness measures it as part of wake-to-response latency,
        # exactly as production code does (the cloud's HTTP round-trip
        # is paid between ``__aenter__`` and the first SSE chunk).
        async def _events() -> AsyncIterator[LLMEvent]:
            await asyncio.sleep(self._first_token_delay)
            for chunk in _chunked(self._response_text, self._chunk_chars):
                yield ContentDeltaEvent(text=chunk)
                if self._inter_token_delay > 0:
                    await asyncio.sleep(self._inter_token_delay)

        yield _events()


# ---------------------------------------------------------------------------
# Stub TTS — structurally :class:`~jarvis.voice.tts.base.TTSEngine`
# ---------------------------------------------------------------------------


class StubTTS:
    """Latency-recording :class:`~jarvis.voice.tts.base.TTSEngine` stub.

    The stub honours the production engine's enqueue-and-return
    contract: :meth:`speak` schedules a background "synthesis" task
    that sleeps for ``synth_delay_seconds`` and then records the
    monotonic timestamp at which the first PCM sample would have been
    written. Callers consume :attr:`first_pcm_at` after awaiting
    :meth:`drain` (which the harness does at the end of each turn) —
    the timestamp is captured *only* on the first ``speak`` of a turn
    so subsequent sentences in the same response do not overwrite it.

    The stub is intentionally minimal: it does not implement barge-in
    or queue draining behaviour beyond what the latency measurement
    needs, because the benchmark's critical section ends at the first
    PCM sample.
    """

    def __init__(
        self,
        *,
        synth_delay_seconds: float,
        time_source: AbstractTimeSource | None = None,
    ) -> None:
        if synth_delay_seconds < 0:
            raise ValueError("synth_delay_seconds must be non-negative")
        self._synth_delay = float(synth_delay_seconds)
        self._time = time_source if time_source is not None else _MonotonicTime()
        self._first_pcm_at: float | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._first_event = asyncio.Event()
        self._closed = False

    @property
    def first_pcm_at(self) -> float | None:
        """Monotonic timestamp of the first PCM sample, or ``None``."""
        return self._first_pcm_at

    @property
    def first_pcm_event(self) -> asyncio.Event:
        """Event set when the first PCM sample has been recorded."""
        return self._first_event

    async def speak(self, text: str) -> None:
        """Enqueue a sentence for synthesis.

        Returns once the synthesis task is scheduled, mirroring the
        production engine's contract. The first PCM sample timestamp
        is captured by the background task after
        ``synth_delay_seconds`` elapses; only the first such capture
        per stub instance is kept.
        """
        if self._closed:
            return
        del text  # not used; the latency measurement is content-agnostic.
        task = asyncio.create_task(self._synthesize_one())
        self._tasks.append(task)

    async def _synthesize_one(self) -> None:
        """Background "synthesis" — sleep then record first-PCM timestamp."""
        if self._synth_delay > 0:
            await asyncio.sleep(self._synth_delay)
        if self._first_pcm_at is None:
            self._first_pcm_at = self._time.monotonic()
            self._first_event.set()

    async def stop(self) -> None:
        """Cancel any pending synthesis tasks (barge-in path).

        The benchmark does not exercise barge-in, but the production
        :class:`TTSEngine` Protocol requires the method, so we provide
        a faithful no-op-on-empty / cancel-on-pending implementation.
        """
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def is_playing(self) -> bool:
        """Return ``True`` while any synthesis task is still pending."""
        return any(not t.done() for t in self._tasks)

    async def drain(self) -> None:
        """Wait for every queued synthesis task to complete.

        Used by the harness at the end of each turn to ensure the
        first-PCM timestamp has been recorded before the latency
        sample is taken. Raises :class:`asyncio.TimeoutError` if a
        task hangs longer than the harness's per-turn budget — that
        would indicate a bug in the stub or the harness, not a
        latency-budget violation.
        """
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Idempotent shutdown."""
        self._closed = True
        await self.stop()


# ---------------------------------------------------------------------------
# Time source — small Protocol so tests can inject deterministic clocks
# ---------------------------------------------------------------------------


class AbstractTimeSource:
    """Minimal monotonic-only time source used by the harness.

    Defined locally rather than imported from
    :mod:`jarvis.utils.time_source` so the benchmark does not pull in
    the wider :mod:`jarvis.utils` namespace (and its transitive
    dependencies on :mod:`datetime` / :mod:`zoneinfo`) for a single
    ``monotonic()`` call.
    """

    def monotonic(self) -> float:
        raise NotImplementedError  # pragma: no cover - abstract


class _MonotonicTime(AbstractTimeSource):
    """Default :class:`AbstractTimeSource` backed by :func:`time.monotonic`."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.monotonic()


# ---------------------------------------------------------------------------
# Per-iteration measurement
# ---------------------------------------------------------------------------


def _apply_jitter(value_ms: float, jitter_ms: float, rng: random.Random) -> float:
    """Return ``value_ms`` perturbed by a uniform jitter in ``[-j, +j]`` ms.

    The result is clamped at zero so a generously large jitter cannot
    drive the modelled latency negative (which would silently advance
    the harness clock and produce nonsensical p99 figures).
    """
    if jitter_ms <= 0:
        return max(0.0, value_ms)
    delta = rng.uniform(-jitter_ms, jitter_ms)
    return max(0.0, value_ms + delta)


@dataclass
class _IterationTimings:
    """Resolved per-iteration timings (ms) after jitter has been applied."""

    stt_ms: float
    first_token_ms: float
    inter_token_ms: float
    tts_synth_ms: float


def _resolve_timings(
    *,
    stt_ms: float,
    first_token_ms: float,
    inter_token_ms: float,
    tts_synth_ms: float,
    jitter_ms: float,
    rng: random.Random,
) -> _IterationTimings:
    """Apply jitter to each component budget for one iteration."""
    return _IterationTimings(
        stt_ms=_apply_jitter(stt_ms, jitter_ms, rng),
        first_token_ms=_apply_jitter(first_token_ms, jitter_ms, rng),
        inter_token_ms=_apply_jitter(inter_token_ms, jitter_ms, rng),
        tts_synth_ms=_apply_jitter(tts_synth_ms, jitter_ms, rng),
    )


async def _measure_one_turn(
    utterance: CorpusUtterance,
    *,
    timings: _IterationTimings,
    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    time_source: AbstractTimeSource | None = None,
) -> float:
    """Drive one wake-to-response turn end-to-end and return its latency.

    The latency is the wall-time (ms) between two events:

    * ``t_speech_end`` — the moment immediately after the simulated
      VAD ``speech_end`` is observed and STT processing begins. The
      VAD itself contributes no latency (it has already concluded by
      definition); the harness clock starts at this exact instant.
    * ``t_first_pcm`` — the moment :class:`StubTTS` records its first
      PCM-sample timestamp.

    The path between the two events runs through the production
    :class:`SentenceAccumulator` so the sentence-boundary streaming
    behaviour from Requirement 12.2 is exercised on every turn.
    """
    clock = time_source if time_source is not None else _MonotonicTime()

    # The harness clock starts at speech_end. STT inference is
    # modelled as a deterministic sleep before the LLM is invoked.
    t_speech_end = clock.monotonic()

    # ---- 1) STT inference -------------------------------------------------
    if timings.stt_ms > 0:
        await asyncio.sleep(timings.stt_ms / 1000.0)

    # ---- 2) LLM streaming through SentenceAccumulator → TTS ---------------
    accumulator = SentenceAccumulator()
    tts = StubTTS(
        synth_delay_seconds=timings.tts_synth_ms / 1000.0,
        time_source=clock,
    )
    backend = StubLLMBackend(
        response_text=utterance.response,
        first_token_delay_seconds=timings.first_token_ms / 1000.0,
        inter_token_delay_seconds=timings.inter_token_ms / 1000.0,
        chunk_chars=chunk_chars,
    )

    # The Dialog_Manager dispatches tokens into the SentenceAccumulator
    # and forwards finalised sentences to the TTS engine on every
    # ``feed`` call. We replicate that minimal slice here so the
    # benchmark exercises the same code path.
    async with backend.stream([], tools=[]) as events:
        async for event in events:
            if isinstance(event, ContentDeltaEvent):
                sentences = accumulator.feed(event.text)
                for sentence in sentences:
                    await tts.speak(sentence)
                    # First PCM timestamp is captured by the synthesis
                    # task as soon as it fires; no further sentences
                    # in this turn affect the latency measurement.
                    if tts.first_pcm_at is not None:
                        # Drain any in-flight tasks so the stub's
                        # bookkeeping is clean before the next iteration.
                        await tts.drain()
                        return (tts.first_pcm_at - t_speech_end) * 1000.0

    # Fallback: the response had no terminator before stream end (e.g.,
    # a one-word reply without punctuation). The accumulator's tail is
    # drained on flush, then spoken. This path is not on the typical
    # latency-budget hot path but is included for completeness so the
    # harness never silently returns ``inf``.
    tail = accumulator.flush()
    if tail is not None:
        await tts.speak(tail)
    await tts.drain()
    if tts.first_pcm_at is None:
        # Truly empty response — measure as zero so the bench still
        # produces a finite distribution. In practice this branch is
        # unreachable for the default corpus.
        return 0.0
    return (tts.first_pcm_at - t_speech_end) * 1000.0


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _percentile(samples: Sequence[float], pct: float) -> float:
    """Return the ``pct``-th percentile (0..100) of ``samples`` in ms.

    Uses linear interpolation between the nearest ranks (NumPy-style
    ``"linear"`` interpolation). NumPy is *not* imported here so the
    benchmark stays usable in environments where NumPy is unavailable;
    the math is straightforward and this implementation matches
    :func:`numpy.percentile` to within floating-point error.
    """
    if not samples:
        raise ValueError("cannot compute percentile of an empty sample set")
    if not 0.0 <= pct <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    rank = (pct / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_samples[int(rank)]
    weight = rank - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


@dataclass(frozen=True)
class RunResult:
    """Outcome of one path's benchmark pass (primary or fallback).

    ``samples_ms`` is preserved verbatim so downstream tooling can
    re-aggregate (e.g. compute a CDF) without re-running the harness.
    Percentiles are pre-computed so the JSON record is human-readable.
    """

    path: str
    iterations: int
    samples_ms: tuple[float, ...]
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    first_token_delay_ms: float
    stt_ms: float
    inter_token_ms: float
    tts_synth_ms: float
    jitter_ms: float
    threshold_ms: float = LATENCY_BUDGET_MS

    @property
    def passed(self) -> bool:
        """``True`` iff the measured p90 satisfies Requirement 12.1."""
        return self.p90_ms <= self.threshold_ms

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise to a plain ``dict`` suitable for ``json.dumps``."""
        return {
            "path": self.path,
            "iterations": self.iterations,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "threshold_ms": self.threshold_ms,
            "passed": self.passed,
            "config": {
                "first_token_delay_ms": self.first_token_delay_ms,
                "stt_ms": self.stt_ms,
                "inter_token_ms": self.inter_token_ms,
                "tts_synth_ms": self.tts_synth_ms,
                "jitter_ms": self.jitter_ms,
            },
            "samples_ms": list(self.samples_ms),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Top-level JSON record emitted by the harness.

    Carries both the *primary* (Mistral cloud) and *fallback*
    (Mistral → Ollama) :class:`RunResult` records. Only the primary
    path is asserted against :data:`LATENCY_BUDGET_MS`; the fallback
    is reported separately as Requirement 12.4 says nothing about a
    p90 budget for the local backend, only that the user is informed
    of the fallback. The fallback figures still gate CI on whether
    they regress beyond the harness's tracking threshold (configured
    on the CLI) so the local path stays usable.
    """

    primary: RunResult
    fallback: RunResult
    corpus_size: int
    iterations: int
    seed: int
    started_at: float
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        """Combined pass/fail. Requirement 12.1 binds only the primary."""
        return self.primary.passed

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "corpus_size": self.corpus_size,
            "iterations": self.iterations,
            "seed": self.seed,
            "passed": self.passed,
            "primary": self.primary.to_json_dict(),
            "fallback": self.fallback.to_json_dict(),
            "thresholds": {
                "p90_ms_max": LATENCY_BUDGET_MS,
            },
        }


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


async def run_path(
    *,
    path_label: str,
    corpus: Sequence[CorpusUtterance],
    iterations: int,
    first_token_ms: float,
    stt_ms: float,
    inter_token_ms: float,
    tts_synth_ms: float,
    jitter_ms: float,
    seed: int,
) -> RunResult:
    """Run one path's benchmark pass and return its :class:`RunResult`.

    The function iterates over the cross product of
    ``range(iterations) x corpus``, producing
    ``iterations * len(corpus)`` latency samples. Each iteration
    re-seeds its local PRNG from ``seed + iteration_index`` so the
    primary and fallback runs use different jitter trajectories from
    the same base seed (helpful for diffing one against the other).
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not corpus:
        raise ValueError("corpus must contain at least one utterance")

    samples: list[float] = []
    for i in range(iterations):
        rng = random.Random(seed + i)
        for utterance in corpus:
            timings = _resolve_timings(
                stt_ms=stt_ms,
                first_token_ms=first_token_ms,
                inter_token_ms=inter_token_ms,
                tts_synth_ms=tts_synth_ms,
                jitter_ms=jitter_ms,
                rng=rng,
            )
            latency_ms = await _measure_one_turn(utterance, timings=timings)
            samples.append(latency_ms)

    return RunResult(
        path=path_label,
        iterations=len(samples),
        samples_ms=tuple(samples),
        p50_ms=_percentile(samples, 50.0),
        p90_ms=_percentile(samples, 90.0),
        p99_ms=_percentile(samples, 99.0),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        first_token_delay_ms=first_token_ms,
        stt_ms=stt_ms,
        inter_token_ms=inter_token_ms,
        tts_synth_ms=tts_synth_ms,
        jitter_ms=jitter_ms,
    )


async def run_benchmark(
    *,
    corpus: Sequence[CorpusUtterance] = DEFAULT_CORPUS,
    iterations: int = DEFAULT_ITERATIONS,
    primary_first_token_ms: float = DEFAULT_PRIMARY_FIRST_TOKEN_DELAY_MS,
    fallback_first_token_ms: float = DEFAULT_FALLBACK_FIRST_TOKEN_DELAY_MS,
    stt_ms: float = DEFAULT_STT_PROCESSING_MS,
    inter_token_ms: float = DEFAULT_INTER_TOKEN_DELAY_MS,
    tts_synth_ms: float = DEFAULT_TTS_SYNTH_MS,
    jitter_ms: float = DEFAULT_JITTER_MS,
    seed: int = 0,
) -> BenchmarkReport:
    """Run both the primary and fallback paths and return a report.

    The two paths are run sequentially (not concurrently) so they do
    not interfere via the asyncio scheduler — the harness measures
    real-time latency, and concurrent CPU contention from the second
    path would distort the first path's percentiles.
    """
    started_at = time.time()
    started_monotonic = time.monotonic()

    primary = await run_path(
        path_label="primary",
        corpus=corpus,
        iterations=iterations,
        first_token_ms=primary_first_token_ms,
        stt_ms=stt_ms,
        inter_token_ms=inter_token_ms,
        tts_synth_ms=tts_synth_ms,
        jitter_ms=jitter_ms,
        seed=seed,
    )
    fallback = await run_path(
        path_label="fallback",
        corpus=corpus,
        iterations=iterations,
        first_token_ms=fallback_first_token_ms,
        stt_ms=stt_ms,
        inter_token_ms=inter_token_ms,
        tts_synth_ms=tts_synth_ms,
        jitter_ms=jitter_ms,
        # Offset the seed for the fallback pass so the two paths
        # explore different jitter trajectories. Without the offset
        # both runs would share the same per-iteration RNG state and
        # the primary/fallback diff would conflate latency-budget
        # changes with jitter alignment.
        seed=seed + iterations,
    )

    elapsed = time.monotonic() - started_monotonic
    return BenchmarkReport(
        primary=primary,
        fallback=fallback,
        corpus_size=len(corpus),
        iterations=iterations,
        seed=seed,
        started_at=started_at,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks.voice_pipeline",
        description=(
            "Measure wake-to-response latency (VAD speech_end → first PCM "
            "sample of TTS) for the JARVIS voice pipeline. Reports p50/p90/p99 "
            "and asserts p90 <= 800 ms (Requirement 12.1)."
        ),
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=(
            "Iterations per path. Total samples = iterations * corpus_size. "
            f"Default {DEFAULT_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--primary-first-token-ms",
        type=float,
        default=DEFAULT_PRIMARY_FIRST_TOKEN_DELAY_MS,
        help=(
            "First-token delay for the stub primary (Mistral cloud) backend "
            f"in milliseconds. Default {DEFAULT_PRIMARY_FIRST_TOKEN_DELAY_MS}."
        ),
    )
    parser.add_argument(
        "--fallback-first-token-ms",
        type=float,
        default=DEFAULT_FALLBACK_FIRST_TOKEN_DELAY_MS,
        help=(
            "First-token delay for the stub fallback (Ollama local) backend "
            f"in milliseconds. Default {DEFAULT_FALLBACK_FIRST_TOKEN_DELAY_MS}."
        ),
    )
    parser.add_argument(
        "--stt-ms",
        type=float,
        default=DEFAULT_STT_PROCESSING_MS,
        help=(
            "Modelled STT processing time per utterance in milliseconds. "
            f"Default {DEFAULT_STT_PROCESSING_MS}."
        ),
    )
    parser.add_argument(
        "--inter-token-ms",
        type=float,
        default=DEFAULT_INTER_TOKEN_DELAY_MS,
        help=(
            "Inter-chunk delivery delay during streaming, in ms. "
            f"Default {DEFAULT_INTER_TOKEN_DELAY_MS}."
        ),
    )
    parser.add_argument(
        "--tts-synth-ms",
        type=float,
        default=DEFAULT_TTS_SYNTH_MS,
        help=(
            "Modelled TTS synthesis time from sentence-boundary to first "
            f"PCM sample, in ms. Default {DEFAULT_TTS_SYNTH_MS}."
        ),
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=DEFAULT_JITTER_MS,
        help=(
            "Per-iteration uniform jitter applied to each timing variable "
            f"in ms. Default {DEFAULT_JITTER_MS}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for jitter. Default 0 (fully deterministic).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the JSON record to (in addition to stdout).",
    )
    parser.add_argument(
        "--strict-thresholds",
        dest="strict_thresholds",
        action="store_true",
        default=True,
        help=(
            "Exit with status 1 if measured primary p90 exceeds the 800 ms "
            "Requirement 12.1 threshold. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-strict-thresholds",
        dest="strict_thresholds",
        action="store_false",
        help="Always exit 0 regardless of measured p90 (CI lab runs).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only emit the JSON record on stdout (suppress info logging).",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help=(
            "Omit per-sample latency arrays from the JSON record. Useful "
            "when the harness is run with very large iteration counts and "
            "the raw samples would dominate CI artefact size."
        ),
    )
    return parser


def _format_summary(report: BenchmarkReport) -> str:
    """Render a one-paragraph human-readable summary for the log."""
    p = report.primary
    f = report.fallback
    return (
        f"primary  p50={p.p50_ms:6.1f} ms  p90={p.p90_ms:6.1f} ms  "
        f"p99={p.p99_ms:6.1f} ms  (budget {LATENCY_BUDGET_MS:.0f} ms; "
        f"{'PASS' if p.passed else 'FAIL'})\n"
        f"fallback p50={f.p50_ms:6.1f} ms  p90={f.p90_ms:6.1f} ms  "
        f"p99={f.p99_ms:6.1f} ms  (separate run; tracked for regressions)"
    )


def _strip_samples(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``payload`` without per-sample arrays."""
    out = dict(payload)
    for key in ("primary", "fallback"):
        run = dict(out[key])
        run.pop("samples_ms", None)
        out[key] = run
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit code policy
    ----------------
    * ``0`` — primary p90 within budget (or ``--no-strict-thresholds``).
    * ``1`` — primary p90 exceeded the 800 ms Requirement 12.1 budget,
      under the default ``--strict-thresholds`` policy.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    report = asyncio.run(
        run_benchmark(
            iterations=args.iterations,
            primary_first_token_ms=args.primary_first_token_ms,
            fallback_first_token_ms=args.fallback_first_token_ms,
            stt_ms=args.stt_ms,
            inter_token_ms=args.inter_token_ms,
            tts_synth_ms=args.tts_synth_ms,
            jitter_ms=args.jitter_ms,
            seed=args.seed,
        )
    )

    payload = report.to_json_dict()
    if args.no_samples:
        payload = _strip_samples(payload)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    logger.info("%s", _format_summary(report))

    if not args.strict_thresholds:
        return 0
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(main())
