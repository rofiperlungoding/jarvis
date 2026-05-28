"""Daily memory compaction: condense old ``chat`` records into ``summary`` records.

This module implements :class:`MemoryCompactor`, the daily background task
described in ``design.md §Memory_Store``::

    Summarization: a daily ``MemoryCompactor`` task condenses old turn-
    records into category summaries via the ``LLM_Backend``.

The compactor satisfies two acceptance criteria from Requirement 10:

* **10.1** — every conversation turn is persisted as a ``Memory_Record``.
  Once those records exceed the configured age threshold (default seven
  days), the compactor replaces the noisy raw chat history with a single
  ``summary`` record so the vector index stays small and so retrieval
  surfaces high-signal context to the LLM_Backend.
* **10.4** — retrieved memory records are forwarded to the LLM_Backend in
  a clearly delimited "memory" section. By keeping summaries on the same
  ChromaDB collection as raw chat records, the existing
  :meth:`MemoryStore.retrieve` path naturally surfaces them: the
  ``Dialog_Manager`` does not need to know that compaction has happened.

Algorithm
---------

A single ``run_once()`` execution performs the following steps:

1. **Snapshot the cut-off.** Compute ``cutoff = time_source.now() - max_age``.
   The ``time_source`` is injectable so unit tests can drive the clock
   deterministically (matching :class:`~jarvis.utils.time_source.TimeSource`
   used elsewhere — Requirement 6.2 / 17.3).
2. **List eligible records.** Call
   :meth:`MemoryStore.list_records(category="chat", older_than=cutoff)`.
   ChromaDB applies the category filter; the store applies the timestamp
   filter post-decode.
3. **Bail out early** when fewer than ``min_records`` (default 2) match.
   A single record is not worth summarising — the summary would just be
   the record itself with extra hallucination risk.
4. **Render a stable transcript.** Sort the records by timestamp and
   concatenate them into a single prompt body. Each turn is emitted on
   its own line so the model sees the natural conversational order, and
   so the prompt is greppable in audit logs if anything goes wrong.
5. **Stream a summary out of the LLM.** Open
   :meth:`LLMBackend.stream` with no tools (this is a non-tool task), drop
   tool-call events defensively, and accumulate text deltas into a
   summary string. Empty / whitespace-only completions are treated as a
   no-op: the compactor leaves the original records in place and logs a
   warning rather than persisting an empty summary that would degrade
   retrieval.
6. **Persist the summary.** Call :meth:`MemoryStore.persist_fact` with
   ``category="summary"``; the resulting record's id is recorded in the
   provenance of every forgotten chat record via ``source_id`` so an
   operator running ``MemoryAdminSkill list`` can trace a summary back
   to the turns it replaced.
7. **Forget the originals.** Each chat record id is fed to
   :meth:`MemoryStore.forget`. The mutation set is best-effort: a
   :class:`Exception` from one ``forget`` is logged and swallowed so the
   rest of the batch still completes. Property 4 / CP4 guarantees that
   subsequent retrievals will not see the deleted records.

The order matters: persist *before* forgetting. If the process crashes
mid-run, the worst case is a duplicated summary on the next pass (which
the LLM will naturally absorb as redundant input) rather than data loss.

Scheduling
----------

The class exposes :meth:`start_daily_task` for production wiring: an
asyncio task that loops forever, waking up every ``interval_seconds``
(default 24 h) to call :meth:`run_once`. The companion :meth:`stop`
cancels the loop cleanly. The loop swallows exceptions out of
``run_once`` so a transient LLM failure does not take down the
application — the next iteration will try again.

Tests construct a :class:`MemoryCompactor` directly and call
:meth:`run_once`; they never need :meth:`start_daily_task`.

Validates: Requirements 10.1, 10.4
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
import logging
from typing import Any

from jarvis.llm.base import LLMBackend, Message, ToolDefinition
from jarvis.memory.store import MemoryRecord, MemoryStore
from jarvis.utils.time_source import SystemTimeSource, TimeSource

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COMPACTION_AGE",
    "DEFAULT_COMPACTION_INTERVAL",
    "DEFAULT_MAX_RECORDS_PER_RUN",
    "DEFAULT_MIN_RECORDS",
    "DEFAULT_SUMMARY_MODEL_KW",
    "MemoryCompactor",
    "MemoryCompactorResult",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


#: Records older than this are eligible for compaction. The seven-day
#: window matches the design doc's "old turn-records" wording: long enough
#: that a user is unlikely to revisit the raw transcript, short enough
#: that the summary remains a useful retrieval target.
DEFAULT_COMPACTION_AGE: timedelta = timedelta(days=7)

#: How often :meth:`MemoryCompactor.start_daily_task` re-runs the
#: compaction. Daily — hence the task name. Tests can pass a smaller
#: value to make the loop test-friendly.
DEFAULT_COMPACTION_INTERVAL: timedelta = timedelta(days=1)

#: Compactor will skip a run if fewer than this many records match. Two
#: is the smallest count where summarisation can plausibly compress;
#: with one record the summary tends to be the record itself.
DEFAULT_MIN_RECORDS: int = 2

#: Hard cap on the number of records summarised in a single LLM call.
#: Prevents runaway prompt sizes when a user has been chatting for
#: months with compaction disabled. The remainder is picked up on the
#: next daily run.
DEFAULT_MAX_RECORDS_PER_RUN: int = 200

#: Default keyword arguments forwarded to :meth:`LLMBackend.stream`. We
#: keep temperature low so the summary is deterministic-ish across runs
#: of the same input, which makes property tests on the compactor
#: tractable.
DEFAULT_SUMMARY_MODEL_KW: dict[str, Any] = {"temperature": 0.2}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryCompactorResult:
    """Outcome of a single :meth:`MemoryCompactor.run_once` invocation.

    Returned to callers (and surfaced in logs) so an operator can verify
    that compaction is making progress without grepping through DEBUG
    output. ``summary_record_id`` is ``None`` when the run was a no-op
    (no records eligible, or the LLM produced an empty summary); in that
    case ``forgotten_ids`` is also empty.
    """

    summary_record_id: str | None
    forgotten_ids: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None  # populated when no summary was produced


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


# Default system prompt for the summarisation call. Kept short — the
# meat is the user message that contains the rendered transcript. The
# prompt is intentionally generic about the persona because the daily
# task runs out of band of any conversation; a witty summary is not a
# requirement.
_DEFAULT_SYSTEM_PROMPT = (
    "You compress old conversation transcripts into short, factual "
    "summaries for a personal assistant's long-term memory. Preserve "
    "user preferences, decisions, and unresolved tasks. Drop chit-chat. "
    "Output a single paragraph; do not use bullet points or headings."
)

_DEFAULT_USER_PROMPT_HEADER = (
    "Summarise the following conversation transcript into one paragraph "
    "for long-term memory. Use third person.\n\nTranscript:\n"
)


class MemoryCompactor:
    """Daily summariser that compacts old ``chat`` records into ``summary`` records.

    Parameters
    ----------
    memory_store:
        The :class:`MemoryStore` whose ``chat`` records will be
        summarised. Must be the same instance used by the
        Dialog_Manager so the new ``summary`` records share the
        ChromaDB collection and surface in retrieval (Requirement 10.4).
    llm_backend:
        Any object satisfying :class:`LLMBackend`. The compactor opens
        a :meth:`LLMBackend.stream` per run with no tools — function
        calling is irrelevant for summarisation and Requirement 19.4
        permits an empty ``tools`` list.
    time_source:
        Injectable :class:`TimeSource`. Defaults to
        :class:`SystemTimeSource`. Tests substitute
        :class:`~jarvis.utils.time_source.FakeTimeSource` to advance
        time deterministically.
    max_age:
        Records older than ``time_source.now() - max_age`` are eligible
        for compaction. Defaults to :data:`DEFAULT_COMPACTION_AGE`
        (7 days).
    min_records:
        Floor on the number of records needed to trigger a run. When
        fewer match, ``run_once`` returns an empty result.
    max_records_per_run:
        Cap on the number of records summarised in a single LLM call.
        See :data:`DEFAULT_MAX_RECORDS_PER_RUN`.
    interval:
        How often :meth:`start_daily_task` re-runs ``run_once``.
        Defaults to one day.
    system_prompt:
        Override for the LLM system prompt. The default is intentionally
        terse and persona-free.
    summary_model_kwargs:
        Backend-specific keyword arguments forwarded to
        :meth:`LLMBackend.stream` (``model``, ``temperature``,
        ``max_tokens`` ...). Defaults to
        :data:`DEFAULT_SUMMARY_MODEL_KW`.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_backend: LLMBackend,
        time_source: TimeSource | None = None,
        *,
        max_age: timedelta = DEFAULT_COMPACTION_AGE,
        min_records: int = DEFAULT_MIN_RECORDS,
        max_records_per_run: int = DEFAULT_MAX_RECORDS_PER_RUN,
        interval: timedelta = DEFAULT_COMPACTION_INTERVAL,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        summary_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(max_age, timedelta):
            raise TypeError("max_age must be a datetime.timedelta")
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        if not isinstance(min_records, int) or min_records < 1:
            raise ValueError("min_records must be a positive int")
        if (
            not isinstance(max_records_per_run, int)
            or max_records_per_run < min_records
        ):
            raise ValueError(
                "max_records_per_run must be an int >= min_records"
            )
        if not isinstance(interval, timedelta):
            raise TypeError("interval must be a datetime.timedelta")
        if interval <= timedelta(0):
            raise ValueError("interval must be positive")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("system_prompt must be a non-empty str")

        self._store = memory_store
        self._llm = llm_backend
        self._time = time_source if time_source is not None else SystemTimeSource()
        self._max_age = max_age
        self._min_records = int(min_records)
        self._max_records_per_run = int(max_records_per_run)
        self._interval = interval
        self._system_prompt = system_prompt
        self._summary_model_kwargs = (
            dict(summary_model_kwargs)
            if summary_model_kwargs is not None
            else dict(DEFAULT_SUMMARY_MODEL_KW)
        )

        # Background-loop bookkeeping.
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def max_age(self) -> timedelta:
        """Cut-off age for chat records eligible for compaction."""
        return self._max_age

    @property
    def interval(self) -> timedelta:
        """Sleep between successive ``run_once`` invocations from the daily task."""
        return self._interval

    @property
    def is_running(self) -> bool:
        """``True`` while :meth:`start_daily_task` has an active loop."""
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> MemoryCompactorResult:
        """Run a single compaction pass.

        Returns a :class:`MemoryCompactorResult` describing the outcome.
        Never raises for "expected" no-op cases (no eligible records,
        empty LLM output); raises for genuinely exceptional failures
        (LLM connection error, ChromaDB I/O error) so the caller —
        typically the daily loop — can decide whether to retry.
        """
        cutoff = self._time.now() - self._max_age
        candidates = await self._store.list_records(
            category="chat",
            older_than=cutoff,
        )
        if len(candidates) < self._min_records:
            logger.debug(
                "MemoryCompactor.run_once: %d candidate(s); below floor %d, skipping",
                len(candidates),
                self._min_records,
            )
            return MemoryCompactorResult(
                summary_record_id=None,
                forgotten_ids=(),
                reason="below_min_records",
            )

        # Sort by timestamp ascending so the LLM sees the conversation
        # in chronological order, and trim to the configured per-run
        # cap. The remainder is picked up on the next daily run.
        candidates.sort(key=lambda r: r.timestamp)
        batch = candidates[: self._max_records_per_run]

        transcript = self._render_transcript(batch)
        summary_text = await self._summarise(transcript)
        if not summary_text or not summary_text.strip():
            logger.warning(
                "MemoryCompactor.run_once: LLM produced an empty summary "
                "for %d records; leaving originals in place",
                len(batch),
            )
            return MemoryCompactorResult(
                summary_record_id=None,
                forgotten_ids=(),
                reason="empty_summary",
            )

        # Persist before forgetting: a crash after this line means we
        # might have a duplicated summary next time, which the LLM will
        # absorb as redundant input. The reverse order would risk data
        # loss.
        summary_record = await self._store.persist_fact(
            content=summary_text.strip(),
            category="summary",
            source_id=batch[0].record_id,
        )

        forgotten: list[str] = []
        for record in batch:
            try:
                removed = await self._store.forget(record.record_id)
            except Exception:  # ChromaDB I/O is the realistic failure here
                logger.exception(
                    "MemoryCompactor.run_once: forget(%s) failed; continuing",
                    record.record_id,
                )
                continue
            if removed:
                forgotten.append(record.record_id)

        logger.info(
            "MemoryCompactor.run_once: summarised %d chat record(s) into %s; "
            "forgot %d",
            len(batch),
            summary_record.record_id,
            len(forgotten),
        )

        return MemoryCompactorResult(
            summary_record_id=summary_record.record_id,
            forgotten_ids=tuple(forgotten),
            reason=None,
        )

    def start_daily_task(self) -> asyncio.Task[None]:
        """Launch a background asyncio task that runs ``run_once`` periodically.

        Returns the created :class:`asyncio.Task` so the caller can
        ``await`` cancellation if desired. Calling this method twice
        without an intervening :meth:`stop` is a no-op — the existing
        task is returned. Exceptions raised inside ``run_once`` are
        caught and logged so a transient LLM failure does not crash
        the application; the next iteration retries.
        """
        if self._task is not None and not self._task.done():
            return self._task

        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._daily_loop(self._stop_event),
            name="MemoryCompactor.daily",
        )
        return self._task

    async def stop(self) -> None:
        """Cancel the daily task started by :meth:`start_daily_task`.

        Idempotent. Safe to call when no task is running.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        if task is None:
            return
        if task.done():
            self._task = None
            self._stop_event = None
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            # We don't re-raise: stop() is a clean-up call invoked
            # during application shutdown, and the loop's own
            # exception handler has already logged anything noteworthy.
            pass
        finally:
            self._task = None
            self._stop_event = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _daily_loop(self, stop_event: asyncio.Event) -> None:
        """Background loop driven by :meth:`start_daily_task`."""
        interval_seconds = self._interval.total_seconds()
        while not stop_event.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "MemoryCompactor daily loop: run_once raised; "
                    "will retry next interval"
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=interval_seconds,
                )
            except TimeoutError:
                continue

    @staticmethod
    def _render_transcript(records: list[MemoryRecord]) -> str:
        """Concatenate chat records into a single LLM-ready transcript.

        Each record's ``content`` is already the ``"User: ...\\nAssistant: ..."``
        string written by :meth:`MemoryStore.persist_turn`. We separate
        successive turns with a blank line so the model can tell them
        apart even when one turn ends without a trailing newline.
        """
        chunks: list[str] = []
        for record in records:
            timestamp = record.timestamp.isoformat()
            chunks.append(f"[{timestamp}]\n{record.content.strip()}")
        return "\n\n".join(chunks)

    async def _summarise(self, transcript: str) -> str:
        """Drive a single LLM stream to produce a summary string."""
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": _DEFAULT_USER_PROMPT_HEADER + transcript,
            },
        ]
        tools: list[ToolDefinition] = []  # summarisation never calls tools

        accumulator: list[str] = []
        async with self._llm.stream(
            messages,
            tools=tools,
            **self._summary_model_kwargs,
        ) as events:
            async for event in events:
                # Defensive dispatch on the event-type discriminator
                # shared by all backends (see ``llm.base``). Tool-call
                # events are unexpected — we passed ``tools=[]`` — but
                # if a backend emits one anyway we drop it rather than
                # crashing the daily loop.
                if getattr(event, "type", None) == "content_delta":
                    text = getattr(event, "text", "")
                    if text:
                        accumulator.append(text)
                # Any other event type is ignored.
        return "".join(accumulator)
