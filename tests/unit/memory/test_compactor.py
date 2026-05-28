"""Unit tests for ``jarvis.memory.compactor``.

Covers:
    * Constructor validation (max_age, min_records, intervals).
    * ``run_once`` no-ops when fewer than ``min_records`` are eligible.
    * ``run_once`` summarises eligible chat records, persists a summary,
      and forgets the originals (Requirements 10.1, 10.4).
    * Records younger than the configured ``max_age`` are left alone.
    * Records of other categories are not eligible for compaction.
    * Empty LLM output leaves originals untouched.
    * Tool-call events from the LLM are ignored.
    * Per-run record cap defers excess records to the next pass.
    * Summary record's source_id provenance points back at the input.
    * ``start_daily_task`` / ``stop`` lifecycle is idempotent.

Validates: Requirements 10.1, 10.4
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import uuid

import pytest

from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)
from jarvis.memory.compactor import (
    DEFAULT_COMPACTION_AGE,
    DEFAULT_COMPACTION_INTERVAL,
    MemoryCompactor,
    MemoryCompactorResult,
)
from jarvis.memory.embedder import HashEmbedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryStore
from jarvis.security.dpapi import NullDPAPI
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedLLM:
    """Minimal :class:`LLMBackend` test double.

    Yields a pre-canned list of events on every ``stream`` call. Records
    the messages, tools, and kwargs passed by the caller so tests can
    assert the compactor wired the call correctly.
    """

    events: list[LLMEvent]
    captured_messages: list[list[Message]] = field(default_factory=list)
    captured_tools: list[list[ToolDefinition]] = field(default_factory=list)
    captured_kwargs: list[dict[str, Any]] = field(default_factory=list)
    call_count: int = 0

    @asynccontextmanager
    async def _stream_cm(self) -> AsyncIterator[AsyncIterator[LLMEvent]]:
        async def gen() -> AsyncIterator[LLMEvent]:
            for event in self.events:
                yield event

        yield gen()

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.captured_messages.append(list(messages))
        self.captured_tools.append(list(tools))
        self.captured_kwargs.append(dict(kwargs))
        return self._stream_cm()


def _content_events(text: str) -> list[LLMEvent]:
    """Return a single content delta carrying ``text``."""
    return [ContentDeltaEvent(text=text)]


def _tool_call_event(skill: str = "Noop") -> ToolCallEvent:
    return ToolCallEvent(
        tool_call=ToolCall(
            id="call-x",
            skill_name=skill,
            arguments={},
            raw_arguments="{}",
        )
    )


# ---------------------------------------------------------------------------
# Fake chromadb stand-in (in-memory, single collection)
# ---------------------------------------------------------------------------


class _FakeCollection:
    """A tiny ChromaDB ``Collection`` substitute keyed by ``id``."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self.rows)

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for i, rid in enumerate(ids):
            self.rows[rid] = {
                "embedding": list(embeddings[i]),
                "document": documents[i],
                "metadata": dict(metadatas[i]),
            }

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        keys = list(ids) if ids is not None else list(self.rows.keys())
        out_ids: list[str] = []
        out_docs: list[str] = []
        out_metas: list[dict[str, Any]] = []
        out_embs: list[list[float]] = []
        for rid in keys:
            row = self.rows.get(rid)
            if row is None:
                continue
            if where is not None:
                metadata = row["metadata"]
                if not all(metadata.get(k) == v for k, v in where.items()):
                    continue
            out_ids.append(rid)
            out_docs.append(row["document"])
            out_metas.append(dict(row["metadata"]))
            out_embs.append(list(row["embedding"]))
        result: dict[str, Any] = {"ids": out_ids}
        if include is None or "documents" in include:
            result["documents"] = out_docs
        if include is None or "metadatas" in include:
            result["metadatas"] = out_metas
        if include is None or "embeddings" in include:
            result["embeddings"] = out_embs
        return result

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        # The compactor never calls retrieve; provide a stub for completeness.
        ids = list(self.rows.keys())[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i]["document"] for i in ids]],
            "metadatas": [[dict(self.rows[i]["metadata"]) for i in ids]],
            "embeddings": [[list(self.rows[i]["embedding"]) for i in ids]],
            "distances": [[0.0 for _ in ids]],
        }

    def delete(self, *, ids: list[str]) -> None:
        for rid in ids:
            self.rows.pop(rid, None)


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> _FakeCollection:
        del metadata, embedding_function
        return self.collections.setdefault(name, _FakeCollection())

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


class _FakeChromaDB:
    """Minimal chromadb shim used as ``chromadb_module`` constructor arg."""

    def __init__(self) -> None:
        self._client = _FakeClient()

    def PersistentClient(self, *, path: str) -> _FakeClient:
        del path
        return self._client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_now() -> datetime:
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def time_source(fake_now: datetime) -> FakeTimeSource:
    return FakeTimeSource(now=fake_now)


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        db_path=tmp_path / "chroma",
        embedder=HashEmbedder(dimension=16),
        dpapi=NullDPAPI(),
        redactor=PIIRedactor.with_defaults(),
        redaction_enabled=False,
        chromadb_module=_FakeChromaDB(),
    )


async def _seed_chat_record(
    store: MemoryStore,
    *,
    content: str,
    when: datetime,
    record_id: str | None = None,
) -> str:
    """Write a single ``chat`` record at a controlled timestamp.

    The public :meth:`MemoryStore.persist_turn` always stamps records
    with ``finished_at`` / ``now``. The compactor's contract is "older
    than threshold", so tests need to inject precisely-timestamped
    records. We reach into the underlying collection to do that — the
    only test-specific affordance in this module.
    """
    rid = record_id or str(uuid.uuid4())
    embedding = store._embedder.embed(content)
    ciphertext = store._dpapi.protect(
        content.encode("utf-8"), entropy=b"jarvis/memory_store/v1"
    )
    document = base64.b64encode(ciphertext).decode("ascii")
    metadata = {
        "category": "chat",
        "timestamp": when.isoformat(),
        "redacted": False,
        "model_name": store._embedder.model_name,
        "prov_source": "turn",
    }
    store._collection.add(
        ids=[rid],
        # chromadb runtime accepts list[list[float]] and dict metadata.
        embeddings=[embedding],  # type: ignore[arg-type]
        documents=[document],
        metadatas=[metadata],  # type: ignore[list-item]
    )
    return rid


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults_match_design_doc(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=_ScriptedLLM(events=[]),
            time_source=time_source,
        )
        assert compactor.max_age == DEFAULT_COMPACTION_AGE
        assert compactor.interval == DEFAULT_COMPACTION_INTERVAL
        assert compactor.is_running is False

    @pytest.mark.parametrize("bad", [timedelta(0), timedelta(seconds=-1)])
    def test_rejects_non_positive_max_age(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        bad: timedelta,
    ) -> None:
        with pytest.raises(ValueError):
            MemoryCompactor(
                memory_store=memory_store,
                llm_backend=_ScriptedLLM(events=[]),
                time_source=time_source,
                max_age=bad,
            )

    def test_rejects_min_records_below_one(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        with pytest.raises(ValueError):
            MemoryCompactor(
                memory_store=memory_store,
                llm_backend=_ScriptedLLM(events=[]),
                time_source=time_source,
                min_records=0,
            )

    def test_rejects_max_records_below_min(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        with pytest.raises(ValueError):
            MemoryCompactor(
                memory_store=memory_store,
                llm_backend=_ScriptedLLM(events=[]),
                time_source=time_source,
                min_records=5,
                max_records_per_run=2,
            )

    def test_rejects_non_positive_interval(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        with pytest.raises(ValueError):
            MemoryCompactor(
                memory_store=memory_store,
                llm_backend=_ScriptedLLM(events=[]),
                time_source=time_source,
                interval=timedelta(0),
            )


# ---------------------------------------------------------------------------
# run_once behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunOnce:
    async def test_noop_when_no_records_match(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        llm = _ScriptedLLM(events=_content_events("ignored"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        assert result == MemoryCompactorResult(
            summary_record_id=None,
            forgotten_ids=(),
            reason="below_min_records",
        )
        assert llm.call_count == 0

    async def test_noop_when_below_min_records(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        # Single eligible chat record; min_records defaults to 2.
        await _seed_chat_record(
            memory_store,
            content="User: hi\nAssistant: hello",
            when=fake_now - timedelta(days=10),
        )
        llm = _ScriptedLLM(events=_content_events("summary"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        assert result.summary_record_id is None
        assert result.reason == "below_min_records"
        assert llm.call_count == 0

    async def test_summarises_old_chat_records_and_forgets_them(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        old_a = await _seed_chat_record(
            memory_store,
            content="User: where is Bandung?\nAssistant: West Java",
            when=fake_now - timedelta(days=10),
        )
        old_b = await _seed_chat_record(
            memory_store,
            content="User: weather there?\nAssistant: warm and humid",
            when=fake_now - timedelta(days=9),
        )
        llm = _ScriptedLLM(
            events=[
                ContentDeltaEvent(text="The user asked about Bandung; "),
                ContentDeltaEvent(text="JARVIS supplied geography and weather."),
            ]
        )
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )

        result = await compactor.run_once()

        assert result.summary_record_id is not None
        assert set(result.forgotten_ids) == {old_a, old_b}
        assert llm.call_count == 1

        # The summary record landed in the same store with category=summary.
        summaries = await memory_store.list_records(category="summary")
        assert len(summaries) == 1
        assert summaries[0].record_id == result.summary_record_id
        assert (
            "Bandung" in summaries[0].content
            and "weather" in summaries[0].content
        )

        # The originals are gone.
        remaining_chats = await memory_store.list_records(category="chat")
        assert remaining_chats == []

    async def test_summary_provenance_points_at_first_input(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        first = await _seed_chat_record(
            memory_store,
            content="User: A\nAssistant: 1",
            when=fake_now - timedelta(days=20),
            record_id="aaaa",
        )
        await _seed_chat_record(
            memory_store,
            content="User: B\nAssistant: 2",
            when=fake_now - timedelta(days=15),
            record_id="bbbb",
        )
        llm = _ScriptedLLM(events=_content_events("two unrelated turns"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        assert result.summary_record_id is not None
        summaries = await memory_store.list_records(category="summary")
        assert len(summaries) == 1
        assert summaries[0].provenance.get("source_id") == first

    async def test_records_within_threshold_are_not_compacted(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        # Two old records (eligible) and one fresh record (untouched).
        old_a = await _seed_chat_record(
            memory_store,
            content="User: old1\nAssistant: a1",
            when=fake_now - timedelta(days=10),
        )
        old_b = await _seed_chat_record(
            memory_store,
            content="User: old2\nAssistant: a2",
            when=fake_now - timedelta(days=8),
        )
        fresh = await _seed_chat_record(
            memory_store,
            content="User: new\nAssistant: today",
            when=fake_now - timedelta(hours=1),
        )
        llm = _ScriptedLLM(events=_content_events("summary"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()

        assert set(result.forgotten_ids) == {old_a, old_b}
        chats = await memory_store.list_records(category="chat")
        assert [r.record_id for r in chats] == [fresh]

    async def test_other_categories_are_not_eligible(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        # Persist an OLD preference fact directly via the store; without
        # any old chat records, the compactor must do nothing.
        await memory_store.persist_fact(
            content="User prefers metric units.",
            category="preference",
        )
        llm = _ScriptedLLM(events=_content_events("noop"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        assert result.summary_record_id is None
        assert llm.call_count == 0
        # Preference record is still there.
        prefs = await memory_store.list_records(category="preference")
        assert len(prefs) == 1

    async def test_empty_summary_leaves_originals_in_place(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        a = await _seed_chat_record(
            memory_store,
            content="User: a\nAssistant: 1",
            when=fake_now - timedelta(days=10),
        )
        b = await _seed_chat_record(
            memory_store,
            content="User: b\nAssistant: 2",
            when=fake_now - timedelta(days=9),
        )
        # Whitespace-only summary.
        llm = _ScriptedLLM(events=_content_events("   \n  "))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        assert result.summary_record_id is None
        assert result.reason == "empty_summary"
        chats = await memory_store.list_records(category="chat")
        assert {r.record_id for r in chats} == {a, b}
        summaries = await memory_store.list_records(category="summary")
        assert summaries == []

    async def test_tool_call_events_are_ignored(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        await _seed_chat_record(
            memory_store,
            content="User: a\nAssistant: 1",
            when=fake_now - timedelta(days=10),
        )
        await _seed_chat_record(
            memory_store,
            content="User: b\nAssistant: 2",
            when=fake_now - timedelta(days=9),
        )
        llm = _ScriptedLLM(
            events=[
                ContentDeltaEvent(text="hello"),
                _tool_call_event(),  # ignored
                ContentDeltaEvent(text=" world"),
            ]
        )
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        result = await compactor.run_once()
        summaries = await memory_store.list_records(category="summary")
        assert len(summaries) == 1
        assert summaries[0].content == "hello world"
        assert result.summary_record_id == summaries[0].record_id

    async def test_per_run_cap_defers_excess_records(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        # Five eligible records; cap to 3 per run.
        ids = []
        for i in range(5):
            ids.append(
                await _seed_chat_record(
                    memory_store,
                    content=f"User: {i}\nAssistant: r{i}",
                    when=fake_now - timedelta(days=10 + i),
                )
            )
        llm = _ScriptedLLM(events=_content_events("compressed"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
            max_records_per_run=3,
        )
        result = await compactor.run_once()
        # Only three should be forgotten this pass.
        assert len(result.forgotten_ids) == 3
        remaining = await memory_store.list_records(category="chat")
        assert len(remaining) == 2

    async def test_llm_messages_carry_system_and_user_roles(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        await _seed_chat_record(
            memory_store,
            content="User: hi\nAssistant: hello",
            when=fake_now - timedelta(days=10),
        )
        await _seed_chat_record(
            memory_store,
            content="User: bye\nAssistant: goodbye",
            when=fake_now - timedelta(days=9),
        )
        llm = _ScriptedLLM(events=_content_events("compressed"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
        )
        await compactor.run_once()
        msgs = llm.captured_messages[0]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "hi" in msgs[1]["content"] and "bye" in msgs[1]["content"]
        # Tools list is empty per Requirement 19.4.
        assert llm.captured_tools[0] == []


# ---------------------------------------------------------------------------
# Daily task lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDailyLoop:
    async def test_start_runs_compaction_and_stop_cancels(
        self,
        memory_store: MemoryStore,
        time_source: FakeTimeSource,
        fake_now: datetime,
    ) -> None:
        await _seed_chat_record(
            memory_store,
            content="User: a\nAssistant: 1",
            when=fake_now - timedelta(days=10),
        )
        await _seed_chat_record(
            memory_store,
            content="User: b\nAssistant: 2",
            when=fake_now - timedelta(days=9),
        )
        llm = _ScriptedLLM(events=_content_events("done"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
            interval=timedelta(seconds=3600),
        )
        compactor.start_daily_task()
        # Yield the loop a few times so the first run completes before
        # the loop blocks on the long sleep.
        for _ in range(20):
            await asyncio.sleep(0)
            if llm.call_count >= 1:
                break
        assert llm.call_count >= 1
        assert compactor.is_running is True
        await compactor.stop()
        assert compactor.is_running is False

    async def test_start_is_idempotent(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        llm = _ScriptedLLM(events=_content_events("ignored"))
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=llm,
            time_source=time_source,
            interval=timedelta(seconds=3600),
        )
        first = compactor.start_daily_task()
        second = compactor.start_daily_task()
        assert first is second
        await compactor.stop()

    async def test_stop_without_start_is_safe(
        self, memory_store: MemoryStore, time_source: FakeTimeSource
    ) -> None:
        compactor = MemoryCompactor(
            memory_store=memory_store,
            llm_backend=_ScriptedLLM(events=[]),
            time_source=time_source,
        )
        await compactor.stop()
        assert compactor.is_running is False
