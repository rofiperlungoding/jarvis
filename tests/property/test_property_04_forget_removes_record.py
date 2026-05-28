"""Property test for Property 4 — ``MemoryStore.forget`` removes the record.

From ``design.md §Correctness Properties``:

    *For any* ``MemoryStore`` ``M`` containing record ``R`` and *for any*
    subsequent query ``Q`` and integer ``K``, after ``M.forget(R.record_id)``,
    no call to ``M.retrieve(Q, K)`` SHALL return a record whose
    ``record_id`` equals ``R.record_id``.

This file implements that universal quantification with Hypothesis. The
strategy generates a corpus of ``MemoryRecord``-shaped inputs (content +
category), persists each via :meth:`MemoryStore.persist_fact` against a
deterministic embedder, picks an arbitrary persisted ``record_id``,
calls :meth:`MemoryStore.forget`, and then re-runs :meth:`retrieve`
across a battery of queries / ``k`` values to assert that the deleted
id never resurfaces.

The test also covers the closed-taxonomy half of the contract: forgetting
a non-existent ``record_id`` returns ``False`` without raising, so a
caller that asks the store to forget an unknown id (e.g. a malformed
``MemoryAdminSkill.forget`` request) sees a clean negative answer
rather than an exception.

The store is wired with:

* :class:`HashEmbedder` — deterministic, dependency-free, so retrieval
  ordering is reproducible across Hypothesis examples (also satisfies
  CP3 in adjacent tests).
* :class:`NullDPAPI` — keeps the encrypted-at-rest invariants on the
  same code path as production while avoiding the Windows-only
  ``win32crypt`` dependency in CI.
* ``_FakeChromaDB`` — an in-memory ChromaDB substitute (mirrors the
  shim used by ``tests/unit/memory/test_compactor.py``) so the property
  test does not pay the multi-second ChromaDB warm-up on every example.

Validates: Requirements 10.5, 10.6, 13.5 (CP4)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from jarvis.memory.embedder import HashEmbedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryStore
from jarvis.security.dpapi import NullDPAPI

# ---------------------------------------------------------------------------
# In-memory ChromaDB stand-in (mirrors ``tests/unit/memory/test_compactor.py``)
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Tiny ChromaDB ``Collection`` substitute keyed by ``id``.

    Only the surface area used by :class:`MemoryStore` is implemented:
    ``count``, ``add``, ``get``, ``query``, and ``delete``. The query
    method returns rows in stored order with constant distances; that
    is sufficient for Property 4 because the property only checks
    *which* record ids are returned, not their ordering.
    """

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
        del query_embeddings, include  # constant-distance stub
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

    # The real ``chromadb`` module exposes ``PersistentClient`` as a
    # callable returning a client; method-name parity is required by
    # :class:`MemoryStore`'s lazy-import path. ``N802`` is suppressed
    # for tests via ``ruff.toml``.
    def PersistentClient(self, *, path: str) -> _FakeClient:
        del path
        return self._client


# ---------------------------------------------------------------------------
# Hypothesis strategy for the persist + forget corpus
# ---------------------------------------------------------------------------


# Free-form printable text for the persisted ``content`` field. We avoid
# surrogates / control chars so the assertion failures are easy to read,
# and we cap the size so Hypothesis spends its budget on shape coverage
# rather than on long strings.
_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFD,
        exclude_categories=("Cs",),  # type: ignore[arg-type]
    ),
    min_size=1,
    max_size=48,
)

# The closed set of ``MemoryRecord.category`` values mirrored from
# ``design.md §Data Models``. Persisting across categories means the
# property holds regardless of what the caller stores.
_categories = st.sampled_from(("preference", "fact", "summary"))


@st.composite
def _record_inputs(
    draw: st.DrawFn,
    *,
    min_records: int = 2,
    max_records: int = 6,
) -> list[tuple[str, str]]:
    """Generate a list of ``(content, category)`` pairs to persist.

    At least two records are produced so the post-forget retrieval has
    something else to return — this exposes regressions where ``forget``
    accidentally clears the whole collection rather than the targeted
    id.
    """
    n = draw(st.integers(min_value=min_records, max_value=max_records))
    return [
        (draw(_safe_text), draw(_categories))
        for _ in range(n)
    ]


# A small set of free-form queries used to probe the post-forget store.
# Including the empty string is deliberate: ``MemoryStore.retrieve``
# accepts it (the embedder treats it as any other input) and we want
# the property to hold for every legal query.
_queries = st.lists(
    st.text(
        alphabet=st.characters(
            min_codepoint=0x20,
            max_codepoint=0xFFFD,
            exclude_categories=("Cs",),  # type: ignore[arg-type]
        ),
        min_size=0,
        max_size=32,
    ),
    min_size=1,
    max_size=4,
)


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> MemoryStore:
    """Construct a fresh :class:`MemoryStore` backed by the in-memory shim.

    Each Hypothesis example gets its own store (and therefore its own
    fake ChromaDB collection) because the property quantifies over
    *fresh* stores: state from a previous example must not bleed into
    the next.
    """
    return MemoryStore(
        db_path=tmp_path / "chroma",
        embedder=HashEmbedder(dimension=16),
        dpapi=NullDPAPI(),  # type: ignore[arg-type]
        redactor=PIIRedactor.with_defaults(),
        # Disable redaction so the persisted text round-trips verbatim
        # — the property is about ``forget``, not redaction.
        redaction_enabled=False,
        chromadb_module=_FakeChromaDB(),
    )


# ---------------------------------------------------------------------------
# Property 4 — forget removes the record
# ---------------------------------------------------------------------------


@given(
    inputs=_record_inputs(),
    queries=_queries,
    target_index=st.integers(min_value=0, max_value=99),
    k_values=st.lists(
        st.integers(min_value=1, max_value=16),
        min_size=1,
        max_size=4,
    ),
)
@settings(
    # Inherit ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile in ``tests/conftest.py``. The
    # health-check suppression handles the small per-example fixed
    # overhead of constructing the store, which Hypothesis would
    # otherwise classify as ``too_slow`` on slower CI runners.
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow),
)
def test_forget_removes_record(
    tmp_path: Path,
    inputs: list[tuple[str, str]],
    queries: list[str],
    target_index: int,
    k_values: list[int],
) -> None:
    """After ``forget(R.record_id)``, no ``retrieve`` returns ``R.record_id``.

    **Validates: Requirements 10.5, 10.6, 13.5 (CP4)**
    """

    async def _run() -> None:
        store = _make_store(tmp_path)

        # Persist every input as a typed fact. Using ``persist_fact``
        # (rather than ``persist_turn``) keeps the call surface compact
        # and exercises the same internal write path
        # (``_persist_record``) that ``persist_turn`` uses.
        record_ids: list[str] = []
        for content, category in inputs:
            record = await store.persist_fact(content=content, category=category)
            record_ids.append(record.record_id)

        # Sanity: the strategy guarantees ``len(inputs) >= 2``, so we
        # have at least two ids to choose from. Pick the target with a
        # modulo so Hypothesis can shrink ``target_index`` independently
        # of the corpus size.
        assert len(record_ids) >= 2
        target_id = record_ids[target_index % len(record_ids)]

        # The record should exist *before* the forget call. We probe
        # via ``list_records`` so we are sure the persistence path
        # produced exactly the ids we collected — a regression that
        # silently dropped writes would otherwise hide behind a
        # vacuously-true "post-forget never returns the id" assertion.
        all_records_pre = await store.list_records()
        ids_pre = {r.record_id for r in all_records_pre}
        assert target_id in ids_pre, (
            "precondition: target id was not persisted before forget"
        )

        # Forgetting an existing id MUST return True (Requirement 10.6).
        forgot = await store.forget(target_id)
        assert forgot is True

        # Property 4: no subsequent retrieve, for any query and any k,
        # returns the deleted record id. We probe with several
        # ``(query, k)`` combinations so the universal quantifier in
        # the property statement is meaningfully exercised even on a
        # fixed corpus.
        for query in queries:
            for k in k_values:
                hits = await store.retrieve(query, k=k)
                returned_ids = {hit.record_id for hit in hits}
                assert target_id not in returned_ids, (
                    f"forgotten id {target_id!r} resurfaced for "
                    f"query={query!r}, k={k}: {returned_ids}"
                )

        # ``list_records`` (the ``MemoryAdminSkill.list`` path) MUST
        # also drop the forgotten id. Without this the
        # ``MemoryAdminSkill`` UI could continue to surface a record
        # that ``retrieve`` no longer returns.
        all_records_post = await store.list_records()
        ids_post = {r.record_id for r in all_records_post}
        assert target_id not in ids_post

        # Forgetting the same id a second time is a no-op that returns
        # False (closed taxonomy: no exception is raised).
        forgot_again = await store.forget(target_id)
        assert forgot_again is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: forgetting an unknown id returns False
# ---------------------------------------------------------------------------


@given(
    inputs=_record_inputs(min_records=1, max_records=4),
    bogus_id=st.text(
        alphabet=st.characters(
            min_codepoint=0x21,
            max_codepoint=0x7E,
            exclude_categories=("Cs",),  # type: ignore[arg-type]
        ),
        min_size=1,
        max_size=32,
    ),
)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_forget_unknown_id_returns_false(
    tmp_path: Path,
    inputs: list[tuple[str, str]],
    bogus_id: str,
) -> None:
    """Forgetting an id that was never persisted returns ``False`` cleanly.

    Property 4 demands that the post-condition ("no retrieve returns
    the deleted id") still hold even when the caller asks the store to
    forget an id that was never written. The cleanest way to express
    this in the closed taxonomy is: ``forget`` returns ``False`` and
    does not raise, and existing records are unaffected.

    **Validates: Requirements 10.5, 10.6, 13.5 (CP4)**
    """

    async def _run() -> None:
        store = _make_store(tmp_path)
        persisted_ids: list[str] = []
        for content, category in inputs:
            record = await store.persist_fact(content=content, category=category)
            persisted_ids.append(record.record_id)

        # Avoid the (astronomically unlikely) collision where Hypothesis
        # generates a string equal to one of our UUID4 ids.
        if bogus_id in persisted_ids:
            return

        result = await store.forget(bogus_id)
        assert result is False

        # Persisted records remain intact.
        all_records = await store.list_records()
        assert {r.record_id for r in all_records} == set(persisted_ids)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: forget rejects malformed ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_rejects_empty_record_id(tmp_path: Path) -> None:
    """``forget`` raises on an empty ``record_id`` per the docstring contract.

    The closed-taxonomy guarantee for "non-existent record_id returns
    False without raising" only applies to *well-formed* identifiers
    (non-empty strings). A blank string is rejected at the type-check
    boundary; this companion test pins that contract so it is not
    relaxed accidentally.
    """
    store = _make_store(tmp_path)
    with pytest.raises(ValueError):
        await store.forget("")


@pytest.mark.asyncio
async def test_forget_rejects_non_string_record_id(tmp_path: Path) -> None:
    """``forget`` raises ``TypeError`` for a non-string ``record_id``.

    Same rationale as :func:`test_forget_rejects_empty_record_id` — pins
    the type-check boundary so a regression cannot quietly accept
    ``None`` or ``int`` ids.
    """
    store = _make_store(tmp_path)
    with pytest.raises(TypeError):
        await store.forget(123)  # type: ignore[arg-type]
