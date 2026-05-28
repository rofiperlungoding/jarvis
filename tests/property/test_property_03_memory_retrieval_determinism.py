"""Property test for Property 3 — Memory_Store retrieval determinism.

From ``design.md §Correctness Properties``:

    *For any* ``MemoryStore`` snapshot ``M``, query string ``Q``, and
    integer ``K``, two consecutive calls to ``M.retrieve(Q, K)`` within
    the same session and embedding model version SHALL return identical
    ordered lists of ``MemoryRecord.record_id``.

This file implements that universal quantification with Hypothesis. The
strategy generates a small set of memory records, persists each via
:meth:`MemoryStore.persist_fact` against a deterministic embedder, then
calls :meth:`MemoryStore.retrieve` *twice* for an arbitrary
``(query, k)`` pair and asserts that the two returned ``record_id``
lists are equal *as ordered sequences* (CP3).

Why a deterministic embedder?
-----------------------------

CP3 only holds when the embedding function is deterministic for a
fixed model version. The production
:class:`~jarvis.memory.embedder.SentenceTransformerEmbedder` satisfies
this — Hugging Face's ``all-MiniLM-L6-v2`` is a deterministic encoder
— but loading the model would cost multiple seconds per Hypothesis
example and pull in PyTorch. Instead we use the SHA-256-counter-mode
:class:`~jarvis.memory.embedder.HashEmbedder` test double, which the
embedder module documents as deterministic and stable across processes
and machines (Property 3's exact precondition).

The store is wired with:

* :class:`HashEmbedder` — deterministic, dependency-free.
* :class:`NullDPAPI` — keeps the encrypted-at-rest invariants on the
  same code path as production while avoiding the Windows-only
  ``win32crypt`` dependency in CI.
* ``_FakeChromaDB`` — an in-memory ChromaDB substitute (mirrors the
  shim used by ``tests/unit/memory/test_compactor.py`` and
  ``tests/property/test_property_04_forget_removes_record.py``) so the
  property test does not pay the multi-second ChromaDB warm-up on
  every Hypothesis example.

Validates: Requirements 10.3, 10.4 (CP3)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from jarvis.memory.embedder import HashEmbedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryStore
from jarvis.security.dpapi import NullDPAPI

# ---------------------------------------------------------------------------
# In-memory ChromaDB stand-in
# ---------------------------------------------------------------------------
#
# Mirrors the shim used by ``tests/unit/memory/test_compactor.py`` and
# ``tests/property/test_property_04_forget_removes_record.py``. Only the
# subset of the ChromaDB ``Collection`` API consumed by
# :class:`MemoryStore` is implemented (``count``, ``add``, ``get``,
# ``query``, ``delete``). The stub deliberately mirrors the real
# library's semantics on the dimensions Property 3 cares about:
#
# * ``query`` is a *pure function* of the stored rows and the query
#   embedding — no hidden randomness, no time dependence. Two calls
#   with identical inputs return identical outputs in identical order.
# * ``query`` returns at most ``n_results`` ids in stable insertion
#   order so the test exercises a non-trivial ordering rather than a
#   single-element collection.
#
# The real ChromaDB engine uses an HNSW index that is also
# deterministic for a fixed dataset (the index is rebuilt purely from
# the embeddings); so the stub is a faithful proxy for the property
# under test.


class _FakeCollection:
    """Tiny ChromaDB ``Collection`` substitute keyed by ``id``."""

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
        """Rank rows by cosine distance against ``query_embeddings[0]``.

        ``HashEmbedder`` produces L2-normalised vectors, so for unit-norm
        vectors cosine distance reduces to ``1 - dot_product``. Sorting
        ascending by distance therefore gives the most-similar-first
        ordering :class:`MemoryStore.retrieve` documents (cf.
        ``_COLLECTION_METRIC = "cosine"`` in ``store.py``).

        Two calls with identical ``query_embeddings`` and an unchanged
        collection produce identical ordered outputs — the precondition
        Property 3 depends on. We use a deterministic stable sort
        (``sorted`` with an explicit key) so ties (e.g. duplicate
        documents) are broken by Python's stable sort on insertion
        order, never by hash randomisation or RNG state.
        """
        del include  # Memory_Store always asks for the same shape.
        if not query_embeddings:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "embeddings": [[]],
                "distances": [[]],
            }
        q = query_embeddings[0]
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for rid, row in self.rows.items():
            emb = row["embedding"]
            # Guard against shape mismatch — should never happen because
            # the embedder is the only writer, but a length-mismatch row
            # would otherwise raise mid-zip and mask the property failure.
            if len(emb) != len(q):
                continue
            dot = 0.0
            for a, b in zip(emb, q, strict=True):
                dot += a * b
            distance = 1.0 - dot
            ranked.append((distance, rid, row))
        ranked.sort(key=lambda triple: triple[0])
        top = ranked[:n_results]
        ids = [rid for _dist, rid, _row in top]
        return {
            "ids": [ids],
            "documents": [[row["document"] for _d, _i, row in top]],
            "metadatas": [[dict(row["metadata"]) for _d, _i, row in top]],
            "embeddings": [[list(row["embedding"]) for _d, _i, row in top]],
            "distances": [[d for d, _i, _r in top]],
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
# Hypothesis strategies
# ---------------------------------------------------------------------------


# Free-form printable text for the persisted ``content`` field. We avoid
# surrogates / control chars so failures are easy to read, and we cap
# the size so Hypothesis spends its budget on shape coverage rather
# than on long strings.
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

# Free-form query strings. Includes the empty string deliberately:
# :meth:`MemoryStore.retrieve` accepts it (the embedder treats it as
# any other input) and CP3 is universally quantified over query
# strings, so the test should not exclude that corner.
_queries = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFD,
        exclude_categories=("Cs",),  # type: ignore[arg-type]
    ),
    min_size=0,
    max_size=32,
)


@st.composite
def _record_inputs(
    draw: st.DrawFn,
    *,
    min_records: int = 1,
    max_records: int = 6,
) -> list[tuple[str, str]]:
    """Generate a list of ``(content, category)`` pairs to persist.

    Hypothesis is allowed to shrink the corpus down to a single
    record because CP3 holds for any non-empty collection (and
    trivially for the empty one — see
    :func:`test_retrieve_on_empty_store_is_deterministic`). Allowing
    duplicate content gives the stable-sort tie-breaking path
    coverage.
    """
    n = draw(st.integers(min_value=min_records, max_value=max_records))
    return [
        (draw(_safe_text), draw(_categories))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path, *, dimension: int = 16) -> MemoryStore:
    """Construct a fresh :class:`MemoryStore` backed by the in-memory shim.

    Each Hypothesis example gets its own store (and therefore its own
    fake ChromaDB collection) so state from a previous example cannot
    bleed into the next — Property 3 quantifies over a *fixed*
    snapshot, and we get that by giving every example a fresh store.

    ``dimension`` is left small (16) so the per-example embed cost is
    in the tens of microseconds. The hash embedder's vector space is
    independent of dimension for the property under test (CP3 only
    cares that the embedding is reproducible), so a small dimension is
    a strict performance win without weakening the test.
    """
    return MemoryStore(
        db_path=tmp_path / "chroma",
        embedder=HashEmbedder(dimension=dimension),
        dpapi=NullDPAPI(),  # type: ignore[arg-type]
        redactor=PIIRedactor.with_defaults(),
        # Disable redaction so the persisted text round-trips verbatim
        # — the property is about retrieval determinism, not redaction
        # (Property 15 covers the redaction path separately).
        redaction_enabled=False,
        chromadb_module=_FakeChromaDB(),
    )


# ---------------------------------------------------------------------------
# Property 3 — Memory retrieval determinism
# ---------------------------------------------------------------------------


@given(
    inputs=_record_inputs(),
    query=_queries,
    k=st.integers(min_value=1, max_value=8),
)
@settings(
    # Inherit ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile in ``tests/conftest.py``. The
    # health-check suppression handles the small per-example fixed
    # overhead of constructing the store, which Hypothesis would
    # otherwise classify as ``too_slow`` on slower CI runners. The
    # ``function_scoped_fixture`` suppression covers ``tmp_path``,
    # which pytest re-creates per test invocation but Hypothesis re-
    # uses across examples — that re-use is what we want here, since
    # the per-example ``_make_store`` call constructs a new ChromaDB
    # client inside the directory regardless.
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow),
)
def test_retrieve_is_deterministic(
    tmp_path: Path,
    inputs: list[tuple[str, str]],
    query: str,
    k: int,
) -> None:
    """``retrieve(Q, K)`` returns identical ordered ``record_id`` lists on consecutive calls.

    The test seeds a fresh :class:`MemoryStore`, persists every
    generated ``(content, category)`` pair, then calls
    :meth:`MemoryStore.retrieve` twice with the same ``(query, k)``
    and asserts that the two ordered lists of ``record_id`` are equal.

    The ``k`` strategy is bounded by ``len(records)`` *inside the
    test body* (rather than via a coupled strategy) so Hypothesis can
    shrink ``inputs`` and ``k`` independently. The constructor
    documentation guarantees that ``retrieve`` clamps ``k`` against
    the collection size, so generating ``k`` values larger than the
    corpus is also legitimate — we exercise that path explicitly to
    cover the clamping branch in :meth:`MemoryStore.retrieve`.

    **Validates: Requirements 10.3, 10.4 (CP3)**
    """

    async def _run() -> None:
        store = _make_store(tmp_path)

        # Persist every input. Using ``persist_fact`` (rather than
        # ``persist_turn``) keeps the call surface compact and
        # exercises the same internal write path (``_persist_record``)
        # that ``persist_turn`` uses, so the property covers both.
        persisted_ids: list[str] = []
        for content, category in inputs:
            record = await store.persist_fact(content=content, category=category)
            persisted_ids.append(record.record_id)

        # First and second retrieval. CP3 only holds within "the same
        # session and embedding model version", so the two calls share
        # the *same store instance* and the same in-memory embedder.
        first = await store.retrieve(query, k=k)
        second = await store.retrieve(query, k=k)

        first_ids = [r.record_id for r in first]
        second_ids = [r.record_id for r in second]

        # The principal CP3 assertion: ordered equality of record ids.
        assert first_ids == second_ids, (
            f"retrieval was non-deterministic for query={query!r}, k={k}:\n"
            f"  first  = {first_ids}\n"
            f"  second = {second_ids}"
        )

        # Defensive companion: every returned id MUST come from the
        # corpus we persisted. Without this the property could be
        # vacuously satisfied by a buggy store that returns the same
        # garbage on both calls. ``len(first_ids)`` is also bounded by
        # the smaller of ``k`` and the corpus size, per the
        # ``retrieve`` docstring.
        persisted_set = set(persisted_ids)
        assert all(rid in persisted_set for rid in first_ids), (
            f"retrieval returned ids not in the persisted corpus: "
            f"{set(first_ids) - persisted_set}"
        )
        assert len(first_ids) <= min(k, len(persisted_ids))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: the empty store still behaves deterministically
# ---------------------------------------------------------------------------


@given(
    query=_queries,
    k=st.integers(min_value=0, max_value=8),
)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_retrieve_on_empty_store_is_deterministic(
    tmp_path: Path,
    query: str,
    k: int,
) -> None:
    """An empty :class:`MemoryStore` returns ``[]`` deterministically.

    CP3 quantifies over *any* snapshot, including the empty one. The
    documented behaviour for ``retrieve`` against an empty collection
    is to return ``[]``, and that must be true on every consecutive
    call. ``k = 0`` is also accepted by the docstring as a no-op short
    circuit, which we cover here.

    **Validates: Requirements 10.3, 10.4 (CP3)**
    """

    async def _run() -> None:
        store = _make_store(tmp_path)
        first = await store.retrieve(query, k=k)
        second = await store.retrieve(query, k=k)
        assert first == [] == second

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: a fixed-corpus determinism check across queries
# ---------------------------------------------------------------------------


@given(
    queries=st.lists(_queries, min_size=2, max_size=6),
    k=st.integers(min_value=1, max_value=8),
)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow))
def test_retrieve_is_deterministic_across_many_queries(
    tmp_path: Path,
    queries: list[str],
    k: int,
) -> None:
    """For a fixed corpus, every query is deterministic on consecutive retrieve.

    The principal property test parametrises both the corpus and the
    query. This companion fixes the corpus across one example and
    quantifies over a *batch* of queries so a regression that depends
    on internal state being mutated by a previous ``retrieve`` call
    (e.g. a future caching bug) cannot hide behind the per-example
    store reset.

    **Validates: Requirements 10.3, 10.4 (CP3)**
    """

    # Fixed corpus: small, well-known, with a tie pair so the stable-
    # sort tie-breaking path is hit on at least some queries. The
    # particular contents do not matter for CP3 — only that the corpus
    # is non-empty and persists reproducibly.
    corpus: list[tuple[str, str]] = [
        ("alpha bravo charlie", "fact"),
        ("delta echo foxtrot", "fact"),
        ("golf hotel india", "preference"),
        # Duplicate content -> identical embedding -> tied distance.
        # The fake ChromaDB stable-sorts on insertion order, mirroring
        # the real engine's deterministic tie-breaking; the property
        # asserts both calls land on the same side of that tie.
        ("alpha bravo charlie", "summary"),
    ]

    async def _run() -> None:
        store = _make_store(tmp_path)
        for content, category in corpus:
            await store.persist_fact(content=content, category=category)

        for query in queries:
            first = [r.record_id for r in await store.retrieve(query, k=k)]
            second = [r.record_id for r in await store.retrieve(query, k=k)]
            assert first == second, (
                f"retrieval was non-deterministic for query={query!r}, k={k}: "
                f"first={first} second={second}"
            )

    asyncio.run(_run())
