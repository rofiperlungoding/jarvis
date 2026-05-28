"""Property test for Property 15 — memory redaction containment.

From ``design.md §Correctness Properties``:

    *For any* turn ``T`` whose user / assistant text contains PII,
    after :meth:`MemoryStore.persist_turn` runs with redaction enabled,
    the decrypted ``MemoryRecord.content`` SHALL contain none of the
    original PII strings as a substring; each match SHALL have been
    replaced by ``[REDACTED:<kind>]``.

This file implements that universal quantification with Hypothesis. The
strategy generates two PII-bearing strings (one for the user side and
one for the assistant side of a turn) using
:func:`tests.strategies.pii_corpus`, persists the turn through a real
:class:`MemoryStore` wired with ``redaction_enabled=True`` and the
default :class:`PIIRedactor`, then reads the record back, decrypts its
ciphertext via :class:`NullDPAPI` (the test-only DPAPI substitute), and
asserts:

1. **Containment** — for every PII sample value in
   :data:`tests.strategies.PII_SAMPLES` that appears in the original
   rendered turn (``"User: {user}\\nAssistant: {assistant}"``), the
   sample SHALL NOT appear as a substring of the decrypted content.
   This is the literal statement of Property 15.
2. **Replacement marker** — when at least one PII sample was inserted,
   the decrypted content contains the ``[REDACTED:`` marker. This
   pins the redactor's replacement contract, which downstream code
   (audit log, MemoryAdminSkill UI, the LLM context window) relies
   on as a stable visual marker.
3. **Functional equivalence** — the decrypted content equals exactly
   what :meth:`PIIRedactor.redact` would produce on the original
   rendered turn. This is a stronger companion that catches
   regressions where the MemoryStore write path forgets to invoke the
   redactor at all (the containment assertion alone would still pass
   in that case if no sample regex matched).
4. **Audit metadata** — the persisted record's ``redacted`` flag is
   ``True`` whenever redaction actually changed the text. This lets
   post-hoc auditors confirm Requirement 10.8 compliance from
   metadata alone, without re-running the redactor.

Why a fake ChromaDB?
--------------------

The test uses the same in-memory ChromaDB shim as
``test_property_03_memory_retrieval_determinism.py`` and
``test_property_04_forget_removes_record.py``. Property 15 is about the
write-time redaction path, not the vector index, and re-warming a real
ChromaDB ``PersistentClient`` for every Hypothesis example would dwarf
the per-example cost of the redactor itself. The shim mirrors the real
engine's semantics for the API surface :class:`MemoryStore` consumes
(``count``/``add``/``get``/``query``/``delete``) so the only piece of
behaviour the test does NOT exercise — vector search ordering — is
also irrelevant to this property.

Why ``NullDPAPI``?
------------------

:class:`jarvis.security.dpapi.NullDPAPI` is the documented test double
for non-Windows CI. It still exercises the same encrypt-then-decrypt
round-trip the production
:class:`jarvis.security.dpapi.WindowsDPAPI` would: the literal plaintext
never appears on disk verbatim (the obfuscation keystream is XORed
in), so the assertion ``"alice@example.com" not in
record.content`` would also fail if the implementation forgot to run
the redactor *before* encryption — exactly the kind of regression
Property 15 is meant to catch.

Validates: Requirements 10.8 (CP — memory redaction containment)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from tests.strategies import PII_SAMPLES, pii_corpus

from jarvis.memory.embedder import HashEmbedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryStore
from jarvis.security.dpapi import NullDPAPI

# ---------------------------------------------------------------------------
# In-memory ChromaDB stand-in (mirrors P03 / P04)
# ---------------------------------------------------------------------------
#
# Keeps the property test cheap by sidestepping ChromaDB's multi-second
# warm-up. Only the API surface :class:`MemoryStore` consumes is
# implemented. Property 15 does not depend on vector ordering, so the
# ``query`` stub returns rows in insertion order with constant
# distances — sufficient for this test (we mostly read back via
# ``list_records`` and ``get``).


class _FakeCollection:
    """Tiny ChromaDB ``Collection`` substitute keyed by ``id``.

    The implementation matches the shim used in
    :mod:`tests.property.test_property_03_memory_retrieval_determinism`
    and :mod:`tests.property.test_property_04_forget_removes_record`. We
    intentionally re-implement it here rather than share via a fixture
    because each property test wants its own ChromaDB stub that cannot
    be poisoned by leftover state from a sibling test run. Centralising
    the shim in ``tests/conftest.py`` would invite that cross-test
    coupling.
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
# Test fixture helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> MemoryStore:
    """Construct a fresh :class:`MemoryStore` with redaction enabled.

    Each Hypothesis example gets its own store (and therefore its own
    fake ChromaDB collection) so persistence state from a previous
    example cannot bleed into the next — Property 15 quantifies over a
    *fresh* turn against a *fresh* store.

    Constructor choices, in order:

    * ``embedder=HashEmbedder(dimension=16)`` — deterministic,
      dependency-free, and small (16-dim vectors keep per-example
      embedding cost in the tens of microseconds).
    * ``dpapi=NullDPAPI()`` — keeps the same encrypt-then-decrypt
      round-trip the production code path uses, without needing
      ``win32crypt`` on non-Windows CI. The ``suppress_warning`` flag
      keeps test output clean; the warning itself is unit-tested
      elsewhere.
    * ``redactor=PIIRedactor.with_defaults()`` — the explicit factory
      makes the "use defaults" intent obvious to readers and matches
      the production wiring in :mod:`jarvis.app`.
    * ``redaction_enabled=True`` — the precondition Property 15
      quantifies over (Requirement 10.8). Without this, the property
      is vacuous.
    * ``chromadb_module=_FakeChromaDB()`` — the in-memory ChromaDB
      shim defined above; keeps Hypothesis examples cheap.
    """
    return MemoryStore(
        db_path=tmp_path / "chroma",
        embedder=HashEmbedder(dimension=16),
        dpapi=NullDPAPI(suppress_warning=True),  # type: ignore[arg-type]
        redactor=PIIRedactor.with_defaults(),
        redaction_enabled=True,
        chromadb_module=_FakeChromaDB(),
    )


def _make_turn(user: str, assistant: str) -> SimpleNamespace:
    """Build a Turn-like object that satisfies :meth:`MemoryStore.persist_turn`.

    ``persist_turn`` only accesses ``user``, ``assistant``,
    ``finished_at``, ``started_at``, ``session_id`` and ``turn_index``
    via ``getattr``; the dialog package's :class:`Turn` dataclass is
    intentionally NOT imported here so the test stays decoupled from
    the dialog wiring (and avoids dragging the persona / tool-call
    machinery into a memory-only property test).

    Both timestamps are pinned to a fixed UTC instant so the
    persistence-then-retrieval round trip produces stable timestamps
    for assertions; the property does not depend on the timestamp's
    value, so any tz-aware UTC datetime would do.
    """
    fixed = datetime(2024, 1, 1, tzinfo=UTC)
    return SimpleNamespace(
        user=user,
        assistant=assistant,
        tool_calls=[],
        started_at=fixed,
        finished_at=fixed,
    )


# ---------------------------------------------------------------------------
# Property 15 — memory redaction containment
# ---------------------------------------------------------------------------


@given(user_text=pii_corpus(), assistant_text=pii_corpus())
@settings(
    # Inherit ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile in ``tests/conftest.py``. Suppress
    # ``function_scoped_fixture`` because pytest's ``tmp_path`` is
    # function-scoped but Hypothesis re-uses it across examples — that
    # is fine here because every example builds its OWN MemoryStore
    # instance (and therefore its own ChromaDB collection) inside that
    # directory, so the fake-chromadb state never leaks across
    # examples. ``too_slow`` is suppressed for the same reason as the
    # other Memory_Store property tests: per-example fixed overhead
    # for store construction can occasionally exceed Hypothesis'
    # default 200 ms watchdog on slower CI runners.
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow),
)
def test_redaction_containment(
    tmp_path: Path,
    user_text: str,
    assistant_text: str,
) -> None:
    """After ``persist_turn`` with redaction enabled, no PII string survives.

    The test exercises the full Memory_Store write path:

    1. ``persist_turn`` runs the rendered turn through
       :meth:`PIIRedactor.redact` (because ``redaction_enabled=True``).
    2. The redacted plaintext is embedded with the configured
       :class:`Embedder` and DPAPI-encrypted via
       :meth:`NullDPAPI.protect` before being handed to ChromaDB.
    3. :meth:`MemoryStore.list_records` re-reads the row, base64-
       decodes the ``documents`` column, calls
       :meth:`NullDPAPI.unprotect` to recover the plaintext, and wraps
       it in a fresh :class:`MemoryRecord`.

    The four assertions below cover both the containment property
    Requirement 10.8 names ("does not contain the PII as a substring")
    and the surrounding contract (replacement marker, redactor
    equivalence, audit metadata) so a regression in any of those
    pieces surfaces immediately.

    **Validates: Requirement 10.8**
    """

    async def _run() -> None:
        store = _make_store(tmp_path)
        turn = _make_turn(user_text, assistant_text)

        records = await store.persist_turn(turn)
        # ``persist_turn`` always emits exactly one record today (the
        # docstring documents the list shape as forward-compatible);
        # pin the contract so a future split into multiple records is
        # noticed by this test rather than silently passing.
        assert len(records) == 1
        record_id = records[0].record_id

        # Read the record back through the public API. ``list_records``
        # exercises the full decrypt path (base64 → DPAPI →
        # MemoryRecord) without depending on the embedder, so the
        # property is unaffected by retrieval ranking quirks.
        all_records = await store.list_records()
        matches = [r for r in all_records if r.record_id == record_id]
        assert len(matches) == 1, (
            f"persisted record {record_id!r} was not returned by list_records; "
            f"got {[r.record_id for r in all_records]!r}"
        )
        decrypted_content = matches[0].content

        # Reconstruct the exact string the redactor saw at write time;
        # ``MemoryStore._render_turn`` is the documented stable
        # representation ("User: <u>\nAssistant: <a>").
        rendered = f"User: {user_text}\nAssistant: {assistant_text}"

        # ----------------------------------------------------------
        # (1) Containment — the literal Property 15 statement.
        # ----------------------------------------------------------
        # ``pii_corpus`` always inserts samples between ``" ".join``
        # separators, so every inserted sample is word-bounded and
        # therefore matched by the default redactor regex. The check
        # below quantifies over the full PII_SAMPLES catalogue and
        # only asserts on samples that actually appear in the
        # rendered turn — which is exactly the universally-quantified
        # form Requirement 10.8 calls out.
        any_sample_present = False
        for kind, sample in PII_SAMPLES:
            if sample in rendered:
                any_sample_present = True
                assert sample not in decrypted_content, (
                    f"PII sample {sample!r} (kind={kind!r}) survived "
                    f"redaction in decrypted content {decrypted_content!r} "
                    f"(rendered input was {rendered!r})"
                )

        # ----------------------------------------------------------
        # (2) Replacement marker — the redactor's documented contract
        # is to substitute ``[REDACTED:<kind>]`` for every match. If
        # any sample appeared in the input, at least one such marker
        # must appear in the output.
        # ----------------------------------------------------------
        if any_sample_present:
            assert "[REDACTED:" in decrypted_content, (
                f"redactor did not insert a [REDACTED:<kind>] marker even "
                f"though {rendered!r} contained PII; got "
                f"{decrypted_content!r}"
            )

        # ----------------------------------------------------------
        # (3) Functional equivalence — decrypted content equals what
        # the redactor would produce on the rendered input. This is
        # the strongest formulation of the property and catches the
        # regression where MemoryStore forgets to invoke the redactor
        # at all (the containment check alone would still pass for
        # samples whose regex happened to fail to match).
        # ----------------------------------------------------------
        expected = PIIRedactor.with_defaults().redact(rendered)
        assert decrypted_content == expected, (
            "decrypted content diverged from the redactor's output:\n"
            f"  input    = {rendered!r}\n"
            f"  expected = {expected!r}\n"
            f"  actual   = {decrypted_content!r}"
        )

        # ----------------------------------------------------------
        # (4) Audit metadata — the ``redacted`` flag must reflect
        # whether the redactor actually changed the text.
        # ----------------------------------------------------------
        assert matches[0].redacted == (rendered != expected), (
            f"redacted flag {matches[0].redacted!r} disagrees with whether "
            f"redaction changed the text "
            f"(input == expected: {rendered == expected!r})"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: redaction-disabled writes preserve PII
# ---------------------------------------------------------------------------


@given(user_text=pii_corpus(), assistant_text=pii_corpus())
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow))
def test_redaction_disabled_preserves_pii(
    tmp_path: Path,
    user_text: str,
    assistant_text: str,
) -> None:
    """With ``redaction_enabled=False`` the same input round-trips verbatim.

    Property 15 quantifies over the *redaction-enabled* code path. To
    prevent the test above from passing vacuously (e.g. because the
    embedder, encryption layer, or ChromaDB shim silently dropped the
    text on the floor) we pin the inverse: when the operator turns
    redaction off, every PII sample present in the input survives the
    persist/retrieve round-trip unchanged. This guarantees that the
    containment assertion in :func:`test_redaction_containment` is
    measuring the redactor's behaviour and not some upstream sanitiser.

    **Validates: Requirement 10.8 (negative case — redaction disabled
    is a no-op).**
    """

    async def _run() -> None:
        store = MemoryStore(
            db_path=tmp_path / "chroma",
            embedder=HashEmbedder(dimension=16),
            dpapi=NullDPAPI(suppress_warning=True),  # type: ignore[arg-type]
            redactor=PIIRedactor.with_defaults(),
            redaction_enabled=False,
            chromadb_module=_FakeChromaDB(),
        )
        records = await store.persist_turn(_make_turn(user_text, assistant_text))
        assert len(records) == 1
        record_id = records[0].record_id

        all_records = await store.list_records()
        matches = [r for r in all_records if r.record_id == record_id]
        assert len(matches) == 1
        decrypted_content = matches[0].content

        rendered = f"User: {user_text}\nAssistant: {assistant_text}"
        # With redaction disabled, the round-trip is the identity.
        assert decrypted_content == rendered, (
            "redaction-disabled write must round-trip verbatim:\n"
            f"  input  = {rendered!r}\n"
            f"  output = {decrypted_content!r}"
        )
        # And the audit flag must reflect that no redaction occurred.
        assert matches[0].redacted is False
        # No marker should appear (sanity: the disabled path did not
        # accidentally invoke the redactor).
        assert "[REDACTED:" not in decrypted_content

    asyncio.run(_run())
