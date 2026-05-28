"""Persistent, encrypted long-term memory store.

This module implements the ``Memory_Store`` component described in
``design.md §Memory_Store``: a ChromaDB-backed vector index of conversation
turns and user facts, where each ``MemoryRecord.content`` is encrypted at
rest with DPAPI but the embedding is computed on plaintext so semantic
retrieval still works (Requirements 10.1-10.8, 13.3, 13.5).

Storage layout
--------------

A single :class:`chromadb.PersistentClient` collection lives under
``${app.data_dir}/memory/chroma/``. Each document in the collection
corresponds to one :class:`MemoryRecord`:

* ``id`` — the ``record_id`` (a UUID4 string).
* ``embedding`` — passed to ChromaDB so the vector index can do nearest-
  neighbour search. Computed on the *plaintext* content so retrieval
  semantics are preserved (otherwise ciphertext bytes would dominate the
  embedding space and nearest-neighbour would be useless).
* ``documents`` — the *base64-encoded ciphertext* produced by
  :meth:`DPAPI.protect` over ``plaintext.encode("utf-8")``. ChromaDB's
  ``documents`` column expects ``str``, so we b64-encode the binary blob
  before handing it over and decode on retrieval. The base64 round-trip
  is lossless and lightweight.
* ``metadatas`` — non-secret descriptors (``timestamp`` ISO 8601 string,
  ``category``, ``provenance_*``, ``redacted`` bool, ``model_name`` of
  the embedder used). The metadata values are limited to the JSON-
  scalar types ChromaDB accepts (``str``, ``int``, ``float``, ``bool``).

The embedding is stored in ChromaDB's native vector index; we do *not*
duplicate it inside ``documents`` or ``metadatas``. Re-embedding at
retrieval time is unnecessary because ChromaDB returns ``embeddings``
on demand and the vector itself is a lossy projection (the design
explicitly notes that the embedding is not considered secret —
operators who disagree can set ``memory.encrypt_embeddings: true``,
which is honoured by passing ``encrypt_embeddings=True`` to the
constructor).

API summary
-----------

* :meth:`MemoryStore.persist_turn` — encrypts the rendered turn (subject
  to PII redaction and incognito mode) and writes one ``chat`` record.
* :meth:`MemoryStore.persist_fact` — writes a single typed fact.
* :meth:`MemoryStore.retrieve` — top-K nearest-neighbour search; the
  query string is embedded the same way as documents and never persisted.
* :meth:`MemoryStore.forget` — removes a single record by id; subsequent
  ``retrieve`` calls SHALL not return it (Property 4 / CP4).
* :meth:`MemoryStore.wipe` — removes every record from the collection,
  satisfying the memory-store half of Requirement 13.5.

Concurrency
-----------

The store is designed for use from a single asyncio loop. The synchronous
ChromaDB calls are off-loaded to the default executor via
:func:`asyncio.to_thread` so the dialog loop never blocks on disk I/O. An
``asyncio.Lock`` serialises mutations so concurrent ``persist_turn`` calls
cannot interleave with a ``forget`` for the same id.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
13.3, 13.5
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
import uuid

from jarvis.memory.embedder import Embedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.security.dpapi import DPAPI

if TYPE_CHECKING:  # pragma: no cover - import-time only
    # ChromaDB is imported lazily at runtime so importing this module
    # does not pay the multi-second ChromaDB / DuckDB / sqlite warm-up.
    # Type-only references are fine here because mypy can resolve the
    # stub even if the runtime package is absent (see ``mypy.ini``).
    import chromadb  # noqa: F401
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DPAPI_ENTROPY",
    "MemoryRecord",
    "MemoryStore",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Name of the ChromaDB collection used for the persistent memory store.
#: Centralised so tests, migrations, and operators reach the same identifier.
DEFAULT_COLLECTION_NAME: Final[str] = "jarvis_memory"

#: DPAPI ``entropy`` value bound to the Memory_Store. Acts as a domain
#: separator so a memory blob cannot be replayed against another DPAPI
#: consumer (e.g. :class:`jarvis.security.credential_store.CredentialStore`)
#: even if both happen to run under the same Windows user account.
DPAPI_ENTROPY: Final[bytes] = b"jarvis/memory_store/v1"

#: Closed set of memory categories. Mirrors the ``MemoryRecord.category``
#: literal in ``design.md §Data Models``.
MemoryCategory = Literal["chat", "preference", "fact", "summary"]

_VALID_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"chat", "preference", "fact", "summary"}
)

#: Sentinel used in metadata to record whether a record was redacted before
#: encryption. Stored alongside the (non-secret) timestamp and category so
#: post-hoc auditing can tell redaction-enabled writes from raw writes.
_METADATA_REDACTED_KEY: Final[str] = "redacted"
_METADATA_TIMESTAMP_KEY: Final[str] = "timestamp"
_METADATA_CATEGORY_KEY: Final[str] = "category"
_METADATA_MODEL_KEY: Final[str] = "model_name"
_METADATA_PROVENANCE_PREFIX: Final[str] = "prov_"

#: ChromaDB distance metric. Cosine distance is consistent with the
#: L2-normalised vectors produced by ``SentenceTransformerEmbedder`` and
#: ``HashEmbedder``; for unit-norm vectors, cosine and dot-product orderings
#: are identical.
_COLLECTION_METRIC: Final[str] = "cosine"


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRecord:
    """A single stored memory item. Mirrors ``design.md §Data Models``.

    The ``content`` field holds **plaintext** at runtime: callers that
    receive a record from :meth:`MemoryStore.retrieve` see the decrypted
    text. The on-disk representation is the DPAPI-protected ciphertext
    of ``content.encode("utf-8")`` (base64-encoded for ChromaDB's
    ``documents`` column). Callers MUST treat ``content`` as sensitive —
    do not log or transmit it without first running it through
    :class:`~jarvis.memory.redactor.PIIRedactor`.

    Attributes
    ----------
    record_id:
        UUID4 string assigned at write time. Stable for the lifetime of
        the record; a :meth:`MemoryStore.forget` call with this id
        removes the record from every subsequent retrieval (Property 4).
    content:
        Plaintext memory content. Empty strings are permitted but
        callers should avoid persisting them — they carry no semantic
        signal and waste an embedding slot.
    embedding:
        The embedding vector computed by the configured
        :class:`~jarvis.memory.embedder.Embedder`. Stored next to the
        ciphertext in ChromaDB's vector index. Returned on retrieval
        for callers that want to do their own re-ranking.
    timestamp:
        Timezone-aware UTC timestamp recorded at write time.
    category:
        One of ``chat``, ``preference``, ``fact``, ``summary``. Used by
        :class:`~jarvis.memory.compactor.MemoryCompactor` to decide
        which records are eligible for daily summarisation.
    provenance:
        Free-form mapping of strings (``session_id``, ``turn_index``,
        ``source``...). Values must be JSON scalars. Stored as
        ChromaDB metadata so they can drive future filtering.
    redacted:
        ``True`` when :class:`PIIRedactor` was applied to ``content``
        before encryption. ``False`` for raw writes. Available so
        auditors can confirm Requirement 10.8 compliance after the
        fact.
    """

    record_id: str
    content: str
    embedding: list[float]
    timestamp: datetime
    category: MemoryCategory
    provenance: dict[str, Any] = field(default_factory=dict)
    redacted: bool = False


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------


class MemoryStore:
    """Persistent, DPAPI-encrypted vector memory store.

    Parameters
    ----------
    db_path:
        Directory under which the ChromaDB persistent client keeps its
        SQLite + Parquet files. Created on demand so callers do not need
        to ``mkdir -p`` first. In production this is the resolved value
        of ``memory.path`` (defaulting to
        ``${app.data_dir}/memory/chroma``).
    embedder:
        Any object satisfying the :class:`Embedder` protocol. Used to
        embed both stored content and retrieval queries; pass the same
        instance for both write and read paths so CP3 (Memory Retrieval
        Determinism) holds.
    dpapi:
        The :class:`DPAPI` envelope. Production deployments pass
        :class:`~jarvis.security.dpapi.WindowsDPAPI`; tests typically
        pass :class:`~jarvis.security.dpapi.NullDPAPI`.
    redactor:
        :class:`PIIRedactor` used when ``redaction_enabled=True``. Always
        non-``None`` so toggling redaction is a flag rather than a
        dependency-injection change. Pass :meth:`PIIRedactor.with_defaults`
        if you don't care about the regex set.
    collection_name:
        Override for the ChromaDB collection name. Defaults to
        :data:`DEFAULT_COLLECTION_NAME`. Tests may use distinct names
        per fixture to avoid collisions when sharing a directory.
    incognito:
        When ``True``, every ``persist_*`` call is a no-op (Requirement
        13.3). Retrieval still works against any pre-existing records,
        which is the intended behaviour: incognito stops new writes
        without forgetting the past.
    redaction_enabled:
        When ``True`` (the default), :meth:`persist_turn` and
        :meth:`persist_fact` push their content through ``redactor``
        before embedding and encryption (Requirement 10.8). When
        ``False``, content is persisted verbatim.
    encrypt_embeddings:
        Reserved for the future. Today the embedding is stored in
        ChromaDB's vector index in plaintext (the design notes the
        embedding as a lossy projection that is not considered secret).
        When ``True``, the constructor logs a warning and otherwise
        proceeds: the toggle exists so the configuration schema can
        round-trip the operator's intent until task 14.x adds the
        encrypted-embedding code path.
    chromadb_module:
        Dependency-injection seam used by tests. When ``None`` the real
        ``chromadb`` package is imported lazily on first use.
    """

    def __init__(
        self,
        db_path: Path,
        embedder: Embedder,
        dpapi: DPAPI,
        redactor: PIIRedactor,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        incognito: bool = False,
        redaction_enabled: bool = True,
        encrypt_embeddings: bool = False,
        chromadb_module: Any | None = None,
    ) -> None:
        # Accept ``str`` for ergonomic call sites; coerce to Path so the
        # rest of the implementation can rely on the typed API. mypy
        # narrows ``db_path`` to ``Path`` from the declared annotation, so
        # the assignment branch is statically unreachable but defensive at
        # runtime; the ``type: ignore`` keeps both views happy.
        if not isinstance(db_path, Path):
            db_path = Path(db_path)  # type: ignore[unreachable]
        if not isinstance(collection_name, str) or not collection_name:
            raise ValueError("collection_name must be a non-empty str")

        self._db_path: Path = db_path
        self._embedder: Embedder = embedder
        self._dpapi: DPAPI = dpapi
        self._redactor: PIIRedactor = redactor
        self._collection_name: str = collection_name
        self._incognito: bool = bool(incognito)
        self._redaction_enabled: bool = bool(redaction_enabled)
        self._encrypt_embeddings: bool = bool(encrypt_embeddings)

        # Ensure the on-disk directory exists *before* ChromaDB opens it.
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._chromadb: Any = (
            chromadb_module if chromadb_module is not None else _import_chromadb()
        )

        # Lock guarding mutations + collection lookups. ChromaDB's
        # ``PersistentClient`` is not documented as thread-safe; we run all
        # operations on the asyncio default executor and serialise through
        # this lock so two writes (or a write and a forget) cannot land
        # concurrently. The lock is intentionally process-local: cross-
        # process coordination is not in scope (the application is a
        # single-process voice assistant).
        self._lock: asyncio.Lock = asyncio.Lock()

        # Open the persistent client. ChromaDB's API has shifted across
        # versions; the ``PersistentClient`` constructor signature is
        # stable since 0.4.x and is documented as the preferred entry
        # point for on-disk storage.
        self._client: ClientAPI = self._chromadb.PersistentClient(
            path=str(self._db_path),
        )

        # ``get_or_create_collection`` is idempotent: a fresh database
        # creates the collection, an existing database reuses it.
        # ``embedding_function=None`` tells ChromaDB to expect callers to
        # provide pre-computed vectors (we always do, since we control
        # the embedder explicitly for CP3 reproducibility).
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": _COLLECTION_METRIC},
            embedding_function=None,
        )

        if encrypt_embeddings:
            # Documented as a forward-compat toggle. We log so an
            # operator who flips it on can see in their logs that the
            # current code path stores plaintext vectors.
            logger.warning(
                "MemoryStore: encrypt_embeddings=True requested but not "
                "implemented in this version; embeddings are stored in "
                "ChromaDB's vector index in plaintext."
            )

        if not dpapi.is_genuine:
            logger.warning(
                "MemoryStore initialised with a non-genuine DPAPI backend; "
                "memory contents are NOT cryptographically protected."
            )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        """Directory backing the ChromaDB persistent client."""
        return self._db_path

    @property
    def collection_name(self) -> str:
        """Name of the underlying ChromaDB collection."""
        return self._collection_name

    @property
    def incognito(self) -> bool:
        """Whether new ``persist_*`` calls are dropped (Requirement 13.3)."""
        return self._incognito

    @property
    def redaction_enabled(self) -> bool:
        """Whether content is run through :class:`PIIRedactor` before write."""
        return self._redaction_enabled

    def set_incognito(self, value: bool) -> None:
        """Toggle incognito mode at runtime.

        The Dialog_Manager flips this on a user-issued ``incognito``
        command (Requirement 13.3). Idempotent; takes effect on the
        next ``persist_*`` call.
        """
        self._incognito = bool(value)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    async def persist_turn(
        self,
        turn: Any,
        persona: Any | None = None,
    ) -> list[MemoryRecord]:
        """Persist a single conversation turn as a ``chat`` record.

        Encodes the turn as ``"User: <user>\\nAssistant: <assistant>"``
        (a stable representation that mirrors what the LLM_Backend will
        see if the record is later loaded into the prompt as memory
        context — Requirement 10.4). Tool calls are not included in the
        embedded text because they are already structured data captured
        by the audit log; including them in the embedding would
        bias retrieval toward function names rather than content.

        Returns the list of records written. Today this is always a
        single record, but the return type is a list so future versions
        can split a turn into multiple typed records (e.g. a separate
        ``preference`` record extracted from the user's utterance) without
        breaking callers.

        When :attr:`incognito` is ``True`` the call is a no-op and an
        empty list is returned.

        Parameters
        ----------
        turn:
            The :class:`Turn` dataclass from ``design.md §Data Models``.
            Accepted as :class:`Any` to keep this module's import graph
            free of the dialog package — the only attributes accessed
            are ``user`` (str), ``assistant`` (str), and ``finished_at``
            (datetime; falls back to ``started_at`` or ``now()``).
        persona:
            Optional :class:`PersonaProfile`. Ignored by the current
            implementation but accepted to match the ``design.md``
            signature so the Dialog_Manager call site remains stable.
        """
        del persona  # accepted for API compatibility; not used today.

        if self._incognito:
            return []

        user_text = self._coerce_str(getattr(turn, "user", None), field_name="user")
        assistant_text = self._coerce_str(
            getattr(turn, "assistant", None), field_name="assistant"
        )
        rendered = self._render_turn(user_text, assistant_text)

        # Some Turn instances may carry a ``finished_at`` timestamp; fall
        # back through ``started_at`` and finally ``now()`` so callers can
        # pass a partially-populated dataclass during early dialog wiring.
        timestamp = (
            getattr(turn, "finished_at", None)
            or getattr(turn, "started_at", None)
            or _utc_now()
        )

        provenance: dict[str, Any] = {}
        for attr in ("session_id", "turn_index"):
            value = getattr(turn, attr, None)
            if value is not None:
                provenance[attr] = value
        provenance.setdefault("source", "turn")

        record = await self._persist_record(
            content=rendered,
            category="chat",
            timestamp=timestamp,
            provenance=provenance,
        )
        return [record]

    async def persist_fact(
        self,
        content: str,
        category: str = "fact",
        source_id: str | None = None,
    ) -> MemoryRecord:
        """Persist a single typed fact / preference / summary record.

        Used by Requirement 10.2 (preference extraction) and by the
        :class:`~jarvis.memory.compactor.MemoryCompactor` daily summary
        task (Requirement 10.1). Returns the persisted record so the
        caller can surface ``record_id`` in the ``MemoryAdminSkill``
        responses (Requirement 10.5).

        When :attr:`incognito` is ``True`` the call is a no-op and the
        method raises :class:`MemoryStoreIncognitoError` would be too
        intrusive for callers; instead we return a *non-persisted*
        record that still round-trips its plaintext to the caller. The
        record's ``record_id`` is a fresh UUID4 so log lines remain
        unambiguous, and ``provenance['persisted']`` is ``False`` so the
        audit log can tell the two cases apart.
        """
        if not isinstance(content, str):
            raise TypeError("content must be a str")
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(_VALID_CATEGORIES)}; "
                f"got {category!r}"
            )

        provenance: dict[str, Any] = {"source": "fact"}
        if source_id is not None:
            if not isinstance(source_id, str):
                raise TypeError("source_id must be a str when provided")
            provenance["source_id"] = source_id

        if self._incognito:
            # Return a synthetic record that the caller can echo back to
            # the user without persisting anything. Embedding is zeros
            # so a stray ``retrieve`` against this record (which never
            # makes it into the index) cannot smuggle data into the
            # vector store.
            return MemoryRecord(
                record_id=str(uuid.uuid4()),
                content=content,
                embedding=[0.0] * self._embedder.dimension,
                timestamp=_utc_now(),
                category=category,  # type: ignore[arg-type]
                provenance={**provenance, "persisted": False},
                redacted=False,
            )

        return await self._persist_record(
            content=content,
            category=category,  # type: ignore[arg-type]
            timestamp=_utc_now(),
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return the top-K records most similar to ``query``.

        ``query`` is embedded with the same configured :class:`Embedder`
        used for writes; this is what makes CP3 (Memory Retrieval
        Determinism) hold. Records are returned in ChromaDB's native
        order (ascending distance, i.e. most similar first), with
        ciphertext decrypted into plaintext ``content``.

        Returns an empty list when the collection is empty or when ``k``
        is zero.

        Raises
        ------
        ValueError:
            If ``k`` is negative, or if ``query`` is not a string.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative int")
        if k == 0:
            return []

        # Embedding the query is CPU-only and fast enough to run on the
        # event loop for the production embedder; for the heavier
        # SentenceTransformer path we still hand it off to a thread so
        # we never block other coroutines.
        embedding = await asyncio.to_thread(self._embedder.embed, query)

        async with self._lock:
            # ChromaDB caps ``n_results`` at the collection size; query
            # for ``min(k, current_count)`` so we don't trigger spurious
            # warnings on tiny collections.
            count = await asyncio.to_thread(self._collection.count)
            if count == 0:
                return []
            n_results = min(k, count)
            raw = await asyncio.to_thread(
                self._collection.query,
                # chromadb runtime accepts list[list[float]] but stubs are narrower.
                query_embeddings=[embedding],  # type: ignore[arg-type]
                n_results=n_results,
                include=["documents", "metadatas", "embeddings", "distances"],
            )

        # ``raw`` is a chromadb QueryResult TypedDict; ``_parse_query_result``
        # accepts the dict-like view since we only read documented keys.
        return self._parse_query_result(raw)  # type: ignore[arg-type]

    async def list_records(
        self,
        *,
        category: MemoryCategory | None = None,
        older_than: datetime | None = None,
    ) -> list[MemoryRecord]:
        """Return every persisted record, optionally filtered.

        Used by the ``MemoryAdminSkill`` ``list`` operation (Requirement
        10.5) and by :class:`~jarvis.memory.compactor.MemoryCompactor`
        when picking ``chat`` records eligible for daily summarisation
        (Requirement 10.4). The implementation is O(N) and decrypts
        every returned record, so callers should pass ``category`` /
        ``older_than`` to keep the working set small.

        Parameters
        ----------
        category:
            When provided, only records whose stored ``category``
            metadata matches are returned. The filter is applied at
            ChromaDB level via the ``where`` clause so we do not pay
            DPAPI decryption cost on records the caller would have
            discarded anyway.
        older_than:
            When provided, only records whose ``timestamp`` is *strictly
            less than* ``older_than`` are returned. We do this filter in
            Python (post-decode) because ChromaDB's ``where`` operators
            (``$lt`` / ``$gt``) are numeric, but our timestamps are
            stored as ISO 8601 strings to keep them human-auditable
            (see ``_build_metadata``). Naive datetimes are treated as
            UTC for the comparison.

        Returns
        -------
        list[MemoryRecord]
            Decrypted records in arbitrary order. Sort with
            ``sorted(..., key=lambda r: r.timestamp)`` if ordering
            matters to the caller.
        """
        if category is not None and category not in _VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(_VALID_CATEGORIES)}; "
                f"got {category!r}"
            )

        threshold = _ensure_aware(older_than) if older_than is not None else None

        where: dict[str, Any] | None = None
        if category is not None:
            where = {_METADATA_CATEGORY_KEY: category}

        async with self._lock:
            count = await asyncio.to_thread(self._collection.count)
            if count == 0:
                return []
            kwargs: dict[str, Any] = {
                "include": ["documents", "metadatas", "embeddings"],
            }
            if where is not None:
                kwargs["where"] = where
            raw = await asyncio.to_thread(self._collection.get, **kwargs)

        ids = list(raw.get("ids") or [])
        docs = list(raw.get("documents") or [])
        metas = list(raw.get("metadatas") or [])
        embeddings = list(raw.get("embeddings") or [])

        records: list[MemoryRecord] = []
        for index, record_id in enumerate(ids):
            document = docs[index] if index < len(docs) else None
            metadata = metas[index] if index < len(metas) else {}
            embedding_row = embeddings[index] if index < len(embeddings) else []
            record = self._record_from_row(
                record_id=record_id,
                document=document,
                # chromadb's stricter Metadata typing widens to dict at runtime.
                metadata=metadata or {},  # type: ignore[arg-type]
                embedding=embedding_row,
            )
            if record is None:
                continue
            if threshold is not None and record.timestamp >= threshold:
                continue
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    async def forget(self, record_id: str) -> bool:
        """Remove the record with ``record_id`` from the collection.

        Returns ``True`` if a record was removed, ``False`` if no such
        id existed. Property 4 / CP4 demands that subsequent
        :meth:`retrieve` calls SHALL not return the deleted record;
        ChromaDB's ``delete`` removes the row from both the document
        store and the vector index, so the property holds by
        construction.

        ``record_id`` is validated to be a non-empty string but is *not*
        required to be a UUID — operators wiring up tests may use
        custom ids.
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a str")
        if not record_id:
            raise ValueError("record_id must be a non-empty str")

        async with self._lock:
            existing = await asyncio.to_thread(
                self._collection.get,
                ids=[record_id],
                include=[],
            )
            ids_returned = list(existing.get("ids") or [])
            if not ids_returned:
                return False
            await asyncio.to_thread(self._collection.delete, ids=[record_id])
        return True

    async def wipe(self) -> None:
        """Remove every record from the underlying collection.

        Implements the memory-store half of Requirement 13.5: a
        "wipe-all" request clears every persisted memory entry. The
        collection itself is recreated empty so subsequent
        :meth:`persist_turn` / :meth:`persist_fact` calls do not need
        to re-initialise it.
        """
        async with self._lock:
            # ``Client.delete_collection`` + ``get_or_create_collection``
            # is the documented idiom for clearing every row. Calling
            # ``collection.delete()`` without an ``ids`` filter raises in
            # newer ChromaDB versions, so we go through the client.
            await asyncio.to_thread(
                self._client.delete_collection, self._collection_name
            )
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self._collection_name,
                metadata={"hnsw:space": _COLLECTION_METRIC},
                embedding_function=None,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _persist_record(
        self,
        *,
        content: str,
        category: MemoryCategory,
        timestamp: datetime,
        provenance: dict[str, Any],
    ) -> MemoryRecord:
        """Common write path for both ``persist_turn`` and ``persist_fact``."""
        was_redacted = False
        plaintext = content
        if self._redaction_enabled:
            redacted = self._redactor.redact(content)
            was_redacted = redacted != content
            plaintext = redacted

        # Embed plaintext (post-redaction). Doing so on the *redacted*
        # text means the embedded vector cannot leak the original PII via
        # nearest-neighbour reconstruction either.
        embedding = await asyncio.to_thread(self._embedder.embed, plaintext)

        record_id = str(uuid.uuid4())
        ciphertext = self._dpapi.protect(
            plaintext.encode("utf-8"), entropy=DPAPI_ENTROPY
        )
        document = base64.b64encode(ciphertext).decode("ascii")

        metadata = self._build_metadata(
            category=category,
            timestamp=timestamp,
            provenance=provenance,
            redacted=was_redacted,
        )

        async with self._lock:
            await asyncio.to_thread(
                self._collection.add,
                ids=[record_id],
                # chromadb runtime accepts list[list[float]] but stubs narrower.
                embeddings=[embedding],  # type: ignore[arg-type]
                documents=[document],
                metadatas=[metadata],
            )

        return MemoryRecord(
            record_id=record_id,
            content=plaintext,
            embedding=embedding,
            timestamp=_ensure_aware(timestamp),
            category=category,
            provenance=dict(provenance),
            redacted=was_redacted,
        )

    def _build_metadata(
        self,
        *,
        category: MemoryCategory,
        timestamp: datetime,
        provenance: dict[str, Any],
        redacted: bool,
    ) -> dict[str, Any]:
        """Render the per-record metadata dict for ChromaDB storage.

        ChromaDB's metadata column accepts only JSON scalars (str, int,
        float, bool). We flatten ``provenance`` keys under a ``prov_``
        prefix and stringify any value that isn't already a JSON scalar.
        """
        metadata: dict[str, Any] = {
            _METADATA_CATEGORY_KEY: category,
            _METADATA_TIMESTAMP_KEY: _ensure_aware(timestamp).isoformat(),
            _METADATA_REDACTED_KEY: bool(redacted),
            _METADATA_MODEL_KEY: self._embedder.model_name,
        }
        for key, value in provenance.items():
            if not isinstance(key, str) or not key:
                continue
            metadata[f"{_METADATA_PROVENANCE_PREFIX}{key}"] = _coerce_metadata_value(
                value
            )
        return metadata

    def _parse_query_result(self, raw: dict[str, Any]) -> list[MemoryRecord]:
        """Translate ChromaDB's batched query result into ``MemoryRecord``s."""
        # ChromaDB returns each field as ``[[...]]`` because ``query``
        # accepts a batch of query embeddings. We always pass exactly
        # one query, so we read row 0 of every column.
        ids_batch = raw.get("ids") or [[]]
        docs_batch = raw.get("documents") or [[]]
        metas_batch = raw.get("metadatas") or [[]]
        embeddings_batch = raw.get("embeddings") or [[]]

        ids = list(ids_batch[0]) if ids_batch else []
        docs = list(docs_batch[0]) if docs_batch else []
        metas = list(metas_batch[0]) if metas_batch else []
        embeddings = list(embeddings_batch[0]) if embeddings_batch else []

        records: list[MemoryRecord] = []
        for index, record_id in enumerate(ids):
            document = docs[index] if index < len(docs) else None
            metadata = metas[index] if index < len(metas) else {}
            embedding_row = embeddings[index] if index < len(embeddings) else []
            record = self._record_from_row(
                record_id=record_id,
                document=document,
                metadata=metadata or {},
                embedding=embedding_row,
            )
            if record is not None:
                records.append(record)
        return records

    def _record_from_row(
        self,
        *,
        record_id: str,
        document: str | None,
        metadata: dict[str, Any],
        embedding: Any,
    ) -> MemoryRecord | None:
        """Decode one ChromaDB row into a :class:`MemoryRecord`.

        Returns ``None`` when the row is unreadable (corrupted ciphertext,
        missing metadata, etc.) so a single bad record cannot poison the
        entire retrieval. The caller logs and continues.
        """
        if document is None:
            logger.warning(
                "MemoryStore.retrieve dropped record %s: missing document",
                record_id,
            )
            return None
        try:
            ciphertext = base64.b64decode(document, validate=True)
        except (ValueError, TypeError):
            logger.warning(
                "MemoryStore.retrieve dropped record %s: invalid base64",
                record_id,
            )
            return None
        try:
            plaintext_bytes = self._dpapi.unprotect(ciphertext, entropy=DPAPI_ENTROPY)
        except Exception:  # DPAPI error surfaces vary across platforms
            logger.exception(
                "MemoryStore.retrieve dropped record %s: DPAPI decryption failed",
                record_id,
            )
            return None
        try:
            plaintext = plaintext_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "MemoryStore.retrieve dropped record %s: ciphertext is not UTF-8",
                record_id,
            )
            return None

        timestamp_raw = metadata.get(_METADATA_TIMESTAMP_KEY)
        timestamp = _parse_timestamp(timestamp_raw)
        category_raw = metadata.get(_METADATA_CATEGORY_KEY, "chat")
        category: MemoryCategory = (
            category_raw if category_raw in _VALID_CATEGORIES else "chat"
        )
        redacted = bool(metadata.get(_METADATA_REDACTED_KEY, False))
        provenance = _provenance_from_metadata(metadata)

        return MemoryRecord(
            record_id=str(record_id),
            content=plaintext,
            embedding=[float(x) for x in (embedding if embedding is not None and len(embedding) > 0 else [])],
            timestamp=timestamp,
            category=category,
            provenance=provenance,
            redacted=redacted,
        )

    @staticmethod
    def _coerce_str(value: Any, *, field_name: str) -> str:
        """Coerce ``value`` to ``str`` or raise ``TypeError`` with context."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        raise TypeError(
            f"Turn.{field_name} must be a str (or None); got "
            f"{type(value).__name__}"
        )

    @staticmethod
    def _render_turn(user: str, assistant: str) -> str:
        """Render a turn into the canonical embedded text form.

        Stable across versions so embeddings remain comparable across
        application upgrades. Empty parts are still included so the
        format is greppable in audit logs.
        """
        return f"User: {user}\nAssistant: {assistant}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _import_chromadb() -> Any:
    """Lazily import the ``chromadb`` package.

    Centralised so the import error surfaces with an actionable message
    and so unit tests can replace the dependency via the
    ``chromadb_module`` constructor argument.
    """
    try:
        import chromadb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised on minimal envs
        raise RuntimeError(
            "MemoryStore requires the `chromadb` package. Install it "
            "(declared in pyproject.toml) or pass a stub via the "
            "chromadb_module constructor argument."
        ) from exc
    return chromadb


def _utc_now() -> datetime:
    """Return the current time as a tz-aware UTC datetime."""
    return datetime.now(tz=UTC)


def _ensure_aware(value: datetime) -> datetime:
    """Return ``value`` as a tz-aware datetime, assuming UTC if naive."""
    if not isinstance(value, datetime):
        # Defence in depth: a stray ``str`` here would silently pass
        # ``isoformat`` later and corrupt the record. Reject loudly.
        raise TypeError(
            f"timestamp must be a datetime; got {type(value).__name__}"
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value


def _parse_timestamp(value: Any) -> datetime:
    """Parse the ISO 8601 timestamp string we wrote at persist time.

    Falls back to ``_utc_now()`` if the value is missing or unparseable
    so a corrupted metadata field cannot crash retrieval.
    """
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return _utc_now()
        return _ensure_aware(parsed)
    return _utc_now()


def _provenance_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Recover the ``provenance`` dict from ChromaDB metadata."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(key, str) and key.startswith(_METADATA_PROVENANCE_PREFIX):
            out[key[len(_METADATA_PROVENANCE_PREFIX) :]] = value
    return out


def _coerce_metadata_value(value: Any) -> Any:
    """Coerce ``value`` to a ChromaDB-compatible JSON scalar.

    ChromaDB accepts ``str``, ``int``, ``float``, ``bool``. Other types
    are stringified so the metadata column never rejects a write.
    ``None`` is converted to the empty string because ChromaDB treats
    ``None`` metadata as "field absent" and we want the field to be
    present (and round-trippable) for auditability.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
