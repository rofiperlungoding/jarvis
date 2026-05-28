"""Append-only SQLite audit log.

The :class:`AuditLog` is the single source of truth for security-relevant
events recorded by the Authorization_Policy, the network egress hook in
``ProviderClient``, and the Voice_Pipeline error / crash paths described in
``design.md §Audit_Log``. It backs Correctness Property 6 (CP9) — the
``confirmation_requested`` entry for any destructive Tool_Call MUST have a
strictly smaller ``id`` than the matching ``executed`` / ``denied`` entry —
which the property tests verify by inspecting the ordered ``id`` column.

Design points
-------------

* **Append-only schema.** The single table ``audit`` matches the
  :class:`AuditEntry` data model in ``design.md``. The primary key is
  ``INTEGER PRIMARY KEY AUTOINCREMENT`` (the explicit ``AUTOINCREMENT`` is
  load-bearing: SQLite will then guarantee a strictly increasing row id even
  across deletes, which is exactly what CP9 needs).

* **Strict insert ordering.** Writers acquire an :class:`asyncio.Lock`
  before issuing the synchronous ``INSERT``. Combined with ``AUTOINCREMENT``,
  this gives a total order over all events in the process, regardless of
  which coroutine called which ``record_*`` method.

* **Async-safe writer.** The synchronous ``sqlite3`` API is wrapped via
  :meth:`asyncio.AbstractEventLoop.run_in_executor` so callers in the dialog
  loop never block on disk I/O. The connection is opened with
  ``check_same_thread=False`` so the executor pool can use it; the lock
  serialises access so we still see at most one in-flight write at a time.

* **Wipe semantics.** Requirement 13.5 mandates that a "wipe-all" request
  clears the audit log within five seconds. :meth:`wipe` truncates the
  ``audit`` table *and* the matching ``sqlite_sequence`` row so the next
  insert restarts the row-id sequence at 1 — important for tests that
  reason about absolute ids and for the documented behaviour of starting
  fresh.

Validates: Requirements 13.4, 13.5, 13.6, 16.5, 17.4
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Final, Literal, cast

from jarvis.utils.time_source import SystemTimeSource, TimeSource

logger = logging.getLogger(__name__)

__all__ = [
    "AuditEntry",
    "AuditKind",
    "AuditLog",
]


# The closed set of audit kinds defined in ``design.md``. Centralising the
# tuple lets us validate at the API boundary without importing ``Literal``
# at runtime everywhere.
AuditKind = Literal[
    "confirmation_requested",
    "executed",
    "denied",
    "policy_violation",
    "network_egress",
    "error",
    "crash",
]

_KIND_VALUES: Final[frozenset[str]] = frozenset(
    {
        "confirmation_requested",
        "executed",
        "denied",
        "policy_violation",
        "network_egress",
        "error",
        "crash",
    }
)


@dataclass(frozen=True)
class AuditEntry:
    """A single immutable row in the audit log.

    Mirrors the ``AuditEntry`` data model in ``design.md``. ``id`` is the
    SQLite row id assigned by ``AUTOINCREMENT`` and therefore strictly
    monotonic across the lifetime of the underlying database file (even
    after :meth:`AuditLog.wipe` resets the counter, the ordering invariant
    holds within a single contiguous run). ``ts`` is a timezone-aware
    :class:`datetime` produced by the configured :class:`TimeSource` —
    callers MUST NOT compare entries from different machines without
    converting to UTC.
    """

    id: int
    ts: datetime
    kind: AuditKind
    skill: str | None
    args_json: str | None
    outcome: str | None
    destination: str | None
    justification: str | None
    run_id: str


# Schema is stored as a single source of truth so the test suite can assert
# we never silently break compatibility with persisted databases.
_SCHEMA_SQL: Final[str] = """\
CREATE TABLE IF NOT EXISTS audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    skill         TEXT,
    args_json     TEXT,
    outcome       TEXT,
    destination   TEXT,
    justification TEXT,
    run_id        TEXT    NOT NULL
)
"""

_INSERT_SQL: Final[str] = (
    "INSERT INTO audit (ts, kind, skill, args_json, outcome, destination, "
    "justification, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

_SELECT_ALL_SQL: Final[str] = (
    "SELECT id, ts, kind, skill, args_json, outcome, destination, "
    "justification, run_id FROM audit ORDER BY id ASC"
)

_COUNT_SQL: Final[str] = "SELECT COUNT(*) FROM audit"


class AuditLog:
    """Append-only SQLite audit log with an async-safe writer.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database. Pass ``":memory:"`` (or
        :class:`pathlib.Path` that resolves to it) for ephemeral test
        databases. Parent directories of file-backed paths are created on
        demand so callers do not need to ``mkdir -p`` first.
    time_source:
        Injectable clock used to stamp every entry. Defaults to
        :class:`SystemTimeSource`. Tests pass a :class:`FakeTimeSource` to
        get deterministic timestamps.
    run_id:
        A stable identifier for the current process run. Persisted alongside
        every entry so that crash-recovery diagnostics (Requirement 17.4)
        can correlate events across a single launch.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        time_source: TimeSource | None = None,
        run_id: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        self._time_source: TimeSource = time_source or SystemTimeSource()
        self._run_id: str = run_id
        self._db_path: Path | str = self._resolve_db_path(db_path)

        # The lock serialises *async-visible* writes so that, even if the
        # threadpool executor reorders the underlying ``cursor.execute``
        # calls, the apparent commit order matches the call order on the
        # event loop — which is what CP9 requires.
        self._lock: asyncio.Lock = asyncio.Lock()
        self._closed: bool = False

        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path if isinstance(self._db_path, str) else str(self._db_path),
            # The connection is reused from the default executor's worker
            # threads. We rely on ``self._lock`` to ensure only one thread
            # touches it at a time.
            check_same_thread=False,
            # ``isolation_level=None`` puts the driver in autocommit mode so
            # every ``execute`` is its own transaction. That is exactly what
            # an append-only journal wants: no implicit BEGIN that could
            # straddle two ``record_*`` calls and obscure ordering.
            isolation_level=None,
        )
        try:
            # ``WAL`` improves concurrent read-while-writing without
            # weakening the write ordering we already enforce via the lock.
            # Best-effort: it is harmless if the pragma fails on exotic
            # filesystems (e.g., some network mounts).
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute(_SCHEMA_SQL)
        except Exception:
            # If schema creation fails we cannot offer the documented
            # behaviour at all; close the partially-initialised connection
            # rather than leaving a leak.
            self._conn.close()
            raise

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve_db_path(db_path: Path | str) -> Path | str:
        """Normalise the input path; create parent directories on demand.

        ``":memory:"`` is preserved as a string sentinel for SQLite's
        in-memory databases. Anything else is coerced to a :class:`Path`
        and its parent is created so the caller does not need to bootstrap
        ``%LOCALAPPDATA%/Jarvis`` themselves on first run.
        """
        if isinstance(db_path, str) and db_path == ":memory:":
            return ":memory:"
        path = Path(db_path)
        if str(path) == ":memory:":
            return ":memory:"
        parent = path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _serialize_args(args: dict[str, object] | str | None) -> str | None:
        """Coerce a Tool_Call argument bundle to a stable JSON string.

        ``args_json`` is documented to be the canonical JSON of the
        Tool_Call arguments. Callers may pass the original argument
        dictionary for convenience; we serialise with ``sort_keys=True`` so
        equality checks in CP9 (matching ``args_json`` between
        ``confirmation_requested`` and ``executed``) are robust against
        Python dict ordering quirks.
        """
        if args is None:
            return None
        if isinstance(args, str):
            return args
        return json.dumps(args, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _row_to_entry(row: tuple[object, ...]) -> AuditEntry:
        (
            entry_id,
            ts_iso,
            kind,
            skill,
            args_json,
            outcome,
            destination,
            justification,
            run_id,
        ) = row
        # ``fromisoformat`` round-trips :meth:`datetime.isoformat` losslessly
        # for the aware UTC values we store. We re-attach UTC defensively in
        # the rare case a caller has written a naive string at a lower level.
        ts = datetime.fromisoformat(str(ts_iso))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        # SQLite returns integers and strings; the cast goes through ``str``
        # so non-int repr's still parse, and ``str()`` accepts any object.
        return AuditEntry(
            id=int(str(entry_id)),
            ts=ts,
            kind=cast("AuditKind", str(kind)),
            skill=None if skill is None else str(skill),
            args_json=None if args_json is None else str(args_json),
            outcome=None if outcome is None else str(outcome),
            destination=None if destination is None else str(destination),
            justification=None if justification is None else str(justification),
            run_id=str(run_id),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AuditLog is closed")

    # ----------------------------------------------------------------- writer

    async def _append(
        self,
        kind: AuditKind,
        *,
        skill: str | None = None,
        args_json: str | None = None,
        outcome: str | None = None,
        destination: str | None = None,
        justification: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Insert one row and return the materialised :class:`AuditEntry`.

        The lock guarantees ordering is preserved; the executor offload
        keeps disk I/O off the event loop. We round-trip back through the
        connection (``last_insert_rowid``) so the caller learns the assigned
        ``id`` and can rely on it for CP9 ordering assertions.
        """
        if kind not in _KIND_VALUES:
            raise ValueError(f"unknown audit kind: {kind!r}")
        self._ensure_open()

        effective_run_id = run_id or self._run_id
        ts = self._time_source.now()
        if ts.tzinfo is None:
            # The TimeSource contract requires aware datetimes, but we
            # defend against a misbehaving fake by upgrading to UTC rather
            # than persisting an ambiguous naive timestamp.
            ts = ts.replace(tzinfo=UTC)
        ts_iso = ts.isoformat()

        loop = asyncio.get_running_loop()
        async with self._lock:
            entry_id = await loop.run_in_executor(
                None,
                self._do_insert,
                ts_iso,
                kind,
                skill,
                args_json,
                outcome,
                destination,
                justification,
                effective_run_id,
            )

        return AuditEntry(
            id=entry_id,
            ts=ts,
            kind=kind,
            skill=skill,
            args_json=args_json,
            outcome=outcome,
            destination=destination,
            justification=justification,
            run_id=effective_run_id,
        )

    def _do_insert(
        self,
        ts_iso: str,
        kind: str,
        skill: str | None,
        args_json: str | None,
        outcome: str | None,
        destination: str | None,
        justification: str | None,
        run_id: str,
    ) -> int:
        """Synchronous insert run from the executor pool."""
        cursor = self._conn.execute(
            _INSERT_SQL,
            (ts_iso, kind, skill, args_json, outcome, destination, justification, run_id),
        )
        # ``lastrowid`` is populated for INTEGER PRIMARY KEY AUTOINCREMENT
        # tables and is exactly the audit id we just wrote.
        rowid = cursor.lastrowid
        if rowid is None:  # pragma: no cover - defensive; sqlite always sets this
            raise RuntimeError("sqlite did not return a lastrowid for the audit insert")
        return int(rowid)

    # -------------------------------------------------------- public recorders

    async def record_confirmation_requested(
        self,
        *,
        skill: str,
        args_json: str | dict[str, object],
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record that the Authorization_Policy asked the user to confirm.

        MUST be called *before* the corresponding ``record_executed`` /
        ``record_denied`` so the strict-id ordering required by CP9 holds.
        """
        return await self._append(
            "confirmation_requested",
            skill=skill,
            args_json=self._serialize_args(args_json),
            run_id=run_id,
        )

    async def record_executed(
        self,
        *,
        skill: str,
        args_json: str | dict[str, object],
        outcome: str,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record a successful Tool_Call dispatch."""
        return await self._append(
            "executed",
            skill=skill,
            args_json=self._serialize_args(args_json),
            outcome=outcome,
            run_id=run_id,
        )

    async def record_denied(
        self,
        *,
        skill: str,
        args_json: str | dict[str, object],
        outcome: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record that the user denied a Destructive_Action confirmation."""
        return await self._append(
            "denied",
            skill=skill,
            args_json=self._serialize_args(args_json),
            outcome=outcome or "denied",
            run_id=run_id,
        )

    async def record_policy_violation(
        self,
        *,
        skill: str | None,
        justification: str,
        args_json: str | dict[str, object] | None = None,
        outcome: str | None = None,
        destination: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record a sandbox / network allowlist breach (Requirement 13.6)."""
        return await self._append(
            "policy_violation",
            skill=skill,
            args_json=self._serialize_args(args_json),
            outcome=outcome,
            destination=destination,
            justification=justification,
            run_id=run_id,
        )

    async def record_network_egress(
        self,
        *,
        destination: str,
        justification: str,
        skill: str | None = None,
        outcome: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record an outbound network call and the user-visible reason."""
        return await self._append(
            "network_egress",
            skill=skill,
            destination=destination,
            justification=justification,
            outcome=outcome,
            run_id=run_id,
        )

    async def record_error(
        self,
        *,
        skill: str | None,
        outcome: str,
        justification: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record an error result returned to the Dialog_Manager.

        ``outcome`` typically holds the ``error_code`` from
        :class:`SkillResult` (e.g., ``"internal_error"``) plus an optional
        traceback id; ``justification`` may carry a human-readable reason.
        """
        return await self._append(
            "error",
            skill=skill,
            outcome=outcome,
            justification=justification,
            run_id=run_id,
        )

    async def record_crash(
        self,
        *,
        outcome: str,
        justification: str | None = None,
        run_id: str | None = None,
    ) -> AuditEntry:
        """Record that the previous run terminated unexpectedly.

        Emitted by the crash-detection flow described in Requirement 17.4
        when the ``last_run.json`` sentinel is found stale on launch.
        """
        return await self._append(
            "crash",
            outcome=outcome,
            justification=justification,
            run_id=run_id,
        )

    # ----------------------------------------------------------------- wipe

    async def wipe(self) -> None:
        """Delete every row and reset the autoincrement counter.

        Implements the audit-log half of Requirement 13.5: when the user
        requests deletion of all stored data, the audit log is erased and
        the row-id sequence restarts at 1 on the next insert.
        """
        self._ensure_open()
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self._do_wipe)

    def _do_wipe(self) -> None:
        # Wrap both statements in a single transaction so we never observe
        # a half-wiped state from a concurrent reader.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("DELETE FROM audit")
            # ``sqlite_sequence`` only exists when the database has at least
            # one ``AUTOINCREMENT`` table, which is true here, but be
            # defensive in case the file was migrated externally.
            self._conn.execute(
                "DELETE FROM sqlite_sequence WHERE name = 'audit'"
            )
            self._conn.execute("COMMIT")
        except sqlite3.OperationalError:
            # Roll back and re-raise; callers can decide whether to retry.
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise

    # -------------------------------------------------------------- readers

    def count(self) -> int:
        """Return the number of rows currently in the log (sync)."""
        self._ensure_open()
        cursor = self._conn.execute(_COUNT_SQL)
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def entries(self) -> list[AuditEntry]:
        """Return all entries in strict id-ascending order (sync).

        Intended for tests and diagnostics; the production hot path never
        scans the full table.
        """
        self._ensure_open()
        cursor = self._conn.execute(_SELECT_ALL_SQL)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    # --------------------------------------------------------------- lifecycle

    @property
    def run_id(self) -> str:
        """Return the run identifier persisted with every entry."""
        return self._run_id

    @property
    def db_path(self) -> Path | str:
        """Return the resolved database path (or ``":memory:"``)."""
        return self._db_path

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call multiple times; subsequent ``record_*`` calls raise
        :class:`RuntimeError`.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover - close errors are non-fatal
            logger.exception("error while closing audit log connection")

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
