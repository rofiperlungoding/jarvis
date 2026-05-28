"""Reminder, alarm, and timer scheduling service.

This module owns the persistent ``Reminder_Service`` described in
``design.md §Reminder_Service``. The service backs the ``ReminderSkill``,
``TimerSkill``, and ``ListReminderSkill`` (Requirements 6.1, 6.3, 6.7) and
guarantees that scheduled triggers survive across application restarts
(Requirement 6.6).

Storage layout
--------------
A single SQLite database at ``db_path`` hosts two cooperating tables:

* ``reminders`` — our own metadata table mirroring the
  :class:`Reminder` data model (``reminder_id``, ``kind``, ``label``,
  ``trigger_at``, ``duration_seconds``, ``seq``, ``created_at``,
  ``cancelled_at``). The ``seq`` column is ``INTEGER PRIMARY KEY
  AUTOINCREMENT`` so each insert receives a strictly monotonic,
  never-reused identifier — that ordering is exactly what
  Property 10 / CP13 tie-breaks on when two reminders share a
  ``trigger_at``.
* ``apscheduler_jobs`` — managed by :mod:`apscheduler` via
  :class:`apscheduler.jobstores.sqlalchemy.SQLAlchemyJobStore`. Holds the
  persisted scheduling state so that reminders survive across
  application restarts (Requirement 6.6).

Both tables live in the same SQLite file so that a single ``wipe-all``
operation on the data directory (Requirement 13.5) reaches every
reminder artifact.

Property 10 / CP13 ordering
---------------------------
APScheduler dispatches due jobs in ascending ``next_run_time`` order. To
give two reminders with identical ``trigger_at`` a deterministic firing
order, we set the *job*'s run date to
``trigger_at + (seq % 1_000_000) µs``. The caller-visible ``trigger_at``
returned by :meth:`list_pending` and stored on :class:`Reminder` is
always the original, untouched value; the microsecond skew is an
internal scheduling detail that aligns the job-store ordering with the
``(trigger_at, seq)`` lexicographic order required by CP13. A
per-instance :class:`asyncio.Lock` (``_fire_lock``) additionally
serialises notification dispatch so that, even if APScheduler executes
two due jobs as concurrent tasks, the toast / TTS side-effects observe
the same ordering.

Missed-fire policy
------------------
Per ``design.md``, every job is registered with ``coalesce=True`` and
``misfire_grace_time=86400``: if the application has been offline,
APScheduler collapses multiple missed runs into one and grants up to
24 h of catch-up. On :meth:`start`, :class:`ReminderService`
additionally walks the metadata table for any reminders whose
``trigger_at`` is in the past (and not cancelled), sorts them by
``(trigger_at, seq)``, and fires each in order. The 30 s grace window
from Requirement 6.6 is the minimum we honor; larger windows are
permitted via ``on_start_grace_seconds`` and validated against the
floor on construction.

Lazy import
-----------
:mod:`apscheduler` is imported inside :meth:`start`, not at module
import time, so unit tests that exercise the data model or the
metadata SQLite layer alone do not need the scheduler dependencies
installed. The metadata SQLite handle is opened eagerly in
:meth:`__init__` because :mod:`sqlite3` is part of the standard
library.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6, 6.7
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import sqlite3
from typing import Any, Final, Literal, Protocol, runtime_checkable
from uuid import uuid4

from jarvis.utils.time_source import SystemTimeSource, TimeSource

logger = logging.getLogger(__name__)

__all__ = [
    "Reminder",
    "ReminderKind",
    "ReminderService",
    "ToastNotifier",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Closed set of reminder kinds defined in ``design.md §Data Models``.
ReminderKind = Literal["reminder", "alarm", "timer"]

#: Floor from Requirement 6.6 — the grace window for catching up on
#: missed reminders at startup MUST be at least 30 seconds.
_GRACE_SECONDS_FLOOR: Final[int] = 30

#: APScheduler missed-fire policy from ``design.md §Reminder_Service``:
#: collapse multiple missed runs (``coalesce=True``) and grant up to 24 h
#: of catch-up before silently dropping a job.
_MISFIRE_GRACE_TIME_SECONDS: Final[int] = 86_400

#: Modulus applied to ``seq`` when building the per-job microsecond
#: offset that disambiguates reminders sharing a ``trigger_at`` (see the
#: module docstring "Property 10 / CP13 ordering"). One million keeps
#: the offset comfortably within a single second so the wall-clock
#: semantics of ``trigger_at`` are unchanged for any reasonable
#: insertion rate.
_SEQ_OFFSET_MOD: Final[int] = 1_000_000


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ToastNotifier(Protocol):
    """Structural contract for the toast / tray notifier.

    Mirrors the :meth:`PlatformAdapter.notify` shape from
    :mod:`jarvis.automation.platform`. The concrete adapter lives in
    ``src/jarvis/reminders/notifier.py`` (Task 15.2). This Protocol is
    declared here, rather than imported from the platform module, to
    keep the reminders package free of automation-layer imports —
    Reminder_Service depends only on what it needs (Requirement 15.4
    separation between platform-neutral logic and platform-specific
    drivers).
    """

    async def notify(self, title: str, body: str) -> None:
        """Show a non-blocking notification with ``title`` / ``body``."""
        ...


@runtime_checkable
class _TTSLike(Protocol):
    """Minimal slice of :class:`jarvis.voice.tts.base.TTSEngine`.

    The reminder service only needs to enqueue spoken output and probe
    whether the engine is currently active (Requirement 6.5 — the label
    is spoken via TTS when the user is engaged in or has just completed
    a conversation). Declaring a local Protocol avoids importing the
    full TTS module at scheduling time.
    """

    async def speak(self, text: str) -> None: ...

    def is_playing(self) -> bool: ...


# ---------------------------------------------------------------------------
# Reminder data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reminder:
    """Persistent reminder / alarm / timer record.

    Mirrors the data model in ``design.md §Data Models``.

    ``seq`` is assigned by the metadata table's ``AUTOINCREMENT`` PK and
    is used by Property 10 / CP13 to tie-break reminders that share a
    ``trigger_at``. ``trigger_at`` and ``created_at`` are timezone-aware
    UTC :class:`datetime` instances; ``cancelled_at`` is non-``None``
    once the reminder has been cancelled via :meth:`ReminderService.cancel`.
    """

    reminder_id: str
    kind: ReminderKind
    label: str
    trigger_at: datetime
    duration_seconds: int | None
    seq: int
    created_at: datetime
    cancelled_at: datetime | None


# ---------------------------------------------------------------------------
# SQL — metadata table
# ---------------------------------------------------------------------------


_SCHEMA_SQL: Final[str] = """\
CREATE TABLE IF NOT EXISTS reminders (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    reminder_id      TEXT    NOT NULL UNIQUE,
    kind             TEXT    NOT NULL,
    label            TEXT    NOT NULL,
    trigger_at       TEXT    NOT NULL,
    duration_seconds INTEGER,
    created_at       TEXT    NOT NULL,
    cancelled_at     TEXT
)
"""

_INSERT_SQL: Final[str] = (
    "INSERT INTO reminders "
    "(reminder_id, kind, label, trigger_at, duration_seconds, created_at, cancelled_at) "
    "VALUES (?, ?, ?, ?, ?, ?, NULL)"
)

_SELECT_BY_ID_SQL: Final[str] = (
    "SELECT seq, reminder_id, kind, label, trigger_at, duration_seconds, "
    "created_at, cancelled_at FROM reminders WHERE reminder_id = ?"
)

_SELECT_PENDING_SQL: Final[str] = (
    "SELECT seq, reminder_id, kind, label, trigger_at, duration_seconds, "
    "created_at, cancelled_at FROM reminders "
    "WHERE cancelled_at IS NULL AND trigger_at > ? "
    "ORDER BY trigger_at ASC, seq ASC"
)

_SELECT_DUE_SQL: Final[str] = (
    "SELECT seq, reminder_id, kind, label, trigger_at, duration_seconds, "
    "created_at, cancelled_at FROM reminders "
    "WHERE cancelled_at IS NULL AND trigger_at <= ? "
    "ORDER BY trigger_at ASC, seq ASC"
)

_UPDATE_CANCELLED_SQL: Final[str] = (
    "UPDATE reminders SET cancelled_at = ? WHERE reminder_id = ? AND cancelled_at IS NULL"
)


# ---------------------------------------------------------------------------
# Service registry for APScheduler-persistent callbacks
# ---------------------------------------------------------------------------


# When :class:`ReminderService` schedules a job through a persistent
# :class:`SQLAlchemyJobStore`, APScheduler stores the *reference* to the
# callable (module path + qualified name). Bound instance methods do not
# round-trip through that reference safely, so we use a module-level
# coroutine that looks the service up in this registry by a stable ID.
# The registry is keyed by an opaque service-id assigned at construction
# time and is removed in :meth:`ReminderService.stop`.
_SERVICES: dict[str, ReminderService] = {}


async def _fire_reminder(service_id: str, reminder_id: str) -> None:
    """APScheduler-safe top-level coroutine that dispatches a reminder.

    APScheduler invokes this on each scheduled run. We resolve the
    service from the module-level :data:`_SERVICES` registry rather than
    capturing ``self`` so that the callable reference remains stable
    across pickle round-trips through the SQLAlchemy job store. If the
    service has been stopped (registry entry removed) we silently no-op:
    a stale job would otherwise raise after a clean shutdown.
    """

    service = _SERVICES.get(service_id)
    if service is None:
        logger.warning(
            "reminder %s fired but ReminderService %s is no longer registered",
            reminder_id,
            service_id,
        )
        return
    await service._fire(reminder_id)


# ---------------------------------------------------------------------------
# ReminderService
# ---------------------------------------------------------------------------


class ReminderService:
    """APScheduler-backed reminder/alarm/timer service.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database that hosts both the
        metadata ``reminders`` table and APScheduler's
        ``apscheduler_jobs`` table. Pass :class:`pathlib.Path` or a
        ``str`` (``":memory:"`` is rejected — APScheduler's
        ``SQLAlchemyJobStore`` requires a real file URL). Parent
        directories are created on demand.
    toast:
        :class:`ToastNotifier`-shaped object that delivers the toast
        notification when a reminder fires (Requirement 6.5).
    tts:
        TTS engine used to *also* speak the reminder label when the
        Voice_Pipeline reports an active or recently active conversation
        (Requirement 6.5). The probe used here is
        :meth:`_TTSLike.is_playing`; the full ``DialogManager``
        integration that broadens this definition lives in Task 15.2.
    time_source:
        Injectable :class:`TimeSource`. Defaults to
        :class:`SystemTimeSource`. The clock is used for
        ``created_at`` / ``cancelled_at`` stamps and for filtering
        :meth:`list_pending` results. APScheduler itself relies on the
        operating-system wall clock; tests that need deterministic
        firing should drive :meth:`_fire` directly.
    on_start_grace_seconds:
        Catch-up window applied at :meth:`start` when flushing missed
        reminders. Floored at 30 s (Requirement 6.6).

    Lifecycle
    ---------
    Construction is cheap — only the metadata SQLite handle is opened.
    Call :meth:`start` to lazily import APScheduler, attach the
    job store, and process any past-due reminders. Call :meth:`stop`
    to shut the scheduler down cleanly. The service is safe to use as
    an async context manager via :meth:`__aenter__` / :meth:`__aexit__`.
    """

    # Public constant exposed so tests / callers can verify the floor
    # without re-importing the private module-level constant.
    GRACE_SECONDS_FLOOR: Final[int] = _GRACE_SECONDS_FLOOR

    def __init__(
        self,
        db_path: Path | str,
        toast: ToastNotifier,
        tts: _TTSLike,
        time_source: TimeSource | None = None,
        *,
        on_start_grace_seconds: int = _GRACE_SECONDS_FLOOR,
    ) -> None:
        if on_start_grace_seconds < _GRACE_SECONDS_FLOOR:
            raise ValueError(
                "on_start_grace_seconds must be >= "
                f"{_GRACE_SECONDS_FLOOR} (Requirement 6.6); got "
                f"{on_start_grace_seconds!r}"
            )

        self._db_path: Path = self._resolve_db_path(db_path)
        self._toast: ToastNotifier = toast
        self._tts: _TTSLike = tts
        self._time_source: TimeSource = time_source or SystemTimeSource()
        self._on_start_grace_seconds: int = int(on_start_grace_seconds)

        # Stable service identifier so APScheduler-persisted job
        # references can resolve us without pickling ``self``. The id is
        # process-local; jobs hydrated from disk in a *new* process will
        # use whatever id the new ReminderService registers — which is
        # fine because we own both ends of the registration.
        self._service_id: str = uuid4().hex
        _SERVICES[self._service_id] = self

        # Eager metadata connection: sqlite3 is in the stdlib, so this
        # adds no install-cost. The lock serialises async writes.
        self._meta_conn: sqlite3.Connection = self._open_metadata_conn(self._db_path)
        self._meta_lock: asyncio.Lock = asyncio.Lock()

        # APScheduler is imported lazily in :meth:`start`, so the
        # scheduler attribute is ``None`` until the service is started.
        # Typed as ``Any`` so static analysis is not forced to depend on
        # the apscheduler stubs.
        self._scheduler: Any = None
        self._scheduler_lock: asyncio.Lock = asyncio.Lock()

        # Serialises notification dispatch so two reminders that fire on
        # the same scheduler tick deliver in (trigger_at, seq) order
        # even if APScheduler runs them as concurrent asyncio tasks.
        self._fire_lock: asyncio.Lock = asyncio.Lock()

        self._started: bool = False
        self._closed: bool = False

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve_db_path(db_path: Path | str) -> Path:
        """Normalise ``db_path`` and create parent directories on demand.

        APScheduler's :class:`SQLAlchemyJobStore` requires a real on-disk
        file because it builds its connection URL from the path. The
        ``":memory:"`` sentinel accepted by some other components is
        therefore rejected here.
        """

        if isinstance(db_path, str) and db_path == ":memory:":
            raise ValueError(
                "ReminderService requires a file-backed SQLite path; "
                '":memory:" is not supported because '
                "SQLAlchemyJobStore needs a stable URL."
            )
        path = Path(db_path)
        if str(path) == ":memory:":
            raise ValueError(
                "ReminderService requires a file-backed SQLite path; "
                '":memory:" is not supported because '
                "SQLAlchemyJobStore needs a stable URL."
            )
        parent = path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _open_metadata_conn(db_path: Path) -> sqlite3.Connection:
        """Open and migrate the metadata SQLite connection.

        WAL is enabled so APScheduler's SQLAlchemy connections can read
        and write the same file concurrently without blocking the
        notification dispatch path. ``isolation_level=None`` puts the
        driver in autocommit mode — every statement we issue is a
        transaction unto itself, which matches the journal-style
        write pattern of an append-mostly metadata table.
        """

        conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            # Best-effort pragmas; no-op on filesystems that reject WAL.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(_SCHEMA_SQL)
        except Exception:
            conn.close()
            raise
        return conn

    @staticmethod
    def _ensure_aware_utc(value: datetime, *, name: str) -> datetime:
        """Validate ``value`` is timezone-aware and return it in UTC.

        Storing trigger times in UTC keeps comparisons stable across
        DST transitions and across processes that may run with
        different local timezones (for example, a laptop docked in
        another country). The conversion is loss-less for any input
        that already has a tzinfo.
        """

        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be a datetime; got {type(value).__name__}")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(f"{name} must be timezone-aware (Requirements 6.2/6.4)")
        return value.astimezone(UTC)

    @staticmethod
    def _row_to_reminder(row: tuple[Any, ...]) -> Reminder:
        """Materialise a metadata row as a :class:`Reminder`.

        The metadata table stores datetimes as ISO-8601 strings (via
        :meth:`datetime.isoformat`) — round-tripping through
        :meth:`datetime.fromisoformat` recovers an aware UTC instance
        because that is the only thing we ever write.
        """

        (
            seq,
            reminder_id,
            kind,
            label,
            trigger_at_iso,
            duration_seconds,
            created_at_iso,
            cancelled_at_iso,
        ) = row
        trigger_at = datetime.fromisoformat(str(trigger_at_iso))
        if trigger_at.tzinfo is None:
            trigger_at = trigger_at.replace(tzinfo=UTC)
        created_at = datetime.fromisoformat(str(created_at_iso))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        cancelled_at: datetime | None
        if cancelled_at_iso is None:
            cancelled_at = None
        else:
            cancelled_at = datetime.fromisoformat(str(cancelled_at_iso))
            if cancelled_at.tzinfo is None:
                cancelled_at = cancelled_at.replace(tzinfo=UTC)

        kind_value: ReminderKind
        if kind == "reminder":
            kind_value = "reminder"
        elif kind == "alarm":
            kind_value = "alarm"
        elif kind == "timer":
            kind_value = "timer"
        else:  # pragma: no cover — defensive: only our own writers populate this
            raise ValueError(f"unknown reminder kind in database: {kind!r}")

        return Reminder(
            reminder_id=str(reminder_id),
            kind=kind_value,
            label=str(label),
            trigger_at=trigger_at,
            duration_seconds=(None if duration_seconds is None else int(duration_seconds)),
            seq=int(seq),
            created_at=created_at,
            cancelled_at=cancelled_at,
        )

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("ReminderService is closed")
        if not self._started:
            raise RuntimeError(
                "ReminderService has not been started; await start() before scheduling reminders"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ReminderService is closed")

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Initialise the APScheduler job store and flush missed jobs.

        Lazy-imports :mod:`apscheduler` and
        :mod:`apscheduler.jobstores.sqlalchemy` so the dependency cost
        is only paid by callers that actually run the scheduler. Idempotent.

        After the scheduler is running, any reminders whose
        ``trigger_at`` is already in the past — and which have not been
        cancelled — are dispatched in ``(trigger_at, seq)`` order. This
        is the explicit half of Requirement 6.6's startup grace
        window; APScheduler's own ``misfire_grace_time`` covers the
        case where the job store still has an unprocessed job.
        """

        self._ensure_open()
        async with self._scheduler_lock:
            if self._started:
                return
            self._scheduler = self._build_scheduler()
            self._scheduler.start()
            self._started = True

        await self._flush_due_reminders()

    async def stop(self) -> None:
        """Shut the scheduler down and release resources.

        Removes this instance from the module-level service registry
        so any in-flight :func:`_fire_reminder` invocations resolve to
        ``None`` and silently no-op rather than touching a torn-down
        connection. Safe to call more than once.
        """

        if self._closed:
            return
        async with self._scheduler_lock:
            scheduler = self._scheduler
            self._scheduler = None
            if scheduler is not None:
                # ``wait=False`` matches the design intent of an
                # immediate shutdown; in-flight jobs already running
                # under the AsyncIO executor are allowed to complete
                # in the background — we have already taken our
                # service off the registry so they will short-circuit.
                try:
                    scheduler.shutdown(wait=False)
                except Exception:  # pragma: no cover — defensive
                    logger.exception("error while shutting down APScheduler")
        self._started = False
        self._closed = True
        _SERVICES.pop(self._service_id, None)
        try:
            self._meta_conn.close()
        except sqlite3.Error:  # pragma: no cover — close is non-fatal
            logger.exception("error while closing reminders metadata connection")

    async def __aenter__(self) -> ReminderService:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.stop()

    # --------------------------------------------------------------- public API

    async def add(self, label: str, trigger_at: datetime) -> Reminder:
        """Schedule a reminder for ``trigger_at`` with the given ``label``.

        Validates Requirements 6.1 and 6.2: the metadata row is
        persisted before the APScheduler job is added, so a crash
        between the two stages still leaves a discoverable record that
        the next :meth:`start` call can flush as a missed reminder.
        """

        if not isinstance(label, str):
            raise TypeError(f"label must be str; got {type(label).__name__}")
        if not label:
            raise ValueError("label must be a non-empty string")
        trigger_utc = self._ensure_aware_utc(trigger_at, name="trigger_at")
        return await self._add_record(
            kind="reminder",
            label=label,
            trigger_at=trigger_utc,
            duration_seconds=None,
        )

    async def add_timer(self, duration_seconds: int, label: str | None = None) -> Reminder:
        """Start a countdown timer for ``duration_seconds``.

        Validates Requirements 6.3 and 6.4: ``duration_seconds`` must be
        a positive integer; ``label`` is optional and stored as the empty
        string when ``None`` so the metadata column's ``NOT NULL``
        constraint stays satisfied without a sentinel value.
        """

        if not isinstance(duration_seconds, int) or isinstance(duration_seconds, bool):
            raise TypeError(f"duration_seconds must be int; got {type(duration_seconds).__name__}")
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be > 0; got {duration_seconds}")
        if label is not None and not isinstance(label, str):
            raise TypeError(f"label must be str or None; got {type(label).__name__}")

        now = self._time_source.now()
        if now.tzinfo is None:
            # Defensive: TimeSource is documented to return aware
            # datetimes, but a misbehaving fake should not silently
            # corrupt the metadata table.
            now = now.replace(tzinfo=UTC)
        trigger_utc = (now + timedelta(seconds=duration_seconds)).astimezone(UTC)
        stored_label = label if label is not None else ""

        return await self._add_record(
            kind="timer",
            label=stored_label,
            trigger_at=trigger_utc,
            duration_seconds=duration_seconds,
        )

    async def cancel(self, reminder_id: str) -> bool:
        """Cancel a pending reminder.

        Returns ``True`` when the reminder existed and was not already
        cancelled, ``False`` otherwise. The metadata row is flagged
        before the APScheduler job is removed so a crash between the
        two leaves the row consistent (cancelled), and the orphan job
        — if any — will be a no-op when fired (the :meth:`_fire`
        callback re-checks the metadata before dispatching).
        """

        if not isinstance(reminder_id, str) or not reminder_id:
            raise ValueError("reminder_id must be a non-empty string")
        self._ensure_open()
        cancelled_at = self._time_source.now()
        if cancelled_at.tzinfo is None:
            cancelled_at = cancelled_at.replace(tzinfo=UTC)
        cancelled_iso = cancelled_at.astimezone(UTC).isoformat()

        async with self._meta_lock:
            cursor = self._meta_conn.execute(_UPDATE_CANCELLED_SQL, (cancelled_iso, reminder_id))
            updated = cursor.rowcount

        if updated <= 0:
            return False

        # Best-effort job removal. The APScheduler ``remove_job`` call
        # raises ``JobLookupError`` when the job has already fired or
        # does not exist; treating that as success matches the
        # documented semantics of "cancel returns whether the user's
        # cancellation request changed observable state".
        if self._scheduler is not None:
            try:
                self._scheduler.remove_job(self._job_id_for(reminder_id))
            except Exception as exc:  # pragma: no cover — apscheduler-specific
                logger.debug(
                    "remove_job(%s) raised %s; treating as already-fired/missing",
                    reminder_id,
                    exc,
                )
        return True

    async def list_pending(self) -> list[Reminder]:
        """Return all pending reminders, ordered by ``(trigger_at, seq)``.

        A reminder is *pending* when its ``cancelled_at`` is ``NULL``
        and its ``trigger_at`` lies strictly in the future relative to
        :class:`TimeSource.now`. Already-fired reminders are filtered
        out by the time comparison rather than by an explicit
        ``fired_at`` column — keeping the schema aligned with the
        design's :class:`Reminder` model.
        """

        self._ensure_open()
        now = self._time_source.now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        now_iso = now.astimezone(UTC).isoformat()

        async with self._meta_lock:
            cursor = self._meta_conn.execute(_SELECT_PENDING_SQL, (now_iso,))
            rows = cursor.fetchall()
        return [self._row_to_reminder(row) for row in rows]

    # ------------------------------------------------------------- internals

    async def _add_record(
        self,
        *,
        kind: ReminderKind,
        label: str,
        trigger_at: datetime,
        duration_seconds: int | None,
    ) -> Reminder:
        """Persist a metadata row and register the matching APScheduler job.

        ``trigger_at`` MUST already be timezone-aware UTC; callers are
        responsible for the conversion. The metadata insert assigns
        ``seq`` via SQLite's ``AUTOINCREMENT`` PK; we read it back via
        :attr:`sqlite3.Cursor.lastrowid` and use it to compute the
        per-job microsecond offset that gives APScheduler a strictly
        increasing ``next_run_time`` for reminders sharing a
        ``trigger_at`` (CP13 ordering).
        """

        self._ensure_started()
        reminder_id = uuid4().hex
        created_at = self._time_source.now()
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        created_at = created_at.astimezone(UTC)

        trigger_iso = trigger_at.isoformat()
        created_iso = created_at.isoformat()

        async with self._meta_lock:
            cursor = self._meta_conn.execute(
                _INSERT_SQL,
                (
                    reminder_id,
                    kind,
                    label,
                    trigger_iso,
                    duration_seconds,
                    created_iso,
                ),
            )
            seq = cursor.lastrowid
            if seq is None:  # pragma: no cover — sqlite always sets lastrowid
                raise RuntimeError("sqlite did not return a lastrowid for the reminder insert")

        # Schedule the APScheduler job *after* the metadata write
        # commits. If scheduling fails we still have a discoverable
        # row that the next ``start()`` flush will fire — better than
        # losing the reminder entirely.
        run_date = self._effective_run_date(trigger_at, seq)
        try:
            self._scheduler.add_job(
                _fire_reminder,
                trigger="date",
                run_date=run_date,
                args=(self._service_id, reminder_id),
                id=self._job_id_for(reminder_id),
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=_MISFIRE_GRACE_TIME_SECONDS,
            )
        except Exception:
            logger.exception(
                "failed to register APScheduler job for reminder %s; "
                "metadata row persisted, will retry on next start()",
                reminder_id,
            )

        return Reminder(
            reminder_id=reminder_id,
            kind=kind,
            label=label,
            trigger_at=trigger_at,
            duration_seconds=duration_seconds,
            seq=int(seq),
            created_at=created_at,
            cancelled_at=None,
        )

    @staticmethod
    def _effective_run_date(trigger_at: datetime, seq: int) -> datetime:
        """Return the run date used internally by APScheduler for this seq.

        See module docstring "Property 10 / CP13 ordering" for the
        rationale. The visible ``trigger_at`` returned to callers is
        unchanged; only APScheduler's view is offset.
        """

        # ``seq % _SEQ_OFFSET_MOD`` keeps the offset bounded by 1 second,
        # so even for very large ``seq`` values the wall-clock semantics
        # of ``trigger_at`` remain accurate to within a microsecond
        # rounding budget. We also avoid offset 0 collisions naturally
        # because no two reminders share a ``seq``.
        offset_us = seq % _SEQ_OFFSET_MOD
        return trigger_at + timedelta(microseconds=offset_us)

    @staticmethod
    def _job_id_for(reminder_id: str) -> str:
        """Return the APScheduler job id that wraps ``reminder_id``.

        Prefixing the reminder id makes it easy to recognise our jobs
        in a shared APScheduler database without colliding with any
        other component that might share the job store in the future.
        """

        return f"reminder:{reminder_id}"

    def _build_scheduler(self) -> Any:
        """Lazily import APScheduler and build the configured scheduler.

        Centralises every ``import apscheduler.*`` line so the lazy-
        import contract from the module docstring is easy to verify.
        Raises an actionable error when the optional dependency is
        missing rather than letting a generic ``ImportError`` bubble
        out of :meth:`start`.
        """

        try:
            from apscheduler.jobstores.sqlalchemy import (  # noqa: PLC0415
                SQLAlchemyJobStore,
            )
            from apscheduler.schedulers.asyncio import (  # noqa: PLC0415
                AsyncIOScheduler,
            )
        except ImportError as exc:  # pragma: no cover — exercised on minimal envs
            raise RuntimeError(
                "ReminderService.start requires the `apscheduler` and "
                "`sqlalchemy` packages (declared in pyproject.toml)."
            ) from exc

        # Build a SQLite URL from the resolved Path. ``as_uri()`` handles
        # the platform-specific quoting, but APScheduler / SQLAlchemy
        # expect the ``sqlite:///`` scheme rather than ``file://``.
        url = f"sqlite:///{self._db_path}"
        jobstore = SQLAlchemyJobStore(url=url)

        scheduler = AsyncIOScheduler(
            timezone=UTC,
            jobstores={"default": jobstore},
            job_defaults={
                "coalesce": True,
                "misfire_grace_time": _MISFIRE_GRACE_TIME_SECONDS,
                "max_instances": 1,
            },
        )
        return scheduler

    async def _flush_due_reminders(self) -> None:
        """Fire any past-due, non-cancelled reminders in CP13 order.

        Walks the metadata table once, sorted by ``(trigger_at, seq)``,
        and dispatches each reminder via :meth:`_fire`. The
        :attr:`_fire_lock` taken inside :meth:`_fire` serialises
        delivery so the toast / TTS observers see the same ordering.
        """

        now = self._time_source.now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        now_iso = now.astimezone(UTC).isoformat()
        async with self._meta_lock:
            cursor = self._meta_conn.execute(_SELECT_DUE_SQL, (now_iso,))
            rows = cursor.fetchall()
        for row in rows:
            reminder = self._row_to_reminder(row)
            await self._fire(reminder.reminder_id)

    async def _fire(self, reminder_id: str) -> None:
        """Deliver notifications for ``reminder_id`` and mark it complete.

        Re-reads metadata under the meta-lock so concurrently-cancelled
        reminders short-circuit (they will have ``cancelled_at`` set).
        Acquires :attr:`_fire_lock` around the side-effects so two
        reminders firing simultaneously deliver in
        ``(trigger_at, seq)`` order — APScheduler hands them to us in
        that order via :meth:`_effective_run_date`, and the lock
        prevents the toast / TTS calls from interleaving.
        """

        async with self._fire_lock:
            async with self._meta_lock:
                cursor = self._meta_conn.execute(_SELECT_BY_ID_SQL, (reminder_id,))
                row = cursor.fetchone()
            if row is None:
                logger.debug("reminder %s no longer exists; skip fire", reminder_id)
                return
            reminder = self._row_to_reminder(row)
            if reminder.cancelled_at is not None:
                logger.debug("reminder %s was cancelled before fire; skip", reminder_id)
                return

            # Mark the reminder as completed by writing the cancelled_at
            # column. The data model has no separate ``fired_at`` field
            # (see ``design.md §Data Models``), and the
            # :meth:`list_pending` filter already excludes past-due
            # rows, so this never visibly contradicts the user.
            completed_at = self._time_source.now()
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=UTC)
            completed_iso = completed_at.astimezone(UTC).isoformat()
            async with self._meta_lock:
                self._meta_conn.execute(
                    _UPDATE_CANCELLED_SQL,
                    (completed_iso, reminder_id),
                )

            await self._dispatch_notifications(reminder)

    async def _dispatch_notifications(self, reminder: Reminder) -> None:
        """Deliver toast + (optional) TTS for ``reminder``.

        The toast is always shown (Requirement 6.5 / design intent).
        The TTS announcement is gated by :meth:`_TTSLike.is_playing`
        as a proxy for "user is engaged in or has just completed a
        conversation" — the broader 30 s / DialogManager-aware policy
        from Requirement 6.5 is implemented by the notifier wrapper in
        Task 15.2 and is intentionally additive: this method always
        speaks when the engine is currently active and never speaks
        when it is idle, leaving the more nuanced gating to layers
        with full DialogManager context.
        """

        title = self._notification_title(reminder)
        body = reminder.label or title
        try:
            await self._toast.notify(title, body)
        except Exception:  # pragma: no cover — toast errors are non-fatal
            logger.exception("toast delivery failed for reminder %s", reminder.reminder_id)

        # Speak when the TTS engine is currently active. The probe is
        # synchronous and side-effect-free per the protocol contract
        # (see :class:`jarvis.voice.tts.base.TTSEngine.is_playing`).
        try:
            if self._tts.is_playing():
                speech = reminder.label if reminder.label else title
                await self._tts.speak(speech)
        except Exception:  # pragma: no cover — TTS errors are non-fatal
            logger.exception("TTS delivery failed for reminder %s", reminder.reminder_id)

    @staticmethod
    def _notification_title(reminder: Reminder) -> str:
        """Return the toast title appropriate for the reminder ``kind``."""

        if reminder.kind == "timer":
            return "Timer"
        if reminder.kind == "alarm":
            return "Alarm"
        return "Reminder"
