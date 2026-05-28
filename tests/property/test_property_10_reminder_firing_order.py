"""Property 10 — Reminder firing order (CP13).

From ``design.md §Correctness Properties``:

    *For any* set of reminders ``{R1, ..., Rn}`` with strictly ordered
    ``(trigger_at, seq)`` keys, when a synthetic monotonically advancing
    clock reaches each trigger time, ``ReminderService`` SHALL deliver
    ``notify`` events in exactly the order induced by ``(trigger_at, seq)``.

This test drives the ordering invariant directly through the production
code path:

* The Hypothesis strategy :func:`tests.strategies.reminder_sets` emits a
  list of ``(trigger_at, seq, label)`` triples already sorted by
  ``(trigger_at, seq)`` with strict total ordering on the keys
  (Property 10's precondition).
* Each example inserts those reminders into the metadata table of a
  freshly-constructed :class:`ReminderService` *in a permuted order*,
  so the seq SQLite assigns through ``INTEGER PRIMARY KEY AUTOINCREMENT``
  (the production ``Reminder.seq``) is decoupled from the strategy's
  abstract seq. This is the half of CP13 that says "seq reflects
  insertion order" — by shuffling the insertion order, we exercise the
  scenario where two reminders sharing a ``trigger_at`` must be
  tie-broken on insertion order.
* The :class:`FakeTimeSource` is then advanced past every persisted
  ``trigger_at`` and the production ``_flush_due_reminders`` path is
  driven. That coroutine is the exact code path Requirement 6.6 names
  for the startup grace window: it walks ``reminders`` filtered to
  ``trigger_at <= now``, sorted by ``(trigger_at, seq)``, and dispatches
  each one through :meth:`ReminderService._fire`. APScheduler is
  bypassed entirely so the test does not depend on real wall-clock time.
* A recording :class:`_RecordingToast` captures the order of
  ``notify(title, body)`` calls. The property's invariant is then a
  simple equality between the toast's call sequence and the
  ``(trigger_at, seq)`` sort of the persisted reminders.

A complementary deterministic test pins the duplicate-``trigger_at``
case explicitly — :func:`reminder_sets` generates strictly increasing
``trigger_at`` values, which trivially satisfies CP13 *without* ever
exercising the ``seq`` tie-breaker. The companion test inserts
multiple reminders that share a single ``trigger_at`` and asserts they
fire in insertion order, closing the gap.

Validates: Requirements 6.2, 6.4 (CP13)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from hypothesis import HealthCheck, given, settings, strategies as st
from tests.strategies import reminder_sets

from jarvis.reminders.service import ReminderService
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Helpers — fake collaborators
# ---------------------------------------------------------------------------


# The metadata-table INSERT mirrors the production
# ``ReminderService._add_record`` path: every column except ``seq``
# (which is ``INTEGER PRIMARY KEY AUTOINCREMENT``) is supplied so SQLite
# assigns the same ``seq`` shape the production code observes. Inlining
# the SQL keeps the test self-contained and decouples it from any
# refactor of the service module's private constant names.
_INSERT_SQL = (
    "INSERT INTO reminders "
    "(reminder_id, kind, label, trigger_at, duration_seconds, "
    "created_at, cancelled_at) "
    "VALUES (?, ?, ?, ?, ?, ?, NULL)"
)


class _RecordingToast:
    """Captures every :meth:`notify` call in arrival order.

    The notification dispatch path inside :meth:`ReminderService._fire`
    is serialised by ``_fire_lock``, so the order this list observes is
    the order in which firing actually happens — exactly what the
    property quantifies over.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def notify(self, title: str, body: str) -> None:
        self.events.append((title, body))


class _SilentTTS:
    """:class:`_TTSLike` stub that never speaks.

    The TTS branch in :meth:`ReminderService._dispatch_notifications`
    is gated by :meth:`is_playing`; returning ``False`` keeps the test
    focused on the toast ordering invariant. :meth:`speak` raises so a
    regression that flips the gate would surface immediately rather
    than silently changing the call sequence.
    """

    async def speak(self, text: str) -> None:  # pragma: no cover — gated off
        raise AssertionError("TTS.speak must not be called when is_playing() is False")

    def is_playing(self) -> bool:
        return False


def _insert_reminder_row(
    service: ReminderService,
    *,
    label: str,
    trigger_at: datetime,
) -> int:
    """Persist a metadata row directly and return the AUTOINCREMENT ``seq``.

    Bypasses the APScheduler job-store leg so the test does not depend
    on real wall-clock time. The columns and serialisation match the
    production ``_add_record`` path exactly so :meth:`_fire` round-trips
    the row through :meth:`_row_to_reminder` without surprises.
    """

    now_iso = service._time_source.now().astimezone(UTC).isoformat()
    cursor = service._meta_conn.execute(
        _INSERT_SQL,
        (
            uuid4().hex,
            "reminder",
            label,
            trigger_at.astimezone(UTC).isoformat(),
            None,
            now_iso,
        ),
    )
    seq = cursor.lastrowid
    assert seq is not None  # SQLite always populates lastrowid for INSERTs
    return int(seq)


def _build_service(tmp_path: Path, fake_clock: FakeTimeSource) -> tuple[
    ReminderService, _RecordingToast
]:
    """Construct a non-started :class:`ReminderService` for a single example.

    The service is constructed with a unique SQLite filename per call so
    Hypothesis examples never share metadata state; the recording toast
    is returned alongside so the caller can inspect the firing order.
    """

    toast = _RecordingToast()
    db_path = tmp_path / f"reminders-{uuid4().hex}.sqlite"
    service = ReminderService(
        db_path=db_path,
        toast=toast,
        tts=_SilentTTS(),
        time_source=fake_clock,
    )
    return service, toast


# ---------------------------------------------------------------------------
# Property 10 — firing order matches (trigger_at, seq)
# ---------------------------------------------------------------------------


@given(
    triples=reminder_sets(min_size=1, max_size=8),
    rnd=st.randoms(use_true_random=False),
)
@settings(
    suppress_health_check=(
        # ``tmp_path`` is function-scoped; reusing it across Hypothesis
        # examples is safe because every example creates a SQLite file
        # under a unique uuid name.
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
def test_reminder_firing_order_matches_trigger_seq_order(
    tmp_path: Path,
    triples: list[tuple[datetime, int, str]],
    rnd: Any,
) -> None:
    """Synthetic clock past every trigger fires notify in (trigger_at, seq) order.

    **Validates: Requirements 6.2, 6.4 (CP13)**
    """

    base_dt = datetime(2024, 1, 1, tzinfo=UTC)
    fake_clock = FakeTimeSource(now=base_dt)
    service, toast = _build_service(tmp_path, fake_clock)
    try:
        # Permute the insertion order so SQLite-assigned seq differs from
        # the strategy's abstract seq. The (trigger_at, seq_assigned)
        # tie-breaker — the contract Property 10 asserts — is on
        # insertion order, not on the strategy's generated seq.
        permuted = list(range(len(triples)))
        rnd.shuffle(permuted)

        # ``persisted`` records (trigger_at, seq_assigned, label) so we
        # can build the expected order from the values SQLite actually
        # assigned. Labels are tagged with the insertion index so any
        # duplicate labels generated by the strategy do not mask an
        # ordering bug — every reminder appears as a unique body.
        persisted: list[tuple[datetime, int, str]] = []
        for insertion_index, source_index in enumerate(permuted):
            trigger_at, _strategy_seq, _strategy_label = triples[source_index]
            label = f"reminder-{insertion_index:04d}-{uuid4().hex[:8]}"
            assigned_seq = _insert_reminder_row(
                service, label=label, trigger_at=trigger_at
            )
            persisted.append((trigger_at, assigned_seq, label))

        # Advance the synthetic clock strictly past every trigger so
        # _SELECT_DUE_SQL (``trigger_at <= now``) returns every row.
        max_trigger = max(t for t, _, _ in persisted)
        delta = (max_trigger - fake_clock.now()).total_seconds() + 1.0
        fake_clock.advance(delta)

        # Drive the production firing path. ``_flush_due_reminders``
        # is the explicit half of Requirement 6.6 — it walks the
        # metadata table sorted by ``(trigger_at, seq)`` and invokes
        # ``_fire`` for each due reminder. ``_fire`` then calls
        # ``_dispatch_notifications`` under ``_fire_lock``, which is
        # what guarantees serialised, ordered toast delivery.
        asyncio.run(service._flush_due_reminders())

        # Expected firing order: sort persisted by (trigger_at, seq).
        expected_labels = [
            label
            for _, _, label in sorted(persisted, key=lambda r: (r[0], r[1]))
        ]
        # Toast bodies equal labels because labels are non-empty, so
        # the ``body = reminder.label or title`` fall-through never
        # fires — see ``ReminderService._dispatch_notifications``.
        actual_labels = [body for _, body in toast.events]

        assert actual_labels == expected_labels, (
            "ReminderService did not fire notify events in (trigger_at, seq) "
            "order:\n"
            f"  expected (sorted by (trigger_at, seq)): {expected_labels!r}\n"
            f"  actual   (toast.events order):           {actual_labels!r}\n"
            f"  persisted reminders: {persisted!r}"
        )

        # Every reminder must have fired exactly once — ``_fire`` marks
        # the row as completed by writing ``cancelled_at``, and the
        # ``_SELECT_DUE_SQL`` filter drops cancelled rows on subsequent
        # flushes. Re-running the flush MUST be a no-op.
        toast.events.clear()
        asyncio.run(service._flush_due_reminders())
        assert toast.events == [], (
            "second _flush_due_reminders fired additional events: "
            f"{toast.events!r}"
        )
    finally:
        # Tear the service down so its sqlite handle is released and
        # the module-level _SERVICES registry stays clean. ``stop()``
        # is safe to call on a never-started service: the
        # ``_scheduler is None`` branch short-circuits the shutdown leg.
        asyncio.run(service.stop())


# ---------------------------------------------------------------------------
# Companion: duplicate-``trigger_at`` exercise of the seq tie-breaker
# ---------------------------------------------------------------------------


def test_duplicate_trigger_at_fires_in_insertion_order(tmp_path: Path) -> None:
    """Reminders sharing a ``trigger_at`` fire in seq (= insertion) order.

    :func:`reminder_sets` generates strictly increasing ``trigger_at``
    values, which trivially satisfies the (trigger_at, seq) ordering
    invariant *without* ever exercising the seq tie-breaker. This
    deterministic test pins the missing case: three reminders share a
    single ``trigger_at`` and the firing order MUST equal the order in
    which they were persisted.

    **Validates: Requirements 6.2, 6.4 (CP13)**
    """

    base_dt = datetime(2024, 1, 1, tzinfo=UTC)
    fake_clock = FakeTimeSource(now=base_dt)
    service, toast = _build_service(tmp_path, fake_clock)
    try:
        shared_trigger = base_dt + timedelta(seconds=30)
        labels_in_insertion_order = [
            "duplicate-A",
            "duplicate-B",
            "duplicate-C",
        ]
        for label in labels_in_insertion_order:
            _insert_reminder_row(
                service, label=label, trigger_at=shared_trigger
            )

        fake_clock.advance(60.0)
        asyncio.run(service._flush_due_reminders())

        actual_labels = [body for _, body in toast.events]
        assert actual_labels == labels_in_insertion_order, (
            "Reminders sharing a trigger_at did not fire in insertion order: "
            f"expected={labels_in_insertion_order!r}, actual={actual_labels!r}"
        )
    finally:
        asyncio.run(service.stop())
