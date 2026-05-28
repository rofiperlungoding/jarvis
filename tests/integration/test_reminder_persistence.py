"""Integration test for reminder persistence across process restart.

This test exercises the cross-process leg of
:class:`~jarvis.reminders.service.ReminderService` end-to-end:

1. A first service instance starts the APScheduler-backed loop, adds a
   reminder via the public :meth:`ReminderService.add` API, then is
   shut down — simulating the application terminating with a pending
   reminder still on disk.
2. A *second* service instance is constructed pointing at the same
   ``db_path``. Its injected :class:`~jarvis.utils.time_source.FakeTimeSource`
   is advanced past the persisted ``trigger_at`` so the reminder is
   now due. Driving :meth:`ReminderService._flush_due_reminders` —
   the explicit half of Requirement 6.6's startup grace window — must
   surface the reminder to the recording :class:`_RecordingToast`.

Why a future-anchored fake clock?
---------------------------------

``ReminderService.add`` requires the scheduler to be started (via
``_ensure_started``), and the AsyncIOScheduler started by
:meth:`start` runs against the real OS wall clock — the fake clock
governs only the metadata ``created_at`` / ``trigger_at`` /
``cancelled_at`` columns and the :meth:`_flush_due_reminders` filter.
By anchoring the fake clock in the year 2099 we guarantee that the
APScheduler job written by ``add`` has a wall-clock ``run_date``
firmly in the future, so the first service's running scheduler will
never accidentally fire the job during the test. The reminder is then
delivered exclusively through the explicit metadata-driven flush path,
which is the code path Requirement 6.6 actually names.

Why call ``_flush_due_reminders`` directly on the second service?
----------------------------------------------------------------

The task brief calls out that explicitly: ``_flush_due_reminders`` is
the deterministic, scheduler-free entry point for the startup grace
window. Driving it without first :meth:`ReminderService.start`-ing the
second service avoids spinning up another AsyncIOScheduler that would
also try to consume the persisted job from the same SQLAlchemy job
store — a second ``start()`` is unnecessary noise for this property.

Validates: Requirements 6.2, 6.6
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from jarvis.reminders.service import Reminder, ReminderService
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Recording fakes
# ---------------------------------------------------------------------------


class _RecordingToast:
    """Captures every :meth:`notify` call in arrival order.

    ``ReminderService._dispatch_notifications`` always calls
    ``notify(title, body)`` exactly once per fired reminder, so the
    list of recorded events is the ground truth this integration test
    asserts against.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def notify(self, title: str, body: str) -> None:
        self.events.append((title, body))


class _RecordingTTS:
    """``_TTSLike`` stub that records ``speak`` invocations.

    The reminder service consults :meth:`is_playing` before deciding
    whether to also speak the label (Requirement 6.5 — only when the
    user is engaged in or just finished a conversation). This test is
    not about the TTS gating — it is about *persistence* — so
    :meth:`is_playing` returns ``False`` and any ``speak`` call would
    be a regression that we would want to surface immediately.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        # Recorded so a regression that flips the gate is visible in
        # the test output rather than silently no-ops.
        self.spoken.append(text)

    def is_playing(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Anchor every fake clock used in this module in the year 2099 so that any
# APScheduler job written by ``ReminderService.add`` has a wall-clock
# ``run_date`` in the real future — the live scheduler will never fire
# the job during the test, isolating the assertion to the metadata-driven
# ``_flush_due_reminders`` path that Requirement 6.6 names.
_FUTURE_BASE: datetime = datetime(2099, 1, 1, 0, 0, 0, tzinfo=UTC)


def _shared_db_path(tmp_path: Path) -> Path:
    """Return the SQLite path shared between the two service instances.

    A single file under ``tmp_path`` keeps both the metadata
    ``reminders`` table and the APScheduler ``apscheduler_jobs`` table
    accessible to each service — exactly the on-disk layout the design
    documents (``design.md §Reminder_Service > Storage layout``).
    """

    return tmp_path / "reminders.sqlite"


def _build_service(
    db_path: Path,
    fake_clock: FakeTimeSource,
    *,
    toast: _RecordingToast | None = None,
    tts: _RecordingTTS | None = None,
) -> tuple[ReminderService, _RecordingToast, _RecordingTTS]:
    """Construct a :class:`ReminderService` wired to recording collaborators.

    Each call gets its own toast / TTS recorder by default so the two
    services in a single test never share notification state — the
    second service's recorder must observe exactly the firing that
    happens after restart, with no leakage from the first service's
    setup phase.
    """

    toast = toast if toast is not None else _RecordingToast()
    tts = tts if tts is not None else _RecordingTTS()
    service = ReminderService(
        db_path=db_path,
        toast=toast,
        tts=tts,
        time_source=fake_clock,
    )
    return service, toast, tts


def _labels(events: Iterable[tuple[str, str]]) -> list[str]:
    """Project a list of ``(title, body)`` notify calls onto bodies only."""

    return [body for _, body in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reminder_added_before_termination_fires_after_restart_within_grace(
    tmp_path: Path,
) -> None:
    """A reminder added pre-termination fires on restart when due.

    The fake clock is advanced past ``trigger_at`` by 15 seconds —
    comfortably within the 30 second grace window from Requirement 6.6
    — and :meth:`ReminderService._flush_due_reminders` (the explicit
    half of that requirement) is driven directly on the second service
    instance. The recording toast captures exactly one event with the
    ``Reminder`` title and the persisted label, proving that
    persistence survived the simulated process termination.

    Validates: Requirements 6.2, 6.6
    """

    db_path = _shared_db_path(tmp_path)

    # ---- Phase 1: original process ----------------------------------
    fake_clock_1 = FakeTimeSource(now=_FUTURE_BASE)
    service1, toast1, tts1 = _build_service(db_path, fake_clock_1)

    await service1.start()
    try:
        trigger_at = fake_clock_1.now() + timedelta(seconds=60)
        reminder = await service1.add("Drink water, sir.", trigger_at)
        # Sanity: the reminder is observable as pending before restart.
        # ``list_pending`` returns rows with ``cancelled_at IS NULL`` and
        # ``trigger_at`` in the future relative to the fake clock.
        pending_pre_restart = await service1.list_pending()
        assert reminder in pending_pre_restart, (
            f"reminder {reminder!r} should be pending in service1 before "
            f"restart; saw {pending_pre_restart!r}"
        )
        # Nothing should have fired yet — the trigger is in the future
        # relative to the fake clock, and APScheduler is on the real
        # wall clock anchored in 2099.
        assert toast1.events == []
        assert tts1.spoken == []
    finally:
        # ``stop()`` shuts the scheduler down, closes the metadata
        # connection, and pops this instance out of the module-level
        # service registry — exactly the cleanup that an OS process
        # exit performs through ``atexit`` / ``__aexit__``.
        await service1.stop()

    # ---- Phase 2: fresh process pointing at the same db_path --------
    # Advance past ``trigger_at`` by 15 s — well within the 30 s grace
    # the design promises. Using a *new* FakeTimeSource (rather than
    # mutating the first one) emphasises that the second service is a
    # fully separate instance that does not share any in-memory state
    # with the first.
    fake_clock_2 = FakeTimeSource(now=trigger_at + timedelta(seconds=15))
    service2, toast2, tts2 = _build_service(db_path, fake_clock_2)
    try:
        # The reminder we persisted in phase 1 is still on disk and
        # the fake clock is past ``trigger_at``: ``_SELECT_DUE_SQL``
        # MUST surface it for ``_fire`` to process.
        await service2._flush_due_reminders()

        # Exactly one toast — the kind tag for a wall-clock reminder is
        # ``"Reminder"`` per ``_notification_title``; the body is the
        # original label.
        assert toast2.events == [("Reminder", "Drink water, sir.")], (
            "Second instance should fire exactly one reminder toast; "
            f"saw {toast2.events!r}"
        )
        # TTS is gated on ``is_playing`` returning ``True``; our stub
        # returns ``False`` so the speak path MUST stay quiet.
        assert tts2.spoken == [], (
            "TTS speak must not be called when the engine is idle "
            f"(is_playing=False); saw {tts2.spoken!r}"
        )

        # Idempotency: re-flushing should be a no-op because ``_fire``
        # marks the row as completed by writing ``cancelled_at``.
        await service2._flush_due_reminders()
        assert toast2.events == [("Reminder", "Drink water, sir.")], (
            "Second flush must be a no-op once the reminder is fired; "
            f"saw {toast2.events!r}"
        )
    finally:
        await service2.stop()


@pytest.mark.asyncio
async def test_reminder_state_round_trips_through_metadata_table(
    tmp_path: Path,
) -> None:
    """The reminder persisted by service1 is visible to service2 verbatim.

    Pins the on-disk persistence contract Requirement 6.2 leans on:
    ``trigger_at``, ``label``, and ``seq`` survive an instance
    boundary unchanged. Without this, a corrupted round-trip would
    masquerade as an ordering bug at the firing layer rather than as
    a persistence bug at the metadata layer.

    Validates: Requirements 6.2, 6.6
    """

    db_path = _shared_db_path(tmp_path)
    fake_clock_1 = FakeTimeSource(now=_FUTURE_BASE)
    service1, _toast1, _tts1 = _build_service(db_path, fake_clock_1)

    await service1.start()
    try:
        trigger_at = fake_clock_1.now() + timedelta(minutes=5)
        original = await service1.add("Renew passport.", trigger_at)
    finally:
        await service1.stop()

    # The second service must see the exact same Reminder row — same
    # id, same label, same trigger_at (UTC), same seq, same
    # ``cancelled_at = None``. ``list_pending`` filters on
    # ``trigger_at > now`` so we keep the second clock *before* the
    # trigger to keep the row in the pending set.
    fake_clock_2 = FakeTimeSource(now=_FUTURE_BASE + timedelta(seconds=1))
    service2, _toast2, _tts2 = _build_service(db_path, fake_clock_2)
    try:
        pending = await service2.list_pending()
        assert len(pending) == 1, (
            f"second service must see exactly one pending reminder; "
            f"saw {pending!r}"
        )
        restored = pending[0]
        # Compare every persisted field individually so a regression
        # surfaces with a precise failure message rather than a generic
        # "Reminder objects differ" line.
        assert restored.reminder_id == original.reminder_id
        assert restored.kind == original.kind == "reminder"
        assert restored.label == original.label == "Renew passport."
        assert restored.trigger_at == original.trigger_at
        assert restored.duration_seconds is None
        assert restored.seq == original.seq
        assert restored.cancelled_at is None
        # ``Reminder`` is a frozen dataclass so structural equality is
        # also expected to hold — included as a final belt-and-braces
        # check.
        assert restored == original
    finally:
        await service2.stop()


@pytest.mark.asyncio
async def test_cancelled_reminder_does_not_fire_after_restart(
    tmp_path: Path,
) -> None:
    """Cancellation persists across the restart boundary.

    A reminder cancelled in the first instance MUST stay cancelled in
    the second. ``_flush_due_reminders`` filters on
    ``cancelled_at IS NULL`` so the toast recorder must see no events.
    This guards against a regression where ``cancel()`` only removes
    the APScheduler job (an in-memory artifact) without flagging the
    metadata row, which would let the row resurface as a missed
    reminder on the next start.

    Validates: Requirements 6.2, 6.6
    """

    db_path = _shared_db_path(tmp_path)
    fake_clock_1 = FakeTimeSource(now=_FUTURE_BASE)
    service1, _toast1, _tts1 = _build_service(db_path, fake_clock_1)

    await service1.start()
    try:
        trigger_at = fake_clock_1.now() + timedelta(seconds=30)
        reminder = await service1.add("Cancelled item.", trigger_at)
        cancelled = await service1.cancel(reminder.reminder_id)
        assert cancelled is True
    finally:
        await service1.stop()

    # Second instance, advanced well past ``trigger_at`` so the row
    # *would* fire if it were not cancelled. The cancellation flag in
    # the metadata row must keep ``_flush_due_reminders`` quiet.
    fake_clock_2 = FakeTimeSource(now=trigger_at + timedelta(seconds=29))
    service2, toast2, tts2 = _build_service(db_path, fake_clock_2)
    try:
        await service2._flush_due_reminders()
        assert toast2.events == [], (
            "Cancelled reminder must not fire after restart; "
            f"saw {toast2.events!r}"
        )
        assert tts2.spoken == []
    finally:
        await service2.stop()


@pytest.mark.asyncio
async def test_reminder_not_yet_due_does_not_fire_after_restart(
    tmp_path: Path,
) -> None:
    """A reminder still in the future after restart stays pending.

    Pins the ``trigger_at <= now`` half of the
    ``_flush_due_reminders`` filter: when the second service's fake
    clock has *not* advanced past ``trigger_at`` (e.g., the user
    relaunches the app shortly after creating the reminder), the
    reminder must remain pending and silent until its real moment
    arrives. Without this guard, a regression that flipped the
    comparison would fire every reminder on every restart.

    Validates: Requirements 6.2, 6.6
    """

    db_path = _shared_db_path(tmp_path)
    fake_clock_1 = FakeTimeSource(now=_FUTURE_BASE)
    service1, _toast1, _tts1 = _build_service(db_path, fake_clock_1)

    await service1.start()
    try:
        trigger_at = fake_clock_1.now() + timedelta(minutes=10)
        reminder = await service1.add("Future event.", trigger_at)
    finally:
        await service1.stop()

    # Restart only 5 seconds later — long before ``trigger_at``.
    fake_clock_2 = FakeTimeSource(now=_FUTURE_BASE + timedelta(seconds=5))
    service2, toast2, tts2 = _build_service(db_path, fake_clock_2)
    try:
        await service2._flush_due_reminders()
        # Nothing fires because nothing is due.
        assert toast2.events == []
        assert tts2.spoken == []
        # The reminder is still observable as pending in the second
        # service: ``list_pending`` filters on ``trigger_at > now``,
        # which our pre-trigger fake clock satisfies.
        pending = await service2.list_pending()
        assert [r.reminder_id for r in pending] == [reminder.reminder_id]
    finally:
        await service2.stop()


@pytest.mark.asyncio
async def test_two_reminders_one_due_one_pending_round_trip(
    tmp_path: Path,
) -> None:
    """Mixed-due/-pending state is partitioned correctly across restart.

    Locks down the interaction between the two filters used by
    Requirement 6.6's startup grace window:

    * ``_SELECT_DUE_SQL`` (``trigger_at <= now AND cancelled_at IS NULL``)
      drives the firing path.
    * ``_SELECT_PENDING_SQL`` (``trigger_at > now AND cancelled_at IS NULL``)
      drives :meth:`list_pending`.

    A regression that swapped the two predicates would either fire the
    pending reminder (false positive) or leave the due reminder
    pending (false negative); this test catches both.

    Validates: Requirements 6.2, 6.6
    """

    db_path = _shared_db_path(tmp_path)
    fake_clock_1 = FakeTimeSource(now=_FUTURE_BASE)
    service1, _toast1, _tts1 = _build_service(db_path, fake_clock_1)

    due_at = _FUTURE_BASE + timedelta(seconds=30)
    later_at = _FUTURE_BASE + timedelta(minutes=10)
    persisted: list[Reminder] = []
    await service1.start()
    try:
        persisted.append(await service1.add("Earlier item.", due_at))
        persisted.append(await service1.add("Later item.", later_at))
    finally:
        await service1.stop()

    # Second service: clock past ``due_at`` (10 s past, well within
    # the 30 s grace) but still well before ``later_at``.
    fake_clock_2 = FakeTimeSource(now=due_at + timedelta(seconds=10))
    service2, toast2, _tts2 = _build_service(db_path, fake_clock_2)
    try:
        await service2._flush_due_reminders()
        # Only the due item fires.
        assert _labels(toast2.events) == ["Earlier item."], (
            "Only the due reminder should fire on restart; "
            f"saw {toast2.events!r}"
        )
        # The later item remains pending and observable as such.
        pending_ids = {r.reminder_id for r in await service2.list_pending()}
        assert pending_ids == {persisted[1].reminder_id}, (
            "Later reminder must remain pending after restart-flush; "
            f"saw pending_ids={pending_ids!r}"
        )
    finally:
        await service2.stop()


# ---------------------------------------------------------------------------
# Smoke: the test module exposes the three helpers under a stable name so
# future siblings (e.g. test_reminder_recovery.py) can reuse them rather than
# diverging on their own copy.
# ---------------------------------------------------------------------------


def test_module_exports_recording_helpers_under_stable_names() -> None:
    """Pin the helper class / function names other tests may depend on.

    Tests that grow alongside the persistence story — for example a
    future check that the restart grace path is a no-op when no
    reminders exist — will want to share these helpers. Renaming any
    of them in this module without updating dependents would break
    those tests silently; this trivial assertion makes the rename
    surface during the originating change.
    """

    # ``inspect.iscoroutinefunction`` is the modern API for the check
    # that ``asyncio.iscoroutinefunction`` provides; staying on the
    # supported entry point keeps the module deprecation-clean on
    # Python 3.14+ where the asyncio shim is being retired.
    assert inspect.iscoroutinefunction(_RecordingToast.notify)
    assert inspect.iscoroutinefunction(_RecordingTTS.speak)
    assert callable(_build_service)
    assert callable(_shared_db_path)
    # Future-anchored constant must stay in the future relative to
    # any sane real wall-clock; otherwise the isolation rationale in
    # the module docstring is invalidated.
    assert _FUTURE_BASE.year >= 2099
