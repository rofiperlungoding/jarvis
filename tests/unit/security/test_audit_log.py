"""Unit tests for ``jarvis.security.audit_log``.

Covers:
    * Monotonic ``id`` ordering across every ``record_*`` method.
    * Every one of the seven recorder methods documented in
      ``design.md §Audit_Log`` produces a row with the matching ``kind``,
      the supplied fields, and a fresh autoincrement id.
    * ``wipe()`` deletes every row and resets the SQLite autoincrement
      counter so the next insert starts again at id=1
      (Requirement 13.5).
    * Concurrent ``record_*`` invocations from the same event loop preserve
      a strict total order of ``id`` values; no two entries share an id and
      every value returned to a coroutine matches the row that was actually
      written to disk.
    * ``args_json`` is canonicalised so callers passing an ordinary dict
      still observe stable equality with the persisted string. This is the
      precondition for CP9 — :class:`AuthorizationPolicy` matches
      ``confirmation_requested`` to its companion ``executed`` / ``denied``
      entry by ``(skill, args_json)``.
    * Schema robustness: the writer rejects unknown ``kind`` values and
      empty ``run_id`` values up-front rather than leaking them into the
      database.

Validates: Requirements 13.1, 13.4, 13.5, 13.6, 16.5, 17.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from jarvis.security.audit_log import AuditEntry, AuditLog
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_source() -> FakeTimeSource:
    """A deterministic clock so entry timestamps are predictable."""
    return FakeTimeSource(now=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))


@pytest.fixture()
def audit_log_factory(
    tmp_path: Path, time_source: FakeTimeSource
) -> Callable[..., AuditLog]:
    """Build a fresh file-backed :class:`AuditLog` per test invocation.

    File-backed instead of ``:memory:`` because the wipe / autoincrement
    test relies on the on-disk ``sqlite_sequence`` row, and because some
    tests open a second sync connection to peek at the schema.
    """

    counter = {"n": 0}

    def _make(**kwargs: Any) -> AuditLog:
        counter["n"] += 1
        db_path = tmp_path / f"audit-{counter['n']}.sqlite"
        return AuditLog(
            db_path,
            time_source=kwargs.pop("time_source", time_source),
            run_id=kwargs.pop("run_id", "test-run"),
            **kwargs,
        )

    return _make


def _run(coro: Awaitable[Any]) -> Any:
    """Synchronously execute a coroutine on a fresh event loop.

    Mirrors the pattern used elsewhere in the test suite (see
    ``tests/unit/skills/test_base.py``) so the file does not require
    ``pytest-asyncio``.
    """
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        AuditLog(tmp_path / "x.sqlite", run_id="")


def test_constructor_rejects_non_string_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        AuditLog(tmp_path / "x.sqlite", run_id=123)  # type: ignore[arg-type]


def test_constructor_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "audit.sqlite"
    log = AuditLog(nested, run_id="r")
    try:
        assert nested.parent.is_dir()
        assert log.count() == 0
    finally:
        log.close()


def test_in_memory_db_path_is_preserved(time_source: FakeTimeSource) -> None:
    log = AuditLog(":memory:", time_source=time_source, run_id="mem")
    try:
        assert log.db_path == ":memory:"
        assert log.run_id == "mem"
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Monotonic ordering and the seven recorder methods
# ---------------------------------------------------------------------------


def test_each_record_method_writes_its_documented_kind(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> list[AuditEntry]:
            return [
                await log.record_confirmation_requested(
                    skill="SendEmailSkill", args_json={"recipient": "a@b"}
                ),
                await log.record_executed(
                    skill="SendEmailSkill",
                    args_json={"recipient": "a@b"},
                    outcome="ok",
                ),
                await log.record_denied(
                    skill="SendEmailSkill",
                    args_json={"recipient": "a@b"},
                ),
                await log.record_policy_violation(
                    skill="ReadFileSkill",
                    justification="path outside sandbox",
                    args_json={"path": "C:/Windows/System32"},
                    outcome="access_denied",
                ),
                await log.record_network_egress(
                    destination="https://api.openweathermap.org",
                    justification="weather lookup",
                    skill="WeatherSkill",
                ),
                await log.record_error(
                    skill="WeatherSkill",
                    outcome="provider_unavailable",
                    justification="upstream 503",
                ),
                await log.record_crash(
                    outcome="crash",
                    justification="last_run.json sentinel was stale",
                ),
            ]

        results = _run(driver())

        kinds = [e.kind for e in results]
        assert kinds == [
            "confirmation_requested",
            "executed",
            "denied",
            "policy_violation",
            "network_egress",
            "error",
            "crash",
        ]

        # Persisted rows match the in-memory results exactly.
        persisted = log.entries()
        assert [e.kind for e in persisted] == kinds
        # Every returned id is reflected in the persisted rows.
        assert [e.id for e in persisted] == [e.id for e in results]
    finally:
        log.close()


def test_ids_are_strictly_increasing_in_call_order(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> list[int]:
            ids: list[int] = []
            for n in range(10):
                e = await log.record_confirmation_requested(
                    skill="SendEmailSkill",
                    args_json={"n": n},
                )
                ids.append(e.id)
            return ids

        ids = _run(driver())
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        # AUTOINCREMENT semantics — first row id is 1, then strictly +1.
        assert ids[0] == 1
        assert ids == list(range(1, 11))
    finally:
        log.close()


def test_unknown_audit_kind_is_rejected(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> None:
            # Use the private writer through ``record_*`` semantics is fine,
            # but the documented public surface is the seven recorders.
            # Reach into ``_append`` only to assert the validation guard.
            await log._append("not_a_kind")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="unknown audit kind"):
            _run(driver())

        # And the database is untouched after the failure.
        assert log.count() == 0
    finally:
        log.close()


# ---------------------------------------------------------------------------
# args_json canonicalisation
# ---------------------------------------------------------------------------


def test_args_json_dict_is_canonicalised_with_sorted_keys(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_confirmation_requested(
                skill="SendEmailSkill",
                args_json={"recipient": "alex@example.invalid", "subject": "hi"},
            )

        entry = _run(driver())

        # Sorted keys + tight separators — the documented canonical form.
        assert entry.args_json == (
            '{"recipient":"alex@example.invalid","subject":"hi"}'
        )
        # And the round-trip through JSON yields the original mapping.
        assert json.loads(entry.args_json or "{}") == {
            "recipient": "alex@example.invalid",
            "subject": "hi",
        }
    finally:
        log.close()


def test_args_json_canonicalisation_is_order_insensitive(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    """CP9 matches ``confirmation_requested`` to ``executed`` by ``args_json``.

    Two calls with the same arguments but different dict-key insertion
    orders MUST yield byte-equal ``args_json`` strings.
    """
    log = audit_log_factory()
    try:

        async def driver() -> tuple[AuditEntry, AuditEntry]:
            a = await log.record_confirmation_requested(
                skill="SendEmailSkill",
                args_json={"a": 1, "b": 2, "c": 3},
            )
            b = await log.record_executed(
                skill="SendEmailSkill",
                args_json={"c": 3, "a": 1, "b": 2},
                outcome="ok",
            )
            return a, b

        first, second = _run(driver())
        assert first.args_json == second.args_json
        # Strict id ordering: confirmation_requested < executed (CP9).
        assert first.id < second.id
    finally:
        log.close()


def test_args_json_string_is_passed_through_verbatim(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:
        raw = '{"already":"serialized"}'

        async def driver() -> AuditEntry:
            return await log.record_executed(
                skill="LaunchAppSkill",
                args_json=raw,
                outcome="ok",
            )

        entry = _run(driver())
        assert entry.args_json == raw
    finally:
        log.close()


def test_none_args_json_remains_none(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_network_egress(
                destination="https://example.invalid",
                justification="health check",
            )

        entry = _run(driver())
        assert entry.args_json is None
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Wipe semantics
# ---------------------------------------------------------------------------


def test_wipe_removes_all_rows_and_resets_autoincrement(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def populate() -> None:
            for n in range(5):
                await log.record_executed(
                    skill="WeatherSkill",
                    args_json={"n": n},
                    outcome="ok",
                )

        _run(populate())
        assert log.count() == 5

        _run(log.wipe())
        assert log.count() == 0
        assert log.entries() == []

        async def reseed() -> AuditEntry:
            return await log.record_executed(
                skill="WeatherSkill",
                args_json={"n": 0},
                outcome="ok",
            )

        # AUTOINCREMENT counter must restart at 1 — Requirement 13.5
        # documents that wipe leaves the audit log in a fresh state.
        post_wipe = _run(reseed())
        assert post_wipe.id == 1
    finally:
        log.close()


def test_wipe_clears_sqlite_sequence_row(
    audit_log_factory: Callable[..., AuditLog],
    tmp_path: Path,
) -> None:
    """Belt-and-braces: the underlying ``sqlite_sequence`` row is gone too."""
    log = audit_log_factory()
    db_path = log.db_path
    try:

        async def populate_then_wipe() -> None:
            await log.record_crash(outcome="crash")
            await log.wipe()

        _run(populate_then_wipe())

        # Open a second connection so we are not reading through the lock.
        side = sqlite3.connect(str(db_path))
        try:
            row = side.execute(
                "SELECT seq FROM sqlite_sequence WHERE name = 'audit'"
            ).fetchone()
            assert row is None
        finally:
            side.close()
    finally:
        log.close()


def test_wipe_on_empty_log_is_a_no_op(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:
        _run(log.wipe())
        assert log.count() == 0
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Async safety / concurrent ordering
# ---------------------------------------------------------------------------


def test_concurrent_record_calls_yield_unique_strictly_ordered_ids(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    """Concurrent recorders never share an id and never lose a row.

    The lock inside :class:`AuditLog` serialises writes regardless of the
    order the event loop dispatches the gathered coroutines, so:

    * Each ``record_*`` call returns a distinct id.
    * Every returned id appears exactly once in :meth:`entries`.
    * The set of ids is contiguous starting from 1.

    The exact mapping from coroutine-launch-order to id depends on the
    event loop's scheduling and is intentionally NOT asserted; CP9 only
    requires per-pair ordering, which the per-task awaits below provide.
    """
    log = audit_log_factory()
    try:
        n = 50

        async def driver() -> list[AuditEntry]:
            tasks = [
                log.record_executed(
                    skill="WeatherSkill",
                    args_json={"i": i},
                    outcome="ok",
                )
                for i in range(n)
            ]
            return await asyncio.gather(*tasks)

        entries = _run(driver())

        ids = [e.id for e in entries]
        assert len(ids) == n
        assert len(set(ids)) == n  # uniqueness
        assert sorted(ids) == list(range(1, n + 1))  # contiguous from 1

        persisted = log.entries()
        assert len(persisted) == n
        # Persisted rows are themselves strictly id-ordered (this is what
        # CP9 leans on when it inspects the audit log post-hoc).
        assert [p.id for p in persisted] == sorted(p.id for p in persisted)

        # Each returned entry can be located in the persisted view by id,
        # with the original payload intact.
        by_id = {p.id: p for p in persisted}
        for e in entries:
            assert by_id[e.id].kind == "executed"
            assert by_id[e.id].args_json == e.args_json
    finally:
        log.close()


def test_per_pair_ordering_is_preserved_when_awaited_sequentially(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    """The pre-condition for CP9: a confirmation/execute pair awaited in
    order has strictly increasing ids regardless of other concurrent
    recorders running in the background."""

    log = audit_log_factory()
    try:

        async def driver() -> tuple[int, int]:
            # Background traffic representing other audit events.
            background = [
                log.record_network_egress(
                    destination=f"https://svc-{i}.example.invalid",
                    justification="bg",
                )
                for i in range(20)
            ]
            bg_task = asyncio.gather(*background)

            confirm = await log.record_confirmation_requested(
                skill="SendEmailSkill",
                args_json={"to": "alex"},
            )
            executed = await log.record_executed(
                skill="SendEmailSkill",
                args_json={"to": "alex"},
                outcome="ok",
            )
            await bg_task
            return confirm.id, executed.id

        confirm_id, executed_id = _run(driver())
        assert confirm_id < executed_id
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Field plumbing per recorder
# ---------------------------------------------------------------------------


def test_record_confirmation_requested_persists_skill_and_args(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_confirmation_requested(
                skill="RunScriptSkill",
                args_json={"script_id": "dailyBackup"},
            )

        entry = _run(driver())
        assert entry.skill == "RunScriptSkill"
        assert entry.args_json == '{"script_id":"dailyBackup"}'
        assert entry.outcome is None
        assert entry.destination is None
    finally:
        log.close()


def test_record_executed_persists_outcome(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_executed(
                skill="LaunchAppSkill",
                args_json={"application": "Spotify"},
                outcome="ok",
            )

        entry = _run(driver())
        assert entry.outcome == "ok"
        assert entry.kind == "executed"
    finally:
        log.close()


def test_record_denied_defaults_outcome_to_denied(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_denied(
                skill="SendMessageSkill",
                args_json={"channel": "slack"},
            )

        entry = _run(driver())
        assert entry.outcome == "denied"
    finally:
        log.close()


def test_record_policy_violation_carries_destination_and_justification(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_policy_violation(
                skill=None,
                justification="destination not on allowlist",
                destination="https://attacker.invalid",
                outcome="blocked",
            )

        entry = _run(driver())
        assert entry.skill is None
        assert entry.justification == "destination not on allowlist"
        assert entry.destination == "https://attacker.invalid"
        assert entry.outcome == "blocked"
    finally:
        log.close()


def test_record_network_egress_persists_destination(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_network_egress(
                destination="https://api.mistral.ai",
                justification="LLM streaming",
                skill="DialogManager",
            )

        entry = _run(driver())
        assert entry.kind == "network_egress"
        assert entry.destination == "https://api.mistral.ai"
        assert entry.justification == "LLM streaming"
        assert entry.skill == "DialogManager"
    finally:
        log.close()


def test_record_error_carries_outcome_and_optional_justification(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_error(
                skill="ReadFileSkill",
                outcome="internal_error:trace-7f3b",
                justification="ZeroDivisionError",
            )

        entry = _run(driver())
        assert entry.kind == "error"
        assert entry.outcome == "internal_error:trace-7f3b"
        assert entry.justification == "ZeroDivisionError"
    finally:
        log.close()


def test_record_crash_writes_minimal_row(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_crash(outcome="crash")

        entry = _run(driver())
        assert entry.kind == "crash"
        assert entry.outcome == "crash"
        assert entry.skill is None
        assert entry.args_json is None
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Run id, timestamps, and lifecycle
# ---------------------------------------------------------------------------


def test_run_id_override_is_persisted_per_call(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory(run_id="default-run")
    try:

        async def driver() -> tuple[AuditEntry, AuditEntry]:
            default = await log.record_executed(
                skill="WeatherSkill", args_json={}, outcome="ok"
            )
            override = await log.record_executed(
                skill="WeatherSkill",
                args_json={},
                outcome="ok",
                run_id="other-run",
            )
            return default, override

        a, b = _run(driver())
        assert a.run_id == "default-run"
        assert b.run_id == "other-run"
    finally:
        log.close()


def test_timestamp_is_taken_from_the_injected_time_source(
    audit_log_factory: Callable[..., AuditLog],
    time_source: FakeTimeSource,
) -> None:
    log = audit_log_factory()
    try:

        async def driver() -> AuditEntry:
            return await log.record_executed(
                skill="WeatherSkill", args_json={}, outcome="ok"
            )

        entry = _run(driver())
        assert entry.ts == time_source.now()
        # And after advancing the fake clock the next entry uses the new time.
        time_source.advance(60)

        next_entry = _run(driver())
        assert next_entry.ts == time_source.now()
        assert next_entry.ts > entry.ts
    finally:
        log.close()


def test_close_makes_subsequent_writes_raise(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    log.close()

    async def driver() -> None:
        await log.record_crash(outcome="crash")

    with pytest.raises(RuntimeError, match="closed"):
        _run(driver())


def test_close_is_idempotent(audit_log_factory: Callable[..., AuditLog]) -> None:
    log = audit_log_factory()
    log.close()
    log.close()  # must not raise


def test_context_manager_closes_on_exit(
    audit_log_factory: Callable[..., AuditLog],
) -> None:
    log = audit_log_factory()
    with log as ctx:
        assert ctx is log
    # ``record_*`` after the context exits must raise.
    with pytest.raises(RuntimeError):
        _run(log.record_crash(outcome="crash"))
