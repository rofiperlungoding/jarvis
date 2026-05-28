"""Unit tests for :mod:`jarvis.skills.builtin.timer`.

The tests pin three behaviours that together pin down Requirements 6.3
and 6.4:

* the manifest exposes the contract the Skill_Registry requires —
  ``TimerSkill`` name, ``destructive=False``, JSON Schema that demands
  a strictly positive integer ``duration_seconds`` and an optional
  ``label`` (Requirement 6.3);
* the executor delegates to :meth:`ReminderService.add_timer` exactly
  once with the supplied arguments and translates the returned
  :class:`Reminder` into a JSON-serialisable success payload (Requirement
  6.4); and
* the executor returns a structured :class:`SkillResult` — never raises —
  when the :class:`ReminderService` is absent from ``ctx.extras`` or when
  the service rejects the arguments after the schema gate.

A complementary registry round-trip test exercises the Mistral subset
gate via :class:`SkillRegistry.register` so the manifest stays Mistral-
compatible (Requirement 19.4 / CP15).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from jarvis.reminders.service import Reminder
from jarvis.skills.base import Skill, SkillContext, SkillManifest
from jarvis.skills.builtin import timer as timer_module
from jarvis.skills.builtin.timer import SKILL, TimerSkill
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Drive a single coroutine to completion under the default loop."""
    return asyncio.run(coro)


def _make_reminder(
    *,
    reminder_id: str = "rid-1",
    label: str = "",
    duration_seconds: int = 60,
    seq: int = 1,
) -> Reminder:
    # Anchor ``trigger_at`` at a fixed instant + ``duration_seconds`` so
    # the test value is realistic-looking without forcing the helper to
    # care about minute / hour overflow for large durations.
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    trigger_at = base + timedelta(seconds=duration_seconds)
    created_at = base
    return Reminder(
        reminder_id=reminder_id,
        kind="timer",
        label=label,
        trigger_at=trigger_at,
        duration_seconds=duration_seconds,
        seq=seq,
        created_at=created_at,
        cancelled_at=None,
    )


class _FakeReminderService:
    """Records ``add_timer`` calls and returns a canned :class:`Reminder`.

    Mirrors the public surface that :class:`TimerSkill` relies on without
    pulling in APScheduler / SQLite. Defaults to returning a successful
    :class:`Reminder`; tests that need failure semantics override
    :attr:`raise_exc`.
    """

    def __init__(self, reminder: Reminder | None = None) -> None:
        self._reminder = reminder or _make_reminder()
        self.calls: list[tuple[int, str | None]] = []
        self.raise_exc: BaseException | None = None

    async def add_timer(
        self, duration_seconds: int, label: str | None = None
    ) -> Reminder:
        self.calls.append((duration_seconds, label))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self._reminder


def _ctx_with_service(service: Any) -> SkillContext:
    return SkillContext(extras={"reminder_service": service})


# ---------------------------------------------------------------------------
# Module export
# ---------------------------------------------------------------------------


def test_module_exports_skill_singleton() -> None:
    # Plugin discovery binds whatever lives at module-level ``SKILL`` so
    # this is the contract every built-in Skill module owes the registry.
    assert isinstance(SKILL, TimerSkill)
    assert isinstance(SKILL, Skill)
    assert SKILL is timer_module.SKILL


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_metadata() -> None:
    m = TimerSkill.manifest
    assert isinstance(m, SkillManifest)
    assert m.name == "TimerSkill"
    assert m.source == "builtin"
    # Setting a timer is local-only state; not destructive.
    assert m.destructive is False
    # The Skill is implemented in pure Python on top of ReminderService,
    # which is platform-neutral, so all three platform tags are advertised.
    assert "windows" in m.platforms


def test_manifest_schema_requires_duration_seconds() -> None:
    schema = TimerSkill.manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["duration_seconds"]
    duration = schema["properties"]["duration_seconds"]
    assert duration["type"] == "integer"
    # Requirement 6.3: strictly greater than zero.
    assert duration["minimum"] == 1
    assert duration.get("exclusiveMinimum") == 0


def test_manifest_schema_label_optional() -> None:
    schema = TimerSkill.manifest.json_schema
    label_props = schema["properties"]["label"]
    assert label_props["type"] == "string"
    assert "label" not in schema["required"]
    # ``additionalProperties: false`` keeps the LLM from sneaking extra
    # fields past the gate (defence in depth on top of the Mistral
    # subset checker).
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Execute happy path
# ---------------------------------------------------------------------------


def test_execute_with_label_persists_via_add_timer() -> None:
    reminder = _make_reminder(
        reminder_id="rid-42",
        label="pasta",
        duration_seconds=600,
        seq=7,
    )
    fake = _FakeReminderService(reminder)

    result = _run(
        TimerSkill().execute(
            {"duration_seconds": 600, "label": "pasta"},
            _ctx_with_service(fake),
        )
    )

    assert fake.calls == [(600, "pasta")]
    assert result.ok is True
    assert result.error_code is None
    assert result.value == {
        "reminder_id": "rid-42",
        "kind": "timer",
        "label": "pasta",
        "duration_seconds": 600,
        "trigger_at": reminder.trigger_at.isoformat(),
        "seq": 7,
    }


def test_execute_without_label_passes_none_to_service() -> None:
    fake = _FakeReminderService(_make_reminder(label=""))

    result = _run(
        TimerSkill().execute({"duration_seconds": 30}, _ctx_with_service(fake))
    )

    # The Skill must NOT inject a default empty string itself —
    # ReminderService.add_timer is the single source of truth for the
    # missing-label normalisation. Keeping ``label=None`` flowing
    # through preserves that contract (Requirement 6.4).
    assert fake.calls == [(30, None)]
    assert result.ok is True
    assert result.value is not None
    assert result.value["label"] == ""


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_execute_returns_internal_error_when_service_missing() -> None:
    # No ``reminder_service`` in extras simulates a wiring bug at
    # bootstrap. The executor must surface a structured failure rather
    # than raising — the Dialog_Manager relies on Property 7 / CP10.
    result = _run(TimerSkill().execute({"duration_seconds": 5}, SkillContext()))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert result.error_message is not None
    assert "reminder service" in result.error_message.lower()


def test_execute_maps_value_error_from_service_to_schema_violation() -> None:
    fake = _FakeReminderService()
    fake.raise_exc = ValueError("duration_seconds must be > 0; got 0")

    result = _run(
        TimerSkill().execute({"duration_seconds": 1}, _ctx_with_service(fake))
    )

    # Even though the registry's JSON-Schema gate would normally catch
    # this, the executor still defends in depth: a ValueError out of
    # add_timer is treated as a schema-level disagreement so the LLM
    # gets a chance to retry rather than seeing an opaque crash.
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert "duration_seconds" in (result.error_message or "")


def test_execute_maps_type_error_from_service_to_schema_violation() -> None:
    fake = _FakeReminderService()
    fake.raise_exc = TypeError("duration_seconds must be int; got str")

    result = _run(
        TimerSkill().execute({"duration_seconds": 1}, _ctx_with_service(fake))
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_execute_propagates_unexpected_exception() -> None:
    # Anything outside ValueError/TypeError is left to bubble up so the
    # Skill_Registry's catch-all converts it into ``internal_error``
    # with a traceback id (Property 7 / CP10).
    fake = _FakeReminderService()
    fake.raise_exc = RuntimeError("scheduler offline")

    with pytest.raises(RuntimeError, match="scheduler offline"):
        _run(TimerSkill().execute({"duration_seconds": 1}, _ctx_with_service(fake)))


# ---------------------------------------------------------------------------
# Registry round-trip — Mistral subset compatibility
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_through_skill_registry() -> None:
    """End-to-end: register, dispatch, verify the success payload.

    Going through :class:`SkillRegistry` exercises the JSON-Schema gate,
    the Mistral subset checker, and the dispatch contract in a single
    test, which is the cheapest way to keep the manifest honest.
    """

    reg = SkillRegistry()
    reg.register(SKILL)

    fake = _FakeReminderService(
        _make_reminder(
            reminder_id="rid-9",
            label="pomodoro",
            duration_seconds=1500,
            seq=3,
        )
    )
    ctx = _ctx_with_service(fake)

    result = _run(
        reg.dispatch(
            "TimerSkill",
            {"duration_seconds": 1500, "label": "pomodoro"},
            ctx,
        )
    )

    assert result.ok is True
    assert fake.calls == [(1500, "pomodoro")]
    assert result.value is not None
    assert result.value["reminder_id"] == "rid-9"
    assert result.value["kind"] == "timer"


def test_registry_rejects_zero_duration_with_schema_violation() -> None:
    """``duration_seconds`` = 0 must be rejected before ``execute`` runs."""

    reg = SkillRegistry()
    reg.register(SKILL)

    fake = _FakeReminderService()
    ctx = _ctx_with_service(fake)

    result = _run(reg.dispatch("TimerSkill", {"duration_seconds": 0}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # The skill must NOT have been invoked — Property 2 / CP2.
    assert fake.calls == []


def test_registry_rejects_negative_duration_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeReminderService()

    result = _run(
        reg.dispatch("TimerSkill", {"duration_seconds": -5}, _ctx_with_service(fake))
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_registry_rejects_missing_duration() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeReminderService()

    result = _run(reg.dispatch("TimerSkill", {"label": "tea"}, _ctx_with_service(fake)))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_registry_rejects_extra_properties() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeReminderService()

    result = _run(
        reg.dispatch(
            "TimerSkill",
            {"duration_seconds": 30, "category": "kitchen"},
            _ctx_with_service(fake),
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []


def test_registry_publishes_mistral_tool_definition() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    [tool] = reg.mistral_tool_definitions()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "TimerSkill"
    params = tool["function"]["parameters"]
    assert params["required"] == ["duration_seconds"]
