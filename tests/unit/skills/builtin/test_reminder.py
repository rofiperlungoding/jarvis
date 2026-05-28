"""Unit tests for :mod:`jarvis.skills.builtin.reminder`.

These tests pin three layers of behaviour for the two skills that
together cover Requirements 6.1, 6.2, and 6.7:

* the manifests advertise the contract the Skill_Registry requires —
  ``ReminderSkill`` / ``ListReminderSkill`` names, ``destructive=False``,
  Mistral-subset-compatible JSON Schemas;
* :class:`ReminderSkill.execute` delegates to
  :meth:`ReminderService.add` exactly once with the parsed
  :class:`datetime` and faithfully serialises the returned
  :class:`Reminder` (Requirements 6.1, 6.2);
* :class:`ListReminderSkill.execute` faithfully forwards the
  :meth:`ReminderService.list_pending` ordering, which is what
  underwrites Property 10 / CP13 at the Skill boundary
  (Requirement 6.7).

A registry round-trip exercises the JSON-Schema gate, the Mistral
subset checker, and the ``mistral_tool_definitions()`` projection in
one go — that is the cheapest way to keep the manifests honest.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from jarvis.reminders.service import Reminder
from jarvis.skills.base import Skill, SkillContext, SkillManifest
from jarvis.skills.builtin import reminder as reminder_module
from jarvis.skills.builtin.reminder import (
    REMINDER_SERVICE_EXTRAS_KEY,
    SKILLS,
    ListReminderSkill,
    ReminderSkill,
)
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
    kind: str = "reminder",
    label: str = "stand up",
    trigger_at: datetime | None = None,
    duration_seconds: int | None = None,
    seq: int = 1,
    created_at: datetime | None = None,
) -> Reminder:
    trigger_at = trigger_at or datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    created_at = created_at or datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
    return Reminder(
        reminder_id=reminder_id,
        kind=kind,  # type: ignore[arg-type]
        label=label,
        trigger_at=trigger_at,
        duration_seconds=duration_seconds,
        seq=seq,
        created_at=created_at,
        cancelled_at=None,
    )


class _FakeReminderService:
    """Minimal :class:`ReminderService` lookalike for test isolation.

    Records ``add`` and ``list_pending`` calls so the tests can pin the
    Skill's exact pass-through behaviour without standing up APScheduler
    or SQLite. The :attr:`add_exc` / :attr:`list_exc` knobs let a single
    test exercise the executor's defensive error mapping.
    """

    def __init__(
        self,
        *,
        added: Reminder | None = None,
        pending: list[Reminder] | None = None,
    ) -> None:
        self._added = added or _make_reminder()
        self._pending = list(pending or [])
        self.add_calls: list[tuple[str, datetime]] = []
        self.list_calls: int = 0
        self.add_exc: BaseException | None = None
        self.list_exc: BaseException | None = None

    async def add(self, label: str, trigger_at: datetime) -> Reminder:
        self.add_calls.append((label, trigger_at))
        if self.add_exc is not None:
            raise self.add_exc
        return self._added

    async def list_pending(self) -> list[Reminder]:
        self.list_calls += 1
        if self.list_exc is not None:
            raise self.list_exc
        return list(self._pending)


def _ctx_with_service(service: Any) -> SkillContext:
    return SkillContext(extras={REMINDER_SERVICE_EXTRAS_KEY: service})


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


def test_module_exports_skills_list() -> None:
    """``SKILLS`` is the convention for modules shipping multiple skills."""

    assert isinstance(SKILLS, list)
    # Two skills in this module: scheduling + listing.
    assert len(SKILLS) == 2
    names = {type(s).__name__ for s in SKILLS}
    assert names == {"ReminderSkill", "ListReminderSkill"}
    # Every entry must satisfy the Skill protocol — that is the
    # contract the registry's ``register`` call relies on.
    for skill in SKILLS:
        assert isinstance(skill, Skill)


def test_extras_key_constant_is_stable() -> None:
    # Pin the extras key so the eventual app.py wiring (Task 19.1)
    # cannot drift away from this module without breaking the tests.
    assert REMINDER_SERVICE_EXTRAS_KEY == "reminder_service"
    assert reminder_module.REMINDER_SERVICE_EXTRAS_KEY == "reminder_service"


# ---------------------------------------------------------------------------
# ReminderSkill manifest
# ---------------------------------------------------------------------------


def test_reminder_manifest_metadata() -> None:
    m = ReminderSkill.manifest
    assert isinstance(m, SkillManifest)
    assert m.name == "ReminderSkill"
    assert m.source == "builtin"
    # Scheduling a reminder is local-only state, not destructive.
    assert m.destructive is False
    assert "windows" in m.platforms


def test_reminder_manifest_schema_requires_label_and_trigger_at() -> None:
    schema = ReminderSkill.manifest.json_schema
    assert schema["type"] == "object"
    assert sorted(schema["required"]) == ["label", "trigger_at"]
    label = schema["properties"]["label"]
    assert label["type"] == "string"
    assert label.get("minLength") == 1
    trigger_at = schema["properties"]["trigger_at"]
    assert trigger_at["type"] == "string"
    # ``date-time`` is the only ``format`` value the Mistral subset
    # accepts; it MUST be present so the LLM produces parseable
    # ISO-8601 instants.
    assert trigger_at["format"] == "date-time"
    # ``additionalProperties: false`` is defence in depth on top of the
    # Mistral subset checker — extra fields are rejected up front.
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# ReminderSkill.execute — happy path
# ---------------------------------------------------------------------------


def test_execute_persists_reminder_and_returns_serialised_payload() -> None:
    trigger = datetime(2024, 6, 1, 9, 30, tzinfo=UTC)
    reminder = _make_reminder(
        reminder_id="rid-42",
        label="dentist",
        trigger_at=trigger,
        seq=7,
    )
    fake = _FakeReminderService(added=reminder)

    result = _run(
        ReminderSkill().execute(
            {"label": "dentist", "trigger_at": "2024-06-01T09:30:00+00:00"},
            _ctx_with_service(fake),
        )
    )

    # Service was called exactly once with the parsed datetime.
    assert len(fake.add_calls) == 1
    label, parsed_trigger = fake.add_calls[0]
    assert label == "dentist"
    assert parsed_trigger == trigger
    # The serialised payload mirrors the freshly-persisted Reminder.
    assert result.ok is True
    assert result.error_code is None
    assert result.value == {
        "reminder_id": "rid-42",
        "kind": "reminder",
        "label": "dentist",
        "trigger_at": "2024-06-01T09:30:00+00:00",
        "duration_seconds": None,
        "seq": 7,
        "created_at": reminder.created_at.isoformat(),
    }


def test_execute_accepts_zulu_suffix_for_trigger_at() -> None:
    """Python 3.11 ``fromisoformat`` accepts the ``Z`` UTC shorthand.

    The LLM frequently emits ``"2024-06-01T09:30:00Z"``; the executor
    must accept it without forcing the model to retry with an explicit
    ``+00:00`` offset.
    """

    fake = _FakeReminderService()

    result = _run(
        ReminderSkill().execute(
            {"label": "ping", "trigger_at": "2024-06-01T09:30:00Z"},
            _ctx_with_service(fake),
        )
    )

    assert result.ok is True
    assert len(fake.add_calls) == 1
    _, parsed = fake.add_calls[0]
    assert parsed == datetime(2024, 6, 1, 9, 30, tzinfo=UTC)


def test_execute_preserves_non_utc_timezone() -> None:
    """A non-UTC offset is forwarded verbatim — the service handles UTC normalisation."""

    fake = _FakeReminderService()
    plus_two = timezone(timedelta(hours=2))

    result = _run(
        ReminderSkill().execute(
            {"label": "lunch", "trigger_at": "2024-06-01T13:00:00+02:00"},
            _ctx_with_service(fake),
        )
    )

    assert result.ok is True
    assert len(fake.add_calls) == 1
    _, parsed = fake.add_calls[0]
    assert parsed == datetime(2024, 6, 1, 13, 0, tzinfo=plus_two)


# ---------------------------------------------------------------------------
# ReminderSkill.execute — failure modes
# ---------------------------------------------------------------------------


def test_execute_returns_internal_error_when_service_missing() -> None:
    # No reminder service in extras simulates a bootstrap wiring bug.
    result = _run(
        ReminderSkill().execute(
            {"label": "x", "trigger_at": "2024-06-01T09:30:00+00:00"},
            SkillContext(),
        )
    )
    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "reminder_service" in (result.error_message or "")


def test_execute_returns_internal_error_when_extras_holds_bogus_value() -> None:
    """A non-service value under the key is treated as missing."""

    result = _run(
        ReminderSkill().execute(
            {"label": "x", "trigger_at": "2024-06-01T09:30:00+00:00"},
            SkillContext(extras={REMINDER_SERVICE_EXTRAS_KEY: object()}),
        )
    )
    assert result.ok is False
    assert result.error_code == "internal_error"


def test_execute_rejects_naive_trigger_at_with_schema_violation() -> None:
    fake = _FakeReminderService()
    result = _run(
        ReminderSkill().execute(
            {"label": "x", "trigger_at": "2024-06-01T09:30:00"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert "timezone" in (result.error_message or "").lower()
    # The service must NOT be touched when the parse fails.
    assert fake.add_calls == []


def test_execute_rejects_unparseable_trigger_at() -> None:
    fake = _FakeReminderService()
    result = _run(
        ReminderSkill().execute(
            {"label": "x", "trigger_at": "tomorrow"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.add_calls == []


def test_execute_rejects_empty_label_via_schema_violation() -> None:
    """Defence in depth: the registry already rejects empty labels."""

    fake = _FakeReminderService()
    result = _run(
        ReminderSkill().execute(
            {"label": "", "trigger_at": "2024-06-01T09:30:00+00:00"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.add_calls == []


def test_execute_maps_value_error_from_service_to_schema_violation() -> None:
    fake = _FakeReminderService()
    fake.add_exc = ValueError("trigger_at must be timezone-aware")

    result = _run(
        ReminderSkill().execute(
            {"label": "x", "trigger_at": "2024-06-01T09:30:00+00:00"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# ListReminderSkill manifest
# ---------------------------------------------------------------------------


def test_list_manifest_metadata() -> None:
    m = ListReminderSkill.manifest
    assert m.name == "ListReminderSkill"
    assert m.destructive is False
    assert m.source == "builtin"


def test_list_manifest_schema_accepts_no_arguments() -> None:
    schema = ListReminderSkill.manifest.json_schema
    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# ListReminderSkill.execute
# ---------------------------------------------------------------------------


def test_list_execute_returns_pending_reminders_in_service_order() -> None:
    """Faithful pass-through of the ``(trigger_at, seq)`` order.

    The underlying SQL query already sorts by ``(trigger_at ASC,
    seq ASC)`` (Property 10 / CP13). The Skill must NOT re-sort or
    filter the list — that would either contradict the service or
    silently hide bugs.
    """

    base = datetime(2024, 6, 1, 9, 30, tzinfo=UTC)
    pending = [
        _make_reminder(
            reminder_id="rid-a",
            kind="reminder",
            label="a",
            trigger_at=base,
            seq=1,
        ),
        _make_reminder(
            reminder_id="rid-b",
            kind="timer",
            label="b",
            trigger_at=base,
            duration_seconds=120,
            seq=2,
        ),
        _make_reminder(
            reminder_id="rid-c",
            kind="reminder",
            label="c",
            trigger_at=base + timedelta(minutes=5),
            seq=3,
        ),
    ]
    fake = _FakeReminderService(pending=pending)

    result = _run(ListReminderSkill().execute({}, _ctx_with_service(fake)))

    assert fake.list_calls == 1
    assert result.ok is True
    assert result.value is not None
    assert [r["reminder_id"] for r in result.value["reminders"]] == [
        "rid-a",
        "rid-b",
        "rid-c",
    ]
    # The timer entry preserves its ``duration_seconds``; the plain
    # reminder serialises ``None`` so the LLM can distinguish kinds.
    timer_entry = result.value["reminders"][1]
    assert timer_entry["kind"] == "timer"
    assert timer_entry["duration_seconds"] == 120
    assert result.value["reminders"][0]["duration_seconds"] is None


def test_list_execute_returns_empty_payload_when_no_reminders() -> None:
    fake = _FakeReminderService(pending=[])
    result = _run(ListReminderSkill().execute({}, _ctx_with_service(fake)))
    assert result.ok is True
    assert result.value == {"reminders": []}


def test_list_execute_returns_internal_error_when_service_missing() -> None:
    result = _run(ListReminderSkill().execute({}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "internal_error"


# ---------------------------------------------------------------------------
# Registry round-trip — Mistral subset compatibility
# ---------------------------------------------------------------------------


def test_skills_register_and_dispatch_through_registry() -> None:
    """End-to-end: register both skills and dispatch each in turn."""

    reg = SkillRegistry()
    for skill in SKILLS:
        reg.register(skill)

    # Two tools are now exposed to the LLM in deterministic name order.
    tool_names = [t["function"]["name"] for t in reg.mistral_tool_definitions()]
    assert tool_names == ["ListReminderSkill", "ReminderSkill"]

    reminder = _make_reminder(reminder_id="rid-x", label="renew passport")
    fake = _FakeReminderService(added=reminder, pending=[reminder])
    ctx = _ctx_with_service(fake)

    add_result = _run(
        reg.dispatch(
            "ReminderSkill",
            {
                "label": "renew passport",
                "trigger_at": reminder.trigger_at.isoformat(),
            },
            ctx,
        )
    )
    assert add_result.ok is True
    assert fake.add_calls == [("renew passport", reminder.trigger_at)]

    list_result = _run(reg.dispatch("ListReminderSkill", {}, ctx))
    assert list_result.ok is True
    assert list_result.value is not None
    assert [r["reminder_id"] for r in list_result.value["reminders"]] == ["rid-x"]


def test_registry_rejects_missing_label_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(ReminderSkill())
    fake = _FakeReminderService()

    result = _run(
        reg.dispatch(
            "ReminderSkill",
            {"trigger_at": "2024-06-01T09:30:00+00:00"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.add_calls == []


def test_registry_rejects_missing_trigger_at_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(ReminderSkill())
    fake = _FakeReminderService()

    result = _run(
        reg.dispatch(
            "ReminderSkill",
            {"label": "x"},
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.add_calls == []


def test_registry_rejects_extra_properties_on_reminder() -> None:
    reg = SkillRegistry()
    reg.register(ReminderSkill())
    fake = _FakeReminderService()

    result = _run(
        reg.dispatch(
            "ReminderSkill",
            {
                "label": "x",
                "trigger_at": "2024-06-01T09:30:00+00:00",
                "category": "kitchen",
            },
            _ctx_with_service(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.add_calls == []


def test_registry_rejects_extra_properties_on_list_reminder() -> None:
    reg = SkillRegistry()
    reg.register(ListReminderSkill())

    result = _run(
        reg.dispatch(
            "ListReminderSkill",
            {"only_kind": "timer"},
            _ctx_with_service(_FakeReminderService()),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_registry_publishes_reminder_skill_function_definition() -> None:
    reg = SkillRegistry()
    reg.register(ReminderSkill())
    [tool] = reg.mistral_tool_definitions()
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "ReminderSkill"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert sorted(params["required"]) == ["label", "trigger_at"]
    assert params["properties"]["trigger_at"]["format"] == "date-time"
