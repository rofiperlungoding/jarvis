"""Unit tests for ``jarvis.skills.builtin.calendar``.

The Skill itself is intentionally thin — it adapts an injected
:class:`~jarvis.automation.providers.calendar.CalendarClient`-shaped
dependency to the :class:`SkillResult` taxonomy and routes the three
operations (``list_today``, ``list_range``, ``create_event``) to the
right client method. The tests exercise the adapter behaviour with a
fake calendar client, so the HTTP transport is never touched here.
Provider-level coverage (bearer-token plumbing, allowlist enforcement,
RFC 3339 formatting) lives in
``tests/unit/automation/providers/test_calendar.py``.

Validates: Requirements 7.5, 7.6, 7.7, 16.1, 16.2
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any

import httpx
from jsonschema import Draft7Validator
import pytest

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.config.schema import AuthorizationConfig, DestructiveOperation
from jarvis.llm.base import ToolCall
from jarvis.llm.mistral_schema import MistralSchemaValidator
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DESTRUCTIVE,
    SAFE,
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import calendar as calendar_module
from jarvis.skills.builtin.calendar import (
    CALENDAR_PROVIDER_KEY,
    SCHEMA,
    SKILL,
    CalendarSkill,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCalendarClient:
    """Records every call and replays a configured outcome.

    The real :class:`CalendarClient` exposes three async methods
    (``list_today``, ``list_range``, ``create_event``); the fake
    mirrors that surface so the Skill cannot tell it apart at runtime.
    """

    def __init__(
        self,
        *,
        list_today_result: list[dict[str, Any]] | None = None,
        list_range_result: list[dict[str, Any]] | None = None,
        create_event_result: dict[str, Any] | None = None,
        raise_on: str | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.list_today_calls: list[None] = []
        self.list_range_calls: list[tuple[datetime, datetime]] = []
        self.create_event_calls: list[tuple[str, datetime, datetime]] = []
        self._list_today_result = list_today_result
        self._list_range_result = list_range_result
        self._create_event_result = create_event_result
        self._raise_on = raise_on
        self._raise = raise_exc

    async def list_today(self) -> list[dict[str, Any]]:
        self.list_today_calls.append(None)
        if self._raise_on == "list_today" and self._raise is not None:
            raise self._raise
        return list(self._list_today_result or [])

    async def list_range(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        self.list_range_calls.append((start, end))
        if self._raise_on == "list_range" and self._raise is not None:
            raise self._raise
        return list(self._list_range_result or [])

    async def create_event(
        self, title: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        self.create_event_calls.append((title, start, end))
        if self._raise_on == "create_event" and self._raise is not None:
            raise self._raise
        return dict(
            self._create_event_result
            or {
                "id": "evt-1",
                "title": title,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "status": "confirmed",
            }
        )


def _make_ctx(client: Any | None = ...) -> SkillContext:
    """Build a :class:`SkillContext` with the calendar provider wired.

    Passing ``None`` for ``client`` simulates a misconfigured provider
    mapping (no ``"calendar"`` entry); passing the sentinel default
    installs a fresh successful-path fake.
    """

    if client is ...:
        client = _FakeCalendarClient()
    providers: dict[str, Any] = {}
    if client is not None:
        providers[CALENDAR_PROVIDER_KEY] = client
    return SkillContext(providers=providers, run_id="calendar-test")


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # Plugin discovery (registry._load_plugin_file) imports the module
    # and reads ``getattr(module, "SKILL", None)``; the constant must
    # exist and resolve to the singleton instance.
    assert isinstance(SKILL, CalendarSkill)
    assert calendar_module.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable: confirm the
    # singleton would be accepted by the registry's ``isinstance`` gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_not_destructive_at_manifest_level() -> None:
    """Two of three operations are read-only; manifest-level destructive
    is False. Per-operation destructive classification lives in the
    authorization config (see test_authorization_policy_classifies_create_event_destructive).
    """
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "CalendarSkill"
    assert manifest.destructive is False
    assert manifest.source == "builtin"
    assert manifest.json_schema is SCHEMA


def test_schema_declares_three_operations() -> None:
    op_field = SCHEMA["properties"]["operation"]
    assert op_field["type"] == "string"
    assert set(op_field["enum"]) == {"list_today", "list_range", "create_event"}
    # ``operation`` is the only unconditionally required field; the
    # per-operation requirements live in the conditional ``allOf`` block.
    assert SCHEMA["required"] == ["operation"]
    assert SCHEMA["additionalProperties"] is False


def test_schema_requires_start_end_for_list_range() -> None:
    """The ``allOf`` conditional makes ``start``/``end`` required when
    ``operation == "list_range"``."""
    blocks = SCHEMA["allOf"]
    range_block = next(
        b for b in blocks if b["if"]["properties"]["operation"]["const"] == "list_range"
    )
    assert set(range_block["then"]["required"]) == {"start", "end"}


def test_schema_requires_title_start_end_for_create_event() -> None:
    """The ``allOf`` conditional makes ``title``/``start``/``end`` required
    when ``operation == "create_event"``."""
    blocks = SCHEMA["allOf"]
    create_block = next(
        b
        for b in blocks
        if b["if"]["properties"]["operation"]["const"] == "create_event"
    )
    assert set(create_block["then"]["required"]) == {"title", "start", "end"}


def test_schema_accepts_list_today_without_extra_fields() -> None:
    """``list_today`` does not require any additional fields."""
    validator = Draft7Validator(SCHEMA)
    assert validator.is_valid({"operation": "list_today"})


def test_schema_rejects_create_event_without_required_fields() -> None:
    validator = Draft7Validator(SCHEMA)
    # create_event without title/start/end must fail
    assert not validator.is_valid({"operation": "create_event"})
    # missing title only
    assert not validator.is_valid(
        {
            "operation": "create_event",
            "start": "2024-01-01T09:00:00+00:00",
            "end": "2024-01-01T10:00:00+00:00",
        }
    )


def test_schema_rejects_list_range_without_bounds() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "list_range"})
    assert not validator.is_valid(
        {"operation": "list_range", "start": "2024-01-01T09:00:00+00:00"}
    )


def test_schema_rejects_unknown_operation() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "delete_event"})


def test_schema_accepts_via_mistral_subset_validator() -> None:
    """The Mistral function-calling subset must accept this schema (CP15).

    ``allOf``/``if``/``then``/``const``/``enum`` are all draft-07
    keywords the subset validator allows, and there is no ``$ref`` or
    ``oneOf`` mixing scalars with objects. If this fails the registry
    would refuse to register the Skill.
    """
    MistralSchemaValidator().validate(SCHEMA)


def test_skill_registers_cleanly() -> None:
    """The registry rejects a Skill whose JSON Schema is malformed; a
    successful registration confirms the Mistral subset + draft-07
    meta-schema accept the manifest."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "CalendarSkill" in registry


# ---------------------------------------------------------------------------
# Authorization wiring (Requirements 16.1, 16.2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_log(tmp_path) -> Any:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=FakeTimeSource(now=datetime(2024, 1, 1, tzinfo=UTC)),
        run_id="calendar-skill-test",
    )
    yield log
    log.close()


def test_authorization_policy_classifies_create_event_destructive(
    audit_log: Any,
) -> None:
    """Requirement 16.1: ``CalendarSkill.create_event`` is registered as a
    destructive operation in the authorization config; the policy must
    classify a Tool_Call with ``operation="create_event"`` as
    Destructive."""
    policy = AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
        hard_coded_destructive_skills=(),
        destructive_operations=(
            DestructiveOperation(
                skill="CalendarSkill",
                op_field="operation",
                op_values=["create_event"],
            ),
        ),
    )

    tool_call = ToolCall(
        id="tc-1",
        skill_name="CalendarSkill",
        arguments={
            "operation": "create_event",
            "title": "Demo",
            "start": "2024-01-01T09:00:00+00:00",
            "end": "2024-01-01T10:00:00+00:00",
        },
        raw_arguments=(
            '{"operation": "create_event", "title": "Demo", '
            '"start": "2024-01-01T09:00:00+00:00", '
            '"end": "2024-01-01T10:00:00+00:00"}'
        ),
    )

    assert policy.classify(tool_call, SKILL.manifest) == DESTRUCTIVE


def test_authorization_policy_classifies_list_today_safe(audit_log: Any) -> None:
    """Read-only operations stay Safe even with the destructive_operations
    config in place; only ``create_event`` is gated."""
    policy = AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
        hard_coded_destructive_skills=(),
        destructive_operations=(
            DestructiveOperation(
                skill="CalendarSkill",
                op_field="operation",
                op_values=["create_event"],
            ),
        ),
    )

    list_today_call = ToolCall(
        id="tc-2",
        skill_name="CalendarSkill",
        arguments={"operation": "list_today"},
        raw_arguments='{"operation": "list_today"}',
    )
    list_range_call = ToolCall(
        id="tc-3",
        skill_name="CalendarSkill",
        arguments={
            "operation": "list_range",
            "start": "2024-01-01T00:00:00+00:00",
            "end": "2024-01-02T00:00:00+00:00",
        },
        raw_arguments=(
            '{"operation": "list_range", '
            '"start": "2024-01-01T00:00:00+00:00", '
            '"end": "2024-01-02T00:00:00+00:00"}'
        ),
    )

    assert policy.classify(list_today_call, SKILL.manifest) == SAFE
    assert policy.classify(list_range_call, SKILL.manifest) == SAFE


def test_default_authorization_config_includes_create_event_destructive() -> None:
    """The default :class:`AuthorizationConfig` ships with a
    ``CalendarSkill.create_event`` destructive entry. This is the link
    between the Skill manifest and the policy gate."""
    cfg = AuthorizationConfig()
    matching = [
        op
        for op in cfg.destructive_operations
        if op.skill == "CalendarSkill"
        and op.op_field == "operation"
        and "create_event" in op.op_values
    ]
    assert matching, (
        "default AuthorizationConfig must include "
        "CalendarSkill.create_event as a destructive operation"
    )


# ---------------------------------------------------------------------------
# list_today
# ---------------------------------------------------------------------------


def test_list_today_dispatches_to_client_and_returns_events() -> None:
    expected = [
        {
            "id": "abc",
            "title": "Standup",
            "start": "2024-01-01T09:00:00Z",
            "end": "2024-01-01T09:30:00Z",
            "html_link": "https://cal/x",
            "status": "confirmed",
        }
    ]
    client = _FakeCalendarClient(list_today_result=expected)
    ctx = _make_ctx(client=client)

    result = _run(SKILL.execute({"operation": "list_today"}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["operation"] == "list_today"
    assert result.value["events"] == expected
    # The skill must call the client exactly once.
    assert len(client.list_today_calls) == 1
    assert client.list_range_calls == []
    assert client.create_event_calls == []


def test_list_today_empty_returns_success() -> None:
    """No events on the day is a successful (empty) result, not an error."""
    client = _FakeCalendarClient(list_today_result=[])
    ctx = _make_ctx(client=client)

    result = _run(SKILL.execute({"operation": "list_today"}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["events"] == []


# ---------------------------------------------------------------------------
# list_range
# ---------------------------------------------------------------------------


def test_list_range_parses_iso_and_dispatches() -> None:
    expected = [
        {
            "id": "evt-2",
            "title": "Quarterly review",
            "start": "2024-01-15T13:00:00Z",
            "end": "2024-01-15T14:00:00Z",
        }
    ]
    client = _FakeCalendarClient(list_range_result=expected)
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "list_range",
                "start": "2024-01-15T00:00:00+00:00",
                "end": "2024-01-15T23:59:59+00:00",
            },
            ctx,
        )
    )

    assert result.ok is True
    assert result.value is not None
    assert result.value["operation"] == "list_range"
    assert result.value["events"] == expected
    assert len(client.list_range_calls) == 1
    start, end = client.list_range_calls[0]
    assert start == datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
    assert end == datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)


def test_list_range_naive_start_returns_schema_violation() -> None:
    """Naive timestamps must be rejected before the client is called."""
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "list_range",
                "start": "2024-01-15T00:00:00",  # naive
                "end": "2024-01-15T23:59:59+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.list_range_calls == []


def test_list_range_naive_end_returns_schema_violation() -> None:
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "list_range",
                "start": "2024-01-15T00:00:00+00:00",
                "end": "2024-01-15T23:59:59",  # naive
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.list_range_calls == []


def test_list_range_end_before_start_returns_schema_violation() -> None:
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "list_range",
                "start": "2024-01-15T12:00:00+00:00",
                "end": "2024-01-15T11:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.list_range_calls == []


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


def test_create_event_dispatches_to_client() -> None:
    """The Authorization_Policy has already obtained user confirmation
    by the time the registry dispatches us; the Skill simply forwards
    to ``client.create_event``."""
    expected = {
        "id": "new-id",
        "title": "Demo Meeting",
        "start": "2024-02-01T09:00:00+00:00",
        "end": "2024-02-01T10:00:00+00:00",
        "status": "confirmed",
    }
    client = _FakeCalendarClient(create_event_result=expected)
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "Demo Meeting",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is True
    assert result.value is not None
    assert result.value["operation"] == "create_event"
    assert result.value["event"] == expected

    assert len(client.create_event_calls) == 1
    title, start, end = client.create_event_calls[0]
    assert title == "Demo Meeting"
    assert start == datetime(2024, 2, 1, 9, 0, 0, tzinfo=UTC)
    assert end == datetime(2024, 2, 1, 10, 0, 0, tzinfo=UTC)


def test_create_event_strips_title_whitespace() -> None:
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "  Padded title  ",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert client.create_event_calls[0][0] == "Padded title"


def test_create_event_blank_title_returns_schema_violation() -> None:
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "   ",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.create_event_calls == []


def test_create_event_end_before_or_equal_start_returns_schema_violation() -> None:
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    # equal start/end is rejected — events must have positive duration
    result_equal = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "Zero",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T09:00:00+00:00",
            },
            ctx,
        )
    )
    assert result_equal.ok is False
    assert result_equal.error_code == "schema_violation"

    result_inverted = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "Inverted",
                "start": "2024-02-01T10:00:00+00:00",
                "end": "2024-02-01T09:00:00+00:00",
            },
            ctx,
        )
    )
    assert result_inverted.ok is False
    assert result_inverted.error_code == "schema_violation"
    assert client.create_event_calls == []


# ---------------------------------------------------------------------------
# Provider error translation
# ---------------------------------------------------------------------------


def test_missing_credentials_returns_missing_credentials() -> None:
    client = _FakeCalendarClient(
        raise_on="list_today",
        raise_exc=ProviderError(
            "missing_credentials",
            "calendar/oauth_token is not set",
            provider="google",
        ),
    )
    ctx = _make_ctx(client=client)

    result = _run(SKILL.execute({"operation": "list_today"}, ctx))

    assert result.ok is False
    assert result.error_code == "missing_credentials"


def test_provider_unavailable_returns_provider_unavailable() -> None:
    client = _FakeCalendarClient(
        raise_on="list_today",
        raise_exc=ProviderError(
            "provider_unavailable",
            "Google Calendar returned HTTP 503",
            provider="google",
        ),
    )
    ctx = _make_ctx(client=client)

    result = _run(SKILL.execute({"operation": "list_today"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_network_policy_violation_returns_access_denied() -> None:
    client = _FakeCalendarClient(
        raise_on="create_event",
        raise_exc=NetworkPolicyViolation(
            destination="https://calendar.example.com/api",
            host="calendar.example.com",
        ),
    )
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "Blocked",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_timeout_returns_timeout() -> None:
    client = _FakeCalendarClient(
        raise_on="list_range",
        raise_exc=httpx.TimeoutException("request took too long"),
    )
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "list_range",
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-01-02T00:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "timeout"


def test_value_error_from_client_returns_schema_violation() -> None:
    client = _FakeCalendarClient(
        raise_on="create_event",
        raise_exc=ValueError("end must be > start"),
    )
    ctx = _make_ctx(client=client)

    result = _run(
        SKILL.execute(
            {
                "operation": "create_event",
                "title": "x",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Misconfiguration paths
# ---------------------------------------------------------------------------


def test_missing_provider_returns_provider_unavailable() -> None:
    """When the bootstrap did not wire a calendar client into
    ``ctx.providers``, the Skill must surface a structured error rather
    than crashing."""
    ctx = _make_ctx(client=None)

    result = _run(SKILL.execute({"operation": "list_today"}, ctx))

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_unknown_operation_returns_schema_violation_when_bypassing_validator() -> None:
    """Defence-in-depth: even if a caller bypasses the registry's
    validator, the Skill must reject an unknown operation with
    ``schema_violation`` rather than mis-routing."""
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    result = _run(SKILL.execute({"operation": "delete_event"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.list_today_calls == []
    assert client.list_range_calls == []
    assert client.create_event_calls == []


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registry_dispatch_invalid_args_returns_schema_violation_without_calling_client() -> (
    None
):
    """Property 2 / CP2: invalid args short-circuit at the registry; the
    Skill executor (and therefore the calendar client) is never invoked."""
    registry = SkillRegistry()
    registry.register(SKILL)
    client = _FakeCalendarClient()
    ctx = _make_ctx(client=client)

    # ``create_event`` without ``title`` violates the conditional schema
    result = _run(
        registry.dispatch(
            "CalendarSkill",
            {
                "operation": "create_event",
                "start": "2024-02-01T09:00:00+00:00",
                "end": "2024-02-01T10:00:00+00:00",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert client.create_event_calls == []


def test_registry_dispatch_list_today_round_trips() -> None:
    """End-to-end smoke: registry validates args, dispatches to the
    Skill, and returns the structured payload."""
    registry = SkillRegistry()
    registry.register(SKILL)
    expected = [{"id": "z", "title": "Lunch", "start": None, "end": None}]
    client = _FakeCalendarClient(list_today_result=expected)
    ctx = _make_ctx(client=client)

    result = _run(registry.dispatch("CalendarSkill", {"operation": "list_today"}, ctx))

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert result.value is not None
    assert result.value["events"] == expected
