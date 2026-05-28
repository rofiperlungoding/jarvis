"""Unit tests for :mod:`jarvis.skills.builtin.send_message`.

Validates Requirements 5.4 (channel/recipient/body schema), 5.5 (same
confirmation flow as ``SendEmailSkill`` — i.e. ``destructive=True``),
5.6 (``missing_credentials`` flow), 16.1 (destructive classification),
and 16.2 (the Skill never short-circuits the confirmation gate).

The tests deliberately exercise the dispatcher pattern via the in-tree
:class:`InMemoryChannelAdapter` so we never reach for a real network
transport. A handful of scenarios use bespoke adapter classes to
exercise the error-mapping table exhaustively (provider errors, network
policy violations, unknown channels, value errors).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import send_message as send_message_module
from jarvis.skills.builtin.send_message import (
    MESSAGING_PROVIDER_KEY,
    SCHEMA,
    SKILL,
    InMemoryChannelAdapter,
    MessageChannelAdapter,
    MessageChannelDispatcher,
    SendMessageSkill,
    UnknownChannelError,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Synchronously run a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


def _ctx_with(dispatcher: Any) -> SkillContext:
    """Build a :class:`SkillContext` with the messaging dispatcher wired in."""
    return SkillContext(
        providers={MESSAGING_PROVIDER_KEY: dispatcher},
        credential_store=object(),
    )


# A bespoke adapter for tests that need to inject specific exceptions.
class _FailingAdapter:
    """Adapter whose ``send`` always raises a configured exception."""

    def __init__(self, name: str, exc: BaseException) -> None:
        self.name = name
        self._exc = exc
        self.calls: list[tuple[str, str]] = []

    async def send(self, recipient: str, body: str) -> Mapping[str, Any]:
        self.calls.append((recipient, body))
        raise self._exc


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, SendMessageSkill)
    assert SKILL.manifest is SendMessageSkill.manifest


def test_module_skill_satisfies_runtime_protocol() -> None:
    assert isinstance(send_message_module.SKILL, Skill)


def test_manifest_marks_skill_destructive() -> None:
    """Requirement 16.1 — confirmation gate applies via destructive flag."""
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "SendMessageSkill"
    assert manifest.destructive is True
    assert manifest.source == "builtin"
    # Messaging adapters are OS-agnostic.
    assert "windows" in manifest.platforms


def test_schema_requires_channel_recipient_body() -> None:
    """Requirement 5.4 fixes the three required string arguments."""
    schema = SKILL.manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["channel", "recipient", "body"]
    assert schema["additionalProperties"] is False
    for field in ("channel", "recipient", "body"):
        prop = schema["properties"][field]
        assert prop["type"] == "string"
        assert prop["minLength"] == 1


def test_schema_constant_matches_manifest_schema() -> None:
    """The exported ``SCHEMA`` is the manifest's JSON Schema."""
    assert SKILL.manifest.json_schema is SCHEMA


# ---------------------------------------------------------------------------
# MessageChannelAdapter Protocol
# ---------------------------------------------------------------------------


def test_in_memory_adapter_satisfies_protocol() -> None:
    adapter = InMemoryChannelAdapter()
    assert isinstance(adapter, MessageChannelAdapter)


def test_in_memory_adapter_records_messages() -> None:
    adapter = InMemoryChannelAdapter("local")
    result = _run(adapter.send("+15555550100", "hello"))

    assert result == {
        "channel": "local",
        "recipient": "+15555550100",
        "delivered": True,
    }
    assert adapter.sent == [
        {"channel": "local", "recipient": "+15555550100", "body": "hello"}
    ]


def test_in_memory_adapter_rejects_empty_recipient() -> None:
    adapter = InMemoryChannelAdapter()
    with pytest.raises(ValueError, match="recipient must be a non-empty"):
        _run(adapter.send("   ", "hi"))


def test_in_memory_adapter_rejects_comma_separated_recipients() -> None:
    adapter = InMemoryChannelAdapter()
    with pytest.raises(ValueError, match="single identifier"):
        _run(adapter.send("alice,bob", "hi"))


def test_in_memory_adapter_can_be_configured_to_fail() -> None:
    boom = ProviderError("provider_unavailable", "down for maintenance")
    adapter = InMemoryChannelAdapter(fail_with=boom)
    with pytest.raises(ProviderError):
        _run(adapter.send("alice", "hi"))
    assert adapter.sent == []


# ---------------------------------------------------------------------------
# MessageChannelDispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_registers_and_lists_channels() -> None:
    a = InMemoryChannelAdapter("sms")
    b = InMemoryChannelAdapter("slack")
    dispatcher = MessageChannelDispatcher([a, b])

    assert dispatcher.channels == ("slack", "sms")
    assert "sms" in dispatcher
    assert "telegram" not in dispatcher
    assert len(dispatcher) == 2
    assert dispatcher.get("sms") is a


def test_dispatcher_rejects_duplicate_channel_names() -> None:
    a = InMemoryChannelAdapter("sms")
    b = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([a])
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register(b)


def test_dispatcher_rejects_non_protocol_objects() -> None:
    dispatcher = MessageChannelDispatcher()
    with pytest.raises(TypeError, match="MessageChannelAdapter"):
        dispatcher.register(object())  # type: ignore[arg-type]


def test_dispatcher_rejects_empty_name() -> None:
    bad = InMemoryChannelAdapter.__new__(InMemoryChannelAdapter)
    # Bypass __init__'s guard to construct an adapter with an invalid
    # name and verify the dispatcher catches it independently.
    bad.name = ""
    bad.sent = []
    bad._fail_with = None
    dispatcher = MessageChannelDispatcher()
    with pytest.raises(ValueError, match="non-empty"):
        dispatcher.register(bad)


def test_dispatcher_send_routes_to_named_adapter() -> None:
    sms = InMemoryChannelAdapter("sms")
    slack = InMemoryChannelAdapter("slack")
    dispatcher = MessageChannelDispatcher([sms, slack])

    _run(dispatcher.send("slack", "#general", "hi"))

    assert slack.sent == [
        {"channel": "slack", "recipient": "#general", "body": "hi"}
    ]
    assert sms.sent == []


def test_dispatcher_get_unknown_channel_raises() -> None:
    dispatcher = MessageChannelDispatcher([InMemoryChannelAdapter("sms")])
    with pytest.raises(UnknownChannelError) as ei:
        dispatcher.get("telegram")
    assert ei.value.channel == "telegram"
    assert ei.value.available == ("sms",)


def test_unknown_channel_message_lists_no_channels_when_empty() -> None:
    dispatcher = MessageChannelDispatcher()
    with pytest.raises(UnknownChannelError) as ei:
        dispatcher.get("sms")
    assert "no messaging channels are configured" in str(ei.value)


# ---------------------------------------------------------------------------
# Successful dispatch
# ---------------------------------------------------------------------------


def test_execute_routes_message_through_dispatcher() -> None:
    sms = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([sms])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "+15555550100", "body": "hello"},
            ctx,
        )
    )

    assert result.ok is True
    assert result.error_code is None
    assert result.value == {
        "channel": "sms",
        "recipient": "+15555550100",
        "delivered": True,
    }
    assert sms.sent == [
        {"channel": "sms", "recipient": "+15555550100", "body": "hello"}
    ]


def test_execute_strips_whitespace_around_channel_and_recipient() -> None:
    sms = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([sms])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": " sms ", "recipient": " +15555550100 ", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is True
    assert sms.sent[0]["recipient"] == "+15555550100"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_execute_without_messaging_provider_returns_missing_credentials() -> None:
    """Requirement 5.6: surface missing_credentials when nothing is wired."""
    ctx = SkillContext(providers={}, credential_store=object())

    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "missing_credentials"
    assert "Messaging is not configured" in (result.error_message or "")


def test_execute_with_non_dispatcher_provider_is_internal_error() -> None:
    ctx = _ctx_with(object())
    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "internal_error"


def test_execute_unknown_channel_returns_not_supported() -> None:
    sms = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([sms])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "telegram", "recipient": "@alex", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.value == {
        "channel": "telegram",
        "available_channels": ["sms"],
    }
    assert "telegram" in (result.error_message or "")


def test_execute_provider_error_missing_credentials_is_propagated() -> None:
    adapter = _FailingAdapter(
        "sms",
        ProviderError(
            "missing_credentials",
            "credential 'sms/api_key' is not set",
            provider="sms",
        ),
    )
    dispatcher = MessageChannelDispatcher([adapter])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "missing_credentials"
    assert "sms/api_key" in (result.error_message or "")


def test_execute_provider_error_provider_unavailable_is_propagated() -> None:
    adapter = _FailingAdapter(
        "slack",
        ProviderError("provider_unavailable", "rate limited"),
    )
    dispatcher = MessageChannelDispatcher([adapter])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "slack", "recipient": "#general", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_execute_network_policy_violation_returns_access_denied() -> None:
    adapter = _FailingAdapter(
        "sms",
        NetworkPolicyViolation(
            destination="https://api.example.invalid",
            host="api.example.invalid",
        ),
    )
    dispatcher = MessageChannelDispatcher([adapter])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "access_denied"
    assert "blocked by network allowlist" in (result.error_message or "")


def test_execute_value_error_is_mapped_to_schema_violation() -> None:
    adapter = _FailingAdapter("sms", ValueError("recipient must be E.164"))
    dispatcher = MessageChannelDispatcher([adapter])
    ctx = _ctx_with(dispatcher)

    result = _run(
        SKILL.execute(
            {"channel": "sms", "recipient": "alice", "body": "hi"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert "E.164" in (result.error_message or "")


def test_execute_unexpected_exception_is_left_to_registry() -> None:
    """Non-mapped exceptions bubble up so the registry's audit path runs."""
    adapter = _FailingAdapter("sms", RuntimeError("kaboom"))
    dispatcher = MessageChannelDispatcher([adapter])
    ctx = _ctx_with(dispatcher)

    with pytest.raises(RuntimeError, match="kaboom"):
        _run(
            SKILL.execute(
                {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
                ctx,
            )
        )


# ---------------------------------------------------------------------------
# Integration with the SkillRegistry
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_via_registry() -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "SendMessageSkill" in registry

    sms = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([sms])
    ctx = _ctx_with(dispatcher)

    result = _run(
        registry.dispatch(
            "SendMessageSkill",
            {"channel": "sms", "recipient": "+15555550100", "body": "hi"},
            ctx,
        )
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert sms.sent == [
        {"channel": "sms", "recipient": "+15555550100", "body": "hi"}
    ]


def test_registry_rejects_missing_required_field_with_schema_violation() -> None:
    """Property 2 / CP2: missing fields short-circuit before execute."""
    registry = SkillRegistry()
    registry.register(SKILL)

    sms = InMemoryChannelAdapter("sms")
    dispatcher = MessageChannelDispatcher([sms])
    ctx = _ctx_with(dispatcher)

    result = _run(
        registry.dispatch(
            "SendMessageSkill",
            {"channel": "sms", "recipient": "+15555550100"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert sms.sent == []  # adapter is never reached on schema violation


def test_registry_rejects_extra_properties() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    dispatcher = MessageChannelDispatcher([InMemoryChannelAdapter("sms")])
    ctx = _ctx_with(dispatcher)

    result = _run(
        registry.dispatch(
            "SendMessageSkill",
            {
                "channel": "sms",
                "recipient": "+15555550100",
                "body": "hi",
                "subject": "smuggled-in field",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_skill_appears_in_mistral_tool_definitions() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    tools = registry.mistral_tool_definitions()
    names = [tool["function"]["name"] for tool in tools]
    assert "SendMessageSkill" in names

    tool = next(t for t in tools if t["function"]["name"] == "SendMessageSkill")
    parameters = tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert set(parameters["required"]) == {"channel", "recipient", "body"}
