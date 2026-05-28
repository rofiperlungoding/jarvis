"""Unit tests for :mod:`jarvis.skills.builtin.media_control`.

Validates Requirements 4.1 (closed action vocabulary) and 4.2 (each
action is dispatched to the platform adapter as the corresponding
media key). Tests use a hand-rolled fake :class:`PlatformAdapter` that
records every ``media_key`` call so the action-to-key mapping is
asserted exhaustively without importing the Windows-only adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    MediaKey,
    PlatformAdapter,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import media_control as media_control_module
from jarvis.skills.builtin.media_control import (
    ACTION_TO_MEDIA_KEY,
    MEDIA_CONTROL_ACTIONS,
    SKILL,
    MediaControlSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that captures every ``media_key`` invocation.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied (every other capability raises
    :class:`PlatformNotSupportedError`) so a misconfigured Skill that
    accidentally calls something other than ``media_key`` would fail
    loudly during the test rather than silently no-op.
    """

    platform_tag = "test"

    def __init__(self) -> None:
        self.calls: list[MediaKey] = []

    async def media_key(self, key: MediaKey) -> None:
        self.calls.append(key)


class _UnsupportedAdapter(BasePlatformAdapter):
    """Adapter whose ``media_key`` always raises ``PlatformNotSupportedError``."""

    platform_tag = "test"

    async def media_key(self, key: MediaKey) -> None:
        raise self._unsupported("media_key", detail=f"no media key support in test (key={key!r})")


class _BoomAdapter(BasePlatformAdapter):
    """Adapter whose ``media_key`` raises an unrelated exception."""

    platform_tag = "test"

    async def media_key(self, key: MediaKey) -> None:
        raise RuntimeError(f"adapter boom for key {key!r}")


def _run(coro: Any) -> Any:
    """Synchronously run a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, MediaControlSkill)
    assert SKILL.manifest is MediaControlSkill.manifest


def test_action_vocabulary_matches_requirement() -> None:
    """Requirement 4.1 fixes the action enum."""
    assert MEDIA_CONTROL_ACTIONS == ("play", "pause", "next", "previous", "stop")
    schema = SKILL.manifest.json_schema
    assert schema["properties"]["action"]["enum"] == list(MEDIA_CONTROL_ACTIONS)


def test_manifest_is_non_destructive_and_windows_only() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "media_control"
    assert manifest.destructive is False
    assert manifest.platforms == ("windows",)
    assert manifest.source == "builtin"
    schema = manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["action"]
    assert schema["additionalProperties"] is False


def test_action_to_media_key_is_total_over_action_vocabulary() -> None:
    """Every accepted action must map to a concrete media-key literal."""
    assert set(ACTION_TO_MEDIA_KEY) == set(MEDIA_CONTROL_ACTIONS)
    # play and pause collapse to the OS toggle, ``previous`` is renamed
    # to the adapter literal, and the others map identity.
    assert ACTION_TO_MEDIA_KEY["play"] == "play_pause"
    assert ACTION_TO_MEDIA_KEY["pause"] == "play_pause"
    assert ACTION_TO_MEDIA_KEY["next"] == "next"
    assert ACTION_TO_MEDIA_KEY["previous"] == "prev"
    assert ACTION_TO_MEDIA_KEY["stop"] == "stop"


# ---------------------------------------------------------------------------
# Successful dispatch — exhaustive over the action enum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected_key"),
    [
        ("play", "play_pause"),
        ("pause", "play_pause"),
        ("next", "next"),
        ("previous", "prev"),
        ("stop", "stop"),
    ],
)
def test_execute_dispatches_each_action_to_correct_media_key(
    action: str, expected_key: MediaKey
) -> None:
    """Requirement 4.2: each action sends the corresponding media key."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"action": action}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert result.value == {"action": action, "media_key": expected_key}
    assert adapter.calls == [expected_key]


def test_execute_calls_adapter_exactly_once() -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    _run(SKILL.execute({"action": "next"}, ctx))

    assert len(adapter.calls) == 1


# ---------------------------------------------------------------------------
# Adapter failure modes
# ---------------------------------------------------------------------------


def test_execute_returns_platform_not_supported_when_adapter_unsupported() -> None:
    adapter = _UnsupportedAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"action": "play"}, ctx))

    assert result.ok is False
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert result.error_message is not None
    assert "media_key" in result.error_message


def test_execute_propagates_unrelated_exceptions() -> None:
    """Non-PlatformNotSupportedError exceptions are left to the registry."""
    adapter = _BoomAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    with pytest.raises(RuntimeError, match="adapter boom"):
        _run(SKILL.execute({"action": "stop"}, ctx))


def test_execute_without_platform_adapter_is_internal_error() -> None:
    result = _run(SKILL.execute({"action": "stop"}, SkillContext()))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    """Defensive: a smuggled-in object that is not a PlatformAdapter is rejected."""
    ctx = SkillContext(platform_adapter=object())

    result = _run(SKILL.execute({"action": "play"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "PlatformAdapter" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Integration with the SkillRegistry
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_via_registry() -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "media_control" in registry

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("media_control", {"action": "next"}, ctx))

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.calls == ["next"]


def test_registry_rejects_invalid_action_with_schema_violation() -> None:
    """Property 2 / CP2: bad enum value is a schema_violation, not a dispatch."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("media_control", {"action": "rewind"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # Executor must NOT have been called.
    assert adapter.calls == []


def test_registry_rejects_extra_properties() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    result = _run(
        registry.dispatch(
            "media_control",
            {"action": "play", "volume": 50},
            SkillContext(platform_adapter=_RecordingAdapter()),
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_skill_appears_in_mistral_tool_definitions() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    tools = registry.mistral_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "media_control" in names

    media_tool = next(t for t in tools if t["function"]["name"] == "media_control")
    assert media_tool["type"] == "function"
    parameters = media_tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["action"]["enum"] == list(MEDIA_CONTROL_ACTIONS)


# ---------------------------------------------------------------------------
# Module reference checks
# ---------------------------------------------------------------------------


def test_module_skill_is_a_protocol_instance() -> None:
    """The exported SKILL satisfies the runtime-checkable Skill Protocol."""
    assert isinstance(media_control_module.SKILL, Skill)


def test_skill_contract_uses_runtime_protocol_for_adapter_check() -> None:
    """A bare `BasePlatformAdapter` (which raises everywhere) still passes the type guard."""
    base = BasePlatformAdapter()
    assert isinstance(base, PlatformAdapter)
    ctx = SkillContext(platform_adapter=base)
    # Even though the adapter satisfies the Protocol, ``media_key`` is
    # unsupported by default — confirms the path through the
    # platform-not-supported branch is reached.
    result = _run(SKILL.execute({"action": "play"}, ctx))
    assert result.error_code == PLATFORM_NOT_SUPPORTED
