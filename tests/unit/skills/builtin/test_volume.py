"""Unit tests for :mod:`jarvis.skills.builtin.volume`.

Validates Requirements 4.3 (closed operation vocabulary), 4.4 (``set``
targets an absolute level), and 4.5 (``increase`` / ``decrease`` shift
the master output volume by the supplied level, defaulting to 10
percent when ``level`` is omitted).

Tests use a hand-rolled fake :class:`PlatformAdapter` that records every
``set_volume``, ``adjust_volume``, and ``hotkey`` call so the
operation-to-adapter mapping is asserted exhaustively without importing
the Windows-only adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    PlatformAdapter,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import volume as volume_module
from jarvis.skills.builtin.volume import (
    DEFAULT_DELTA_PCT,
    MUTE_HOTKEY,
    SKILL,
    VOLUME_OPERATIONS,
    VolumeSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that captures volume / hotkey invocations.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied (every other capability raises
    :class:`PlatformNotSupportedError`) so a misconfigured Skill that
    accidentally calls something other than the volume / hotkey methods
    would fail loudly during the test rather than silently no-op.
    """

    platform_tag = "test"

    def __init__(self) -> None:
        self.set_calls: list[int] = []
        self.adjust_calls: list[int] = []
        self.hotkey_calls: list[tuple[str, ...]] = []

    async def set_volume(self, level_pct: int) -> None:
        self.set_calls.append(level_pct)

    async def adjust_volume(self, delta_pct: int) -> None:
        self.adjust_calls.append(delta_pct)

    async def hotkey(self, *keys: str) -> None:
        self.hotkey_calls.append(tuple(keys))


class _UnsupportedAdapter(BasePlatformAdapter):
    """Adapter whose volume / hotkey methods always raise unsupported."""

    platform_tag = "test"

    async def set_volume(self, level_pct: int) -> None:
        raise self._unsupported(
            "set_volume", detail=f"no audio endpoint (level={level_pct})"
        )

    async def adjust_volume(self, delta_pct: int) -> None:
        raise self._unsupported(
            "adjust_volume", detail=f"no audio endpoint (delta={delta_pct})"
        )

    async def hotkey(self, *keys: str) -> None:
        raise self._unsupported("hotkey", detail=f"no input device (keys={keys!r})")


class _BoomAdapter(BasePlatformAdapter):
    """Adapter whose volume methods raise an unrelated exception."""

    platform_tag = "test"

    async def set_volume(self, level_pct: int) -> None:
        raise RuntimeError(f"adapter boom set_volume({level_pct})")

    async def adjust_volume(self, delta_pct: int) -> None:
        raise RuntimeError(f"adapter boom adjust_volume({delta_pct})")

    async def hotkey(self, *keys: str) -> None:
        raise RuntimeError(f"adapter boom hotkey({keys!r})")


def _run(coro: Any) -> Any:
    """Synchronously run a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, VolumeSkill)
    assert SKILL.manifest is VolumeSkill.manifest


def test_operation_vocabulary_matches_requirement() -> None:
    """Requirement 4.3 fixes the operation enum."""
    assert VOLUME_OPERATIONS == ("set", "increase", "decrease", "mute", "unmute")
    schema = SKILL.manifest.json_schema
    assert schema["properties"]["operation"]["enum"] == list(VOLUME_OPERATIONS)


def test_default_delta_matches_requirement() -> None:
    """Requirement 4.5 fixes the default delta at 10 percent."""
    assert DEFAULT_DELTA_PCT == 10


def test_mute_hotkey_uses_volumemute() -> None:
    """Windows ``VK_VOLUME_MUTE`` is exposed via pyautogui's ``volumemute``."""
    assert MUTE_HOTKEY == "volumemute"


def test_manifest_is_non_destructive_and_windows_only() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "volume"
    assert manifest.destructive is False
    assert manifest.platforms == ("windows",)
    assert manifest.source == "builtin"
    schema = manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["operation"]
    assert schema["additionalProperties"] is False


def test_level_property_is_bounded_to_zero_through_one_hundred() -> None:
    """Requirements 4.4/4.5 restrict ``level`` to ``[0, 100]``."""
    level_schema = SKILL.manifest.json_schema["properties"]["level"]
    assert level_schema["type"] == "integer"
    assert level_schema["minimum"] == 0
    assert level_schema["maximum"] == 100


# ---------------------------------------------------------------------------
# set — absolute volume target (Requirement 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 25, 50, 75, 100])
def test_set_dispatches_set_volume_with_supplied_level(level: int) -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "set", "level": level}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert result.value == {"operation": "set", "level": level}
    assert adapter.set_calls == [level]
    assert adapter.adjust_calls == []
    assert adapter.hotkey_calls == []


def test_set_calls_adapter_exactly_once() -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    _run(SKILL.execute({"operation": "set", "level": 42}, ctx))

    assert len(adapter.set_calls) == 1


# ---------------------------------------------------------------------------
# increase / decrease — relative volume shift (Requirement 4.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "level", "expected_delta"),
    [
        ("increase", 5, 5),
        ("increase", 25, 25),
        ("increase", 100, 100),
        ("decrease", 5, -5),
        ("decrease", 25, -25),
        ("decrease", 100, -100),
    ],
)
def test_increase_decrease_with_explicit_level(
    operation: str, level: int, expected_delta: int
) -> None:
    """Requirement 4.5: explicit level is forwarded as a signed delta."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": operation, "level": level}, ctx))

    assert result.ok is True
    assert result.value == {
        "operation": operation,
        "delta": expected_delta,
        "magnitude": level,
    }
    assert adapter.adjust_calls == [expected_delta]
    assert adapter.set_calls == []
    assert adapter.hotkey_calls == []


@pytest.mark.parametrize(
    ("operation", "expected_delta"),
    [("increase", DEFAULT_DELTA_PCT), ("decrease", -DEFAULT_DELTA_PCT)],
)
def test_increase_decrease_default_delta_when_level_omitted(
    operation: str, expected_delta: int
) -> None:
    """Requirement 4.5: omitted level defaults to 10 percent."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": operation}, ctx))

    assert result.ok is True
    assert result.value == {
        "operation": operation,
        "delta": expected_delta,
        "magnitude": DEFAULT_DELTA_PCT,
    }
    assert adapter.adjust_calls == [expected_delta]


# ---------------------------------------------------------------------------
# mute / unmute — single OS toggle key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ["mute", "unmute"])
def test_mute_and_unmute_press_volumemute_hotkey(operation: str) -> None:
    """Both operations press the OS volume-mute toggle key."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": operation}, ctx))

    assert result.ok is True
    assert result.value == {"operation": operation, "hotkey": MUTE_HOTKEY}
    assert adapter.hotkey_calls == [(MUTE_HOTKEY,)]
    assert adapter.set_calls == []
    assert adapter.adjust_calls == []


def test_mute_ignores_supplied_level() -> None:
    """Requirement 4.3: ``level`` is meaningless for mute / unmute."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    # The schema does not forbid a level on mute, but the executor must
    # not forward it to a set/adjust call. Schema validation through the
    # registry permits the extra field as long as ``additionalProperties``
    # only excludes unknown keys, not optional ones; ``level`` is a known
    # optional property.
    result = _run(SKILL.execute({"operation": "mute", "level": 50}, ctx))

    assert result.ok is True
    assert adapter.set_calls == []
    assert adapter.adjust_calls == []
    assert adapter.hotkey_calls == [(MUTE_HOTKEY,)]


# ---------------------------------------------------------------------------
# Adapter failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "set", "level": 25},
        {"operation": "increase", "level": 5},
        {"operation": "decrease"},
        {"operation": "mute"},
        {"operation": "unmute"},
    ],
)
def test_execute_returns_platform_not_supported_when_adapter_unsupported(
    args: dict[str, Any],
) -> None:
    adapter = _UnsupportedAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute(args, ctx))

    assert result.ok is False
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert result.error_message is not None


def test_execute_propagates_unrelated_exceptions() -> None:
    """Non-PlatformNotSupportedError exceptions are left to the registry."""
    adapter = _BoomAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    with pytest.raises(RuntimeError, match="adapter boom"):
        _run(SKILL.execute({"operation": "set", "level": 30}, ctx))


def test_execute_without_platform_adapter_is_internal_error() -> None:
    result = _run(SKILL.execute({"operation": "set", "level": 30}, SkillContext()))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    """Defensive: a smuggled-in object that is not a PlatformAdapter is rejected."""
    ctx = SkillContext(platform_adapter=object())

    result = _run(SKILL.execute({"operation": "mute"}, ctx))

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
    assert "volume" in registry

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("volume", {"operation": "set", "level": 60}, ctx))

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.set_calls == [60]


def test_registry_rejects_invalid_operation_with_schema_violation() -> None:
    """Property 2 / CP2: bad enum value is a schema_violation, not a dispatch."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("volume", {"operation": "max"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # Executor must NOT have been called.
    assert adapter.set_calls == []
    assert adapter.adjust_calls == []
    assert adapter.hotkey_calls == []


def test_registry_rejects_set_without_level_with_schema_violation() -> None:
    """Requirement 4.4: ``set`` without ``level`` is a schema violation."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("volume", {"operation": "set"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.set_calls == []


@pytest.mark.parametrize("level", [-1, 101, 200])
def test_registry_rejects_out_of_range_level_with_schema_violation(
    level: int,
) -> None:
    """Requirements 4.4/4.5: ``level`` must be inside ``[0, 100]``."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch("volume", {"operation": "set", "level": level}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.set_calls == []


def test_registry_rejects_extra_properties() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "volume",
            {"operation": "increase", "level": 5, "extra": "nope"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.adjust_calls == []


def test_skill_appears_in_mistral_tool_definitions() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    tools = registry.mistral_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "volume" in names

    volume_tool = next(t for t in tools if t["function"]["name"] == "volume")
    assert volume_tool["type"] == "function"
    parameters = volume_tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["operation"]["enum"] == list(VOLUME_OPERATIONS)


# ---------------------------------------------------------------------------
# Module reference checks
# ---------------------------------------------------------------------------


def test_module_skill_is_a_protocol_instance() -> None:
    """The exported SKILL satisfies the runtime-checkable Skill Protocol."""
    assert isinstance(volume_module.SKILL, Skill)


def test_skill_contract_uses_runtime_protocol_for_adapter_check() -> None:
    """A bare ``BasePlatformAdapter`` (raises everywhere) hits the unsupported branch."""
    base = BasePlatformAdapter()
    assert isinstance(base, PlatformAdapter)
    ctx = SkillContext(platform_adapter=base)
    # Even though the adapter satisfies the Protocol, ``set_volume`` is
    # unsupported by default — confirms the path through the
    # platform-not-supported branch is reached.
    result = _run(SKILL.execute({"operation": "set", "level": 40}, ctx))
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert isinstance(result.error_message, str)
    assert "set_volume" in result.error_message


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "set", "level": 0},
        {"operation": "set", "level": 100},
        {"operation": "increase"},
        {"operation": "decrease"},
        {"operation": "increase", "level": 0},
        {"operation": "mute"},
        {"operation": "unmute"},
    ],
)
def test_registry_accepts_all_valid_argument_shapes(args: dict[str, Any]) -> None:
    """Spot-check that every documented invocation passes schema validation."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("volume", args, ctx))

    assert result.ok is True
