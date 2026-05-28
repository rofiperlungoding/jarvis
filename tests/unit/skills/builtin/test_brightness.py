"""Unit tests for :mod:`jarvis.skills.builtin.brightness`.

Covers Requirements 4.6 (operation enum + level bounds), 4.7 (delegation
to :meth:`PlatformAdapter.set_brightness`), and 4.8 (the
``not_supported`` branch when the active display does not implement WMI
brightness). Uses a hand-rolled fake :class:`PlatformAdapter` so the
tests do not import the Windows-only adapter and run on every platform.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.automation.platform import BasePlatformAdapter, PlatformAdapter
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import brightness as brightness_module
from jarvis.skills.builtin.brightness import (
    BRIGHTNESS_OPERATIONS,
    DEFAULT_DELTA_PCT,
    SKILL,
    BrightnessSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter implementations
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that tracks every brightness call and stores a fake state.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied: every other capability raises
    :class:`PlatformNotSupportedError`, so a Skill that accidentally
    reaches for an unrelated method would fail loudly.
    """

    platform_tag = "test"

    def __init__(self, initial: int = 50) -> None:
        self.current: int = initial
        self.set_calls: list[int] = []
        self.get_calls: int = 0

    async def get_brightness(self) -> int:
        self.get_calls += 1
        return self.current

    async def set_brightness(self, level_pct: int) -> None:
        self.set_calls.append(level_pct)
        self.current = level_pct


class _SetUnsupportedAdapter(BasePlatformAdapter):
    """``set_brightness`` raises :class:`PlatformNotSupportedError`."""

    platform_tag = "test"

    def __init__(self, initial: int = 50) -> None:
        self.current = initial
        self.get_calls = 0

    async def get_brightness(self) -> int:
        self.get_calls += 1
        return self.current

    async def set_brightness(self, level_pct: int) -> None:
        raise self._unsupported(
            "set_brightness",
            detail="WmiMonitorBrightnessMethods missing",
        )


class _GetUnsupportedAdapter(BasePlatformAdapter):
    """``get_brightness`` raises :class:`PlatformNotSupportedError`."""

    platform_tag = "test"

    async def get_brightness(self) -> int:
        raise self._unsupported(
            "get_brightness",
            detail="WmiMonitorBrightness returned no instances",
        )

    async def set_brightness(self, level_pct: int) -> None:  # pragma: no cover
        raise AssertionError("set_brightness must not be called when get fails")


class _BoomAdapter(BasePlatformAdapter):
    """Adapter that raises an unrelated exception (registry → ``internal_error``)."""

    platform_tag = "test"

    async def get_brightness(self) -> int:  # pragma: no cover - depends on path
        raise RuntimeError("read boom")

    async def set_brightness(self, level_pct: int) -> None:
        raise RuntimeError(f"adapter boom for level {level_pct!r}")


def _run(coro: Any) -> Any:
    """Synchronously run a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, BrightnessSkill)
    assert SKILL.manifest is BrightnessSkill.manifest


def test_module_skill_satisfies_protocol() -> None:
    assert isinstance(brightness_module.SKILL, Skill)


def test_operation_vocabulary_matches_requirement() -> None:
    """Requirement 4.6 fixes the operation enum."""
    assert BRIGHTNESS_OPERATIONS == ("set", "increase", "decrease")
    schema = SKILL.manifest.json_schema
    assert schema["properties"]["operation"]["enum"] == list(BRIGHTNESS_OPERATIONS)


def test_default_delta_is_ten() -> None:
    """Mirrors the VolumeSkill precedent ('default delta 10')."""
    assert DEFAULT_DELTA_PCT == 10


def test_manifest_is_non_destructive_and_windows_only() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "brightness"
    assert manifest.destructive is False
    assert manifest.platforms == ("windows",)
    assert manifest.source == "builtin"
    schema = manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["operation"]
    assert schema["additionalProperties"] is False
    level_schema = schema["properties"]["level"]
    assert level_schema["type"] == "integer"
    assert level_schema["minimum"] == 0
    assert level_schema["maximum"] == 100


# ---------------------------------------------------------------------------
# operation = "set"
# ---------------------------------------------------------------------------


def test_set_writes_level_to_adapter() -> None:
    """Requirement 4.7: 'set' delegates to ``set_brightness``."""
    adapter = _RecordingAdapter(initial=50)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "set", "level": 75}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert result.value == {"operation": "set", "level": 75}
    assert adapter.set_calls == [75]
    # 'set' does not need the current value.
    assert adapter.get_calls == 0


def test_set_at_boundary_values() -> None:
    adapter = _RecordingAdapter(initial=50)
    ctx = SkillContext(platform_adapter=adapter)

    _run(SKILL.execute({"operation": "set", "level": 0}, ctx))
    _run(SKILL.execute({"operation": "set", "level": 100}, ctx))

    assert adapter.set_calls == [0, 100]


def test_set_without_level_returns_schema_violation() -> None:
    """Requirement 14.5: missing required field surfaces as schema_violation."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "set"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert "level" in (result.error_message or "")
    assert result.value == {"missing": "level"}
    # No side effect when the precondition fails.
    assert adapter.set_calls == []


# ---------------------------------------------------------------------------
# operation = "increase" / "decrease"
# ---------------------------------------------------------------------------


def test_increase_uses_default_delta_when_level_omitted() -> None:
    adapter = _RecordingAdapter(initial=40)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "increase"}, ctx))

    assert result.ok is True
    assert result.value == {
        "operation": "increase",
        "level": 50,
        "previous_level": 40,
        "delta": DEFAULT_DELTA_PCT,
    }
    assert adapter.get_calls == 1
    assert adapter.set_calls == [50]


def test_decrease_uses_default_delta_when_level_omitted() -> None:
    adapter = _RecordingAdapter(initial=40)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "decrease"}, ctx))

    assert result.ok is True
    assert result.value == {
        "operation": "decrease",
        "level": 30,
        "previous_level": 40,
        "delta": DEFAULT_DELTA_PCT,
    }
    assert adapter.set_calls == [30]


def test_increase_respects_explicit_level() -> None:
    adapter = _RecordingAdapter(initial=20)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "increase", "level": 25}, ctx))

    assert result.ok is True
    assert result.value["level"] == 45
    assert result.value["delta"] == 25
    assert adapter.set_calls == [45]


def test_increase_clamps_above_one_hundred() -> None:
    adapter = _RecordingAdapter(initial=95)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "increase", "level": 25}, ctx))

    assert result.ok is True
    assert result.value["level"] == 100
    assert adapter.set_calls == [100]


def test_decrease_clamps_below_zero() -> None:
    adapter = _RecordingAdapter(initial=5)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "decrease"}, ctx))

    assert result.ok is True
    assert result.value["level"] == 0
    assert adapter.set_calls == [0]


# ---------------------------------------------------------------------------
# Requirement 4.8: not_supported branch
# ---------------------------------------------------------------------------


def test_set_not_supported_when_platform_raises() -> None:
    """Requirement 4.8: WMI failure surfaces as ``not_supported``."""
    adapter = _SetUnsupportedAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "set", "level": 75}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.error_message is not None
    assert "brightness" in result.error_message.lower()
    assert result.value is not None
    assert result.value["capability"] == "set_brightness"
    assert result.value["platform"] == "test"


def test_increase_not_supported_when_get_fails() -> None:
    """The ``not_supported`` mapping covers both read and write paths."""
    adapter = _GetUnsupportedAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "increase"}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.value is not None
    assert result.value["capability"] == "get_brightness"


def test_increase_not_supported_when_set_fails_after_read() -> None:
    """If set fails after a successful get, we still surface ``not_supported``."""
    adapter = _SetUnsupportedAdapter(initial=40)
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"operation": "increase"}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert adapter.get_calls == 1
    assert result.value is not None
    assert result.value["capability"] == "set_brightness"


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


def test_execute_without_platform_adapter_is_internal_error() -> None:
    result = _run(SKILL.execute({"operation": "set", "level": 50}, SkillContext()))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    ctx = SkillContext(platform_adapter=object())

    result = _run(SKILL.execute({"operation": "set", "level": 50}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "PlatformAdapter" in (result.error_message or "")


def test_execute_propagates_unrelated_exceptions_from_set() -> None:
    """Non-PlatformNotSupportedError exceptions are left to the registry."""
    adapter = _BoomAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    with pytest.raises(RuntimeError, match="adapter boom"):
        _run(SKILL.execute({"operation": "set", "level": 25}, ctx))


# ---------------------------------------------------------------------------
# Integration with the SkillRegistry
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_via_registry() -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "brightness" in registry

    adapter = _RecordingAdapter(initial=30)
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch("brightness", {"operation": "set", "level": 80}, ctx)
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.set_calls == [80]


def test_registry_rejects_invalid_operation_with_schema_violation() -> None:
    """Property 2 / CP2: bad enum value is a schema_violation, not a dispatch."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("brightness", {"operation": "toggle"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # Executor must NOT have been called.
    assert adapter.set_calls == []


def test_registry_rejects_level_out_of_range() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch("brightness", {"operation": "set", "level": 150}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.set_calls == []


def test_registry_rejects_extra_properties() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    result = _run(
        registry.dispatch(
            "brightness",
            {"operation": "set", "level": 50, "monitor": "primary"},
            SkillContext(platform_adapter=adapter),
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_skill_appears_in_mistral_tool_definitions() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    tools = registry.mistral_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "brightness" in names

    tool = next(t for t in tools if t["function"]["name"] == "brightness")
    assert tool["type"] == "function"
    parameters = tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["operation"]["enum"] == list(BRIGHTNESS_OPERATIONS)


# ---------------------------------------------------------------------------
# Adapter satisfying the Protocol but with no overrides
# ---------------------------------------------------------------------------


def test_default_base_adapter_yields_not_supported() -> None:
    """A bare ``BasePlatformAdapter`` satisfies the Protocol but raises
    :class:`PlatformNotSupportedError` for every method, which the
    Skill must translate into ``not_supported`` rather than letting it
    bubble up as an unhandled exception."""
    base = BasePlatformAdapter()
    assert isinstance(base, PlatformAdapter)

    ctx = SkillContext(platform_adapter=base)
    result = _run(SKILL.execute({"operation": "set", "level": 50}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.value is not None
    assert result.value["capability"] == "set_brightness"
