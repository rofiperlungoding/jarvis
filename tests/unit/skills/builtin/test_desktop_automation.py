"""Unit tests for :mod:`jarvis.skills.builtin.desktop_automation`.

Validates Requirements 9.6 (closed action vocabulary with typed payload
fields), 9.7 (each action is dispatched to the platform adapter), and
15.4 (platform-not-supported translation).

Tests use a hand-rolled fake :class:`PlatformAdapter` that records every
``click``, ``type_text``, ``hotkey``, and ``focus_window`` call so the
action-to-method mapping is asserted exhaustively without importing the
Windows-only adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    MouseButton,
    PlatformAdapter,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import desktop_automation as desktop_automation_module
from jarvis.skills.builtin.desktop_automation import (
    DEFAULT_MOUSE_BUTTON,
    DESKTOP_AUTOMATION_ACTIONS,
    MOUSE_BUTTONS,
    SKILL,
    DesktopAutomationSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that captures every UI-automation invocation.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied (every other capability raises
    :class:`PlatformNotSupportedError`) so a misconfigured Skill that
    accidentally calls something other than the four UI methods would
    fail loudly during the test rather than silently no-op.
    """

    platform_tag = "test"

    def __init__(self) -> None:
        self.click_calls: list[tuple[int, int, MouseButton]] = []
        self.type_calls: list[str] = []
        self.hotkey_calls: list[tuple[str, ...]] = []
        self.focus_calls: list[str] = []

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        self.click_calls.append((x, y, button))

    async def type_text(self, text: str) -> None:
        self.type_calls.append(text)

    async def hotkey(self, *keys: str) -> None:
        self.hotkey_calls.append(tuple(keys))

    async def focus_window(self, title_pattern: str) -> None:
        self.focus_calls.append(title_pattern)


class _UnsupportedAdapter(BasePlatformAdapter):
    """Adapter whose UI methods always raise ``PlatformNotSupportedError``."""

    platform_tag = "test"

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        raise self._unsupported(
            "click", detail=f"no display server (x={x}, y={y}, button={button!r})"
        )

    async def type_text(self, text: str) -> None:
        raise self._unsupported(
            "type_text", detail=f"no input device (len(text)={len(text)})"
        )

    async def hotkey(self, *keys: str) -> None:
        raise self._unsupported("hotkey", detail=f"no input device (keys={keys!r})")

    async def focus_window(self, title_pattern: str) -> None:
        raise self._unsupported(
            "focus_window", detail=f"no window manager (pattern={title_pattern!r})"
        )


class _RejectingAdapter(BasePlatformAdapter):
    """Adapter whose UI methods raise ``ValueError`` for payload issues."""

    platform_tag = "test"

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        raise ValueError(f"unsupported mouse button: {button!r}")

    async def type_text(self, text: str) -> None:
        raise ValueError(f"unsafe text payload (len={len(text)})")

    async def hotkey(self, *keys: str) -> None:
        raise ValueError(f"unsafe hotkey key: {keys!r}")

    async def focus_window(self, title_pattern: str) -> None:
        raise ValueError("title_pattern must not be empty after sanitisation")


class _BoomAdapter(BasePlatformAdapter):
    """Adapter whose UI methods raise an unrelated exception."""

    platform_tag = "test"

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        raise RuntimeError(f"adapter boom click({x}, {y}, {button!r})")

    async def type_text(self, text: str) -> None:
        raise RuntimeError(f"adapter boom type_text({text!r})")

    async def hotkey(self, *keys: str) -> None:
        raise RuntimeError(f"adapter boom hotkey({keys!r})")

    async def focus_window(self, title_pattern: str) -> None:
        raise RuntimeError(f"adapter boom focus_window({title_pattern!r})")


def _run(coro: Any) -> Any:
    """Synchronously run a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, DesktopAutomationSkill)
    assert SKILL.manifest is DesktopAutomationSkill.manifest


def test_action_vocabulary_matches_requirement() -> None:
    """Requirement 9.6 fixes the action enum."""
    assert DESKTOP_AUTOMATION_ACTIONS == ("click", "type", "hotkey", "focus_window")
    schema = SKILL.manifest.json_schema
    assert schema["properties"]["action"]["enum"] == list(DESKTOP_AUTOMATION_ACTIONS)


def test_mouse_buttons_match_platform_literal() -> None:
    """Schema enum and adapter ``MouseButton`` literal must agree."""
    assert MOUSE_BUTTONS == ("left", "right", "middle")
    schema = SKILL.manifest.json_schema
    assert schema["properties"]["button"]["enum"] == list(MOUSE_BUTTONS)


def test_default_mouse_button_is_left() -> None:
    """``pyautogui.click`` defaults to left; mirror that for terseness."""
    assert DEFAULT_MOUSE_BUTTON == "left"


def test_manifest_is_non_destructive_and_windows_only() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "desktop_automation"
    assert manifest.destructive is False
    assert manifest.platforms == ("windows",)
    assert manifest.source == "builtin"
    schema = manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["action"]
    assert schema["additionalProperties"] is False


def test_schema_lists_typed_payload_fields() -> None:
    """Requirement 9.6: each action has its own typed payload fields."""
    properties = SKILL.manifest.json_schema["properties"]
    assert properties["x"]["type"] == "integer"
    assert properties["y"]["type"] == "integer"
    assert properties["text"]["type"] == "string"
    assert properties["keys"]["type"] == "array"
    assert properties["keys"]["items"]["type"] == "string"
    assert properties["keys"]["minItems"] == 1
    assert properties["title_pattern"]["type"] == "string"
    assert properties["title_pattern"]["minLength"] == 1


# ---------------------------------------------------------------------------
# click — coordinate + button payload (Requirement 9.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "button"),
    [
        (0, 0, "left"),
        (100, 200, "right"),
        (1920, 1080, "middle"),
        (-50, 75, "left"),
    ],
)
def test_click_dispatches_to_adapter_click(x: int, y: int, button: MouseButton) -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(
        SKILL.execute(
            {"action": "click", "x": x, "y": y, "button": button}, ctx
        )
    )

    assert result.ok is True
    assert result.error_code is None
    assert result.value == {"action": "click", "x": x, "y": y, "button": button}
    assert adapter.click_calls == [(x, y, button)]
    assert adapter.type_calls == []
    assert adapter.hotkey_calls == []
    assert adapter.focus_calls == []


def test_click_defaults_to_left_button_when_omitted() -> None:
    """Omitting ``button`` falls through to the documented default."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"action": "click", "x": 10, "y": 20}, ctx))

    assert result.ok is True
    assert result.value == {
        "action": "click",
        "x": 10,
        "y": 20,
        "button": DEFAULT_MOUSE_BUTTON,
    }
    assert adapter.click_calls == [(10, 20, DEFAULT_MOUSE_BUTTON)]


def test_click_calls_adapter_exactly_once() -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    _run(SKILL.execute({"action": "click", "x": 1, "y": 2}, ctx))

    assert len(adapter.click_calls) == 1


# ---------------------------------------------------------------------------
# type — text payload (Requirement 9.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "hello", "Hello, sir.", "multi\nline\ttabs", "unicode: café 🎯"],
)
def test_type_dispatches_to_adapter_type_text(text: str) -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"action": "type", "text": text}, ctx))

    assert result.ok is True
    assert result.value == {"action": "type", "length": len(text)}
    assert adapter.type_calls == [text]
    assert adapter.click_calls == []
    assert adapter.hotkey_calls == []
    assert adapter.focus_calls == []


# ---------------------------------------------------------------------------
# hotkey — keys payload (Requirement 9.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        ["ctrl", "c"],
        ["ctrl", "shift", "t"],
        ["alt", "tab"],
        ["enter"],
        ["win", "d"],
    ],
)
def test_hotkey_dispatches_to_adapter_hotkey(keys: list[str]) -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"action": "hotkey", "keys": keys}, ctx))

    assert result.ok is True
    assert result.value == {"action": "hotkey", "keys": keys}
    assert adapter.hotkey_calls == [tuple(keys)]
    assert adapter.click_calls == []
    assert adapter.type_calls == []
    assert adapter.focus_calls == []


def test_hotkey_preserves_key_order() -> None:
    """The chord order matters; the executor must pass keys verbatim."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    _run(SKILL.execute({"action": "hotkey", "keys": ["ctrl", "shift", "p"]}, ctx))

    assert adapter.hotkey_calls == [("ctrl", "shift", "p")]


# ---------------------------------------------------------------------------
# focus_window — title pattern payload (Requirement 9.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    ["Notepad", ".*Visual Studio.*", "Chrome", "Project — JARVIS"],
)
def test_focus_window_dispatches_to_adapter_focus_window(pattern: str) -> None:
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(
        SKILL.execute({"action": "focus_window", "title_pattern": pattern}, ctx)
    )

    assert result.ok is True
    assert result.value == {"action": "focus_window", "title_pattern": pattern}
    assert adapter.focus_calls == [pattern]
    assert adapter.click_calls == []
    assert adapter.type_calls == []
    assert adapter.hotkey_calls == []


# ---------------------------------------------------------------------------
# Adapter failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        {"action": "click", "x": 10, "y": 20},
        {"action": "type", "text": "hello"},
        {"action": "hotkey", "keys": ["ctrl", "c"]},
        {"action": "focus_window", "title_pattern": "Notepad"},
    ],
)
def test_execute_returns_platform_not_supported_when_adapter_unsupported(
    args: dict[str, Any],
) -> None:
    """Requirement 15.4: capability unsupported on this platform."""
    adapter = _UnsupportedAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute(args, ctx))

    assert result.ok is False
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert result.error_message is not None


@pytest.mark.parametrize(
    "args",
    [
        {"action": "click", "x": 10, "y": 20, "button": "left"},
        {"action": "type", "text": "hello"},
        {"action": "hotkey", "keys": ["ctrl", "c"]},
        {"action": "focus_window", "title_pattern": "Notepad"},
    ],
)
def test_execute_translates_value_error_to_schema_violation(
    args: dict[str, Any],
) -> None:
    """Adapter-level ``ValueError`` becomes ``schema_violation``.

    The Dialog_Manager can then trigger the standard LLM retry loop
    (Requirement 14.5) with a corrected payload.
    """
    adapter = _RejectingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute(args, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert result.error_message is not None
    assert result.value == {"action": args["action"]}


def test_execute_propagates_unrelated_exceptions() -> None:
    """Non-PlatformNotSupportedError / non-ValueError bubbles up."""
    adapter = _BoomAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    with pytest.raises(RuntimeError, match="adapter boom"):
        _run(SKILL.execute({"action": "click", "x": 0, "y": 0}, ctx))


def test_execute_without_platform_adapter_is_internal_error() -> None:
    result = _run(
        SKILL.execute({"action": "click", "x": 0, "y": 0}, SkillContext())
    )

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    """Defensive: a smuggled-in object that is not a PlatformAdapter is rejected."""
    ctx = SkillContext(platform_adapter=object())

    result = _run(SKILL.execute({"action": "type", "text": "hi"}, ctx))

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
    assert "desktop_automation" in registry

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "desktop_automation",
            {"action": "click", "x": 100, "y": 200, "button": "right"},
            ctx,
        )
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.click_calls == [(100, 200, "right")]


def test_registry_rejects_invalid_action_with_schema_violation() -> None:
    """Property 2 / CP2: bad enum value is a schema_violation, not a dispatch."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("desktop_automation", {"action": "drag"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.click_calls == []


def test_registry_rejects_click_without_coordinates() -> None:
    """Requirement 9.6: ``click`` requires ``x`` and ``y``."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("desktop_automation", {"action": "click"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.click_calls == []


def test_registry_rejects_click_missing_y() -> None:
    """Conditional ``required`` enforces both coordinates."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch("desktop_automation", {"action": "click", "x": 10}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.click_calls == []


def test_registry_rejects_type_without_text() -> None:
    """Requirement 9.6: ``type`` requires ``text``."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("desktop_automation", {"action": "type"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.type_calls == []


def test_registry_rejects_hotkey_without_keys() -> None:
    """Requirement 9.6: ``hotkey`` requires ``keys``."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("desktop_automation", {"action": "hotkey"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.hotkey_calls == []


def test_registry_rejects_hotkey_with_empty_keys_list() -> None:
    """``minItems: 1`` rejects an empty chord."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "desktop_automation", {"action": "hotkey", "keys": []}, ctx
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.hotkey_calls == []


def test_registry_rejects_focus_window_without_pattern() -> None:
    """Requirement 9.6: ``focus_window`` requires ``title_pattern``."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch("desktop_automation", {"action": "focus_window"}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.focus_calls == []


def test_registry_rejects_focus_window_with_empty_pattern() -> None:
    """``minLength: 1`` rejects an empty pattern."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "desktop_automation",
            {"action": "focus_window", "title_pattern": ""},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.focus_calls == []


def test_registry_rejects_invalid_button_with_schema_violation() -> None:
    """Schema enum rejects unknown mouse buttons."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "desktop_automation",
            {"action": "click", "x": 1, "y": 2, "button": "double-left"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.click_calls == []


def test_registry_rejects_extra_properties() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(
        registry.dispatch(
            "desktop_automation",
            {"action": "type", "text": "hi", "extra": "nope"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.type_calls == []


def test_skill_appears_in_mistral_tool_definitions() -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    tools = registry.mistral_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "desktop_automation" in names

    tool = next(t for t in tools if t["function"]["name"] == "desktop_automation")
    assert tool["type"] == "function"
    parameters = tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["action"]["enum"] == list(
        DESKTOP_AUTOMATION_ACTIONS
    )


@pytest.mark.parametrize(
    "args",
    [
        {"action": "click", "x": 0, "y": 0},
        {"action": "click", "x": 100, "y": 200, "button": "left"},
        {"action": "click", "x": 50, "y": 75, "button": "right"},
        {"action": "click", "x": 50, "y": 75, "button": "middle"},
        {"action": "type", "text": ""},
        {"action": "type", "text": "hello world"},
        {"action": "hotkey", "keys": ["ctrl", "c"]},
        {"action": "hotkey", "keys": ["enter"]},
        {"action": "focus_window", "title_pattern": "Notepad"},
        {"action": "focus_window", "title_pattern": ".*Chrome.*"},
    ],
)
def test_registry_accepts_all_valid_argument_shapes(args: dict[str, Any]) -> None:
    """Spot-check that every documented invocation passes schema validation."""
    registry = SkillRegistry()
    registry.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)
    result = _run(registry.dispatch("desktop_automation", args, ctx))

    assert result.ok is True


# ---------------------------------------------------------------------------
# Module reference checks
# ---------------------------------------------------------------------------


def test_module_skill_is_a_protocol_instance() -> None:
    """The exported SKILL satisfies the runtime-checkable Skill Protocol."""
    assert isinstance(desktop_automation_module.SKILL, Skill)


def test_skill_contract_uses_runtime_protocol_for_adapter_check() -> None:
    """A bare ``BasePlatformAdapter`` (raises everywhere) hits the unsupported branch."""
    base = BasePlatformAdapter()
    assert isinstance(base, PlatformAdapter)
    ctx = SkillContext(platform_adapter=base)

    result = _run(SKILL.execute({"action": "click", "x": 1, "y": 2}, ctx))

    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert isinstance(result.error_message, str)
    assert "click" in result.error_message
