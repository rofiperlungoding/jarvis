"""Unit tests for :mod:`jarvis.automation.platform`.

Exercises the Protocol shape, the value-type guards, and the default
``BasePlatformAdapter`` behaviour (Requirements 15.2, 15.3, 15.4).
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    PlatformAdapter,
    PlatformNotSupportedError,
    ProcessHandle,
    ScriptResult,
)
from jarvis.skills.base import ERROR_CODES

# ---------------------------------------------------------------------------
# Cross-module invariant: error code matches the SkillResult enum
# ---------------------------------------------------------------------------


def test_platform_not_supported_constant_matches_skill_error_codes() -> None:
    """The literal must round-trip through ``SkillResult.error_code``."""
    assert PLATFORM_NOT_SUPPORTED == "platform_not_supported"
    assert PLATFORM_NOT_SUPPORTED in ERROR_CODES


# ---------------------------------------------------------------------------
# PlatformNotSupportedError
# ---------------------------------------------------------------------------


def test_platform_not_supported_error_carries_error_code() -> None:
    err = PlatformNotSupportedError("set_brightness")
    assert err.error_code == PLATFORM_NOT_SUPPORTED
    assert err.capability == "set_brightness"
    assert err.platform is None
    assert err.detail is None
    assert "set_brightness" in str(err)


def test_platform_not_supported_error_with_platform_and_detail() -> None:
    err = PlatformNotSupportedError(
        "set_brightness", platform="windows", detail="WMI method missing"
    )
    msg = str(err)
    assert "set_brightness" in msg
    assert "windows" in msg
    assert "WMI method missing" in msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capability": ""},  # empty
        {"capability": 1},  # wrong type
    ],
)
def test_platform_not_supported_error_rejects_bad_capability(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PlatformNotSupportedError(**kwargs)  # type: ignore[arg-type]


def test_platform_not_supported_error_rejects_non_string_platform() -> None:
    with pytest.raises(TypeError):
        PlatformNotSupportedError("x", platform=123)  # type: ignore[arg-type]


def test_platform_not_supported_error_rejects_non_string_detail() -> None:
    with pytest.raises(TypeError):
        PlatformNotSupportedError("x", detail=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProcessHandle
# ---------------------------------------------------------------------------


def test_process_handle_minimal_construction() -> None:
    h = ProcessHandle(pid=42, executable_or_uri="notepad.exe")
    assert h.pid == 42
    assert h.executable_or_uri == "notepad.exe"
    assert h.detached is False
    assert dict(h.metadata) == {}


def test_process_handle_metadata_is_snapshotted() -> None:
    """Mutating the original mapping must not change the frozen handle."""
    src = {"window_title": "Untitled - Notepad"}
    h = ProcessHandle(pid=1, executable_or_uri="notepad.exe", metadata=src)
    src["window_title"] = "MUTATED"
    assert h.metadata["window_title"] == "Untitled - Notepad"


def test_process_handle_rejects_negative_pid() -> None:
    with pytest.raises(ValueError):
        ProcessHandle(pid=-1, executable_or_uri="x")


def test_process_handle_rejects_bool_pid() -> None:
    with pytest.raises(TypeError):
        ProcessHandle(pid=True, executable_or_uri="x")  # type: ignore[arg-type]


def test_process_handle_rejects_empty_executable() -> None:
    with pytest.raises(ValueError):
        ProcessHandle(pid=1, executable_or_uri="")


def test_process_handle_rejects_non_bool_detached() -> None:
    with pytest.raises(TypeError):
        ProcessHandle(
            pid=1,
            executable_or_uri="x",
            detached="yes",  # type: ignore[arg-type]
        )


def test_process_handle_rejects_non_mapping_metadata() -> None:
    with pytest.raises(TypeError):
        ProcessHandle(
            pid=1,
            executable_or_uri="x",
            metadata=["a", "b"],  # type: ignore[arg-type]
        )


def test_process_handle_is_frozen() -> None:
    h = ProcessHandle(pid=1, executable_or_uri="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.pid = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScriptResult
# ---------------------------------------------------------------------------


def test_script_result_minimal_success() -> None:
    r = ScriptResult(exit_code=0, stdout="ok\n", stderr="", duration_ms=15)
    assert r.exit_code == 0
    assert r.stdout == "ok\n"
    assert r.stderr == ""
    assert r.duration_ms == 15
    assert r.timed_out is False


def test_script_result_timeout_marker() -> None:
    r = ScriptResult(
        exit_code=-1,
        stdout="",
        stderr="killed",
        duration_ms=60_000,
        timed_out=True,
    )
    assert r.timed_out is True


def test_script_result_rejects_bool_exit_code() -> None:
    with pytest.raises(TypeError):
        ScriptResult(
            exit_code=True,  # type: ignore[arg-type]
            stdout="",
            stderr="",
            duration_ms=0,
        )


def test_script_result_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        ScriptResult(exit_code=0, stdout="", stderr="", duration_ms=-1)


def test_script_result_rejects_non_string_streams() -> None:
    with pytest.raises(TypeError):
        ScriptResult(
            exit_code=0,
            stdout=b"bytes",  # type: ignore[arg-type]
            stderr="",
            duration_ms=0,
        )


def test_script_result_is_frozen() -> None:
    r = ScriptResult(exit_code=0, stdout="", stderr="", duration_ms=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.exit_code = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------

# Method names the Protocol must expose. Using a set keeps the assertion
# stable against future re-orderings.
_REQUIRED_METHODS = frozenset(
    {
        "launch_app",
        "media_key",
        "set_volume",
        "adjust_volume",
        "get_brightness",
        "set_brightness",
        "notify",
        "click",
        "type_text",
        "hotkey",
        "focus_window",
        "run_script",
    }
)


def test_protocol_declares_all_required_methods() -> None:
    declared = {
        name
        for name in dir(PlatformAdapter)
        if not name.startswith("_") and callable(getattr(PlatformAdapter, name))
    }
    missing = _REQUIRED_METHODS - declared
    assert missing == set(), f"PlatformAdapter is missing methods: {missing}"


def test_protocol_methods_are_async_on_base_adapter() -> None:
    """All capabilities must be async so the dialog loop can await them."""
    adapter = BasePlatformAdapter()
    for name in _REQUIRED_METHODS:
        attr = getattr(adapter, name)
        assert inspect.iscoroutinefunction(attr), (
            f"BasePlatformAdapter.{name} must be a coroutine function"
        )


# ---------------------------------------------------------------------------
# BasePlatformAdapter — every method raises platform-not-supported
# ---------------------------------------------------------------------------


def test_base_adapter_satisfies_protocol_via_isinstance() -> None:
    """Structural :class:`Protocol` check covers the runtime contract."""
    assert isinstance(BasePlatformAdapter(), PlatformAdapter)


def test_base_adapter_default_platform_tag() -> None:
    assert BasePlatformAdapter.platform_tag == "unknown"


@pytest.mark.asyncio
async def test_base_launch_app_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError) as excinfo:
        await a.launch_app("notepad.exe", [])
    assert excinfo.value.capability == "launch_app"
    assert excinfo.value.platform == "unknown"
    assert excinfo.value.error_code == PLATFORM_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_base_media_key_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.media_key("play_pause")


@pytest.mark.asyncio
async def test_base_set_and_adjust_volume_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.set_volume(50)
    with pytest.raises(PlatformNotSupportedError):
        await a.adjust_volume(-10)


@pytest.mark.asyncio
async def test_base_brightness_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.get_brightness()
    with pytest.raises(PlatformNotSupportedError):
        await a.set_brightness(80)


@pytest.mark.asyncio
async def test_base_notify_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.notify("Reminder", "Standup in 5")


@pytest.mark.asyncio
async def test_base_click_type_hotkey_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.click(10, 20, "left")
    with pytest.raises(PlatformNotSupportedError):
        await a.type_text("hello")
    with pytest.raises(PlatformNotSupportedError):
        await a.hotkey("ctrl", "c")


@pytest.mark.asyncio
async def test_base_focus_window_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.focus_window(".*Notepad.*")


@pytest.mark.asyncio
async def test_base_run_script_unsupported() -> None:
    a = BasePlatformAdapter()
    with pytest.raises(PlatformNotSupportedError):
        await a.run_script("powershell", Path("c:/scripts/x.ps1"), 5.0)


# ---------------------------------------------------------------------------
# Subclassing — overriding only a subset still works
# ---------------------------------------------------------------------------


class _PartialAdapter(BasePlatformAdapter):
    """Adapter that only implements ``notify``."""

    platform_tag = "test"

    def __init__(self) -> None:
        self.notify_calls: list[tuple[str, str]] = []

    async def notify(self, title: str, body: str) -> None:
        self.notify_calls.append((title, body))


@pytest.mark.asyncio
async def test_subclass_can_override_subset_of_methods() -> None:
    a = _PartialAdapter()
    # Override invoked normally.
    await a.notify("hello", "world")
    assert a.notify_calls == [("hello", "world")]
    # Non-overridden method still raises with the subclass's platform tag.
    with pytest.raises(PlatformNotSupportedError) as excinfo:
        await a.set_brightness(50)
    assert excinfo.value.platform == "test"
    assert excinfo.value.capability == "set_brightness"


@pytest.mark.asyncio
async def test_subclass_can_partially_signal_unsupported() -> None:
    """A method that *can* run sometimes can still raise via ``_unsupported``."""

    class FlakyBrightness(BasePlatformAdapter):
        platform_tag = "windows"

        async def set_brightness(self, level_pct: int) -> None:  # pragma: no cover
            raise self._unsupported("set_brightness", detail="WmiMonitorBrightness not implemented")

    with pytest.raises(PlatformNotSupportedError) as excinfo:
        await FlakyBrightness().set_brightness(50)
    assert excinfo.value.platform == "windows"
    assert "WmiMonitorBrightness" in (excinfo.value.detail or "")
