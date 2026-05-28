"""Unit tests for :mod:`tests.fakes.fake_platform`.

These tests pin the contract the rest of the test suite relies on:

* every :class:`PlatformAdapter` capability appends a typed record to a
  matching public ``*_calls`` list;
* the fake structurally satisfies
  :class:`~jarvis.automation.platform.PlatformAdapter` so consumers can
  type their fixtures against the real Protocol;
* the ``force_unsupported`` toggle raises
  :class:`~jarvis.automation.platform.PlatformNotSupportedError` for the
  flagged method only;
* the ``force_error`` toggle is one-shot: the next call raises and the
  call after it succeeds again.

Validates: Requirements 15.2, 15.3, 15.4
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.fake_platform import (
    FAKE_PLATFORM_TAG,
    BrightnessCall,
    ClickCall,
    FakePlatformAdapter,
    LaunchCall,
    RunScriptCall,
    all_method_names,
)

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    PlatformAdapter,
    PlatformNotSupportedError,
    ProcessHandle,
    ScriptResult,
)

# ---------------------------------------------------------------------------
# Structural conformance
# ---------------------------------------------------------------------------


def test_fake_satisfies_platform_adapter_protocol() -> None:
    """The fake must be a structural :class:`PlatformAdapter` instance."""
    adapter = FakePlatformAdapter()
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform_tag == FAKE_PLATFORM_TAG


def test_fake_starts_with_empty_call_lists(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    for name in (
        "launch_calls",
        "media_key_calls",
        "set_volume_calls",
        "adjust_volume_calls",
        "brightness_calls",
        "notify_calls",
        "click_calls",
        "type_calls",
        "hotkey_calls",
        "focus_calls",
        "run_script_calls",
    ):
        assert getattr(a, name) == []


def test_all_method_names_matches_protocol_surface() -> None:
    """The recogniser set must mirror every PlatformAdapter capability."""
    expected = {
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
    assert set(all_method_names()) == expected


# ---------------------------------------------------------------------------
# Recording semantics — one happy-path test per method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_app_records_arguments_and_returns_default_handle(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    handle = await a.launch_app("notepad.exe", ["/A", "demo.txt"])
    assert a.launch_calls == [LaunchCall("notepad.exe", ("/A", "demo.txt"))]
    assert isinstance(handle, ProcessHandle)
    assert handle.executable_or_uri == "notepad.exe"
    assert handle.pid > 0


@pytest.mark.asyncio
async def test_launch_app_uses_configured_handle() -> None:
    custom = ProcessHandle(pid=99, executable_or_uri="explorer.exe", detached=True)
    a = FakePlatformAdapter(next_process_handle=custom)
    handle = await a.launch_app("explorer.exe", [])
    assert handle is custom


@pytest.mark.asyncio
async def test_media_key_records_argument(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.media_key("play_pause")
    await a.media_key("next")
    assert a.media_key_calls == ["play_pause", "next"]


@pytest.mark.asyncio
async def test_volume_methods_record_levels_and_deltas(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.set_volume(40)
    await a.adjust_volume(-10)
    assert a.set_volume_calls == [40]
    assert a.adjust_volume_calls == [-10]


@pytest.mark.asyncio
async def test_brightness_get_returns_default_value(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    assert await a.get_brightness() == 50
    assert a.brightness_calls == [BrightnessCall("get", None)]


@pytest.mark.asyncio
async def test_brightness_get_returns_configured_value() -> None:
    a = FakePlatformAdapter(brightness_value=80)
    assert await a.get_brightness() == 80


@pytest.mark.asyncio
async def test_brightness_set_records_and_updates_observable_state(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.set_brightness(75)
    assert a.brightness_calls == [BrightnessCall("set", 75)]
    # A subsequent ``get`` reflects the value the test set, mirroring a
    # real adapter where set/get round-trip.
    assert await a.get_brightness() == 75
    assert a.brightness_calls == [
        BrightnessCall("set", 75),
        BrightnessCall("get", None),
    ]


@pytest.mark.asyncio
async def test_brightness_set_is_clamped_in_observable_value(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    """``brightness_value`` mirrors the [0, 100] clamp of a real adapter."""
    a = fake_platform_adapter
    await a.set_brightness(250)
    assert a.brightness_value == 100
    await a.set_brightness(-30)
    assert a.brightness_value == 0


@pytest.mark.asyncio
async def test_notify_records_title_and_body(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.notify("Reminder", "Standup in 5")
    assert a.notify_calls == [("Reminder", "Standup in 5")]


@pytest.mark.asyncio
async def test_click_type_hotkey_record_arguments(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.click(10, 20, "left")
    await a.type_text("hello")
    await a.hotkey("ctrl", "c")
    assert a.click_calls == [ClickCall(10, 20, "left")]
    assert a.type_calls == ["hello"]
    assert a.hotkey_calls == [("ctrl", "c")]


@pytest.mark.asyncio
async def test_focus_window_records_pattern(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.focus_window(".*Notepad.*")
    assert a.focus_calls == [".*Notepad.*"]


@pytest.mark.asyncio
async def test_run_script_records_arguments_and_returns_default(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    path = Path("c:/scripts/demo.ps1")
    result = await a.run_script("powershell", path, 5.0)
    assert a.run_script_calls == [RunScriptCall("powershell", path, 5.0)]
    assert isinstance(result, ScriptResult)
    assert result.exit_code == 0
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_run_script_uses_configured_result() -> None:
    canned = ScriptResult(
        exit_code=2, stdout="", stderr="boom", duration_ms=12, timed_out=False
    )
    a = FakePlatformAdapter(next_script_result=canned)
    result = await a.run_script("python", Path("x.py"), 1.0)
    assert result is canned


# ---------------------------------------------------------------------------
# force_unsupported toggles
# ---------------------------------------------------------------------------


_UNSUPPORTED_CASES: list[tuple[str, tuple[object, ...]]] = [
    ("launch_app", ("notepad.exe", [])),
    ("media_key", ("play_pause",)),
    ("set_volume", (10,)),
    ("adjust_volume", (-5,)),
    ("get_brightness", ()),
    ("set_brightness", (40,)),
    ("notify", ("title", "body")),
    ("click", (1, 2, "left")),
    ("type_text", ("hello",)),
    ("hotkey", ("ctrl", "c")),
    ("focus_window", (".*",)),
    ("run_script", ("powershell", Path("x.ps1"), 1.0)),
]


@pytest.mark.parametrize(("method", "args"), _UNSUPPORTED_CASES)
@pytest.mark.asyncio
async def test_force_unsupported_raises_for_flagged_method(
    fake_platform_adapter: FakePlatformAdapter,
    method: str,
    args: tuple[object, ...],
) -> None:
    """Flagged methods raise ``PlatformNotSupportedError`` (Requirement 15.4)."""
    a = fake_platform_adapter
    a.force_unsupported(method)
    with pytest.raises(PlatformNotSupportedError) as excinfo:
        await getattr(a, method)(*args)
    assert excinfo.value.capability == method
    assert excinfo.value.platform == FAKE_PLATFORM_TAG
    assert excinfo.value.error_code == PLATFORM_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_force_unsupported_records_call_before_raising(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    """A failing call must still leave a trace in the call list."""
    a = fake_platform_adapter
    a.force_unsupported("notify")
    with pytest.raises(PlatformNotSupportedError):
        await a.notify("title", "body")
    assert a.notify_calls == [("title", "body")]


@pytest.mark.asyncio
async def test_force_unsupported_only_affects_flagged_method(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    a.force_unsupported("notify")
    # Other methods continue to work.
    await a.media_key("play_pause")
    assert a.media_key_calls == ["play_pause"]
    with pytest.raises(PlatformNotSupportedError):
        await a.notify("t", "b")


@pytest.mark.asyncio
async def test_clear_unsupported_restores_flagged_method(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    a.force_unsupported("media_key")
    a.clear_unsupported("media_key")
    await a.media_key("play_pause")
    assert a.media_key_calls == ["play_pause"]


@pytest.mark.asyncio
async def test_clear_unsupported_no_args_clears_everything(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    a.force_unsupported("media_key", "notify")
    a.clear_unsupported()
    await a.media_key("play_pause")
    await a.notify("t", "b")
    assert a.media_key_calls == ["play_pause"]
    assert a.notify_calls == [("t", "b")]


def test_force_unsupported_rejects_unknown_method(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    with pytest.raises(ValueError, match="Unknown PlatformAdapter method"):
        fake_platform_adapter.force_unsupported("bogus")


# ---------------------------------------------------------------------------
# force_error toggles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_error_raises_then_clears(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    """``force_error`` is one-shot: the next call after the failure succeeds."""
    a = fake_platform_adapter
    boom = OSError("device offline")
    a.force_error("set_volume", boom)
    with pytest.raises(OSError, match="device offline"):
        await a.set_volume(20)
    # Next call goes through normally.
    await a.set_volume(30)
    assert a.set_volume_calls == [20, 30]


@pytest.mark.asyncio
async def test_force_error_runs_after_recording(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    a.force_error("type_text", RuntimeError("xx"))
    with pytest.raises(RuntimeError):
        await a.type_text("hello")
    assert a.type_calls == ["hello"]


def test_force_error_rejects_unknown_method(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    with pytest.raises(ValueError, match="Unknown PlatformAdapter method"):
        fake_platform_adapter.force_error("bogus", RuntimeError("x"))


def test_force_error_rejects_non_exception(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    with pytest.raises(TypeError):
        fake_platform_adapter.force_error("notify", "not an exception")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reset() clears state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_clears_calls_and_toggles(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    a = fake_platform_adapter
    await a.media_key("play_pause")
    a.force_unsupported("notify")
    a.force_error("set_volume", RuntimeError("x"))

    a.reset()

    assert a.media_key_calls == []
    # No toggles remaining: both calls succeed.
    await a.notify("t", "b")
    await a.set_volume(40)
    assert a.notify_calls == [("t", "b")]
    assert a.set_volume_calls == [40]


# ---------------------------------------------------------------------------
# Argument validation — defensive guards on the recorder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_app_rejects_non_string_args(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    with pytest.raises(TypeError):
        await fake_platform_adapter.launch_app("x.exe", [1, 2])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_run_script_rejects_string_path(
    fake_platform_adapter: FakePlatformAdapter,
) -> None:
    with pytest.raises(TypeError):
        await fake_platform_adapter.run_script(
            "powershell",
            "x.ps1",  # type: ignore[arg-type]
            1.0,
        )


def test_constructor_rejects_brightness_out_of_range() -> None:
    with pytest.raises(ValueError):
        FakePlatformAdapter(brightness_value=150)
