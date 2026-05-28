"""Unit tests for :mod:`jarvis.automation.windows_adapter`.

These tests exercise the cross-platform surface of
:class:`WindowsAdapter` — :class:`InputSanitizer`, target resolution,
input validation, command construction, and the ``run_script`` timeout
path — without touching any Win32 / WMI / pyautogui APIs. The
Win32-specific paths (``media_key``, ``set_volume``, ``set_brightness``,
``notify``, ``click``, ``type_text``, ``hotkey``, ``focus_window``) are
covered by integration tests that run on Windows hosts only.

The module under test imports its native dependencies lazily, so these
tests pass on Linux CI as long as we don't drive the OS-specific paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.automation.platform import (
    BasePlatformAdapter,
    PlatformAdapter,
    PlatformNotSupportedError,
    ProcessHandle,
    ScriptResult,
)
from jarvis.automation.windows_adapter import (
    DEFAULT_MAX_TITLE_LENGTH,
    DEFAULT_MAX_TYPE_LENGTH,
    InputSanitizer,
    WindowsAdapter,
)

# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------


class TestInputSanitizer:
    def test_passthrough_clean_text(self) -> None:
        s = InputSanitizer()
        result = s.sanitize_text("hello world")
        assert result.text == "hello world"
        assert result.stripped_chars == 0
        assert result.truncated is False
        assert result.original_length == len("hello world")

    def test_strips_null_byte(self) -> None:
        s = InputSanitizer()
        result = s.sanitize_text("hello\x00world")
        assert "\x00" not in result.text
        assert result.text == "helloworld"
        assert result.stripped_chars == 1

    def test_keeps_standard_whitespace(self) -> None:
        s = InputSanitizer()
        result = s.sanitize_text("a\tb\nc\rd")
        assert result.text == "a\tb\nc\rd"
        assert result.stripped_chars == 0

    def test_strips_unicode_format_overrides(self) -> None:
        # U+202E RIGHT-TO-LEFT OVERRIDE is a notorious BiDi attack code point.
        s = InputSanitizer()
        result = s.sanitize_text("safe\u202etxt")
        assert "\u202e" not in result.text
        assert result.stripped_chars == 1

    def test_truncates_oversize_input(self) -> None:
        s = InputSanitizer(max_type_length=10)
        result = s.sanitize_text("x" * 25)
        assert len(result.text) == 10
        assert result.truncated is True
        assert result.original_length == 25

    def test_title_pattern_uses_separate_cap(self) -> None:
        s = InputSanitizer(max_title_length=8)
        result = s.sanitize_title_pattern("y" * 50)
        assert len(result.text) == 8
        assert result.truncated is True

    def test_rejects_non_string_input(self) -> None:
        s = InputSanitizer()
        with pytest.raises(TypeError):
            s.sanitize_text(123)  # type: ignore[arg-type]

    def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError):
            InputSanitizer(max_type_length=0)
        with pytest.raises(ValueError):
            InputSanitizer(max_title_length=-1)

    def test_log_action_emits_structured_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        s = InputSanitizer()
        sanitized = s.sanitize_text("hi\x00")
        with caplog.at_level("INFO", logger="jarvis.automation.windows_adapter"):
            s.log_action("type_text", sanitized=sanitized, extra_field="abc")
        assert any(
            "windows_adapter" in record.getMessage()
            and getattr(record, "action", None) == "type_text"
            and getattr(record, "extra_field", None) == "abc"
            and getattr(record, "stripped_chars", None) == 1
            for record in caplog.records
        )

    def test_default_limits_match_module_constants(self) -> None:
        s = InputSanitizer()
        assert s.max_type_length == DEFAULT_MAX_TYPE_LENGTH
        assert s.max_title_length == DEFAULT_MAX_TITLE_LENGTH


# ---------------------------------------------------------------------------
# WindowsAdapter — construction and Protocol shape
# ---------------------------------------------------------------------------


class TestWindowsAdapterShape:
    def test_inherits_base_platform_adapter(self) -> None:
        adapter = WindowsAdapter()
        assert isinstance(adapter, BasePlatformAdapter)
        assert adapter.platform_tag == "windows"

    def test_satisfies_platform_adapter_protocol(self) -> None:
        adapter = WindowsAdapter()
        # The Protocol is runtime-checkable; structural conformance must
        # hold so the Skill_Registry can type its dependency on it.
        assert isinstance(adapter, PlatformAdapter)

    def test_application_registry_is_snapshotted(self) -> None:
        registry = {"chrome": "C:/chrome.exe"}
        adapter = WindowsAdapter(application_registry=registry)
        registry["chrome"] = "EVIL"
        assert adapter.application_registry["chrome"] == "C:/chrome.exe"

    def test_pyautogui_pause_validation(self) -> None:
        with pytest.raises(ValueError):
            WindowsAdapter(pyautogui_pause=-1.0)
        with pytest.raises(TypeError):
            WindowsAdapter(pyautogui_pause="0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Target resolution for launch_app
# ---------------------------------------------------------------------------


class TestTargetResolution:
    def test_resolves_registered_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = WindowsAdapter(
            application_registry={"chrome": "C:/Program Files/chrome.exe"}
        )
        resolved = adapter._resolve_target("chrome")
        assert resolved == "C:/Program Files/chrome.exe"

    def test_resolves_uri_unchanged(self) -> None:
        adapter = WindowsAdapter()
        assert adapter._resolve_target("ms-settings:") == "ms-settings:"
        assert adapter._resolve_target("https://example.org") == "https://example.org"

    def test_resolves_windows_path_unchanged(self) -> None:
        adapter = WindowsAdapter()
        assert (
            adapter._resolve_target("C:/Windows/notepad.exe")
            == "C:/Windows/notepad.exe"
        )
        assert (
            adapter._resolve_target("D:\\tools\\app.exe") == "D:\\tools\\app.exe"
        )

    def test_unknown_name_falls_through(self) -> None:
        adapter = WindowsAdapter()
        # No registry entry, no path, no scheme → returned as-is so
        # subprocess can search PATH.
        assert adapter._resolve_target("notepad.exe") == "notepad.exe"

    def test_expands_environment_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_TEST_VAR", "abc")
        adapter = WindowsAdapter(
            application_registry={"app": "C:/tools/%MY_TEST_VAR%/exe"}
        )
        assert adapter._resolve_target("app") == "C:/tools/abc/exe"

    def test_rejects_empty_target(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            adapter._resolve_target("")

    def test_drive_letter_not_treated_as_uri(self) -> None:
        adapter = WindowsAdapter()
        assert adapter._is_uri("C:/Windows") is False
        assert adapter._is_uri("ms-settings:") is True


# ---------------------------------------------------------------------------
# media_key validation (the Win32 call itself is mocked)
# ---------------------------------------------------------------------------


class TestMediaKeyValidation:
    def test_rejects_unknown_key(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(adapter.media_key("rewind"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Volume / brightness clamping
# ---------------------------------------------------------------------------


class TestVolumeClamp:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(-50, 0), (0, 0), (50, 50), (100, 100), (250, 100)],
    )
    def test_clamp_pct_bounds(self, value: int, expected: int) -> None:
        assert WindowsAdapter._clamp_pct(value) == expected

    def test_clamp_pct_rejects_bool(self) -> None:
        with pytest.raises(TypeError):
            WindowsAdapter._clamp_pct(True)

    def test_clamp_pct_rejects_float(self) -> None:
        with pytest.raises(TypeError):
            WindowsAdapter._clamp_pct(50.0)  # type: ignore[arg-type]


class TestAdjustVolumeValidation:
    def test_rejects_non_int(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.adjust_volume("up"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# click / hotkey / focus_window input validation
# ---------------------------------------------------------------------------


class TestClickValidation:
    def test_rejects_non_int_coords(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.click(1.0, 2, "left"))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            asyncio.run(adapter.click(True, 2, "left"))

    def test_rejects_unknown_button(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(adapter.click(1, 2, "centre"))  # type: ignore[arg-type]


class TestHotkeyValidation:
    def test_requires_at_least_one_key(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(adapter.hotkey())

    def test_rejects_non_string_key(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.hotkey("ctrl", 5))  # type: ignore[arg-type]

    def test_rejects_empty_string_key(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.hotkey("ctrl", ""))

    def test_rejects_control_char_in_key(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(adapter.hotkey("ctrl", "\x00"))


class TestFocusWindowValidation:
    def test_rejects_pattern_that_sanitises_to_empty(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(adapter.focus_window("\x00\x01"))


# ---------------------------------------------------------------------------
# notify input validation
# ---------------------------------------------------------------------------


class TestNotifyValidation:
    def test_rejects_non_string_arguments(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.notify(123, "body"))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            asyncio.run(adapter.notify("title", None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# launch_app argument validation and resolution
# ---------------------------------------------------------------------------


class TestLaunchAppArgValidation:
    def test_rejects_non_list_args(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.launch_app("notepad.exe", "x"))  # type: ignore[arg-type]

    def test_rejects_non_string_in_args(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.launch_app("notepad.exe", ["a", 5]))  # type: ignore[list-item]


class TestLaunchAppExecutable:
    """Drives the executable path through a stubbed ``subprocess.Popen``."""

    def test_returns_process_handle_for_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WindowsAdapter()

        fake_proc = MagicMock()
        fake_proc.pid = 4242
        popen_mock = MagicMock(return_value=fake_proc)
        monkeypatch.setattr(subprocess, "Popen", popen_mock)

        handle = asyncio.run(adapter.launch_app("/usr/bin/true", []))
        assert isinstance(handle, ProcessHandle)
        assert handle.pid == 4242
        assert handle.detached is False
        # The command list is forwarded with shell=False.
        args, kwargs = popen_mock.call_args
        assert kwargs["shell"] is False
        assert args[0][0] == "/usr/bin/true"


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------


class TestRunScriptCommandBuilder:
    def test_powershell_command(self) -> None:
        path = Path("C:/x.ps1")
        cmd = WindowsAdapter._build_command("powershell", path)
        assert cmd[0] == "powershell.exe"
        assert "-File" in cmd
        assert cmd[-1] == str(path)

    def test_python_command_uses_current_interpreter(self) -> None:
        path = Path("C:/x.py")
        cmd = WindowsAdapter._build_command("python", path)
        assert cmd[0] == sys.executable
        assert cmd[-1] == str(path)

    def test_batch_command(self) -> None:
        path = Path("C:/x.bat")
        cmd = WindowsAdapter._build_command("batch", path)
        assert cmd[0] == "cmd.exe"
        assert "/c" in cmd
        assert cmd[-1] == str(path)

    def test_unknown_interpreter_raises(self) -> None:
        with pytest.raises(ValueError):
            WindowsAdapter._build_command("ruby", Path("x"))  # type: ignore[arg-type]


class TestRunScriptArgValidation:
    def test_rejects_non_path(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(TypeError):
            asyncio.run(adapter.run_script("python", "x.py", 5))  # type: ignore[arg-type]

    def test_rejects_non_positive_timeout(self) -> None:
        adapter = WindowsAdapter()
        with pytest.raises(ValueError):
            asyncio.run(
                adapter.run_script("python", Path("x.py"), 0)
            )
        with pytest.raises(ValueError):
            asyncio.run(
                adapter.run_script("python", Path("x.py"), -1.0)
            )


class TestRunScriptExecution:
    def test_returns_completed_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WindowsAdapter()

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "ok\n"
        completed.stderr = ""
        run_mock = MagicMock(return_value=completed)
        monkeypatch.setattr(subprocess, "run", run_mock)

        result = asyncio.run(
            adapter.run_script("python", Path("x.py"), 5.0)
        )
        assert isinstance(result, ScriptResult)
        assert result.exit_code == 0
        assert result.stdout == "ok\n"
        assert result.timed_out is False

    def test_timeout_returns_script_result_with_timed_out_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WindowsAdapter()

        def _raise_timeout(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=1.0,
                                            output=b"partial",
                                            stderr=b"")

        monkeypatch.setattr(subprocess, "run", _raise_timeout)

        result = asyncio.run(
            adapter.run_script("python", Path("x.py"), 1.0)
        )
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.stdout == "partial"

    def test_timeout_handles_string_streams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When ``text=True`` is in effect and the process produced no
        # bytes, ``TimeoutExpired`` may carry already-decoded strings
        # (or ``None``). Both paths must produce a valid ScriptResult.
        adapter = WindowsAdapter()

        def _raise_timeout(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=0.5,
                                            output=None, stderr=None)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)

        result = asyncio.run(
            adapter.run_script("python", Path("x.py"), 0.5)
        )
        assert result.timed_out is True
        assert result.stdout == ""
        assert result.stderr == ""


# ---------------------------------------------------------------------------
# Default unsupported behaviour delegates correctly
# ---------------------------------------------------------------------------


class TestUnsupportedRouting:
    """If the WMI brightness call returns no instances we MUST raise
    PlatformNotSupportedError (Requirement 4.8)."""

    def test_set_brightness_raises_when_methods_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WindowsAdapter()
        # Simulate a host where ``WmiMonitorBrightnessMethods()`` returns
        # an empty list (the documented "unsupported" signal).
        monkeypatch.setattr(
            adapter, "_brightness_methods", lambda: ([], [])
        )
        with pytest.raises(PlatformNotSupportedError) as excinfo:
            asyncio.run(adapter.set_brightness(50))
        assert excinfo.value.capability == "set_brightness"
        assert excinfo.value.platform == "windows"

    def test_get_brightness_raises_when_query_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = WindowsAdapter()
        monkeypatch.setattr(
            adapter, "_brightness_methods", lambda: ([], [])
        )
        with pytest.raises(PlatformNotSupportedError):
            asyncio.run(adapter.get_brightness())
