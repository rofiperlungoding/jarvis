"""Windows implementation of :class:`PlatformAdapter`.

This module provides :class:`WindowsAdapter`, the concrete
:class:`~jarvis.automation.platform.PlatformAdapter` that backs every
host-OS-touching Skill on Windows hosts (Requirements 2.2, 2.3, 4.2, 4.4,
4.5, 4.7, 4.8, 9.3, 9.6, 9.7, 9.8, 13.6, 15.1, 15.2). The adapter is the
single integration point for the otherwise scattered Win32 / WMI / COM
plumbing — every other module in the codebase consumes the abstraction
through the Protocol, never the concrete class — which keeps the
"platform abstraction interface" requirement (Requirement 15.2) auditable
in one file.

Architectural notes
-------------------

* **Lazy native imports.** ``pywin32``, ``pycaw``, ``wmi``, ``pyautogui``,
  ``pywinauto``, and ``win10toast`` are imported *inside* the methods that
  use them, never at module load. This is load-bearing: Linux CI runs the
  same test matrix with these wheels absent (``sys_platform == 'win32'``
  markers in :file:`pyproject.toml`), and the module MUST import cleanly
  there so unit tests for :class:`InputSanitizer`, error mapping, and
  helper logic still execute outside Windows.
* **Async surface.** Every public method is :keyword:`async` and offloads
  blocking work via :func:`asyncio.to_thread`. The Win32 calls are
  inherently synchronous, but the Dialog_Manager event loop must remain
  responsive for the audio pipeline (Requirements 1.x), so we never call
  ``ctypes`` / ``subprocess`` / COM directly on the loop thread.
* **Error semantics.** The contract from
  :mod:`jarvis.automation.platform` is preserved: capability-genuinely-
  -unsupported failures (e.g. an external monitor that does not implement
  ``WmiMonitorBrightnessMethods``) raise
  :class:`PlatformNotSupportedError`; transient OS errors propagate
  unchanged for the Skill exception barrier to translate; and
  :meth:`run_script` reports timeouts via ``ScriptResult.timed_out``
  rather than raising.
* **Input sanitisation.** All free-form text destined for ``pyautogui`` /
  ``pywinauto`` flows through :class:`InputSanitizer` first, which
  strips dangerous control characters, caps the length of typed strings,
  and emits a structured action-log record so the audit trail captures
  every UI manipulation the assistant performs (Requirement 13.6).

Validates: Requirements 2.2, 2.3, 4.2, 4.4, 4.5, 4.7, 4.8, 9.3, 9.6, 9.7,
9.8, 13.6, 15.1, 15.2.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Final
import unicodedata

from jarvis.automation.platform import (
    BasePlatformAdapter,
    MediaKey,
    MouseButton,
    ProcessHandle,
    ScriptInterpreter,
    ScriptResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InputSanitizer",
    "SanitizedText",
    "WindowsAdapter",
]


# ---------------------------------------------------------------------------
# Win32 virtual key codes
# ---------------------------------------------------------------------------
# We hard-code the constants instead of importing them from ``win32con``
# so the module imports without ``pywin32``. Values match
# ``WinUser.h``.

_VK_MEDIA_NEXT_TRACK: Final[int] = 0xB0
_VK_MEDIA_PREV_TRACK: Final[int] = 0xB1
_VK_MEDIA_STOP: Final[int] = 0xB2
_VK_MEDIA_PLAY_PAUSE: Final[int] = 0xB3

# ``KEYEVENTF_EXTENDEDKEY`` flags the call as an extended key (the media
# transport keys live on the extended portion of a 101+ keyboard); the
# ``KEYEVENTF_KEYUP`` flag releases the key. Using 0 for the down event
# matches the Win32 ``keybd_event`` documentation.
_KEYEVENTF_EXTENDEDKEY: Final[int] = 0x0001
_KEYEVENTF_KEYUP: Final[int] = 0x0002

_MEDIA_KEY_TO_VK: Final[Mapping[MediaKey, int]] = {
    "play_pause": _VK_MEDIA_PLAY_PAUSE,
    "next": _VK_MEDIA_NEXT_TRACK,
    "prev": _VK_MEDIA_PREV_TRACK,
    "stop": _VK_MEDIA_STOP,
}


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------

#: Maximum length of a single ``type_text`` payload. Anything longer is
#: almost certainly a misuse (the Dialog_Manager rarely needs to type more
#: than a sentence) and risks pinning the keyboard for seconds while
#: ``pyautogui`` types one character at a time. The value is a sane
#: defence-in-depth bound, not a hard product limit; callers that need to
#: paste large blocks should use a clipboard skill instead.
DEFAULT_MAX_TYPE_LENGTH: Final[int] = 10_000

#: Maximum length of a window-title regex / substring. Catches the most
#: common foot-gun (an LLM dumping a whole transcript into a focus window
#: pattern) without preventing legitimate use.
DEFAULT_MAX_TITLE_LENGTH: Final[int] = 512

# A safe whitelist of control characters: the standard whitespace
# code points the user might legitimately want to type.
_ALLOWED_CONTROLS: Final[frozenset[str]] = frozenset({"\t", "\n", "\r"})


@dataclass(frozen=True)
class SanitizedText:
    """Result of running a free-form string through :class:`InputSanitizer`.

    Attributes
    ----------
    text:
        The sanitised payload safe to forward to ``pyautogui`` /
        ``pywinauto``. Always a ``str``; never ``None``.
    truncated:
        ``True`` if :attr:`text` was shortened to fit ``max_length``.
    stripped_chars:
        Count of code points removed because they were considered unsafe
        control / format characters. ``0`` for already-clean inputs.
    original_length:
        Length of the input before any sanitisation. Logged so the audit
        trail can show what was attempted, even when the payload is
        ultimately benign.
    """

    text: str
    truncated: bool
    stripped_chars: int
    original_length: int


class InputSanitizer:
    """Sanitise free-form text before forwarding it to UI automation.

    The Dialog_Manager funnels every ``DesktopAutomationSkill`` payload
    through this class for two related reasons:

    1. **Defence in depth.** ``pyautogui.typewrite`` happily emits ``\\x00``
       bytes and other control characters, which can drive applications
       into surprising states (e.g. opening a context menu via the
       ``Apps`` key, sending Win+R, …). Stripping these protects the user
       from a misheard transcript or an LLM hallucination embedding a
       stray escape.
    2. **Structured audit logging.** Every sanitisation call emits a log
       record with the action kind, a length-prefixed sample, and how
       many characters were stripped, so an operator can trace any UI
       manipulation back to its origin (Requirement 13.6).

    The class is deliberately tiny and stateless so the same instance can
    be shared across the application; it holds nothing more than two
    bounds.
    """

    __slots__ = ("_max_title_length", "_max_type_length")

    def __init__(
        self,
        *,
        max_type_length: int = DEFAULT_MAX_TYPE_LENGTH,
        max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
    ) -> None:
        if not isinstance(max_type_length, int) or max_type_length <= 0:
            raise ValueError("max_type_length must be a positive integer")
        if not isinstance(max_title_length, int) or max_title_length <= 0:
            raise ValueError("max_title_length must be a positive integer")
        self._max_type_length = max_type_length
        self._max_title_length = max_title_length

    @property
    def max_type_length(self) -> int:
        return self._max_type_length

    @property
    def max_title_length(self) -> int:
        return self._max_title_length

    def sanitize_text(self, text: str) -> SanitizedText:
        """Sanitise text destined for :meth:`PlatformAdapter.type_text`.

        Removes Unicode ``Cc`` / ``Cf`` (control / format) characters
        except the standard whitespace set (``\\t``, ``\\n``, ``\\r``)
        and caps the result at :attr:`max_type_length` code points.
        """
        return self._sanitize(text, max_length=self._max_type_length)

    def sanitize_title_pattern(self, pattern: str) -> SanitizedText:
        """Sanitise a window-title pattern for :meth:`focus_window`.

        Same scrubbing rules as :meth:`sanitize_text`, but with a tighter
        length cap appropriate for window titles.
        """
        return self._sanitize(pattern, max_length=self._max_title_length)

    @staticmethod
    def _scrub(text: str) -> tuple[str, int]:
        """Strip Cc/Cf code points (except common whitespace).

        Returns the cleaned string and the number of code points dropped.
        Implemented as a single pass so the cost stays linear in the
        input length even for adversarial payloads.
        """
        out: list[str] = []
        stripped = 0
        for ch in text:
            if ch in _ALLOWED_CONTROLS:
                out.append(ch)
                continue
            cat = unicodedata.category(ch)
            # Cc = control, Cf = format (e.g. RLO, BiDi overrides),
            # Co = private use, Cs = surrogate. All are unsafe to feed to
            # the GUI automation libraries.
            if cat[0] == "C":
                stripped += 1
                continue
            out.append(ch)
        return "".join(out), stripped

    def _sanitize(self, text: str, *, max_length: int) -> SanitizedText:
        if not isinstance(text, str):
            raise TypeError("InputSanitizer expects a str")
        original_length = len(text)
        cleaned, stripped = self._scrub(text)
        truncated = False
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            truncated = True
        return SanitizedText(
            text=cleaned,
            truncated=truncated,
            stripped_chars=stripped,
            original_length=original_length,
        )

    def log_action(
        self,
        action: str,
        *,
        sanitized: SanitizedText | None = None,
        **fields: Any,
    ) -> None:
        """Emit a structured ``windows_adapter`` action log record.

        Records the action kind, any sanitisation metadata, and arbitrary
        additional fields the caller cares to attach (coordinates, button
        names, script ids …). The structured fields are passed via the
        ``extra`` mapping so log handlers that consume JSON (the design's
        ``LogRedactionFilter`` chain) keep them as separate keys rather
        than fold everything into the message string.
        """
        extra: dict[str, Any] = {"action": action, **fields}
        if sanitized is not None:
            extra.update(
                {
                    "input_length": sanitized.original_length,
                    "kept_length": len(sanitized.text),
                    "stripped_chars": sanitized.stripped_chars,
                    "truncated": sanitized.truncated,
                }
            )
        logger.info("windows_adapter action=%s", action, extra=extra)


# ---------------------------------------------------------------------------
# WindowsAdapter
# ---------------------------------------------------------------------------


# A URI-like target is a non-Windows-drive scheme followed by ``:``. We
# carve out drive letters (``C:\``, ``D:/``) so absolute paths don't get
# mistaken for URIs.
_URI_SCHEME_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9+.\-]+:"
)
_WINDOWS_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z]:[\\/]"
)


@dataclass
class WindowsAdapter(BasePlatformAdapter):
    """Concrete Windows implementation of :class:`PlatformAdapter`.

    Parameters
    ----------
    sanitizer:
        Optional :class:`InputSanitizer` instance. Defaults to a fresh
        sanitizer with the module-level limits. Sharing a sanitizer
        across adapters is supported and encouraged for tests that want
        to inspect logged actions.
    application_registry:
        Optional mapping of registered application names to executable
        paths or URIs (matches ``[automation.application_registry]`` in
        ``default.toml``). The :class:`~jarvis.skills.builtin.LaunchAppSkill`
        normally resolves the registry before calling :meth:`launch_app`,
        but the adapter accepts the same lookup for callers (and tests)
        that pass a registered name straight through.
    pyautogui_pause:
        Optional override for ``pyautogui.PAUSE`` — the inter-action
        sleep applied by ``pyautogui``. Defaults to ``0.0`` so the
        Dialog_Manager does not feel artificially sluggish; tests can set
        a positive value to verify rate-limiting behaviour without
        relying on real wall-clock delays.
    """

    sanitizer: InputSanitizer = field(default_factory=InputSanitizer)
    application_registry: Mapping[str, str] = field(default_factory=dict)
    pyautogui_pause: float = 0.0

    #: Tag baked into :class:`PlatformNotSupportedError` for diagnostic
    #: messages. Required by the contract in
    #: :class:`BasePlatformAdapter`.
    platform_tag: str = "windows"

    def __post_init__(self) -> None:
        # Snapshot the registry so callers cannot retroactively mutate
        # adapter state. Done in ``__post_init__`` because the dataclass
        # is not frozen (we want to allow tests to override
        # :attr:`pyautogui_pause` after construction).
        self.application_registry = dict(self.application_registry)
        if not isinstance(self.pyautogui_pause, (int, float)):
            raise TypeError("pyautogui_pause must be a number")
        if self.pyautogui_pause < 0:
            raise ValueError("pyautogui_pause must be non-negative")

    # ----------------------------------------------------------------- helpers

    def _resolve_target(self, target: str) -> str:
        """Resolve ``target`` against the application registry, if applicable.

        Returns the registry value when ``target`` is a registered name;
        otherwise returns ``target`` unchanged. Path / URI strings always
        fall through unmodified so callers that already resolved through
        a higher-level skill don't double-resolve.
        """
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        # A registered name is exactly the literal key — never a path
        # nor a URI, so we only consult the registry when neither pattern
        # matches. ``os.path.expandvars`` is applied to registry values
        # so entries like ``%USERNAME%`` work out of the box.
        if self._is_uri(target) or self._is_windows_path(target):
            return os.path.expandvars(target)
        if target in self.application_registry:
            return os.path.expandvars(self.application_registry[target])
        # Fall through: subprocess will search PATH at launch time.
        return os.path.expandvars(target)

    @staticmethod
    def _is_uri(target: str) -> bool:
        if _WINDOWS_PATH_RE.match(target):
            return False
        return bool(_URI_SCHEME_RE.match(target))

    @staticmethod
    def _is_windows_path(target: str) -> bool:
        return bool(_WINDOWS_PATH_RE.match(target)) or target.startswith(("\\\\", "/"))

    # ----------------------------------------------------------------- launch

    async def launch_app(
        self, executable_or_uri: str, args: list[str]
    ) -> ProcessHandle:
        """Launch an executable, registered app, or URI handler."""
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise TypeError("args must be a list[str]")
        resolved = self._resolve_target(executable_or_uri)
        is_uri = self._is_uri(resolved)
        self.sanitizer.log_action(
            "launch_app",
            target=resolved,
            kind="uri" if is_uri else "command",
            arg_count=len(args),
        )

        def _spawn() -> tuple[int, bool]:
            if is_uri:
                # ``os.startfile`` is the canonical Windows way to invoke
                # a URI handler (or a document). It returns ``None`` and
                # detaches, so we surface ``pid=0`` + ``detached=True``.
                # Wrapped in a hasattr check so the module is still
                # importable on non-Windows even if a test exercises
                # ``launch_app`` (which it shouldn't, but the failure
                # mode should be obvious).
                if hasattr(os, "startfile"):
                    os.startfile(resolved, "open")
                    return 0, True
                # Fallback: use ``cmd /c start`` via subprocess. The
                # empty title argument prevents ``start`` from treating
                # the first quoted string as a window title.
                proc = subprocess.Popen(
                    ["cmd", "/c", "start", "", resolved, *args],
                    shell=False,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                )
                return proc.pid, True
            # Plain executable. We deliberately use the list form (no
            # ``shell=True``) so Skill arguments can never be folded into
            # a shell command line.
            proc = subprocess.Popen(
                [resolved, *args],
                shell=False,
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
            )
            return proc.pid, False

        try:
            pid, detached = await asyncio.to_thread(_spawn)
        except FileNotFoundError as exc:
            # Surface as the standard "not found" error; the LaunchAppSkill
            # maps this onto the Error Taxonomy.
            raise FileNotFoundError(
                f"could not launch {resolved!r}: {exc}"
            ) from exc
        return ProcessHandle(
            pid=pid,
            executable_or_uri=resolved,
            detached=detached,
        )

    # ----------------------------------------------------------------- media

    async def media_key(self, key: MediaKey) -> None:
        """Press a transport-control media key via Win32 ``keybd_event``."""
        if key not in _MEDIA_KEY_TO_VK:
            # Defensive: ``MediaKey`` is a ``Literal``, but the Skill
            # exception barrier still needs a clear runtime error if a
            # malformed payload escapes schema validation.
            raise ValueError(f"unsupported media key: {key!r}")
        vk = _MEDIA_KEY_TO_VK[key]
        self.sanitizer.log_action("media_key", key=key, vk=vk)

        def _press() -> None:
            # Lazy import — module must remain importable on Linux CI.
            import ctypes  # noqa: PLC0415

            user32 = ctypes.windll.user32
            # Press, then release, the same key. Without the release,
            # Windows will queue a stuck key state for the next
            # ``keybd_event`` caller.
            user32.keybd_event(vk, 0, _KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(
                vk, 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0
            )

        await asyncio.to_thread(_press)

    # ----------------------------------------------------------------- volume

    @staticmethod
    def _clamp_pct(value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("level/delta must be an int")
        return max(0, min(100, value))

    def _audio_endpoint(self) -> Any:
        """Activate the master audio endpoint via pycaw / COM.

        Imported lazily so the module is loadable on non-Windows.
        """
        from comtypes import CLSCTX_ALL  # noqa: PLC0415
        from pycaw.pycaw import (  # noqa: PLC0415
            AudioUtilities,
            IAudioEndpointVolume,
        )

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return interface.QueryInterface(IAudioEndpointVolume)

    async def set_volume(self, level_pct: int) -> None:
        """Set master output volume to ``level_pct`` percent."""
        clamped = self._clamp_pct(level_pct)
        self.sanitizer.log_action(
            "set_volume", level_pct=clamped, requested=level_pct
        )

        def _set() -> None:
            volume = self._audio_endpoint()
            # ``SetMasterVolumeLevelScalar`` takes a float in [0.0, 1.0].
            volume.SetMasterVolumeLevelScalar(clamped / 100.0, None)

        await asyncio.to_thread(_set)

    async def adjust_volume(self, delta_pct: int) -> None:
        """Adjust master output volume by ``delta_pct`` percentage points."""
        if not isinstance(delta_pct, int) or isinstance(delta_pct, bool):
            raise TypeError("delta_pct must be an int")
        self.sanitizer.log_action("adjust_volume", delta_pct=delta_pct)

        def _adjust() -> None:
            volume = self._audio_endpoint()
            current_scalar = float(volume.GetMasterVolumeLevelScalar())
            current_pct = round(current_scalar * 100)
            target_pct = max(0, min(100, current_pct + delta_pct))
            volume.SetMasterVolumeLevelScalar(target_pct / 100.0, None)

        await asyncio.to_thread(_adjust)

    # ------------------------------------------------------------- brightness

    def _brightness_methods(self) -> tuple[Any, Any]:
        """Return ``(brightness_query, brightness_methods)`` from WMI.

        Either side may be empty if the active monitor's driver does not
        implement the WMI brightness contract; callers must handle that.
        """
        import wmi  # noqa: PLC0415

        wmi_iface = wmi.WMI(namespace="wmi")
        brightness = wmi_iface.WmiMonitorBrightness()
        methods = wmi_iface.WmiMonitorBrightnessMethods()
        return brightness, methods

    async def get_brightness(self) -> int:
        """Return the active display's current brightness in ``[0, 100]``."""
        self.sanitizer.log_action("get_brightness")

        def _read() -> int:
            brightness, _ = self._brightness_methods()
            if not brightness:
                raise self._unsupported(
                    "get_brightness",
                    detail="WmiMonitorBrightness returned no instances",
                )
            level = int(brightness[0].CurrentBrightness)
            return max(0, min(100, level))

        return await asyncio.to_thread(_read)

    async def set_brightness(self, level_pct: int) -> None:
        """Set the active display's brightness, raising on unsupported displays."""
        clamped = self._clamp_pct(level_pct)
        self.sanitizer.log_action(
            "set_brightness", level_pct=clamped, requested=level_pct
        )

        def _set() -> None:
            _, methods = self._brightness_methods()
            if not methods:
                raise self._unsupported(
                    "set_brightness",
                    detail="WmiMonitorBrightnessMethods missing",
                )
            # ``WmiSetBrightness`` signature: (Brightness, Timeout) where
            # Timeout=0 means "apply immediately and persist".
            methods[0].WmiSetBrightness(clamped, 0)

        await asyncio.to_thread(_set)

    # --------------------------------------------------------------- notify

    async def notify(self, title: str, body: str) -> None:
        """Show a toast notification via ``win10toast``."""
        if not isinstance(title, str) or not isinstance(body, str):
            raise TypeError("title and body must be strings")
        # Sanitise to keep stray control characters out of the toast and
        # to log a structured action record.
        sanitized_title = self.sanitizer.sanitize_title_pattern(title)
        sanitized_body = self.sanitizer.sanitize_text(body)
        self.sanitizer.log_action(
            "notify",
            title_length=sanitized_title.original_length,
            body_length=sanitized_body.original_length,
        )

        def _show() -> None:
            from win10toast import ToastNotifier  # noqa: PLC0415

            toaster = ToastNotifier()
            # ``threaded=True`` returns immediately; we still wrap the
            # whole call in ``to_thread`` so any one-shot setup cost
            # stays off the event loop.
            toaster.show_toast(
                sanitized_title.text,
                sanitized_body.text,
                duration=5,
                threaded=True,
            )

        await asyncio.to_thread(_show)

    # ----------------------------------------------------------------- input

    def _configure_pyautogui(self) -> Any:
        """Import ``pyautogui`` lazily and apply our pause override."""
        import pyautogui  # noqa: PLC0415

        # ``PAUSE`` is a module-level global. Setting it on every call is
        # cheap and lets tests override the value mid-flight.
        pyautogui.PAUSE = float(self.pyautogui_pause)
        # ``FAILSAFE`` defaults to True (mouse to top-left aborts). We
        # leave it on; the assistant should never fight that.
        return pyautogui

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        """Click ``button`` at absolute screen coordinates ``(x, y)``."""
        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or not isinstance(y, int)
            or isinstance(y, bool)
        ):
            raise TypeError("x and y must be integers")
        if button not in ("left", "right", "middle"):
            raise ValueError(f"unsupported mouse button: {button!r}")
        self.sanitizer.log_action("click", x=x, y=y, button=button)

        def _click() -> None:
            pyautogui = self._configure_pyautogui()
            pyautogui.click(x=x, y=y, button=button)

        await asyncio.to_thread(_click)

    async def type_text(self, text: str) -> None:
        """Type ``text`` into the focused control, after sanitisation."""
        sanitized = self.sanitizer.sanitize_text(text)
        self.sanitizer.log_action("type_text", sanitized=sanitized)

        def _type() -> None:
            pyautogui = self._configure_pyautogui()
            # ``typewrite`` only handles ASCII reliably; for the broader
            # Unicode range we fall back to ``write`` (which uses the
            # clipboard). pyautogui exposes ``write`` as an alias on
            # all supported versions; we guard with ``getattr`` so any
            # future API change degrades to a clear AttributeError.
            writer = getattr(pyautogui, "write", pyautogui.typewrite)
            writer(sanitized.text, interval=0.0)

        await asyncio.to_thread(_type)

    async def hotkey(self, *keys: str) -> None:
        """Press ``keys`` as a chord (e.g. ``"ctrl"``, ``"c"``)."""
        if not keys:
            raise ValueError("hotkey requires at least one key")
        if not all(isinstance(k, str) and k for k in keys):
            raise TypeError("all hotkey keys must be non-empty strings")
        # Defence in depth: keys themselves should never contain control
        # characters. Build a sanitised tuple and reject if anything was
        # stripped — a hotkey with embedded control bytes is unambiguously
        # an attack rather than a typo.
        cleaned: list[str] = []
        for key in keys:
            scrubbed, stripped = InputSanitizer._scrub(key)
            if stripped or len(scrubbed) > 32:
                raise ValueError(f"unsafe hotkey key: {key!r}")
            cleaned.append(scrubbed)
        self.sanitizer.log_action("hotkey", keys=tuple(cleaned))

        def _press() -> None:
            pyautogui = self._configure_pyautogui()
            pyautogui.hotkey(*cleaned)

        await asyncio.to_thread(_press)

    async def focus_window(self, title_pattern: str) -> None:
        """Focus the topmost window whose title matches ``title_pattern``."""
        sanitized = self.sanitizer.sanitize_title_pattern(title_pattern)
        if not sanitized.text:
            raise ValueError("title_pattern must not be empty after sanitisation")
        self.sanitizer.log_action("focus_window", sanitized=sanitized)

        def _focus() -> None:
            from pywinauto import Desktop  # noqa: PLC0415

            window = Desktop(backend="uia").window(title_re=sanitized.text)
            # ``set_focus`` raises if the window is not present;
            # ``ElementNotFoundError`` is a pywinauto-specific exception
            # that the Skill barrier translates into ``internal_error``.
            window.set_focus()

        await asyncio.to_thread(_focus)

    # ------------------------------------------------------------- run_script

    @staticmethod
    def _build_command(
        interpreter: ScriptInterpreter, script_path: Path
    ) -> list[str]:
        if interpreter == "powershell":
            return [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
        if interpreter == "python":
            # Use the same interpreter that's running JARVIS so
            # third-party imports inside the script resolve via the
            # same site-packages.
            return [sys.executable, str(script_path)]
        if interpreter == "batch":
            return ["cmd.exe", "/c", str(script_path)]
        raise ValueError(f"unsupported interpreter: {interpreter!r}")

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        """Execute ``script_path`` with ``interpreter`` under ``timeout_s``."""
        if not isinstance(script_path, Path):
            raise TypeError("script_path must be a pathlib.Path")
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        cmd = self._build_command(interpreter, script_path)
        self.sanitizer.log_action(
            "run_script",
            interpreter=interpreter,
            script_path=str(script_path),
            timeout_s=float(timeout_s),
        )

        def _run() -> ScriptResult:
            start = time.monotonic()
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return ScriptResult(
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr,
                    duration_ms=duration_ms,
                    timed_out=True,
                )
            duration_ms = int((time.monotonic() - start) * 1000)
            return ScriptResult(
                exit_code=int(completed.returncode),
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                duration_ms=duration_ms,
                timed_out=False,
            )

        return await asyncio.to_thread(_run)
