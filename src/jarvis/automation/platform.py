"""Platform abstraction for the Automation_Service.

This module defines :class:`PlatformAdapter`, the OS-agnostic Protocol that
the built-in Skills (``LaunchAppSkill``, ``MediaControlSkill``,
``VolumeSkill``, ``BrightnessSkill``, ``DesktopAutomationSkill``,
``RunScriptSkill``, ``ReminderService``'s notifier, …) call into for any
side effect that touches the host OS. It mirrors the shape sketched in
``design.md §Automation_Service``:

* one runtime-checkable :class:`Protocol` listing every capability the
  Skills depend on (``launch_app``, ``media_key``, ``set_volume``,
  ``adjust_volume``, ``get_brightness``, ``set_brightness``, ``notify``,
  ``click``, ``type_text``, ``hotkey``, ``focus_window``, ``run_script``);
* small, frozen value types (:class:`ProcessHandle`, :class:`ScriptResult`)
  that flow back to the Skills;
* a default :class:`BasePlatformAdapter` whose every method raises
  :class:`PlatformNotSupportedError`, so concrete adapters such as
  :class:`~jarvis.automation.windows_adapter.WindowsAdapter` only have to
  override the methods they actually implement (Requirement 15.4).

Why a Protocol *and* a base class
---------------------------------

The Protocol is the contract the rest of the codebase imports (Skills are
typed against ``PlatformAdapter``, never against the concrete adapter).
The base class is a convenience for adapter implementers and for tests
— it lets a test build a "mostly unsupported" stub by overriding one or
two methods. A class deriving from :class:`BasePlatformAdapter`
automatically satisfies the Protocol because the Protocol is
:func:`runtime_checkable` and structural.

Error reporting contract
------------------------

When a capability is unavailable on the current platform — either because
the adapter is an unrelated platform (``WindowsAdapter`` on Linux) or
because the underlying API is missing (e.g. an external monitor that
ignores ``WmiMonitorBrightnessMethods``) — the adapter raises
:class:`PlatformNotSupportedError`. The skill wrapping the call catches
this exception and converts it into
``SkillResult.error("platform_not_supported", ...)``, matching the
error taxonomy in ``design.md`` (Requirements 4.8, 15.4, and the
``platform_not_supported`` row of the error table).

The ``"platform_not_supported"`` literal on
:attr:`PlatformNotSupportedError.error_code` is shared with the
``SkillErrorCode`` enum in :mod:`jarvis.skills.base` to keep the two
sides of the boundary in sync.

Validates: Requirements 15.2, 15.3, 15.4
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable

__all__ = [
    "PLATFORM_NOT_SUPPORTED",
    "BasePlatformAdapter",
    "MediaKey",
    "MouseButton",
    "PlatformAdapter",
    "PlatformNotSupportedError",
    "ProcessHandle",
    "ScriptInterpreter",
    "ScriptResult",
]


# ---------------------------------------------------------------------------
# Shared literals
# ---------------------------------------------------------------------------

#: Error code returned to the Dialog_Manager whenever a platform capability
#: is unavailable. The literal is intentionally duplicated from
#: :data:`jarvis.skills.base.SkillErrorCode` instead of imported, both to
#: avoid a circular dependency (``skills`` depends on ``automation`` via the
#: Skill executors) and because the value is part of the public on-the-wire
#: contract; if the string ever changes, the change must be coordinated
#: across both modules deliberately.
PLATFORM_NOT_SUPPORTED: Final[Literal["platform_not_supported"]] = "platform_not_supported"

#: Closed set of media-key actions the Dialog_Manager / MediaControlSkill
#: may request. Mirrors ``design.md §Automation_Service``. Using a
#: :class:`Literal` lets static type-checkers refuse mistyped strings at
#: skill-implementation time rather than relying on the per-method runtime
#: validation that ``WindowsAdapter`` performs.
MediaKey = Literal["play_pause", "next", "prev", "stop"]

#: Mouse-button selector accepted by :meth:`PlatformAdapter.click`. The
#: values map cleanly onto pyautogui's button names on Windows and onto
#: equivalent constants on macOS / Linux when those adapters are written.
MouseButton = Literal["left", "right", "middle"]

#: Interpreter selector accepted by :meth:`PlatformAdapter.run_script`,
#: matching the values defined in the script catalog
#: (``design.md §Automation_Service > script catalog``).
ScriptInterpreter = Literal["powershell", "python", "batch"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlatformNotSupportedError(RuntimeError):
    """Raised by adapters when a requested capability is unavailable.

    Skills wrapping a :class:`PlatformAdapter` call SHOULD catch this
    exception and translate it into ``SkillResult.error(error_code, ...)``
    where ``error_code`` is :data:`PLATFORM_NOT_SUPPORTED`. The exception
    carries the same string on :attr:`error_code` so call sites that
    bridge the two boundaries can pattern-match without re-deriving the
    literal.

    The optional ``capability`` and ``platform`` fields are surfaced to
    the user verbatim by the Dialog_Manager when present, so they MUST
    NOT contain credentials or other sensitive data.
    """

    error_code: Final[Literal["platform_not_supported"]] = PLATFORM_NOT_SUPPORTED

    def __init__(
        self,
        capability: str,
        *,
        platform: str | None = None,
        detail: str | None = None,
    ) -> None:
        if not isinstance(capability, str) or not capability:
            raise ValueError("PlatformNotSupportedError.capability must be a non-empty string")
        if platform is not None and not isinstance(platform, str):
            raise TypeError("PlatformNotSupportedError.platform must be a string")
        if detail is not None and not isinstance(detail, str):
            raise TypeError("PlatformNotSupportedError.detail must be a string")
        self.capability: str = capability
        self.platform: str | None = platform
        self.detail: str | None = detail
        bits: list[str] = [f"capability {capability!r} is not supported"]
        if platform:
            bits.append(f"on platform {platform!r}")
        if detail:
            bits.append(f"({detail})")
        super().__init__(" ".join(bits))


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessHandle:
    """Reference to a process spawned by :meth:`PlatformAdapter.launch_app`.

    The handle is intentionally minimal: the assistant rarely needs to
    interact with the spawned process beyond knowing whether the spawn
    succeeded. A dedicated value type (rather than the raw OS pid)
    exists so future adapters that wrap, e.g., a sandboxed process or a
    UWP app activation can attach extra metadata without breaking
    existing call sites.

    Attributes
    ----------
    pid:
        Operating-system process id. Negative values are forbidden so
        ``pid == -1`` cannot be confused with a "process detached"
        sentinel; adapters that genuinely cannot determine the pid (e.g.
        URI-handler activations on some Windows configurations) MUST
        report ``pid=0`` and set :attr:`detached` to ``True``.
    executable_or_uri:
        The string the caller passed to ``launch_app``. Echoed back
        verbatim so audit-log entries and skill responses can refer to it
        without holding a reference to the original Tool_Call arguments.
    detached:
        ``True`` when the adapter could not (or chose not to) keep a
        process handle around — typical for ``shell=True`` URI
        activations on Windows.
    metadata:
        Open-ended, immutable mapping the adapter MAY use to surface
        extra diagnostic data (window title, app id, …). Empty by
        default so the type stays cheap to construct in the common case.
    """

    pid: int
    executable_or_uri: str
    detached: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pid, int) or isinstance(self.pid, bool):
            raise TypeError("ProcessHandle.pid must be an int")
        if self.pid < 0:
            raise ValueError("ProcessHandle.pid must be non-negative")
        if not isinstance(self.executable_or_uri, str) or not self.executable_or_uri:
            raise ValueError("ProcessHandle.executable_or_uri must be a non-empty string")
        if not isinstance(self.detached, bool):
            raise TypeError("ProcessHandle.detached must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ProcessHandle.metadata must be a Mapping[str, str]")
        # Dataclass equality compares dict contents, but a mutable dict
        # would let callers retroactively mutate a "frozen" instance via
        # the metadata reference. Snapshot into an immutable dict.
        # ``object.__setattr__`` is required because the dataclass is
        # frozen.
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ScriptResult:
    """Outcome of :meth:`PlatformAdapter.run_script`.

    The fields mirror what :class:`subprocess.CompletedProcess` exposes,
    plus a measured ``duration_ms`` and a ``timed_out`` flag set when the
    interpreter did not finish within the budget the caller passed to
    ``run_script``. The Skill wrapping this adapter call (``RunScriptSkill``)
    consults ``timed_out`` to surface ``SkillResult.error("timeout", ...)``
    per the error taxonomy.

    Attributes
    ----------
    exit_code:
        Process exit code. ``0`` indicates success by convention, but the
        Skill is free to consider a non-zero exit a failure depending on
        the script catalog entry.
    stdout / stderr:
        Captured text streams, decoded with the system's preferred
        encoding by the adapter. Empty strings are valid and distinct
        from ``None``; we never use ``None`` here so consumers can treat
        the output uniformly.
    duration_ms:
        Wall-clock execution time. Always non-negative; reported even on
        timeout so audit-log entries can record how long the script ran
        before being killed.
    timed_out:
        ``True`` if the adapter killed the process because the run
        exceeded the requested ``timeout_s``; the Skill maps this to
        ``"timeout"`` rather than ``"internal_error"``.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("ScriptResult.exit_code must be an int")
        if not isinstance(self.stdout, str):
            raise TypeError("ScriptResult.stdout must be a string")
        if not isinstance(self.stderr, str):
            raise TypeError("ScriptResult.stderr must be a string")
        if not isinstance(self.duration_ms, int) or isinstance(self.duration_ms, bool):
            raise TypeError("ScriptResult.duration_ms must be an int")
        if self.duration_ms < 0:
            raise ValueError("ScriptResult.duration_ms must be non-negative")
        if not isinstance(self.timed_out, bool):
            raise TypeError("ScriptResult.timed_out must be a bool")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PlatformAdapter(Protocol):
    """Cross-platform interface for OS-level side effects.

    Every method is :keyword:`async` so the Dialog_Manager can run several
    Skills concurrently without one blocking call (e.g. a slow brightness
    WMI roundtrip) stalling the audio loop. Adapters that delegate to a
    blocking library SHOULD off-load the call to a thread pool via
    :func:`asyncio.to_thread` rather than blocking the event loop.

    Failure semantics
    -----------------

    * Capability genuinely unavailable on the current platform → raise
      :class:`PlatformNotSupportedError`. The Skill maps this to
      ``SkillResult.error("platform_not_supported", ...)``.
    * Transient OS-level failure → raise the underlying ``OSError`` /
      adapter-specific exception unchanged. The Skill registry's
      exception barrier converts these into
      ``SkillResult.error("internal_error", ...)``.
    * ``run_script`` exceeding its time budget → return a
      :class:`ScriptResult` with ``timed_out=True`` rather than raising;
      the Skill translates this into ``SkillResult.error("timeout", ...)``.

    The Skills layer never inspects exception types beyond
    :class:`PlatformNotSupportedError`, so adapters can choose the most
    natural exception type for their backend without breaking callers.
    """

    async def launch_app(self, executable_or_uri: str, args: list[str]) -> ProcessHandle:
        """Launch the given executable, URI handler, or registered app.

        ``executable_or_uri`` may be an absolute path, a Windows
        ``ms-`` URI (e.g. ``ms-settings:``), or a registered application
        name resolved via the configured ``automation.application_registry``.
        ``args`` is forwarded verbatim to the spawned process.
        """
        ...

    async def media_key(self, key: MediaKey) -> None:
        """Press a transport-control media key (play/pause/next/prev/stop)."""
        ...

    async def set_volume(self, level_pct: int) -> None:
        """Set the master output volume, clamped to ``[0, 100]``."""
        ...

    async def adjust_volume(self, delta_pct: int) -> None:
        """Adjust the master output volume by ``delta_pct`` percentage points."""
        ...

    async def get_brightness(self) -> int:
        """Return the active display brightness as an integer in ``[0, 100]``."""
        ...

    async def set_brightness(self, level_pct: int) -> None:
        """Set the active display brightness, clamped to ``[0, 100]``."""
        ...

    async def notify(self, title: str, body: str) -> None:
        """Show a non-blocking system notification with ``title`` / ``body``."""
        ...

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        """Click ``button`` at the absolute screen coordinates ``(x, y)``."""
        ...

    async def type_text(self, text: str) -> None:
        """Type ``text`` into the focused control as if entered by keyboard."""
        ...

    async def hotkey(self, *keys: str) -> None:
        """Press the given keys as a chord (e.g. ``"ctrl"``, ``"c"``)."""
        ...

    async def focus_window(self, title_pattern: str) -> None:
        """Focus the topmost window whose title matches ``title_pattern``."""
        ...

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        """Execute ``script_path`` with ``interpreter`` under a timeout.

        Returns a :class:`ScriptResult` whose ``timed_out`` is ``True`` if
        the process was killed for exceeding ``timeout_s``.
        """
        ...


# ---------------------------------------------------------------------------
# Default adapter
# ---------------------------------------------------------------------------


class BasePlatformAdapter:
    """Default :class:`PlatformAdapter` implementation that supports nothing.

    Concrete adapters (``WindowsAdapter`` today; ``MacAdapter`` /
    ``LinuxAdapter`` later — Requirement 15.3) inherit from this class and
    override only the methods the underlying OS exposes. Every method that
    is not overridden raises :class:`PlatformNotSupportedError`, which the
    Skill layer translates into the user-visible
    ``platform_not_supported`` error code (Requirement 15.4).

    Subclasses MAY also call into the base implementation to deliberately
    surface the same error from a partially-supported method — for
    example, ``WindowsAdapter.set_brightness`` falls back to
    ``raise PlatformNotSupportedError("set_brightness", ...)`` (via
    :meth:`_unsupported`) when the WMI call returns an HRESULT indicating
    the monitor does not implement the WMI brightness method.
    """

    #: Platform tag included in :class:`PlatformNotSupportedError` raised
    #: by the default implementations. Subclasses override this with the
    #: tag of the concrete platform (``"windows"``, ``"darwin"``,
    #: ``"linux"``) so the error message points at the right adapter.
    platform_tag: str = "unknown"

    def _unsupported(
        self,
        capability: str,
        *,
        detail: str | None = None,
    ) -> PlatformNotSupportedError:
        """Build a :class:`PlatformNotSupportedError` tagged with this adapter.

        Subclasses can call this from a partially-implemented method to
        signal that *this particular invocation* is not supported (for
        example, brightness on a monitor whose driver lacks the WMI
        method) without having to remember the platform tag and the
        error literal.
        """
        return PlatformNotSupportedError(capability, platform=self.platform_tag, detail=detail)

    # Each method below is ``async`` and immediately raises so that
    # subclasses overriding only a subset still satisfy the awaitable
    # contract of the Protocol. We deliberately keep parameter names
    # matching the Protocol so static type-checkers report mismatches
    # when subclasses change names.

    async def launch_app(self, executable_or_uri: str, args: list[str]) -> ProcessHandle:
        raise self._unsupported(
            "launch_app",
            detail=f"executable_or_uri={executable_or_uri!r}",
        )

    async def media_key(self, key: MediaKey) -> None:
        raise self._unsupported("media_key", detail=f"key={key!r}")

    async def set_volume(self, level_pct: int) -> None:
        raise self._unsupported("set_volume", detail=f"level_pct={level_pct}")

    async def adjust_volume(self, delta_pct: int) -> None:
        raise self._unsupported("adjust_volume", detail=f"delta_pct={delta_pct}")

    async def get_brightness(self) -> int:
        raise self._unsupported("get_brightness")

    async def set_brightness(self, level_pct: int) -> None:
        raise self._unsupported("set_brightness", detail=f"level_pct={level_pct}")

    async def notify(self, title: str, body: str) -> None:
        raise self._unsupported("notify", detail=f"title={title!r}")

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        raise self._unsupported("click", detail=f"x={x}, y={y}, button={button!r}")

    async def type_text(self, text: str) -> None:
        raise self._unsupported("type_text", detail=f"len(text)={len(text)}")

    async def hotkey(self, *keys: str) -> None:
        raise self._unsupported("hotkey", detail=f"keys={keys!r}")

    async def focus_window(self, title_pattern: str) -> None:
        raise self._unsupported("focus_window", detail=f"title_pattern={title_pattern!r}")

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        raise self._unsupported(
            "run_script",
            detail=f"interpreter={interpreter!r}, script_path={script_path!s}",
        )
