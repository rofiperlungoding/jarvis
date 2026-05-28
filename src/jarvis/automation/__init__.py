"""Automation_Service package.

Re-exports the cross-platform :class:`PlatformAdapter` Protocol and its
helper value types so callers can ``from jarvis.automation import
PlatformAdapter`` without reaching into :mod:`jarvis.automation.platform`
directly.
"""

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    MediaKey,
    MouseButton,
    PlatformAdapter,
    PlatformNotSupportedError,
    ProcessHandle,
    ScriptInterpreter,
    ScriptResult,
)
from jarvis.automation.scripts import (
    DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ScriptCatalog,
)

__all__ = [
    "DEFAULT_SCRIPT_TIMEOUT_SECONDS",
    "PLATFORM_NOT_SUPPORTED",
    "BasePlatformAdapter",
    "MediaKey",
    "MouseButton",
    "PlatformAdapter",
    "PlatformNotSupportedError",
    "ProcessHandle",
    "ScriptCatalog",
    "ScriptInterpreter",
    "ScriptResult",
]
