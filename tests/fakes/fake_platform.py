"""In-memory :class:`PlatformAdapter` recording every call.

This fake replaces :class:`~jarvis.automation.windows_adapter.WindowsAdapter`
in skill / dialog integration tests. It satisfies the
:class:`~jarvis.automation.platform.PlatformAdapter` Protocol structurally
by inheriting :class:`~jarvis.automation.platform.BasePlatformAdapter` and
overriding every capability with a recorder that:

* appends a typed record to one of the public ``*_calls`` lists so tests
  can assert exact side-effect arguments without hitting any real OS API
  (Requirement 15.2, "Automation_Service SHALL isolate Windows-specific
  calls behind a platform abstraction interface");
* returns a sensible default value (``ProcessHandle`` for ``launch_app``,
  ``50`` for ``get_brightness``, an ``exit_code=0`` :class:`ScriptResult`
  for ``run_script``) so call sites can exercise the success path without
  configuring anything (Requirement 15.3, "the corresponding Skills SHALL
  function on those platforms without changes");
* honours per-method ``force_unsupported`` toggles so tests can simulate
  the ``platform_not_supported`` branch on any capability (Requirement
  15.4) and per-method ``force_error`` toggles so tests can drive the
  Skill_Registry's exception barrier without subclassing.

The fake is intentionally synchronous in its bookkeeping so tests can read
``adapter.launch_calls`` immediately after ``await adapter.launch_app(...)``
returns; the underlying methods stay :keyword:`async` to match the
Protocol contract.

Validates: Requirements 15.2, 15.3, 15.4
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pytest

from jarvis.automation.platform import (
    BasePlatformAdapter,
    MediaKey,
    MouseButton,
    ProcessHandle,
    ScriptInterpreter,
    ScriptResult,
)

__all__ = [
    "FAKE_PLATFORM_TAG",
    "BrightnessCall",
    "ClickCall",
    "FakePlatformAdapter",
    "LaunchCall",
    "RunScriptCall",
    "all_method_names",
    "fake_platform_adapter",
]


# ---------------------------------------------------------------------------
# Constants and call-record dataclasses
# ---------------------------------------------------------------------------


#: Platform tag attached to :class:`PlatformNotSupportedError` raised by the
#: fake. Distinct from ``"windows" | "darwin" | "linux"`` so error messages
#: clearly point at the test double rather than a real adapter.
FAKE_PLATFORM_TAG: Final[str] = "fake"


#: Set of method names recognised by :meth:`FakePlatformAdapter.force_unsupported`
#: and :meth:`FakePlatformAdapter.force_error`. Mirrors every capability on
#: :class:`~jarvis.automation.platform.PlatformAdapter`; a closed set lets
#: tests fail loudly if they typo a method name.
_METHOD_NAMES: Final[frozenset[str]] = frozenset(
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


@dataclass(frozen=True)
class LaunchCall:
    """Recorded :meth:`FakePlatformAdapter.launch_app` invocation."""

    executable_or_uri: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ClickCall:
    """Recorded :meth:`FakePlatformAdapter.click` invocation."""

    x: int
    y: int
    button: MouseButton


@dataclass(frozen=True)
class BrightnessCall:
    """Recorded brightness-related invocation.

    A single list captures both ``get_brightness`` and ``set_brightness``
    so tests can assert the *order* in which the two are invoked (e.g.
    a "raise brightness by 10%" skill calls ``get`` then ``set``).
    """

    operation: Literal["get", "set"]
    level_pct: int | None


@dataclass(frozen=True)
class RunScriptCall:
    """Recorded :meth:`FakePlatformAdapter.run_script` invocation."""

    interpreter: ScriptInterpreter
    script_path: Path
    timeout_s: float


# ---------------------------------------------------------------------------
# FakePlatformAdapter
# ---------------------------------------------------------------------------


class FakePlatformAdapter(BasePlatformAdapter):
    """Recording :class:`PlatformAdapter` for skill / dialog tests.

    Each capability on :class:`~jarvis.automation.platform.PlatformAdapter`
    is overridden so the call is appended to one of the public
    ``*_calls`` lists; the method then either returns the configured
    default value or raises a configured failure.

    Attributes
    ----------
    launch_calls / media_key_calls / set_volume_calls / adjust_volume_calls
    / brightness_calls / notify_calls / click_calls / type_calls /
    hotkey_calls / focus_calls / run_script_calls:
        Public, mutable lists of typed records — one per invocation in
        the order calls were made. Tests should treat them as
        append-only; clearing them between phases is supported via
        :meth:`reset`.
    brightness_value:
        Value returned from :meth:`get_brightness`. Mutable so a test
        can simulate "after a successful ``set_brightness(80)``, the
        next ``get_brightness`` returns 80" by setting the attribute
        from a callback. Defaults to ``50`` per the task brief.
    next_process_handle:
        :class:`ProcessHandle` returned from the next
        :meth:`launch_app`. ``None`` (the default) means the fake
        synthesises a deterministic handle from the call arguments.
    next_script_result:
        :class:`ScriptResult` returned from the next :meth:`run_script`.
        ``None`` (the default) means the fake returns a successful
        ``exit_code=0`` result.

    Behaviour toggles
    -----------------

    * :meth:`force_unsupported` flags a method so the *next* invocation
      raises :class:`PlatformNotSupportedError`. The flag persists until
      :meth:`clear_unsupported` is called or :meth:`reset` is invoked.
    * :meth:`force_error` registers an arbitrary :class:`BaseException`
      for a method; the next invocation raises it. The exception
      reference is consumed (one-shot) so tests do not have to remember
      to clear it.

    Why both toggles?
    -----------------

    * ``platform_not_supported`` is a *contract-level* failure — the
      capability genuinely is not implemented. The Skill registry maps
      it to ``SkillResult.error("platform_not_supported", ...)``.
    * ``force_error`` covers the *runtime* failures that bubble up as
      ``OSError`` / ``RuntimeError`` and are mapped to
      ``"internal_error"`` by the registry's exception barrier
      (Requirement 17.1, CP10).
    """

    platform_tag: str = FAKE_PLATFORM_TAG

    def __init__(
        self,
        *,
        brightness_value: int = 50,
        next_process_handle: ProcessHandle | None = None,
        next_script_result: ScriptResult | None = None,
    ) -> None:
        if not isinstance(brightness_value, int) or isinstance(brightness_value, bool):
            raise TypeError("brightness_value must be an int")
        if not 0 <= brightness_value <= 100:
            raise ValueError("brightness_value must be in [0, 100]")

        # Public call-record lists. Names match the spec task brief
        # exactly so test code can read like the requirement text:
        # ``adapter.launch_calls``, ``adapter.media_key_calls``, …
        self.launch_calls: list[LaunchCall] = []
        self.media_key_calls: list[MediaKey] = []
        self.set_volume_calls: list[int] = []
        self.adjust_volume_calls: list[int] = []
        self.brightness_calls: list[BrightnessCall] = []
        self.notify_calls: list[tuple[str, str]] = []
        self.click_calls: list[ClickCall] = []
        self.type_calls: list[str] = []
        self.hotkey_calls: list[tuple[str, ...]] = []
        self.focus_calls: list[str] = []
        self.run_script_calls: list[RunScriptCall] = []

        self.brightness_value: int = brightness_value
        self.next_process_handle: ProcessHandle | None = next_process_handle
        self.next_script_result: ScriptResult | None = next_script_result

        # One-shot failure registries. ``set`` lets tests assert that a
        # method was flagged without scanning a list, and rules out
        # double-flagging.
        self._force_unsupported: set[str] = set()
        self._force_errors: dict[str, BaseException] = {}

    # -----------------------------------------------------------------
    # Test-control helpers
    # -----------------------------------------------------------------

    def force_unsupported(self, *methods: str) -> None:
        """Make subsequent calls to ``methods`` raise ``PlatformNotSupportedError``.

        Each method name MUST appear on
        :class:`~jarvis.automation.platform.PlatformAdapter`; the fake
        rejects unknown names so a typo in a test fails loudly instead
        of silently no-op'ing.
        """
        for name in methods:
            if name not in _METHOD_NAMES:
                raise ValueError(
                    f"Unknown PlatformAdapter method {name!r}; "
                    f"expected one of {sorted(_METHOD_NAMES)!r}"
                )
            self._force_unsupported.add(name)

    def clear_unsupported(self, *methods: str) -> None:
        """Remove the ``force_unsupported`` flag from the given methods.

        Passing no arguments clears every flag.
        """
        if not methods:
            self._force_unsupported.clear()
            return
        for name in methods:
            self._force_unsupported.discard(name)

    def force_error(self, method: str, error: BaseException) -> None:
        """Make the *next* call to ``method`` raise ``error``.

        The mapping is one-shot — the exception is removed from the
        registry as soon as it is raised — so tests can simulate a
        single transient failure without touching the rest of the run.
        """
        if method not in _METHOD_NAMES:
            raise ValueError(
                f"Unknown PlatformAdapter method {method!r}; "
                f"expected one of {sorted(_METHOD_NAMES)!r}"
            )
        if not isinstance(error, BaseException):
            raise TypeError("force_error 'error' must be an exception instance")
        self._force_errors[method] = error

    def reset(self) -> None:
        """Clear every recorded call and every behaviour toggle."""
        self.launch_calls.clear()
        self.media_key_calls.clear()
        self.set_volume_calls.clear()
        self.adjust_volume_calls.clear()
        self.brightness_calls.clear()
        self.notify_calls.clear()
        self.click_calls.clear()
        self.type_calls.clear()
        self.hotkey_calls.clear()
        self.focus_calls.clear()
        self.run_script_calls.clear()
        self._force_unsupported.clear()
        self._force_errors.clear()

    # -----------------------------------------------------------------
    # Internal: per-call gating
    # -----------------------------------------------------------------

    def _maybe_raise(self, method: str) -> None:
        """Honour ``force_unsupported`` / ``force_error`` for ``method``.

        Called at the *top* of every overridden capability (after the
        argument has been recorded) so a failing call still leaves a
        trace in the corresponding ``*_calls`` list — tests asserting
        "the skill records the attempt before mapping the error" rely
        on this ordering.
        """
        if method in self._force_unsupported:
            raise self._unsupported(method, detail="forced by FakePlatformAdapter")
        injected = self._force_errors.pop(method, None)
        if injected is not None:
            raise injected

    # -----------------------------------------------------------------
    # PlatformAdapter overrides
    # -----------------------------------------------------------------

    async def launch_app(self, executable_or_uri: str, args: list[str]) -> ProcessHandle:
        if not isinstance(executable_or_uri, str) or not executable_or_uri:
            raise ValueError("launch_app expects a non-empty executable_or_uri string")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise TypeError("launch_app expects args: list[str]")
        self.launch_calls.append(LaunchCall(executable_or_uri, tuple(args)))
        self._maybe_raise("launch_app")
        if self.next_process_handle is not None:
            return self.next_process_handle
        # Synthesise a deterministic handle so consumers see a non-zero
        # pid (matching real adapters) without having to configure one.
        return ProcessHandle(
            pid=4242,
            executable_or_uri=executable_or_uri,
            detached=False,
            metadata={"source": "fake"},
        )

    async def media_key(self, key: MediaKey) -> None:
        self.media_key_calls.append(key)
        self._maybe_raise("media_key")

    async def set_volume(self, level_pct: int) -> None:
        if not isinstance(level_pct, int) or isinstance(level_pct, bool):
            raise TypeError("set_volume level_pct must be an int")
        self.set_volume_calls.append(level_pct)
        self._maybe_raise("set_volume")

    async def adjust_volume(self, delta_pct: int) -> None:
        if not isinstance(delta_pct, int) or isinstance(delta_pct, bool):
            raise TypeError("adjust_volume delta_pct must be an int")
        self.adjust_volume_calls.append(delta_pct)
        self._maybe_raise("adjust_volume")

    async def get_brightness(self) -> int:
        self.brightness_calls.append(BrightnessCall("get", None))
        self._maybe_raise("get_brightness")
        return self.brightness_value

    async def set_brightness(self, level_pct: int) -> None:
        if not isinstance(level_pct, int) or isinstance(level_pct, bool):
            raise TypeError("set_brightness level_pct must be an int")
        self.brightness_calls.append(BrightnessCall("set", level_pct))
        self._maybe_raise("set_brightness")
        # Mirror the real adapter's contract: a successful ``set`` is
        # immediately observable via ``get_brightness``. Tests that
        # need richer behaviour can override ``brightness_value``
        # directly between calls.
        self.brightness_value = max(0, min(100, level_pct))

    async def notify(self, title: str, body: str) -> None:
        if not isinstance(title, str):
            raise TypeError("notify title must be a string")
        if not isinstance(body, str):
            raise TypeError("notify body must be a string")
        self.notify_calls.append((title, body))
        self._maybe_raise("notify")

    async def click(self, x: int, y: int, button: MouseButton) -> None:
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError("click x must be an int")
        if not isinstance(y, int) or isinstance(y, bool):
            raise TypeError("click y must be an int")
        self.click_calls.append(ClickCall(x, y, button))
        self._maybe_raise("click")

    async def type_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("type_text expects a string")
        self.type_calls.append(text)
        self._maybe_raise("type_text")

    async def hotkey(self, *keys: str) -> None:
        if not all(isinstance(k, str) for k in keys):
            raise TypeError("hotkey keys must all be strings")
        self.hotkey_calls.append(tuple(keys))
        self._maybe_raise("hotkey")

    async def focus_window(self, title_pattern: str) -> None:
        if not isinstance(title_pattern, str):
            raise TypeError("focus_window title_pattern must be a string")
        self.focus_calls.append(title_pattern)
        self._maybe_raise("focus_window")

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        if not isinstance(script_path, Path):
            raise TypeError("run_script script_path must be a pathlib.Path")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            raise TypeError("run_script timeout_s must be a number")
        if timeout_s < 0:
            raise ValueError("run_script timeout_s must be non-negative")
        self.run_script_calls.append(RunScriptCall(interpreter, script_path, float(timeout_s)))
        self._maybe_raise("run_script")
        if self.next_script_result is not None:
            return self.next_script_result
        return ScriptResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=0,
            timed_out=False,
        )


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_platform_adapter() -> FakePlatformAdapter:
    """Return a freshly-constructed :class:`FakePlatformAdapter` for one test.

    Each test gets its own instance so call recordings cannot leak
    between tests. Tests that need a different ``brightness_value`` /
    ``next_process_handle`` / ``next_script_result`` SHOULD construct
    the fake directly rather than mutating the fixture; the toggles
    that *are* expected to be flipped during a test
    (:meth:`FakePlatformAdapter.force_unsupported`,
    :meth:`FakePlatformAdapter.force_error`) operate on the fixture
    instance in-place.

    Pytest finds this fixture via ``conftest.py`` re-exports next to the
    test packages that consume it (``tests/unit/fakes/conftest.py``,
    ``tests/integration/conftest.py``); declaring it on the fake module
    keeps the fake's public surface in one place.
    """
    return FakePlatformAdapter()


# ---------------------------------------------------------------------------
# Re-export convenience for tests importing only the recorder types
# ---------------------------------------------------------------------------


def all_method_names() -> Iterable[str]:
    """Return a sorted view of every method name the fake recognises.

    Exposed for property-style tests that want to enumerate every
    capability without re-deriving the list.
    """
    return sorted(_METHOD_NAMES)
