"""Unit tests for :mod:`jarvis.skills.builtin.launch_app`.

Pins the behaviours that together cover Requirements 2.1-2.5:

* the manifest exposes the contract the Skill_Registry requires —
  ``LaunchAppSkill`` name, ``destructive=False``, JSON Schema with a
  single required ``application`` string field (Requirement 2.1);
* a registered ``application`` resolves through the registry and is
  forwarded to :meth:`PlatformAdapter.launch_app` (Requirements 2.2,
  2.3);
* an unknown ``application`` returns ``not_supported`` and carries a
  clarification payload back to the Dialog_Manager (Requirement 2.4);
* a successful launch returns a payload that lets the Dialog_Manager
  confirm the action by application name (Requirement 2.5);
* the executor returns a structured :class:`SkillResult` — never raises
  — when the :class:`PlatformAdapter` or application registry are
  missing from ``ctx``.

A complementary registry round-trip test exercises the JSON-Schema gate
and the Mistral subset checker via :class:`SkillRegistry.register` so
the manifest stays Mistral-compatible (Requirement 19.4 / CP15).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    PlatformAdapter,
    ProcessHandle,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import launch_app as launch_app_module
from jarvis.skills.builtin.launch_app import (
    APPLICATION_REGISTRY_EXTRAS_KEY,
    SKILL,
    LaunchAppSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter implementations
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that records every ``launch_app`` invocation.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied (every other capability raises
    :class:`PlatformNotSupportedError`) so a misconfigured Skill that
    accidentally calls something other than ``launch_app`` would fail
    loudly during the test rather than silently no-op.
    """

    platform_tag = "test"

    def __init__(self, *, pid: int = 4242, detached: bool = False) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._pid = pid
        self._detached = detached

    async def launch_app(
        self, executable_or_uri: str, args: list[str]
    ) -> ProcessHandle:
        self.calls.append((executable_or_uri, list(args)))
        return ProcessHandle(
            pid=self._pid,
            executable_or_uri=executable_or_uri,
            detached=self._detached,
        )


class _UnsupportedAdapter(BasePlatformAdapter):
    """Adapter whose ``launch_app`` always raises ``PlatformNotSupportedError``."""

    platform_tag = "test"

    async def launch_app(
        self, executable_or_uri: str, args: list[str]
    ) -> ProcessHandle:
        raise self._unsupported(
            "launch_app",
            detail=f"no launch support in test (target={executable_or_uri!r})",
        )


class _MissingFileAdapter(BasePlatformAdapter):
    """Adapter that simulates a registry entry pointing at a missing file."""

    platform_tag = "test"

    async def launch_app(
        self, executable_or_uri: str, args: list[str]
    ) -> ProcessHandle:
        raise FileNotFoundError(f"could not launch {executable_or_uri!r}")


class _BoomAdapter(BasePlatformAdapter):
    """Adapter whose ``launch_app`` raises an unrelated exception."""

    platform_tag = "test"

    async def launch_app(
        self, executable_or_uri: str, args: list[str]
    ) -> ProcessHandle:
        raise RuntimeError("adapter boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Drive a single coroutine to completion under the default loop."""
    return asyncio.run(coro)


def _ctx(
    *,
    adapter: PlatformAdapter | None = None,
    registry: dict[str, str] | None = None,
) -> SkillContext:
    """Build a :class:`SkillContext` with the optional registry/adapter."""
    extras: dict[str, Any] = {}
    if registry is not None:
        extras[APPLICATION_REGISTRY_EXTRAS_KEY] = registry
    return SkillContext(platform_adapter=adapter, extras=extras)


_DEFAULT_REGISTRY: dict[str, str] = {
    "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "vscode": "C:/Users/test/AppData/Local/Programs/Microsoft VS Code/Code.exe",
    "spotify": "spotify:",
}


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, LaunchAppSkill)
    assert isinstance(SKILL, Skill)
    assert SKILL is launch_app_module.SKILL
    assert SKILL.manifest is LaunchAppSkill.manifest


def test_manifest_metadata() -> None:
    m = LaunchAppSkill.manifest
    assert isinstance(m, SkillManifest)
    assert m.name == "LaunchAppSkill"
    assert m.source == "builtin"
    # Launching an application is non-destructive (Requirement 16.1
    # lists destructive Skills explicitly; LaunchAppSkill is absent).
    assert m.destructive is False
    assert m.platforms == ("windows",)


def test_manifest_schema_requires_single_application_field() -> None:
    """Requirement 2.1: argument schema requires a single string field 'application'."""
    schema = LaunchAppSkill.manifest.json_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["application"]
    assert schema["additionalProperties"] is False
    application = schema["properties"]["application"]
    assert application["type"] == "string"
    # ``minLength: 1`` keeps the empty-string case in the schema gate
    # rather than the unknown-application path.
    assert application["minLength"] == 1
    # No other properties — the schema is a single-field object.
    assert set(schema["properties"]) == {"application"}


# ---------------------------------------------------------------------------
# Successful dispatch — Requirements 2.2, 2.3, 2.5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_target"),
    [
        ("chrome", _DEFAULT_REGISTRY["chrome"]),
        ("vscode", _DEFAULT_REGISTRY["vscode"]),
        ("spotify", _DEFAULT_REGISTRY["spotify"]),
    ],
)
def test_execute_resolves_registered_name_via_adapter(
    name: str, expected_target: str
) -> None:
    """Requirement 2.2 + 2.3: registered names resolve through the adapter."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": name}, ctx))

    assert result.ok is True
    assert result.error_code is None
    assert adapter.calls == [(expected_target, [])]
    assert result.value is not None
    # Requirement 2.5: the success payload carries the application
    # name so the Dialog_Manager can phrase a natural confirmation.
    assert result.value["application"] == name
    assert result.value["target"] == expected_target


def test_execute_payload_carries_process_handle_metadata() -> None:
    """The success payload includes pid + detached for diagnostics."""
    adapter = _RecordingAdapter(pid=1234, detached=False)
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["pid"] == 1234
    assert result.value["detached"] is False


def test_execute_supports_user_defined_registry_entries() -> None:
    """Requirement 2.3: user-defined entries map spoken names to executables."""
    custom_registry = {"my-tool": "C:/tools/mytool.exe"}
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=custom_registry)

    result = _run(SKILL.execute({"application": "my-tool"}, ctx))

    assert result.ok is True
    assert adapter.calls == [("C:/tools/mytool.exe", [])]


def test_execute_supports_uri_handler_registry_entries() -> None:
    """Requirement 2.3: URI handlers (e.g. spotify:) are valid targets."""
    adapter = _RecordingAdapter(detached=True)
    ctx = _ctx(adapter=adapter, registry={"spotify": "spotify:"})

    result = _run(SKILL.execute({"application": "spotify"}, ctx))

    assert result.ok is True
    assert adapter.calls == [("spotify:", [])]


def test_execute_calls_adapter_exactly_once() -> None:
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    _run(SKILL.execute({"application": "chrome"}, ctx))

    assert len(adapter.calls) == 1


# ---------------------------------------------------------------------------
# Unknown application — Requirement 2.4
# ---------------------------------------------------------------------------


def test_execute_unknown_application_returns_not_supported() -> None:
    """Requirement 2.4: unknown application returns an error result."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": "unknown-app"}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.error_message is not None
    assert "unknown-app" in result.error_message
    # Requirement 2.4: the adapter MUST NOT be called for unknown apps.
    assert adapter.calls == []


def test_execute_unknown_application_payload_supports_clarification() -> None:
    """Requirement 2.4: payload carries known names so the dialog can clarify."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": "firefox"}, ctx))

    assert result.value is not None
    assert result.value["application"] == "firefox"
    assert result.value["needs_clarification"] is True
    # Sorted, deterministic list of registered names so the
    # Dialog_Manager can phrase a stable clarification ("I know about
    # chrome, spotify, and vscode — which one did you mean?").
    assert result.value["known_applications"] == ["chrome", "spotify", "vscode"]


def test_execute_unknown_application_with_empty_registry() -> None:
    """An empty registry still yields a structured ``not_supported`` result."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry={})

    result = _run(SKILL.execute({"application": "anything"}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.value is not None
    assert result.value["known_applications"] == []
    assert adapter.calls == []


def test_execute_application_match_is_case_sensitive() -> None:
    """Registry lookups are case-sensitive (mirrors dict semantics)."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry={"chrome": "C:/chrome.exe"})

    result = _run(SKILL.execute({"application": "Chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# Adapter failure modes
# ---------------------------------------------------------------------------


def test_execute_returns_platform_not_supported_when_adapter_unsupported() -> None:
    """A platform that cannot launch surfaces ``platform_not_supported``."""
    adapter = _UnsupportedAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert result.error_message is not None
    assert "launch_app" in result.error_message


def test_execute_returns_internal_error_when_target_missing_on_disk() -> None:
    """A registry entry pointing at a missing file is ``internal_error``."""
    adapter = _MissingFileAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert result.value is not None
    assert result.value["application"] == "chrome"
    assert result.value["target"] == _DEFAULT_REGISTRY["chrome"]


def test_execute_propagates_unrelated_exceptions() -> None:
    """Non-Platform / non-FileNotFound exceptions bubble up to the registry."""
    adapter = _BoomAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))

    with pytest.raises(RuntimeError, match="adapter boom"):
        _run(SKILL.execute({"application": "chrome"}, ctx))


# ---------------------------------------------------------------------------
# Context misconfiguration
# ---------------------------------------------------------------------------


def test_execute_without_platform_adapter_is_internal_error() -> None:
    ctx = SkillContext(extras={APPLICATION_REGISTRY_EXTRAS_KEY: dict(_DEFAULT_REGISTRY)})

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    """A smuggled-in object that is not a PlatformAdapter is rejected."""
    ctx = SkillContext(
        platform_adapter=object(),
        extras={APPLICATION_REGISTRY_EXTRAS_KEY: dict(_DEFAULT_REGISTRY)},
    )

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "PlatformAdapter" in (result.error_message or "")


def test_execute_without_application_registry_is_internal_error() -> None:
    """A missing registry signals a wiring bug at bootstrap."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert APPLICATION_REGISTRY_EXTRAS_KEY in (result.error_message or "")
    # The adapter must not be called when the registry is unavailable.
    assert adapter.calls == []


def test_execute_with_non_mapping_registry_is_internal_error() -> None:
    """A registry that isn't a Mapping[str, str] is rejected."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(
        platform_adapter=adapter,
        extras={APPLICATION_REGISTRY_EXTRAS_KEY: ["chrome", "vscode"]},
    )

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert adapter.calls == []


def test_execute_with_non_string_registry_values_is_internal_error() -> None:
    """A registry with non-string values is rejected as misconfigured."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(
        platform_adapter=adapter,
        extras={APPLICATION_REGISTRY_EXTRAS_KEY: {"chrome": 12345}},
    )

    result = _run(SKILL.execute({"application": "chrome"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# Integration with the SkillRegistry — manifest is Mistral-compatible
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_via_registry() -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "LaunchAppSkill" in registry.names

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))
    result = _run(
        registry.dispatch("LaunchAppSkill", {"application": "chrome"}, ctx)
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.calls == [(_DEFAULT_REGISTRY["chrome"], [])]


def test_registry_rejects_missing_application_with_schema_violation() -> None:
    """Missing required field is ``schema_violation``, not a dispatch."""
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))
    result = _run(reg.dispatch("LaunchAppSkill", {}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    # Executor must NOT have been called.
    assert adapter.calls == []


def test_registry_rejects_extra_properties_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))
    result = _run(
        reg.dispatch(
            "LaunchAppSkill",
            {"application": "chrome", "args": ["--incognito"]},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_registry_rejects_empty_application_with_schema_violation() -> None:
    """``minLength: 1`` keeps the empty-string case in the schema gate."""
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))
    result = _run(reg.dispatch("LaunchAppSkill", {"application": ""}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_registry_rejects_non_string_application_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, registry=dict(_DEFAULT_REGISTRY))
    result = _run(
        reg.dispatch("LaunchAppSkill", {"application": 12345}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_skill_appears_in_mistral_tool_definitions() -> None:
    """The manifest passes the Mistral function-calling subset checker."""
    reg = SkillRegistry()
    reg.register(SKILL)

    tools = reg.mistral_tool_definitions()
    names = {t["function"]["name"] for t in tools}
    assert "LaunchAppSkill" in names

    launch_tool = next(
        t for t in tools if t["function"]["name"] == "LaunchAppSkill"
    )
    assert launch_tool["type"] == "function"
    parameters = launch_tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["application"]
    assert parameters["properties"]["application"]["type"] == "string"
