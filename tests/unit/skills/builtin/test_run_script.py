"""Unit tests for :mod:`jarvis.skills.builtin.run_script`.

Pins the behaviours that together cover Requirements 9.1, 9.2, 9.3,
9.4, 9.5, 9.8, 16.1, and 16.2:

* the manifest exposes the contract the Skill_Registry / Authorization
  policy require — ``RunScriptSkill`` name, ``destructive=True`` so
  every invocation is gated by user confirmation (Requirements 9.2 /
  16.1), JSON Schema with a single required ``script_id`` string field
  (Requirement 9.1) and ``additionalProperties: false`` so the LLM
  cannot smuggle inline script text through (Requirement 9.5);
* a registered ``script_id`` resolves through the
  :class:`ScriptCatalog` and is forwarded to
  :meth:`PlatformAdapter.run_script` (Requirements 9.2, 9.3);
* the success payload carries stdout/stderr/exit code so the
  Dialog_Manager can read the script's output back to the user
  (Requirement 9.3);
* an unknown ``script_id`` returns ``script_not_found`` and carries a
  clarification payload back to the Dialog_Manager (Requirement 9.4);
* a script that exceeds its 60 s budget surfaces ``timeout`` and
  preserves the captured streams (Requirement 9.8);
* the executor returns a structured :class:`SkillResult` — never raises
  — when the :class:`PlatformAdapter` or :class:`ScriptCatalog` are
  missing from ``ctx``;
* a complementary registry round-trip test exercises the JSON-Schema
  gate, the Mistral subset checker, and end-to-end dispatch through
  :class:`SkillRegistry`.

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.8, 16.1, 16.2
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    BasePlatformAdapter,
    PlatformAdapter,
    ScriptInterpreter,
    ScriptResult,
)
from jarvis.automation.scripts import (
    DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ScriptCatalog,
)
from jarvis.config.schema import ScriptCatalogEntry
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin import run_script as run_script_module
from jarvis.skills.builtin.run_script import (
    SCHEMA,
    SCRIPT_CATALOG_EXTRAS_KEY,
    SKILL,
    RunScriptSkill,
)
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fake PlatformAdapter implementations
# ---------------------------------------------------------------------------


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that records every ``run_script`` invocation.

    Inheriting from :class:`BasePlatformAdapter` keeps the Protocol
    surface satisfied (every other capability raises
    :class:`PlatformNotSupportedError`) so a misconfigured Skill that
    accidentally calls something other than ``run_script`` would fail
    loudly during the test rather than silently no-op.
    """

    platform_tag = "test"

    def __init__(
        self,
        *,
        result: ScriptResult | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[ScriptInterpreter, Path, float]] = []
        self._result = result or ScriptResult(
            exit_code=0, stdout="ok\n", stderr="", duration_ms=12
        )
        self._raise_exc = raise_exc

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        self.calls.append((interpreter, script_path, timeout_s))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


class _UnsupportedAdapter(BasePlatformAdapter):
    """Adapter whose ``run_script`` always raises ``PlatformNotSupportedError``."""

    platform_tag = "test"

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        raise self._unsupported(
            "run_script",
            detail=(
                f"no run_script support in test "
                f"(interpreter={interpreter!r}, path={script_path!s})"
            ),
        )


class _BoomAdapter(BasePlatformAdapter):
    """Adapter whose ``run_script`` raises an unrelated exception."""

    platform_tag = "test"

    async def run_script(
        self,
        interpreter: ScriptInterpreter,
        script_path: Path,
        timeout_s: float,
    ) -> ScriptResult:
        raise OSError("simulated adapter failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Drive a single coroutine to completion under the default loop."""
    return asyncio.run(coro)


def _entry(
    interpreter: ScriptInterpreter = "powershell",
    path: str = "C:/scripts/sample.ps1",
    description: str = "",
) -> ScriptCatalogEntry:
    return ScriptCatalogEntry(
        interpreter=interpreter, path=path, description=description
    )


def _catalog(
    adapter: PlatformAdapter,
    entries: dict[str, ScriptCatalogEntry] | None = None,
) -> ScriptCatalog:
    return ScriptCatalog(
        entries
        if entries is not None
        else {
            "backup": _entry(
                path="C:/scripts/backup.ps1", description="Daily backup"
            ),
            "deploy": _entry(
                interpreter="python",
                path="C:/scripts/deploy.py",
                description="Deploy site",
            ),
        },
        adapter,
    )


def _ctx(
    *,
    adapter: PlatformAdapter | None = None,
    catalog: ScriptCatalog | None = None,
    extras: dict[str, Any] | None = None,
) -> SkillContext:
    """Build a :class:`SkillContext` with the optional catalog/adapter."""
    final_extras: dict[str, Any] = dict(extras or {})
    if catalog is not None:
        final_extras[SCRIPT_CATALOG_EXTRAS_KEY] = catalog
    return SkillContext(platform_adapter=adapter, extras=final_extras)


# ---------------------------------------------------------------------------
# Module-level exports / manifest
# ---------------------------------------------------------------------------


def test_module_exposes_singleton_skill() -> None:
    """Plugin loaders look up the top-level ``SKILL`` attribute."""
    assert isinstance(SKILL, RunScriptSkill)
    assert isinstance(SKILL, Skill)
    assert SKILL is run_script_module.SKILL
    assert SKILL.manifest is RunScriptSkill.manifest


def test_module_exposes_extras_key_constant() -> None:
    """The extras key is module-level so callers can share it."""
    assert SCRIPT_CATALOG_EXTRAS_KEY == "script_catalog"
    assert run_script_module.SCRIPT_CATALOG_EXTRAS_KEY is SCRIPT_CATALOG_EXTRAS_KEY


def test_manifest_metadata() -> None:
    m = RunScriptSkill.manifest
    assert isinstance(m, SkillManifest)
    assert m.name == "RunScriptSkill"
    assert m.source == "builtin"
    # Requirement 9.2 / 16.1 — every invocation requires confirmation.
    assert m.destructive is True
    # Requirement 9.8 — manifest budget mirrors the 60 s adapter ceiling.
    assert m.timeout_seconds == DEFAULT_SCRIPT_TIMEOUT_SECONDS == 60.0


def test_manifest_schema_requires_script_id() -> None:
    """Requirement 9.1 — argument schema requires a single ``script_id`` string."""
    schema = RunScriptSkill.manifest.json_schema
    assert schema is SCHEMA
    assert schema["type"] == "object"
    assert schema["required"] == ["script_id"]
    # Requirement 9.5 — additionalProperties:false stops the LLM from
    # smuggling arbitrary script text via extra fields.
    assert schema["additionalProperties"] is False
    script_id = schema["properties"]["script_id"]
    assert script_id["type"] == "string"
    # ``minLength: 1`` keeps the empty-string case in the schema gate
    # rather than the unknown-id path.
    assert script_id["minLength"] == 1
    # No other properties — the schema is a single-field object so the
    # surface area for the model to abuse is minimal.
    assert set(schema["properties"]) == {"script_id"}


# ---------------------------------------------------------------------------
# Successful dispatch — Requirements 9.2, 9.3
# ---------------------------------------------------------------------------


def test_execute_resolves_registered_script_via_adapter() -> None:
    """Requirement 9.2 — a registered id resolves through the catalog."""
    adapter = _RecordingAdapter()
    catalog = _catalog(adapter)
    ctx = _ctx(adapter=adapter, catalog=catalog)

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is True
    assert result.error_code is None
    # Requirement 9.3 — the catalog forwards the declared interpreter,
    # path, and the 60 s default timeout to the platform adapter.
    assert adapter.calls == [
        ("powershell", Path("C:/scripts/backup.ps1"), 60.0),
    ]


def test_execute_success_payload_carries_stdout_stderr_exit_code() -> None:
    """Requirement 9.3 — captured outputs travel back to the dialog layer."""
    adapter = _RecordingAdapter(
        result=ScriptResult(
            exit_code=0,
            stdout="hello world\n",
            stderr="warning: foo\n",
            duration_ms=42,
        )
    )
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["script_id"] == "backup"
    assert result.value["exit_code"] == 0
    assert result.value["stdout"] == "hello world\n"
    assert result.value["stderr"] == "warning: foo\n"
    assert result.value["duration_ms"] == 42
    assert result.value["timed_out"] is False


@pytest.mark.parametrize(
    ("interpreter", "path"),
    [
        ("powershell", "C:/scripts/job.ps1"),
        ("python", "C:/scripts/job.py"),
        ("batch", "C:/scripts/job.bat"),
    ],
)
def test_execute_supports_every_interpreter(
    interpreter: ScriptInterpreter, path: str
) -> None:
    """Requirement 9.3 — all three supported interpreters dispatch correctly."""
    adapter = _RecordingAdapter()
    catalog = _catalog(
        adapter,
        entries={"job": _entry(interpreter=interpreter, path=path)},
    )
    ctx = _ctx(adapter=adapter, catalog=catalog)

    result = _run(SKILL.execute({"script_id": "job"}, ctx))

    assert result.ok is True
    assert adapter.calls[0][0] == interpreter
    assert adapter.calls[0][1] == Path(path)


def test_execute_calls_adapter_exactly_once() -> None:
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert len(adapter.calls) == 1


def test_execute_non_zero_exit_is_reported_as_success_with_payload() -> None:
    """A script that runs to completion but reports failure is ok=True.

    The closed :class:`SkillResult` taxonomy has no "non_zero_exit" code,
    so a script that finished within its budget is reported as a
    successful invocation; the Dialog_Manager decides how to communicate
    the script's own failure report (Requirement 9.3 — capture exit
    code + streams).
    """
    adapter = _RecordingAdapter(
        result=ScriptResult(
            exit_code=2,
            stdout="",
            stderr="ENOENT: missing file\n",
            duration_ms=99,
        )
    )
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "deploy"}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["exit_code"] == 2
    assert result.value["stderr"] == "ENOENT: missing file\n"


# ---------------------------------------------------------------------------
# Unknown script_id — Requirement 9.4
# ---------------------------------------------------------------------------


def test_execute_unknown_script_id_returns_script_not_found() -> None:
    """Requirement 9.4 — unknown ids surface as ``script_not_found``."""
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "nope"}, ctx))

    assert result.ok is False
    assert result.error_code == "script_not_found"
    assert result.error_message is not None
    assert "nope" in result.error_message
    # Adapter MUST NOT be called when the id is unknown.
    assert adapter.calls == []


def test_execute_unknown_script_payload_supports_clarification() -> None:
    """Payload carries known ids so the dialog can clarify."""
    adapter = _RecordingAdapter()
    catalog = _catalog(adapter)
    ctx = _ctx(adapter=adapter, catalog=catalog)

    result = _run(SKILL.execute({"script_id": "ship-it"}, ctx))

    assert result.value is not None
    assert result.value["script_id"] == "ship-it"
    assert result.value["needs_clarification"] is True
    # Carry the registered ids in insertion order so the dialog layer
    # can phrase a stable clarification.
    assert result.value["known_script_ids"] == ["backup", "deploy"]


def test_execute_unknown_script_with_empty_catalog() -> None:
    """An empty catalog still yields a structured ``script_not_found``."""
    adapter = _RecordingAdapter()
    catalog = ScriptCatalog({}, adapter)
    ctx = _ctx(adapter=adapter, catalog=catalog)

    result = _run(SKILL.execute({"script_id": "anything"}, ctx))

    assert result.ok is False
    assert result.error_code == "script_not_found"
    assert result.value is not None
    assert result.value["known_script_ids"] == []
    assert adapter.calls == []


def test_execute_does_not_treat_arbitrary_text_as_script_body() -> None:
    """Requirement 9.5 — arbitrary script text is rejected as unknown id.

    Even a value that *looks* like inline PowerShell is passed straight
    through to the catalog lookup; the lookup miss surfaces as
    ``script_not_found`` rather than the shell being invoked.
    """
    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(
        SKILL.execute({"script_id": "Write-Host 'pwned'"}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "script_not_found"
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# Timeout — Requirement 9.8
# ---------------------------------------------------------------------------


def test_execute_timeout_surfaces_timeout_error_with_streams_preserved() -> None:
    """Requirement 9.8 — ``ScriptResult.timed_out`` maps to ``timeout``."""
    adapter = _RecordingAdapter(
        result=ScriptResult(
            exit_code=-1,
            stdout="partial output\n",
            stderr="killed\n",
            duration_ms=60_000,
            timed_out=True,
        )
    )
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.error_message is not None
    # The error message references the 60 s ceiling so the user
    # understands why the script was killed.
    assert "60" in result.error_message
    assert result.value is not None
    assert result.value["script_id"] == "backup"
    assert result.value["timed_out"] is True
    # Captured partial streams travel back so the user can still see
    # what happened before the kill.
    assert result.value["stdout"] == "partial output\n"
    assert result.value["stderr"] == "killed\n"
    assert result.value["duration_ms"] == 60_000


# ---------------------------------------------------------------------------
# Adapter failure modes
# ---------------------------------------------------------------------------


def test_execute_returns_platform_not_supported_when_adapter_unsupported() -> None:
    """A platform that cannot run scripts surfaces ``platform_not_supported``."""
    adapter = _UnsupportedAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == PLATFORM_NOT_SUPPORTED
    assert result.error_message is not None
    assert "run_script" in result.error_message


def test_execute_translates_adapter_oserror_to_internal_error() -> None:
    """Adapter-level OS errors surface as ``internal_error`` with diagnostics."""
    adapter = _BoomAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert result.error_message is not None
    assert "backup" in result.error_message
    assert result.value is not None
    assert result.value["script_id"] == "backup"


# ---------------------------------------------------------------------------
# Context misconfiguration
# ---------------------------------------------------------------------------


def test_execute_without_platform_adapter_is_internal_error() -> None:
    """A missing platform adapter is a wiring bug — ``internal_error``."""
    adapter = _RecordingAdapter()
    catalog = _catalog(adapter)
    # Build the context without an adapter on the SkillContext.
    ctx = SkillContext(extras={SCRIPT_CATALOG_EXTRAS_KEY: catalog})

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "platform_adapter" in (result.error_message or "")
    # The catalog must not be invoked when the adapter is missing.
    assert adapter.calls == []


def test_execute_with_non_protocol_adapter_is_internal_error() -> None:
    """A smuggled-in object that is not a :class:`PlatformAdapter` is rejected."""
    adapter = _RecordingAdapter()
    catalog = _catalog(adapter)
    ctx = SkillContext(
        platform_adapter=object(),
        extras={SCRIPT_CATALOG_EXTRAS_KEY: catalog},
    )

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "PlatformAdapter" in (result.error_message or "")
    assert adapter.calls == []


def test_execute_without_script_catalog_is_internal_error() -> None:
    """A missing catalog signals a wiring bug at bootstrap."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(platform_adapter=adapter)

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert SCRIPT_CATALOG_EXTRAS_KEY in (result.error_message or "")
    assert adapter.calls == []


def test_execute_with_non_catalog_extras_is_internal_error() -> None:
    """A wrong-shaped extras entry is a wiring bug."""
    adapter = _RecordingAdapter()
    ctx = SkillContext(
        platform_adapter=adapter,
        extras={SCRIPT_CATALOG_EXTRAS_KEY: {"backup": "C:/scripts/backup.ps1"}},
    )

    result = _run(SKILL.execute({"script_id": "backup"}, ctx))

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
    assert "RunScriptSkill" in registry.names

    adapter = _RecordingAdapter()
    catalog = _catalog(adapter)
    ctx = _ctx(adapter=adapter, catalog=catalog)
    result = _run(
        registry.dispatch("RunScriptSkill", {"script_id": "backup"}, ctx)
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert adapter.calls == [
        ("powershell", Path("C:/scripts/backup.ps1"), 60.0),
    ]


def test_registry_rejects_missing_script_id_with_schema_violation() -> None:
    """Missing required field is ``schema_violation``, not a dispatch."""
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))
    result = _run(reg.dispatch("RunScriptSkill", {}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_registry_rejects_extra_properties_with_schema_violation() -> None:
    """Requirement 9.5 — the LLM cannot smuggle ``script`` / ``command``.

    ``additionalProperties: false`` stops a Tool_Call carrying a free-form
    ``script`` field at the registry's argument-validation gate so the
    Skill is never even invoked with the smuggled text.
    """
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))
    result = _run(
        reg.dispatch(
            "RunScriptSkill",
            {"script_id": "backup", "script": "Write-Host 'pwned'"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_registry_rejects_empty_script_id_with_schema_violation() -> None:
    """``minLength: 1`` keeps the empty-string case in the schema gate."""
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))
    result = _run(reg.dispatch("RunScriptSkill", {"script_id": ""}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert adapter.calls == []


def test_registry_rejects_non_string_script_id_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)

    adapter = _RecordingAdapter()
    ctx = _ctx(adapter=adapter, catalog=_catalog(adapter))
    result = _run(
        reg.dispatch("RunScriptSkill", {"script_id": 12345}, ctx)
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
    assert "RunScriptSkill" in names

    run_tool = next(
        t for t in tools if t["function"]["name"] == "RunScriptSkill"
    )
    assert run_tool["type"] == "function"
    parameters = run_tool["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["script_id"]
    assert parameters["properties"]["script_id"]["type"] == "string"
    assert parameters["additionalProperties"] is False
