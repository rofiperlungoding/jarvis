"""Unit tests for ``jarvis.skills.registry.SkillRegistry``.

Covers the four registry responsibilities sketched in
``design.md §Skill_Registry``:

* plugin discovery (Requirement 14.1),
* manifest validation (Requirements 14.2, 14.3),
* tool-call dispatch with schema gating, exception isolation, and
  policy-violation audit logging (Requirements 13.6, 14.4, 14.5,
  17.1), and
* Mistral tool-definition publishing (Requirement 19.4).

Hand-written unit tests are paired with a hypothesis-driven property test
in ``tests/property/`` for full CP2 / CP15 coverage; this file exercises
the registry's hand-shaped contract.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.registry import (
    NetworkPolicyViolation,
    PolicyViolation,
    SandboxViolation,
    SkillRegistrationError,
    SkillRegistry,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Mirror the pattern used in ``test_audit_log.py`` to avoid pytest-asyncio."""
    return asyncio.run(coro)


def _basic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "n": {"type": "integer", "minimum": 0},
        },
        "required": ["q"],
        "additionalProperties": False,
    }


class _EchoSkill:
    """Returns an ``echo`` value containing the supplied query."""

    def __init__(self, name: str = "echo") -> None:
        self.calls = 0
        self.last_args: dict[str, Any] | None = None
        self.last_ctx: SkillContext | None = None
        self.manifest = SkillManifest(
            name=name,
            description="echoes its input back",
            json_schema=_basic_schema(),
        )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        self.calls += 1
        self.last_args = args
        self.last_ctx = ctx
        return SkillResult.success(value={"echo": args["q"]})


class _BoomSkill:
    """Always raises a generic exception inside ``execute``."""

    def __init__(self) -> None:
        self.manifest = SkillManifest(
            name="boom",
            description="always fails",
            json_schema={"type": "object", "properties": {}, "additionalProperties": True},
        )
        self.calls = 0

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        self.calls += 1
        raise RuntimeError("kaboom")


class _SandboxSkill:
    """Raises :class:`SandboxViolation` for paths outside the sandbox."""

    def __init__(self) -> None:
        self.manifest = SkillManifest(
            name="read_path",
            description="pretends to read a path",
            json_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        raise SandboxViolation(
            f"path escape: {args['path']}",
            justification="path outside allowed_directories",
        )


class _NotAResultSkill:
    """Returns the wrong type from ``execute`` to exercise the safety net."""

    def __init__(self) -> None:
        self.manifest = SkillManifest(
            name="bad_return",
            description="returns wrong type",
            json_schema={"type": "object"},
        )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> Any:
        return "not a SkillResult"


class _RecordingAuditLog:
    """Minimal stand-in for ``jarvis.security.audit_log.AuditLog``.

    Captures the kwargs every ``record_*`` call receives so tests can
    assert the registry forwards them faithfully.
    """

    def __init__(self) -> None:
        self.policy_violations: list[dict[str, Any]] = []

    async def record_policy_violation(self, **kwargs: Any) -> None:
        self.policy_violations.append(kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_validates_skill_shape() -> None:
    reg = SkillRegistry()

    # Missing manifest entirely.
    class NotASkill:
        execute = lambda self, a, c: None  # noqa: E731

    with pytest.raises(SkillRegistrationError):
        reg.register(NotASkill())  # type: ignore[arg-type]


def test_register_rejects_invalid_json_schema() -> None:
    reg = SkillRegistry()

    class BrokenSchemaSkill:
        manifest = SkillManifest(
            name="broken",
            description="broken schema",
            # ``required`` must be a list of strings; a dict here makes the
            # JSON Schema itself ill-formed at the meta-schema level.
            json_schema={"type": "object", "required": {"q": True}},  # type: ignore[dict-item]
        )

        async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
            return SkillResult.success()

    with pytest.raises(SkillRegistrationError, match="invalid JSON Schema"):
        reg.register(BrokenSchemaSkill())  # type: ignore[arg-type]


def test_register_rejects_non_mistral_subset_schema() -> None:
    reg = SkillRegistry()

    class RemoteRefSkill:
        manifest = SkillManifest(
            name="remote_ref",
            description="schema uses remote $ref",
            json_schema={
                "type": "object",
                "properties": {
                    "q": {"$ref": "https://example.invalid/schema.json"}
                },
            },
        )

        async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
            return SkillResult.success()

    with pytest.raises(SkillRegistrationError, match="Mistral-compatible"):
        reg.register(RemoteRefSkill())  # type: ignore[arg-type]


def test_register_rejects_duplicate_name() -> None:
    reg = SkillRegistry()
    reg.register(_EchoSkill())  # type: ignore[arg-type]
    with pytest.raises(SkillRegistrationError, match="already registered"):
        reg.register(_EchoSkill())  # type: ignore[arg-type]


def test_register_records_skill_and_exposes_membership() -> None:
    reg = SkillRegistry()
    skill = _EchoSkill()
    reg.register(skill)  # type: ignore[arg-type]
    assert "echo" in reg
    assert reg.get("echo") is skill
    assert reg.names == ["echo"]
    assert len(reg) == 1


# ---------------------------------------------------------------------------
# Mistral tool definitions
# ---------------------------------------------------------------------------


def test_mistral_tool_definitions_round_trip() -> None:
    reg = SkillRegistry()
    reg.register(_EchoSkill("echo"))  # type: ignore[arg-type]
    reg.register(_EchoSkill("zecho"))  # type: ignore[arg-type]

    tools = reg.mistral_tool_definitions()
    assert [t["function"]["name"] for t in tools] == ["echo", "zecho"]

    for tool in tools:
        assert tool["type"] == "function"
        params = tool["function"]["parameters"]
        assert params["type"] == "object"

    # Round-trip through JSON without information loss (Property 12).
    serialised = json.dumps(tools)
    deserialised = json.loads(serialised)
    assert deserialised == tools


# ---------------------------------------------------------------------------
# Dispatch — schema gating
# ---------------------------------------------------------------------------


def test_dispatch_unknown_skill_returns_internal_error() -> None:
    reg = SkillRegistry()
    result = _run(reg.dispatch("nonexistent", {}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "nonexistent" in (result.error_message or "")


def test_dispatch_invalid_args_returns_schema_violation_without_invoking_executor() -> None:
    reg = SkillRegistry()
    skill = _EchoSkill()
    reg.register(skill)  # type: ignore[arg-type]

    # ``q`` is required and ``additionalProperties`` is false, so an empty
    # args dict triggers two distinct schema errors.
    result = _run(reg.dispatch("echo", {}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert skill.calls == 0
    assert isinstance(result.value, dict)
    assert "errors" in result.value


def test_dispatch_valid_args_invokes_executor_exactly_once() -> None:
    reg = SkillRegistry()
    skill = _EchoSkill()
    reg.register(skill)  # type: ignore[arg-type]

    result = _run(reg.dispatch("echo", {"q": "hi"}, SkillContext()))
    assert result.ok is True
    assert result.value == {"echo": "hi"}
    assert skill.calls == 1


# ---------------------------------------------------------------------------
# Dispatch — exception isolation (Property 7 / CP10)
# ---------------------------------------------------------------------------


def test_dispatch_executor_exception_returns_internal_error() -> None:
    reg = SkillRegistry()
    boom = _BoomSkill()
    reg.register(boom)  # type: ignore[arg-type]

    result = _run(reg.dispatch("boom", {}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "internal_error"
    # The traceback id is exposed in both the message and the value bag
    # so operators can correlate logs regardless of which surface they
    # see first.
    assert isinstance(result.value, dict)
    assert "traceback_id" in result.value
    assert result.value["exception_type"] == "RuntimeError"
    assert "kaboom" in (result.error_message or "")


def test_dispatch_executor_returning_wrong_type_yields_internal_error() -> None:
    reg = SkillRegistry()
    bad = _NotAResultSkill()
    reg.register(bad)  # type: ignore[arg-type]

    result = _run(reg.dispatch("bad_return", {}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "internal_error"


def test_dispatch_propagates_cancellation() -> None:
    """``asyncio.CancelledError`` MUST escape the registry untouched."""

    class _Cancellable:
        manifest = SkillManifest(
            name="cancellable",
            description="raises CancelledError",
            json_schema={"type": "object"},
        )

        async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
            raise asyncio.CancelledError()

    reg = SkillRegistry()
    reg.register(_Cancellable())  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        _run(reg.dispatch("cancellable", {}, SkillContext()))


# ---------------------------------------------------------------------------
# Dispatch — policy violations (Requirement 13.6)
# ---------------------------------------------------------------------------


def test_dispatch_policy_violation_records_audit_and_returns_access_denied() -> None:
    reg = SkillRegistry()
    reg.register(_SandboxSkill())  # type: ignore[arg-type]

    audit = _RecordingAuditLog()
    ctx = SkillContext(audit_log=audit, run_id="run-123")  # type: ignore[arg-type]

    result = _run(reg.dispatch("read_path", {"path": "C:/secret"}, ctx))

    assert result.ok is False
    assert result.error_code == "access_denied"

    assert len(audit.policy_violations) == 1
    entry = audit.policy_violations[0]
    assert entry["skill"] == "read_path"
    assert entry["justification"] == "path outside allowed_directories"
    assert entry["outcome"] == "access_denied"
    assert entry["run_id"] == "run-123"
    # ``args_json`` is a canonical JSON string so audit consumers can
    # match it byte-for-byte against the Tool_Call args.
    assert entry["args_json"] == '{"path":"C:/secret"}'


def test_dispatch_policy_violation_without_audit_log_still_returns_access_denied() -> None:
    reg = SkillRegistry()
    reg.register(_SandboxSkill())  # type: ignore[arg-type]

    # No audit_log on the context — the registry must still surface the
    # access_denied result rather than crashing on a missing dependency.
    result = _run(reg.dispatch("read_path", {"path": "C:/x"}, SkillContext()))
    assert result.error_code == "access_denied"


def test_network_policy_violation_subclass_is_handled_too() -> None:
    """``NetworkPolicyViolation`` must travel through the same path."""

    class _NetSkill:
        manifest = SkillManifest(
            name="netcall",
            description="contacts a host",
            json_schema={"type": "object"},
        )

        async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
            raise NetworkPolicyViolation(
                "destination not in allowlist",
                justification="evil.example.com not in network allowlist",
            )

    reg = SkillRegistry()
    reg.register(_NetSkill())  # type: ignore[arg-type]
    audit = _RecordingAuditLog()

    result = _run(
        reg.dispatch(
            "netcall",
            {},
            SkillContext(audit_log=audit, run_id="r"),  # type: ignore[arg-type]
        )
    )
    assert result.error_code == "access_denied"
    assert audit.policy_violations[0]["justification"] == (
        "evil.example.com not in network allowlist"
    )


# ---------------------------------------------------------------------------
# Dispatch — duration back-fill
# ---------------------------------------------------------------------------


def test_dispatch_back_fills_duration_ms_when_executor_reports_zero() -> None:
    reg = SkillRegistry(monotonic=iter([0.0, 0.250]).__next__)
    reg.register(_EchoSkill())  # type: ignore[arg-type]
    result = _run(reg.dispatch("echo", {"q": "x"}, SkillContext()))
    assert result.ok is True
    assert result.duration_ms == 250


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _write_plugin(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_discover_loads_plugin_with_skill_attribute(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(
        plugin_dir / "hello.py",
        '''
from jarvis.skills.base import SkillManifest, SkillResult


class _HelloSkill:
    manifest = SkillManifest(
        name="hello",
        description="says hello",
        json_schema={
            "type": "object",
            "properties": {"who": {"type": "string"}},
            "required": ["who"],
        },
    )

    async def execute(self, args, ctx):
        return SkillResult.success(value={"greeting": f"hi, {args['who']}"})


SKILL = _HelloSkill()
''',
    )

    reg = SkillRegistry()
    reg.discover([plugin_dir])
    assert "hello" in reg

    result = _run(reg.dispatch("hello", {"who": "world"}, SkillContext()))
    assert result.ok is True
    assert result.value == {"greeting": "hi, world"}


def test_discover_skips_files_starting_with_underscore(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(
        plugin_dir / "_helper.py",
        "raise RuntimeError('this should never execute')\n",
    )

    reg = SkillRegistry()
    # No exception => underscore-prefixed file was skipped without import.
    reg.discover([plugin_dir])
    assert reg.names == []


def test_discover_continues_after_failed_plugin(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(
        plugin_dir / "broken.py",
        "raise SyntaxError('intentional')\n",
    )
    _write_plugin(
        plugin_dir / "good.py",
        '''
from jarvis.skills.base import SkillManifest, SkillResult


class _G:
    manifest = SkillManifest(
        name="good",
        description="good plugin",
        json_schema={"type": "object"},
    )

    async def execute(self, args, ctx):
        return SkillResult.success()


SKILL = _G()
''',
    )

    reg = SkillRegistry()
    with caplog.at_level("ERROR"):
        reg.discover([plugin_dir])

    # The broken plugin must not prevent the good one from registering.
    assert "good" in reg
    assert "broken" not in reg


def test_discover_ignores_missing_or_non_directory_paths(tmp_path: Path) -> None:
    reg = SkillRegistry()
    # Mix of: nonexistent dir, regular file, and an empty directory.
    file_path = tmp_path / "file.py"
    file_path.write_text("# not a directory", encoding="utf-8")
    reg.discover([tmp_path / "missing", file_path, tmp_path / "empty"])
    assert reg.names == []


def test_policy_violation_default_justification_is_message() -> None:
    """Sanity: omitting ``justification`` falls back to the message."""
    pv = PolicyViolation("nope")
    assert pv.justification == "nope"
