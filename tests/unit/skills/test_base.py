"""Unit tests for ``jarvis.skills.base``.

Covers the shape and invariants of :class:`SkillManifest`,
:class:`SkillResult`, :class:`SkillContext`, and the structural
:class:`Skill` Protocol. The Error Taxonomy table in ``design.md`` lists
exactly eleven error codes; we pin that count here so a future addition
must travel through both the design doc and this test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jarvis.skills.base import (
    ERROR_CODES,
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

# ---------------------------------------------------------------------------
# Error code enumeration
# ---------------------------------------------------------------------------


def test_error_codes_are_exactly_eleven() -> None:
    # Pinned by the Error Taxonomy table in design.md.
    assert len(ERROR_CODES) == 11
    # Defensive: the tuple is a set-like declaration; entries must be unique.
    assert len(set(ERROR_CODES)) == 11


def test_error_codes_match_design_taxonomy() -> None:
    expected = {
        "schema_violation",
        "missing_credentials",
        "not_supported",
        "access_denied",
        "file_too_large",
        "script_not_found",
        "timeout",
        "provider_unavailable",
        "internal_error",
        "platform_not_supported",
        "rate_limited",
    }
    assert set(ERROR_CODES) == expected


# ---------------------------------------------------------------------------
# SkillManifest
# ---------------------------------------------------------------------------


def _basic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }


def test_manifest_defaults() -> None:
    m = SkillManifest(name="echo", description="echo back", json_schema=_basic_schema())
    assert m.name == "echo"
    assert m.destructive is False
    assert m.timeout_seconds == 30.0
    assert m.platforms == ("windows",)
    assert m.source == "builtin"


def test_manifest_is_frozen() -> None:
    m = SkillManifest(name="echo", description="echo", json_schema=_basic_schema())
    # FrozenInstanceError is a subclass of AttributeError; either is fine.
    with pytest.raises((AttributeError, TypeError)):
        m.name = "other"  # type: ignore[misc]


def test_manifest_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        SkillManifest(name="", description="x", json_schema=_basic_schema())


def test_manifest_rejects_non_mapping_schema() -> None:
    with pytest.raises(TypeError):
        SkillManifest(name="x", description="x", json_schema="not-a-dict")  # type: ignore[arg-type]


def test_manifest_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        SkillManifest(
            name="x", description="x", json_schema=_basic_schema(), timeout_seconds=0
        )
    with pytest.raises(ValueError):
        SkillManifest(
            name="x",
            description="x",
            json_schema=_basic_schema(),
            timeout_seconds=-1.0,
        )


def test_manifest_rejects_empty_platforms() -> None:
    with pytest.raises(ValueError):
        SkillManifest(
            name="x", description="x", json_schema=_basic_schema(), platforms=()
        )


def test_manifest_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        SkillManifest(
            name="x",
            description="x",
            json_schema=_basic_schema(),
            source="hostile",  # type: ignore[arg-type]
        )


def test_manifest_accepts_user_and_mcp_sources() -> None:
    for s in ("builtin", "user", "mcp"):
        m = SkillManifest(
            name="x",
            description="x",
            json_schema=_basic_schema(),
            source=s,  # type: ignore[arg-type]
        )
        assert m.source == s


def test_manifest_equality_is_value_based() -> None:
    a = SkillManifest(name="x", description="d", json_schema=_basic_schema())
    b = SkillManifest(name="x", description="d", json_schema=_basic_schema())
    assert a == b


# ---------------------------------------------------------------------------
# SkillResult
# ---------------------------------------------------------------------------


def test_result_success_factory() -> None:
    r = SkillResult.success({"answer": 42}, duration_ms=12)
    assert r.ok is True
    assert r.value == {"answer": 42}
    assert r.error_code is None
    assert r.error_message is None
    assert r.duration_ms == 12


def test_result_error_factory() -> None:
    r = SkillResult.error(
        "schema_violation", "args.q missing", value={"path": "$.q"}, duration_ms=3
    )
    assert r.ok is False
    assert r.error_code == "schema_violation"
    assert r.error_message == "args.q missing"
    assert r.value == {"path": "$.q"}
    assert r.duration_ms == 3


def test_result_rejects_unknown_error_code() -> None:
    with pytest.raises(ValueError):
        SkillResult(
            ok=False,
            value=None,
            error_code="boom",  # type: ignore[arg-type]
            error_message="x",
            duration_ms=0,
        )


def test_result_rejects_success_with_error_code() -> None:
    with pytest.raises(ValueError):
        SkillResult(
            ok=True,
            value=None,
            error_code="internal_error",
            error_message=None,
            duration_ms=0,
        )


def test_result_rejects_failure_without_error_code() -> None:
    with pytest.raises(ValueError):
        SkillResult(
            ok=False,
            value=None,
            error_code=None,
            error_message="boom",
            duration_ms=0,
        )


def test_result_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        SkillResult.success(duration_ms=-1)


def test_result_is_frozen() -> None:
    r = SkillResult.success()
    with pytest.raises((AttributeError, TypeError)):
        r.ok = False  # type: ignore[misc]


@pytest.mark.parametrize("code", list(ERROR_CODES))
def test_result_accepts_every_documented_error_code(code: str) -> None:
    r = SkillResult.error(code, "msg")  # type: ignore[arg-type]
    assert r.error_code == code


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------


def test_context_defaults_are_inert() -> None:
    ctx = SkillContext()
    assert ctx.audit_log is None
    assert ctx.time_source is None
    assert ctx.platform_adapter is None
    assert ctx.credential_store is None
    assert ctx.llm_backend is None
    assert ctx.providers == {}
    assert ctx.allowed_directories == ()
    assert ctx.incognito is False
    assert ctx.run_id is None
    assert ctx.extras == {}


def test_context_carries_dependencies() -> None:
    sentinel_adapter = object()
    sentinel_store = object()
    sentinel_llm = object()
    providers = {"weather": object()}
    ctx = SkillContext(
        platform_adapter=sentinel_adapter,
        credential_store=sentinel_store,
        llm_backend=sentinel_llm,
        providers=providers,
        allowed_directories=(Path("C:/Users/u/Documents"),),
        incognito=True,
        run_id="run-1",
        extras={"mcp_session": "abc"},
    )
    assert ctx.platform_adapter is sentinel_adapter
    assert ctx.credential_store is sentinel_store
    assert ctx.llm_backend is sentinel_llm
    assert ctx.providers["weather"] is providers["weather"]
    assert ctx.allowed_directories == (Path("C:/Users/u/Documents"),)
    assert ctx.incognito is True
    assert ctx.run_id == "run-1"
    assert ctx.extras["mcp_session"] == "abc"


def test_context_is_frozen() -> None:
    ctx = SkillContext()
    with pytest.raises((AttributeError, TypeError)):
        ctx.incognito = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Skill protocol
# ---------------------------------------------------------------------------


class _GoodSkill:
    """Minimal Skill implementation for protocol conformance tests."""

    manifest = SkillManifest(
        name="echo",
        description="echo back the supplied query",
        json_schema=_basic_schema(),
    )

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        return SkillResult.success({"echo": args["q"]})


class _MissingExecute:
    manifest = SkillManifest(
        name="x", description="x", json_schema=_basic_schema()
    )


class _MissingManifest:
    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        return SkillResult.success()


def test_protocol_recognises_compliant_skill() -> None:
    assert isinstance(_GoodSkill(), Skill)


def test_protocol_rejects_skill_missing_execute() -> None:
    assert not isinstance(_MissingExecute(), Skill)


def test_protocol_rejects_skill_missing_manifest() -> None:
    assert not isinstance(_MissingManifest(), Skill)


def test_skill_execute_round_trip() -> None:
    skill = _GoodSkill()
    result = asyncio.run(skill.execute({"q": "hello"}, SkillContext()))
    assert result.ok is True
    assert result.value == {"echo": "hello"}
