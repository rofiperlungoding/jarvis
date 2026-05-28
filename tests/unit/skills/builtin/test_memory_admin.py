"""Unit tests for :mod:`jarvis.skills.builtin.memory_admin`.

These tests pin three layers of behaviour that together cover
Requirements 10.5, 10.6, 13.5, 16.1, and 16.2:

* the manifest advertises the contract the Skill_Registry requires —
  ``MemoryAdminSkill`` name, ``destructive=False`` at the manifest
  level (per-operation gating happens in
  :class:`AuthorizationPolicy`), and a Mistral-subset-compatible JSON
  Schema with three operations and conditional required-fields;
* :meth:`MemoryAdminSkill.execute` delegates to the injected
  :class:`MemoryStore` exactly once per call, faithfully serialises
  the response, and never re-prompts on ``forget`` (the
  Authorization_Policy has already obtained user confirmation by the
  time the registry dispatches us);
* the :class:`AuthorizationPolicy` reads the ``operation``
  discriminator out of the Tool_Call arguments and classifies
  ``MemoryAdminSkill.forget`` as Destructive while keeping ``list``
  and ``search`` Safe (Requirement 16.1, 16.2).

A registry round-trip exercises the JSON-Schema gate, the Mistral
subset checker, and the ``mistral_tool_definitions()`` projection in
one go — that is the cheapest way to keep the manifest honest.

Validates: Requirements 10.5, 10.6, 13.5, 16.1, 16.2
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
import pytest

from jarvis.llm.base import ToolCall
from jarvis.llm.mistral_schema import MistralSchemaValidator
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DEFAULT_DESTRUCTIVE_SKILLS,
    DESTRUCTIVE,
    SAFE,
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import Skill, SkillContext, SkillManifest
from jarvis.skills.builtin import memory_admin as memory_admin_module
from jarvis.skills.builtin.memory_admin import (
    MEMORY_ADMIN_K_CAP,
    MEMORY_ADMIN_K_DEFAULT,
    MEMORY_ADMIN_LIST_CAP,
    MEMORY_STORE_EXTRAS_KEY,
    SCHEMA,
    SKILL,
    MemoryAdminSkill,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[Any]) -> Any:
    """Synchronously drive a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)  # type: ignore[arg-type]


class _FakeMemoryRecord:
    """Minimal :class:`MemoryRecord` lookalike for test isolation.

    The real :class:`~jarvis.memory.store.MemoryRecord` is a frozen
    dataclass; we mirror only the attributes :func:`_serialize_record`
    reads so the Skill cannot tell the fake apart at runtime.
    """

    def __init__(
        self,
        *,
        record_id: str,
        content: str,
        category: str = "chat",
        timestamp: datetime | None = None,
        redacted: bool = False,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.record_id = record_id
        self.content = content
        self.category = category
        self.timestamp = timestamp or datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        self.redacted = redacted
        self.provenance = dict(provenance or {})


class _FakeMemoryStore:
    """Records every call and replays a configured outcome.

    The Skill talks to three :class:`MemoryStore` coroutines:
    ``list_records``, ``retrieve``, and ``forget``. The fake mirrors
    those signatures so the executor cannot tell it apart at runtime.
    """

    def __init__(
        self,
        *,
        listed: list[_FakeMemoryRecord] | None = None,
        retrieved: list[_FakeMemoryRecord] | None = None,
        forget_result: bool = True,
        list_exc: BaseException | None = None,
        retrieve_exc: BaseException | None = None,
        forget_exc: BaseException | None = None,
    ) -> None:
        self._listed = list(listed or [])
        self._retrieved = list(retrieved or [])
        self._forget_result = forget_result
        self.list_calls: list[str | None] = []
        self.retrieve_calls: list[tuple[str, int]] = []
        self.forget_calls: list[str] = []
        self.list_exc = list_exc
        self.retrieve_exc = retrieve_exc
        self.forget_exc = forget_exc

    async def list_records(
        self,
        *,
        category: str | None = None,
        older_than: datetime | None = None,
    ) -> list[_FakeMemoryRecord]:
        del older_than  # unused by the Skill
        self.list_calls.append(category)
        if self.list_exc is not None:
            raise self.list_exc
        if category is None:
            return list(self._listed)
        return [r for r in self._listed if r.category == category]

    async def retrieve(self, query: str, k: int = 5) -> list[_FakeMemoryRecord]:
        self.retrieve_calls.append((query, k))
        if self.retrieve_exc is not None:
            raise self.retrieve_exc
        return list(self._retrieved[:k])

    async def forget(self, record_id: str) -> bool:
        self.forget_calls.append(record_id)
        if self.forget_exc is not None:
            raise self.forget_exc
        return self._forget_result


def _ctx_with_store(store: Any) -> SkillContext:
    return SkillContext(extras={MEMORY_STORE_EXTRAS_KEY: store})


# ---------------------------------------------------------------------------
# Module exports / surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    """Plugin discovery (registry._load_plugin_file) imports the module
    and reads ``getattr(module, "SKILL", None)``; the constant must
    exist and resolve to the singleton instance."""
    assert isinstance(SKILL, MemoryAdminSkill)
    assert memory_admin_module.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    """The :class:`Skill` Protocol is runtime-checkable; confirm the
    singleton would be accepted by the registry's ``isinstance`` gate."""
    assert isinstance(SKILL, Skill)


def test_extras_key_constant_is_stable() -> None:
    """Pin the extras key so the eventual app.py wiring (Task 19.1)
    cannot drift away from this module without breaking the tests."""
    assert MEMORY_STORE_EXTRAS_KEY == "memory_store"
    assert memory_admin_module.MEMORY_STORE_EXTRAS_KEY == "memory_store"


def test_k_constants_are_consistent() -> None:
    # Defaults align with MemoryStore.retrieve and stay below the cap.
    assert MEMORY_ADMIN_K_DEFAULT == 5
    assert MEMORY_ADMIN_K_CAP == 10
    assert MEMORY_ADMIN_K_DEFAULT <= MEMORY_ADMIN_K_CAP
    assert MEMORY_ADMIN_LIST_CAP > 0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_metadata() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # The Skill name is the public Mistral function name; pinned because
    # the destructive_skills config refers to "MemoryAdminSkill.forget".
    assert manifest.name == "MemoryAdminSkill"
    # Per-operation gating means the manifest itself is non-destructive.
    assert manifest.destructive is False
    assert manifest.source == "builtin"
    assert manifest.json_schema is SCHEMA
    # The Skill is OS-agnostic; declare every supported platform so
    # Requirement 15.4 does not block it on macOS / Linux builds.
    for tag in ("windows", "linux", "darwin"):
        assert tag in manifest.platforms


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def test_schema_declares_three_operations() -> None:
    op_field = SCHEMA["properties"]["operation"]
    assert op_field["type"] == "string"
    assert set(op_field["enum"]) == {"list", "search", "forget"}
    # ``operation`` is the only unconditionally required field.
    assert SCHEMA["required"] == ["operation"]
    assert SCHEMA["additionalProperties"] is False


def test_schema_requires_query_for_search() -> None:
    """The ``allOf`` conditional makes ``query`` required when
    ``operation == "search"``."""
    blocks = SCHEMA["allOf"]
    search_block = next(
        b for b in blocks if b["if"]["properties"]["operation"]["const"] == "search"
    )
    assert search_block["then"]["required"] == ["query"]


def test_schema_requires_record_id_for_forget() -> None:
    """The ``allOf`` conditional makes ``record_id`` required when
    ``operation == "forget"`` (Requirement 10.5)."""
    blocks = SCHEMA["allOf"]
    forget_block = next(
        b for b in blocks if b["if"]["properties"]["operation"]["const"] == "forget"
    )
    assert forget_block["then"]["required"] == ["record_id"]


def test_schema_accepts_list_without_extra_fields() -> None:
    validator = Draft7Validator(SCHEMA)
    assert validator.is_valid({"operation": "list"})


def test_schema_accepts_list_with_category_filter() -> None:
    validator = Draft7Validator(SCHEMA)
    for category in ("chat", "preference", "fact", "summary"):
        assert validator.is_valid({"operation": "list", "category": category}), category


def test_schema_rejects_unknown_category() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "list", "category": "kitchen"})


def test_schema_rejects_search_without_query() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "search"})


def test_schema_rejects_forget_without_record_id() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "forget"})


def test_schema_rejects_unknown_operation() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "wipe_everything"})


def test_schema_caps_k_at_ten() -> None:
    validator = Draft7Validator(SCHEMA)
    assert validator.is_valid(
        {"operation": "search", "query": "x", "k": MEMORY_ADMIN_K_CAP}
    )
    assert not validator.is_valid(
        {"operation": "search", "query": "x", "k": MEMORY_ADMIN_K_CAP + 1}
    )
    assert not validator.is_valid({"operation": "search", "query": "x", "k": 0})


def test_schema_rejects_extra_properties() -> None:
    validator = Draft7Validator(SCHEMA)
    assert not validator.is_valid({"operation": "list", "limit": 50})


def test_schema_accepts_via_mistral_subset_validator() -> None:
    """The Mistral function-calling subset must accept this schema (CP15).

    ``allOf`` / ``if`` / ``then`` / ``const`` / ``enum`` are all
    draft-07 keywords the subset validator allows, and there is no
    ``$ref`` or ``oneOf`` mixing scalars with objects. If this fails
    the registry would refuse to register the Skill.
    """
    MistralSchemaValidator().validate(SCHEMA)


def test_skill_registers_cleanly() -> None:
    """A successful registration confirms the Mistral subset + draft-07
    meta-schema accept the manifest."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "MemoryAdminSkill" in registry


# ---------------------------------------------------------------------------
# Authorization wiring (Requirements 16.1, 16.2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_log(tmp_path: Path) -> Any:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=FakeTimeSource(now=datetime(2024, 1, 1, tzinfo=UTC)),
        run_id="memory-admin-test",
    )
    yield log
    log.close()


def test_default_destructive_skills_includes_memory_admin_forget() -> None:
    """Requirement 16.1 explicitly enumerates ``MemoryAdminSkill.forget``."""
    assert "MemoryAdminSkill.forget" in DEFAULT_DESTRUCTIVE_SKILLS


def test_authorization_policy_classifies_forget_as_destructive(
    audit_log: Any,
) -> None:
    """The default :data:`DEFAULT_DESTRUCTIVE_SKILLS` carries
    ``MemoryAdminSkill.forget``; a Tool_Call with
    ``operation="forget"`` must be classified as Destructive."""
    policy = AuthorizationPolicy(allowlist=TrustedActionAllowlist(), audit=audit_log)
    tool_call = ToolCall(
        id="tc-1",
        skill_name="MemoryAdminSkill",
        arguments={"operation": "forget", "record_id": "rec-1"},
        raw_arguments='{"operation": "forget", "record_id": "rec-1"}',
    )
    assert policy.classify(tool_call, SKILL.manifest) == DESTRUCTIVE


def test_authorization_policy_classifies_list_and_search_safe(
    audit_log: Any,
) -> None:
    """Read-only operations stay Safe even with the default
    ``MemoryAdminSkill.forget`` entry in place."""
    policy = AuthorizationPolicy(allowlist=TrustedActionAllowlist(), audit=audit_log)
    list_call = ToolCall(
        id="tc-2",
        skill_name="MemoryAdminSkill",
        arguments={"operation": "list"},
        raw_arguments='{"operation": "list"}',
    )
    search_call = ToolCall(
        id="tc-3",
        skill_name="MemoryAdminSkill",
        arguments={"operation": "search", "query": "home address"},
        raw_arguments='{"operation": "search", "query": "home address"}',
    )
    assert policy.classify(list_call, SKILL.manifest) == SAFE
    assert policy.classify(search_call, SKILL.manifest) == SAFE


# ---------------------------------------------------------------------------
# list operation
# ---------------------------------------------------------------------------


def test_list_returns_serialised_records() -> None:
    records = [
        _FakeMemoryRecord(
            record_id="rid-1",
            content="User likes tea",
            category="preference",
            timestamp=datetime(2024, 1, 5, tzinfo=UTC),
            redacted=False,
            provenance={"source": "fact"},
        ),
        _FakeMemoryRecord(
            record_id="rid-2",
            content="User: hi\nAssistant: hello",
            category="chat",
            timestamp=datetime(2024, 1, 10, tzinfo=UTC),
            redacted=False,
            provenance={"source": "turn", "session_id": "s1"},
        ),
    ]
    fake = _FakeMemoryStore(listed=records)

    result = _run(SKILL.execute({"operation": "list"}, _ctx_with_store(fake)))

    assert result.ok is True
    assert result.value is not None
    assert result.value["operation"] == "list"
    assert result.value["category"] is None
    assert result.value["total"] == 2
    assert result.value["returned"] == 2
    # Newest-first ordering — rid-2 (Jan 10) before rid-1 (Jan 5).
    serialized = result.value["records"]
    assert [r["record_id"] for r in serialized] == ["rid-2", "rid-1"]
    # Each entry carries the public fields and skips embeddings.
    first = serialized[0]
    assert first["category"] == "chat"
    assert first["content"] == "User: hi\nAssistant: hello"
    assert first["redacted"] is False
    assert first["timestamp"] == "2024-01-10T00:00:00+00:00"
    assert first["provenance"] == {"source": "turn", "session_id": "s1"}
    assert "embedding" not in first
    # Service was called exactly once with no filter.
    assert fake.list_calls == [None]


def test_list_with_category_filter_dispatches_to_store() -> None:
    fake = _FakeMemoryStore(
        listed=[
            _FakeMemoryRecord(
                record_id="rid-1",
                content="prefers metric",
                category="preference",
            )
        ]
    )

    result = _run(
        SKILL.execute(
            {"operation": "list", "category": "preference"},
            _ctx_with_store(fake),
        )
    )

    assert result.ok is True
    assert result.value is not None
    assert result.value["category"] == "preference"
    assert [r["record_id"] for r in result.value["records"]] == ["rid-1"]
    assert fake.list_calls == ["preference"]


def test_list_truncates_to_cap() -> None:
    records = [
        _FakeMemoryRecord(
            record_id=f"rid-{i:03d}",
            content=f"content {i}",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=i),
        )
        for i in range(MEMORY_ADMIN_LIST_CAP + 5)
    ]
    fake = _FakeMemoryStore(listed=records)

    result = _run(SKILL.execute({"operation": "list"}, _ctx_with_store(fake)))

    assert result.ok is True
    assert result.value is not None
    assert result.value["total"] == MEMORY_ADMIN_LIST_CAP + 5
    assert result.value["returned"] == MEMORY_ADMIN_LIST_CAP
    assert len(result.value["records"]) == MEMORY_ADMIN_LIST_CAP


def test_list_returns_empty_payload_when_store_is_empty() -> None:
    fake = _FakeMemoryStore(listed=[])
    result = _run(SKILL.execute({"operation": "list"}, _ctx_with_store(fake)))
    assert result.ok is True
    assert result.value == {
        "operation": "list",
        "category": None,
        "total": 0,
        "returned": 0,
        "records": [],
    }


def test_list_rejects_unknown_category_with_schema_violation() -> None:
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "list", "category": "kitchen"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.list_calls == []


# ---------------------------------------------------------------------------
# search operation
# ---------------------------------------------------------------------------


def test_search_dispatches_to_store_with_default_k() -> None:
    records = [
        _FakeMemoryRecord(record_id="rid-1", content="Bandung is in West Java"),
    ]
    fake = _FakeMemoryStore(retrieved=records)

    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "where is Bandung"},
            _ctx_with_store(fake),
        )
    )

    assert result.ok is True
    assert result.value is not None
    assert result.value["operation"] == "search"
    assert result.value["query"] == "where is Bandung"
    assert result.value["k"] == MEMORY_ADMIN_K_DEFAULT
    assert result.value["returned"] == 1
    assert [r["record_id"] for r in result.value["records"]] == ["rid-1"]
    # Service was called exactly once with the default k.
    assert fake.retrieve_calls == [("where is Bandung", MEMORY_ADMIN_K_DEFAULT)]


def test_search_honours_explicit_k() -> None:
    records = [
        _FakeMemoryRecord(record_id=f"rid-{i}", content=str(i)) for i in range(3)
    ]
    fake = _FakeMemoryStore(retrieved=records)

    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x", "k": 3},
            _ctx_with_store(fake),
        )
    )

    assert result.ok is True
    assert result.value is not None
    assert result.value["k"] == 3
    assert fake.retrieve_calls == [("x", 3)]


def test_search_clamps_k_at_cap() -> None:
    """The JSON Schema rejects k > cap up front, but a hand-rolled
    SkillContext (tests, future plugins) may bypass the registry's
    validator. The Skill clamps defensively at the cap."""
    fake = _FakeMemoryStore(retrieved=[])
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x", "k": 999},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is True
    assert result.value is not None
    assert result.value["k"] == MEMORY_ADMIN_K_CAP
    assert fake.retrieve_calls == [("x", MEMORY_ADMIN_K_CAP)]


def test_search_rejects_blank_query_with_schema_violation() -> None:
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "   "},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.retrieve_calls == []


def test_search_rejects_negative_k_with_schema_violation() -> None:
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x", "k": 0},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.retrieve_calls == []


def test_search_rejects_boolean_k_with_schema_violation() -> None:
    """``bool`` is a subclass of ``int``; the Skill rejects explicitly."""
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x", "k": True},  # type: ignore[dict-item]
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.retrieve_calls == []


def test_search_returns_empty_payload_when_store_finds_nothing() -> None:
    fake = _FakeMemoryStore(retrieved=[])
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is True
    assert result.value is not None
    assert result.value["returned"] == 0
    assert result.value["records"] == []


# ---------------------------------------------------------------------------
# forget operation
# ---------------------------------------------------------------------------


def test_forget_dispatches_and_reports_success() -> None:
    """The Authorization_Policy has already obtained user confirmation
    by the time the registry dispatches us; the Skill simply forwards
    to ``store.forget`` (Requirement 10.6)."""
    fake = _FakeMemoryStore(forget_result=True)
    result = _run(
        SKILL.execute(
            {"operation": "forget", "record_id": "rid-42"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is True
    assert result.value == {
        "operation": "forget",
        "record_id": "rid-42",
        "forgotten": True,
    }
    assert fake.forget_calls == ["rid-42"]


def test_forget_returns_success_with_forgotten_false_on_missing_record() -> None:
    """``forgotten=False`` is a successful outcome — the closed error
    taxonomy has no "not_found" code, and the user still hears an
    acknowledgement."""
    fake = _FakeMemoryStore(forget_result=False)
    result = _run(
        SKILL.execute(
            {"operation": "forget", "record_id": "missing"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is True
    assert result.value is not None
    assert result.value["forgotten"] is False
    assert fake.forget_calls == ["missing"]


def test_forget_rejects_blank_record_id_with_schema_violation() -> None:
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "forget", "record_id": "  "},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.forget_calls == []


# ---------------------------------------------------------------------------
# Misconfigured / missing dependencies
# ---------------------------------------------------------------------------


def test_execute_returns_internal_error_when_store_missing() -> None:
    """No memory store in extras simulates a bootstrap wiring bug."""
    result = _run(SKILL.execute({"operation": "list"}, SkillContext()))
    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "memory" in (result.error_message or "").lower()


def test_execute_returns_internal_error_when_extras_holds_bogus_value() -> None:
    """A non-store value under the key is treated as missing."""
    result = _run(
        SKILL.execute(
            {"operation": "list"},
            SkillContext(extras={MEMORY_STORE_EXTRAS_KEY: object()}),
        )
    )
    assert result.ok is False
    assert result.error_code == "internal_error"


def test_execute_rejects_unknown_operation_with_schema_violation() -> None:
    """Defence-in-depth: even if the schema gate is bypassed (direct
    invocation), the executor refuses to dispatch unknown ops."""
    fake = _FakeMemoryStore()
    result = _run(
        SKILL.execute(
            {"operation": "wipe_everything"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.list_calls == []
    assert fake.retrieve_calls == []
    assert fake.forget_calls == []


def test_execute_maps_value_error_from_store_to_schema_violation() -> None:
    fake = _FakeMemoryStore(retrieve_exc=ValueError("k must be a non-negative int"))
    result = _run(
        SKILL.execute(
            {"operation": "search", "query": "x"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Registry round-trip — Mistral subset compatibility
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_through_registry() -> None:
    """End-to-end: register the skill and dispatch each operation."""
    reg = SkillRegistry()
    reg.register(SKILL)

    [tool] = reg.mistral_tool_definitions()
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "MemoryAdminSkill"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["operation"]

    fake = _FakeMemoryStore(
        listed=[_FakeMemoryRecord(record_id="rid-1", content="x")],
        retrieved=[_FakeMemoryRecord(record_id="rid-1", content="x")],
        forget_result=True,
    )
    ctx = _ctx_with_store(fake)

    list_result = _run(reg.dispatch("MemoryAdminSkill", {"operation": "list"}, ctx))
    assert list_result.ok is True
    assert fake.list_calls == [None]

    search_result = _run(
        reg.dispatch(
            "MemoryAdminSkill",
            {"operation": "search", "query": "tea"},
            ctx,
        )
    )
    assert search_result.ok is True
    assert fake.retrieve_calls == [("tea", MEMORY_ADMIN_K_DEFAULT)]

    forget_result = _run(
        reg.dispatch(
            "MemoryAdminSkill",
            {"operation": "forget", "record_id": "rid-1"},
            ctx,
        )
    )
    assert forget_result.ok is True
    assert fake.forget_calls == ["rid-1"]


def test_registry_rejects_search_without_query_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeMemoryStore()
    result = _run(
        reg.dispatch(
            "MemoryAdminSkill",
            {"operation": "search"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.retrieve_calls == []


def test_registry_rejects_forget_without_record_id_with_schema_violation() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeMemoryStore()
    result = _run(
        reg.dispatch(
            "MemoryAdminSkill",
            {"operation": "forget"},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.forget_calls == []


def test_registry_rejects_extra_properties() -> None:
    reg = SkillRegistry()
    reg.register(SKILL)
    fake = _FakeMemoryStore()
    result = _run(
        reg.dispatch(
            "MemoryAdminSkill",
            {"operation": "list", "limit": 999},
            _ctx_with_store(fake),
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
