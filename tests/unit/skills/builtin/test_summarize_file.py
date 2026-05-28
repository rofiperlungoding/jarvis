"""Unit tests for :mod:`jarvis.skills.builtin.summarize_file`.

Covers Requirement 8 acceptance criteria 3, 4, 6, and 7. The Skill
delegates the read (and therefore the sandbox / size / extension
checks) to :class:`~jarvis.skills.builtin.read_file.ReadFileSkill`, so
the sandbox-soundness assertions in
:mod:`tests.unit.skills.builtin.test_read_file` already cover the
sandbox half of the contract; the tests here focus on what
``SummarizeFileSkill`` adds on top:

* JSON Schema correctness (path required, optional ``max_words``,
  ``additionalProperties: false``).
* Streamed LLM call shape (``messages``, empty ``tools``, message
  ordering).
* Word-budget enforcement (Requirement 8.4): the returned summary
  never exceeds ``max_words`` tokens, even when the model overruns.
* Default ``max_words`` = 200 (Requirement 8.3).
* Pass-through of upstream error codes (``access_denied``,
  ``file_too_large``, ``not_supported``) so the closed taxonomy stays
  consistent across both Skills.
* Missing-LLM handling and stream failure handling.
* Argument validation defence-in-depth (the registry already enforces
  the schema, but direct callers must still see clean failures).

Validates: Requirements 8.3, 8.4, 8.6, 8.7
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import pytest

from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)
from jarvis.skills.base import Skill, SkillContext, SkillManifest, SkillResult
from jarvis.skills.builtin import summarize_file as summarize_file_module
from jarvis.skills.builtin.read_file import MAX_FILE_SIZE_BYTES
from jarvis.skills.builtin.summarize_file import (
    DEFAULT_MAX_WORDS,
    MAX_WORDS_HARD_CAP,
    SCHEMA,
    SKILL,
    SummarizeFileSkill,
    _build_summarisation_messages,
    _truncate_to_words,
)
from jarvis.skills.registry import SandboxViolation, SkillRegistry

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[Any]) -> Any:
    """Drive a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _make_text_file(parent: Path, name: str, content: str) -> Path:
    """Write ``content`` to ``parent/name`` and return the file path.

    Uses :meth:`Path.write_bytes` so newlines survive intact on Windows
    (the Skill reads in binary mode and decodes as UTF-8).
    """
    path = parent / name
    path.write_bytes(content.encode("utf-8"))
    return path


@dataclass
class _RecordedCall:
    """One observed invocation of :meth:`FakeLLMBackend.stream`."""

    messages: list[Message]
    tools: list[ToolDefinition]
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeLLMBackend:
    """Minimal :class:`LLMBackend` test double.

    The Skill only needs ``stream(messages, *, tools)`` to return an
    async context manager whose body yields an async iterator of
    :class:`LLMEvent` values. We capture the call so tests can assert
    the prompt shape (Requirement 8.4 — summarisation prompt) and the
    empty ``tools`` list (the Skill must never let a misbehaving model
    invoke side-effecting tools through the summariser path).
    """

    def __init__(
        self,
        events: list[LLMEvent] | None = None,
        *,
        enter_exc: BaseException | None = None,
        iter_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[_RecordedCall] = []
        self._events = events if events is not None else []
        self._enter_exc = enter_exc
        self._iter_exc = iter_exc

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> Any:
        # Snapshot the per-call configuration so subsequent mutation
        # does not retro-actively alter an in-flight stream.
        self.calls.append(
            _RecordedCall(
                messages=list(messages), tools=list(tools), kwargs=dict(kwargs)
            )
        )
        events = list(self._events)
        enter_exc = self._enter_exc
        iter_exc = self._iter_exc

        @asynccontextmanager
        async def _cm() -> AsyncIterator[AsyncIterator[LLMEvent]]:
            if enter_exc is not None:
                raise enter_exc

            async def _events() -> AsyncIterator[LLMEvent]:
                for ev in events:
                    yield ev
                if iter_exc is not None:
                    raise iter_exc

            yield _events()

        return _cm()


def _make_ctx(
    *allowed_dirs: Path,
    llm: FakeLLMBackend | None = None,
) -> SkillContext:
    """Build a :class:`SkillContext` with the supplied sandbox + LLM.

    Each entry is canonicalised via ``realpath`` so a symlinked
    ``tmp_path`` (common on macOS where ``/tmp`` -> ``/private/tmp``)
    does not accidentally fail the sandbox check inside the Skill.
    """
    canonical = tuple(Path(os.path.realpath(str(d))) for d in allowed_dirs)
    return SkillContext(
        allowed_directories=canonical,
        llm_backend=llm,
        run_id="summarize-file-test",
    )


def _content_events(*chunks: str) -> list[LLMEvent]:
    """Wrap raw text chunks as :class:`ContentDeltaEvent` instances."""
    return [ContentDeltaEvent(text=c) for c in chunks]


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # Plugin discovery (registry._load_plugin_file) imports the module
    # and reads ``getattr(module, "SKILL", None)``; the constant must
    # exist and resolve to the singleton instance.
    assert isinstance(SKILL, SummarizeFileSkill)
    assert summarize_file_module.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable; confirm the
    # singleton would be accepted by the registry's isinstance gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_non_destructive_with_expected_name() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # Requirement 8.3 anchors the wording.
    assert manifest.name == "SummarizeFileSkill"
    # Summarisation is not destructive — it neither modifies the
    # filesystem nor sends data outside the local machine beyond the
    # (already-audited) LLM call.
    assert manifest.destructive is False
    assert manifest.source == "builtin"
    # The manifest's schema must be the same dict as the public
    # ``SCHEMA`` constant so consumers (Mistral tool publishing,
    # tests, docs) agree.
    assert manifest.json_schema is SCHEMA


def test_default_max_words_is_two_hundred() -> None:
    # Requirement 8.3: ``max_words`` defaults to 200.
    assert DEFAULT_MAX_WORDS == 200


def test_schema_requires_path_with_optional_max_words() -> None:
    # Requirement 8.3: schema requires "path" + optional "max_words".
    assert SCHEMA["required"] == ["path"]
    assert SCHEMA["properties"]["path"]["type"] == "string"
    max_words_schema = SCHEMA["properties"]["max_words"]
    assert max_words_schema["type"] == "integer"
    assert max_words_schema["minimum"] == 1
    assert max_words_schema["maximum"] == MAX_WORDS_HARD_CAP
    # ``additionalProperties: false`` keeps the LLM from smuggling
    # extra fields through the Skill.
    assert SCHEMA["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Prompt / call-shape assertions
# ---------------------------------------------------------------------------


def test_default_max_words_used_when_omitted(tmp_path: Path) -> None:
    """Requirement 8.3: ``max_words`` defaults to 200 when omitted."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True, result.error_message
    assert result.value is not None
    assert result.value["max_words"] == DEFAULT_MAX_WORDS
    # The system prompt must reflect the budget so the model knows
    # where to stop.
    system_msg = llm.calls[0].messages[0]
    assert system_msg["role"] == "system"
    assert str(DEFAULT_MAX_WORDS) in system_msg["content"]


def test_explicit_max_words_overrides_default(tmp_path: Path) -> None:
    """Requirement 8.3: caller-supplied ``max_words`` is honoured."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    llm = FakeLLMBackend(_content_events("brief"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": 50}, ctx))

    assert result.ok is True, result.error_message
    assert result.value is not None
    assert result.value["max_words"] == 50
    # Both the system AND user prompts reference the budget.
    system_msg = llm.calls[0].messages[0]
    user_msg = llm.calls[0].messages[1]
    assert "50" in system_msg["content"]
    assert "50" in user_msg["content"]


def test_summary_call_uses_empty_tools_list(tmp_path: Path) -> None:
    """Summarisation must not let the model invoke other Skills.

    The Skill always passes ``tools=[]`` so a misbehaving local
    backend cannot trick the summariser into chaining a Tool_Call to,
    e.g., ``SendEmailSkill``.
    """
    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    _run(SKILL.execute({"path": str(path)}, ctx))

    assert len(llm.calls) == 1
    assert llm.calls[0].tools == []


def test_summary_call_includes_document_content(tmp_path: Path) -> None:
    """The user prompt must contain the document text."""
    payload = "Some unique sentinel text 5d3a5e6c."
    path = _make_text_file(tmp_path, "doc.txt", payload)
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    _run(SKILL.execute({"path": str(path)}, ctx))

    user_msg = llm.calls[0].messages[1]
    assert user_msg["role"] == "user"
    assert payload in user_msg["content"]


def test_messages_start_with_system_then_user(tmp_path: Path) -> None:
    """Property 11 / CP14 only constrains the dialog loop, but the
    summariser still uses a stable two-message ordering so the
    document never lands ahead of the summarisation instructions.
    """
    path = _make_text_file(tmp_path, "doc.txt", "x")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    _run(SKILL.execute({"path": str(path)}, ctx))

    msgs = llm.calls[0].messages
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Streaming / output assembly
# ---------------------------------------------------------------------------


def test_concatenates_streamed_content_deltas(tmp_path: Path) -> None:
    """The Skill must concatenate streamed deltas verbatim."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    llm = FakeLLMBackend(_content_events("Hello", " ", "summary."))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "Hello summary."
    assert result.value["truncated"] is False
    assert result.value["word_count"] == 2


def test_ignores_unexpected_tool_call_events(tmp_path: Path) -> None:
    """Tool-call events leaking into the summariser must be ignored.

    A misbehaving local backend could emit a ``tool_call`` event even
    though we passed ``tools=[]``; the Skill must drop it rather than
    crash or invoke a side-effecting Skill.
    """
    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    rogue_call = ToolCallEvent(
        tool_call=ToolCall(
            id="t-1",
            skill_name="SendEmailSkill",
            arguments={},
            raw_arguments="{}",
        )
    )
    events: list[LLMEvent] = [
        ContentDeltaEvent(text="Real "),
        rogue_call,
        ContentDeltaEvent(text="summary."),
    ]
    llm = FakeLLMBackend(events)
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "Real summary."


def test_empty_content_deltas_are_tolerated(tmp_path: Path) -> None:
    """Mistral occasionally emits zero-length deltas as heartbeats."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello.")
    llm = FakeLLMBackend(_content_events("", "ok", ""))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "ok"


# ---------------------------------------------------------------------------
# Word-budget enforcement (Requirement 8.4)
# ---------------------------------------------------------------------------


def test_summary_truncated_when_model_overruns(tmp_path: Path) -> None:
    """Requirement 8.4: summary SHALL NOT exceed ``max_words`` words."""
    path = _make_text_file(tmp_path, "doc.txt", "doc")
    # Model emits 12 words; budget is 5.
    llm = FakeLLMBackend(
        _content_events("one two three four five six seven eight nine ten eleven twelve")
    )
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": 5}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "one two three four five"
    assert result.value["word_count"] == 5
    assert result.value["truncated"] is True


def test_summary_under_budget_is_not_marked_truncated(tmp_path: Path) -> None:
    path = _make_text_file(tmp_path, "doc.txt", "doc")
    llm = FakeLLMBackend(_content_events("alpha beta gamma"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": 200}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "alpha beta gamma"
    assert result.value["word_count"] == 3
    assert result.value["truncated"] is False


def test_truncate_to_words_normalises_whitespace() -> None:
    """``_truncate_to_words`` collapses runs of whitespace to single spaces."""
    summary, count, truncated = _truncate_to_words(
        "  one\ttwo\n\nthree   four  ", 10
    )
    assert summary == "one two three four"
    assert count == 4
    assert truncated is False


def test_truncate_to_words_at_exact_budget() -> None:
    summary, count, truncated = _truncate_to_words("a b c d", 4)
    assert summary == "a b c d"
    assert count == 4
    assert truncated is False


def test_truncate_to_words_one_over_budget() -> None:
    summary, count, truncated = _truncate_to_words("a b c d e", 4)
    assert summary == "a b c d"
    assert count == 4
    assert truncated is True


# ---------------------------------------------------------------------------
# Empty-content shortcut
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_summary_without_calling_llm(tmp_path: Path) -> None:
    """An empty file must not waste an LLM turn (defensive)."""
    path = _make_text_file(tmp_path, "empty.txt", "")
    llm = FakeLLMBackend(_content_events("should-not-appear"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == ""
    assert result.value["word_count"] == 0
    assert result.value["truncated"] is False
    # The LLM must not have been invoked.
    assert llm.calls == []


def test_whitespace_only_file_returns_empty_summary(tmp_path: Path) -> None:
    path = _make_text_file(tmp_path, "blank.txt", "   \n\t  \n")
    llm = FakeLLMBackend(_content_events("should-not-appear"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == ""
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Read-side error pass-through (Requirements 8.6, 8.7, 8.5)
# ---------------------------------------------------------------------------


def test_path_outside_sandbox_raises_sandbox_violation(tmp_path: Path) -> None:
    """Requirement 8.6: a path outside allowed dirs must be blocked.

    The underlying :class:`ReadFileSkill` raises
    :class:`SandboxViolation`, which propagates unchanged so the
    registry can record the single ``policy_violation`` audit entry.
    """
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = _make_text_file(outside, "secret.txt", "leaked")

    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(inside, llm=llm)

    with pytest.raises(SandboxViolation):
        _run(SKILL.execute({"path": str(secret)}, ctx))

    # The LLM must not have been called once the sandbox check fails.
    assert llm.calls == []


def test_oversized_file_returns_file_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.7: > 5 MB → ``file_too_large`` (passed through)."""
    path = _make_text_file(tmp_path, "huge.txt", "x")  # tiny on disk
    real_getsize = os.path.getsize

    def fake_getsize(p: Any) -> int:
        if os.path.realpath(os.fspath(p)) == os.path.realpath(str(path)):
            return MAX_FILE_SIZE_BYTES + 1
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "file_too_large"
    # The LLM must not have been called when the read short-circuits.
    assert llm.calls == []


def test_unsupported_extension_returns_not_supported(tmp_path: Path) -> None:
    """Requirement 8.5: only documented formats are honoured."""
    path = _make_text_file(tmp_path, "image.exe", "binary-blob")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert llm.calls == []


def test_missing_file_returns_internal_error(tmp_path: Path) -> None:
    missing = tmp_path / "ghost.txt"
    assert not missing.exists()
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(missing)}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# LLM availability / failure handling
# ---------------------------------------------------------------------------


def test_missing_llm_backend_returns_internal_error(tmp_path: Path) -> None:
    """A context without an LLM backend cannot summarise."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello.")
    ctx = _make_ctx(tmp_path, llm=None)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "llm" in (result.error_message or "").lower()


def test_llm_stream_failure_surfaces_internal_error(tmp_path: Path) -> None:
    """Backend failures map to ``internal_error`` in the closed taxonomy."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello.")
    llm = FakeLLMBackend(enter_exc=RuntimeError("backend-down"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "backend-down" in (result.error_message or "")


def test_llm_iteration_failure_surfaces_internal_error(tmp_path: Path) -> None:
    """An exception raised mid-stream is also captured cleanly."""
    path = _make_text_file(tmp_path, "doc.txt", "Hello.")
    llm = FakeLLMBackend(
        _content_events("partial "),
        iter_exc=RuntimeError("mid-stream-failure"),
    )
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    assert "mid-stream-failure" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Argument validation (defence-in-depth; the registry already validates)
# ---------------------------------------------------------------------------


def test_blank_path_returns_schema_violation(tmp_path: Path) -> None:
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": "   "}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert llm.calls == []


def test_max_words_zero_returns_schema_violation(tmp_path: Path) -> None:
    path = _make_text_file(tmp_path, "doc.txt", "Hi")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": 0}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert llm.calls == []


def test_max_words_above_hard_cap_returns_schema_violation(tmp_path: Path) -> None:
    path = _make_text_file(tmp_path, "doc.txt", "Hi")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(
        SKILL.execute(
            {"path": str(path), "max_words": MAX_WORDS_HARD_CAP + 1},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert llm.calls == []


def test_max_words_bool_returns_schema_violation(tmp_path: Path) -> None:
    """Booleans must NOT slip through as integers."""
    path = _make_text_file(tmp_path, "doc.txt", "Hi")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": True}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Helper: prompt builder
# ---------------------------------------------------------------------------


def test_build_summarisation_messages_shape() -> None:
    msgs = _build_summarisation_messages("the body", 42)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "42" in msgs[0]["content"]
    assert "42" in msgs[1]["content"]
    assert "the body" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Result payload shape
# ---------------------------------------------------------------------------


def test_success_payload_carries_metadata(tmp_path: Path) -> None:
    """Successful results include path, extension, size, and budget info."""
    payload = "First line.\nSecond line."
    path = _make_text_file(tmp_path, "doc.txt", payload)
    llm = FakeLLMBackend(_content_events("Brief summary."))
    ctx = _make_ctx(tmp_path, llm=llm)

    result = _run(SKILL.execute({"path": str(path), "max_words": 25}, ctx))

    assert result.ok is True
    assert result.value is not None
    # Canonical realpath is reported so downstream consumers can rely
    # on it for caching / dedup.
    assert result.value["path"] == os.path.realpath(str(path))
    assert result.value["extension"] == ".txt"
    assert result.value["size_bytes"] == path.stat().st_size
    assert result.value["max_words"] == 25
    assert result.value["summary"] == "Brief summary."
    assert result.value["word_count"] == 2
    assert result.value["truncated"] is False


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_through_registry(tmp_path: Path) -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "SummarizeFileSkill" in registry

    path = _make_text_file(tmp_path, "doc.txt", "Hello world.")
    llm = FakeLLMBackend(_content_events("Brief."))
    ctx = _make_ctx(tmp_path, llm=llm)
    result = _run(
        registry.dispatch("SummarizeFileSkill", {"path": str(path)}, ctx)
    )

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert result.value is not None
    assert result.value["summary"] == "Brief."


def test_registry_translates_sandbox_violation_to_access_denied(
    tmp_path: Path,
) -> None:
    """Requirement 8.6 + 13.6: blocked path → ``access_denied`` via registry."""
    registry = SkillRegistry()
    registry.register(SKILL)

    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = _make_text_file(outside, "secret.txt", "leaked")

    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(inside, llm=llm)
    result = _run(
        registry.dispatch("SummarizeFileSkill", {"path": str(secret)}, ctx)
    )

    assert result.ok is False
    assert result.error_code == "access_denied"
    assert llm.calls == []


def test_registry_rejects_extra_properties(tmp_path: Path) -> None:
    """``additionalProperties: false`` blocks LLM-smuggled fields."""
    registry = SkillRegistry()
    registry.register(SKILL)

    path = _make_text_file(tmp_path, "doc.txt", "Hi")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)
    result = _run(
        registry.dispatch(
            "SummarizeFileSkill",
            {"path": str(path), "tone": "formal"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_registry_rejects_missing_path(tmp_path: Path) -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)
    result = _run(registry.dispatch("SummarizeFileSkill", {}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_registry_rejects_max_words_above_cap(tmp_path: Path) -> None:
    """The registry's JSON Schema validator must catch the hard cap."""
    registry = SkillRegistry()
    registry.register(SKILL)

    path = _make_text_file(tmp_path, "doc.txt", "Hi")
    llm = FakeLLMBackend(_content_events("ok"))
    ctx = _make_ctx(tmp_path, llm=llm)
    result = _run(
        registry.dispatch(
            "SummarizeFileSkill",
            {"path": str(path), "max_words": MAX_WORDS_HARD_CAP + 5},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"
