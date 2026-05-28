"""Unit tests for ``jarvis.dialog.manager.DialogManager``.

Covers the contract documented in ``design.md Â§Dialog_Manager`` and the
acceptance bullets of task 13.4:

* Persona system prompt is always ``messages[0]`` (Property 11 / CP14).
* Empty / low-confidence transcripts short-circuit to "please repeat"
  WITHOUT invoking ``LLMBackend.stream`` (Property 13).
* Top-K memories are embedded under a delimited "memory" section.
* Streaming tokens are forwarded to TTS at sentence boundaries.
* Tool dispatch loop classifies each call, requests confirmation on
  destructive ones, retries up to ``max_tool_retry`` times on
  ``schema_violation``, and re-enters the LLM until no more tool calls.
* Acknowledgement timer fires after ``acknowledge_after_ms``.
* Memory persistence is skipped under ``state.incognito``.
* The persona guard rewrites forbidden self-references.

The tests use lightweight in-process fakes for every dependency. No
real network, ChromaDB, or Mistral SDK is involved.

Validates: Requirements 1.4, 1.6, 1.8, 10.4, 11.1, 11.3, 12.2, 12.3,
13.3, 14.5, 16.2, 16.3, 19.4, 19.5
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import (
    DEFAULT_MEMORY_K,
    DEFAULT_MIN_CONFIDENCE,
    AssistantResponse,
    DialogManager,
)
from jarvis.dialog.persona import default_jarvis_persona
from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)
from jarvis.memory.store import MemoryRecord
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTTS:
    """Records every speak/stop/aclose; never blocks."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_calls: int = 0
        self.aclose_calls: int = 0
        self._playing: bool = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:
        self.stop_calls += 1
        self._playing = False

    def is_playing(self) -> bool:
        return self._playing

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _ScriptedBackend:
    """Plays back a fixed sequence of streaming responses.

    Each "round" of the dialog loop pops one ``list[LLMEvent]`` from the
    front of the script and yields the events in order. Calls beyond the
    script raise :class:`AssertionError` so tests catch unintended LLM
    invocations.
    """

    def __init__(self, *, rounds: Sequence[Sequence[LLMEvent]]) -> None:
        self._rounds: list[list[LLMEvent]] = [list(r) for r in rounds]
        self.calls: list[
            tuple[list[Message], list[ToolDefinition], dict[str, Any]]
        ] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[LLMEvent]]:
        # Snapshot the messages list so later mutations from the dialog
        # loop don't pollute our recorded call history. The recorded
        # entries reuse the caller's TypedDict shapes verbatim.
        self.calls.append((list(messages), list(tools), dict(kwargs)))
        if not self._rounds:
            raise AssertionError(
                "_ScriptedBackend received an unexpected stream() call"
            )
        events = self._rounds.pop(0)
        return _scripted_stream_cm(events)


@contextlib.asynccontextmanager
async def _scripted_stream_cm(
    events: list[LLMEvent],
) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    """An async context manager that yields a fixed event iterator."""
    yield _aiter(events)


async def _aiter(events: list[LLMEvent]) -> AsyncIterator[LLMEvent]:
    for e in events:
        yield e


class _StubMemoryStore:
    """In-memory MemoryStore stand-in.

    Records ``persist_turn`` calls so we can assert incognito gating
    and returns a fixed list from ``retrieve``.
    """

    def __init__(self, *, retrieved: list[MemoryRecord] | None = None) -> None:
        self.retrieved: list[MemoryRecord] = list(retrieved or [])
        self.persisted_turns: list[Any] = []
        self.retrieve_calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, k: int = 5) -> list[MemoryRecord]:
        self.retrieve_calls.append((query, k))
        return list(self.retrieved)

    async def persist_turn(self, turn: Any, persona: Any | None = None) -> list[Any]:
        del persona
        self.persisted_turns.append(turn)
        return []


class _StubConfirmationDialog:
    def __init__(self, *, response: str = "yes") -> None:
        self.response = response
        self.prompts: list[str] = []

    async def ask_user(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class _ImmediateSkill:
    """Minimal Skill stand-in returning a canned :class:`SkillResult`."""

    def __init__(
        self,
        *,
        name: str,
        result: SkillResult,
        destructive: bool = False,
        observed_calls: list[dict[str, Any]] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.manifest = SkillManifest(
            name=name,
            description=f"{name} test fixture",
            json_schema={"type": "object", "additionalProperties": True},
            destructive=destructive,
        )
        self._result = result
        self.observed_calls: list[dict[str, Any]] = (
            observed_calls if observed_calls is not None else []
        )
        self._delay = delay

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        self.observed_calls.append(dict(args))
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _aware_now() -> datetime:
    return datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def time_source() -> FakeTimeSource:
    return FakeTimeSource(now=_aware_now())


@pytest.fixture()
def audit_log(tmp_path: Path, time_source: FakeTimeSource) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=time_source,
        run_id="test-run",
    )
    yield log
    log.close()


@pytest.fixture()
def policy(audit_log: AuditLog) -> AuthorizationPolicy:
    return AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )


@pytest.fixture()
def state(time_source: FakeTimeSource) -> ConversationState:
    return ConversationState(
        session_id="sess-1",
        started_at=time_source.now(),
    )


def _build_manager(
    *,
    backend: Any,
    skills: SkillRegistry | None = None,
    memory: _StubMemoryStore | None = None,
    policy: AuthorizationPolicy,
    audit_log: AuditLog,
    tts: _FakeTTS | None = None,
    time_source: FakeTimeSource,
    confirmation_dialog: Any | None = None,
    config: DialogConfig | None = None,
) -> tuple[DialogManager, _FakeTTS, _StubMemoryStore, SkillRegistry]:
    skills = skills or SkillRegistry()
    memory = memory or _StubMemoryStore()
    tts = tts or _FakeTTS()
    persona = default_jarvis_persona()
    manager = DialogManager(
        backend=backend,
        skills=skills,
        memory=memory,  # type: ignore[arg-type]
        policy=policy,
        persona=persona,
        tts=tts,
        audit_log=audit_log,
        config=config,
        confirmation_dialog=confirmation_dialog,
        time_source=time_source,
    )
    return manager, tts, memory, skills


def _good_transcript(text: str = "Hello there", *, confidence: float = 0.9) -> Transcript:
    return Transcript(
        text=text,
        confidence=confidence,
        started_at=_aware_now(),
        duration_ms=500,
        language="en",
    )


# ===========================================================================
# Property 13: empty / low-confidence transcripts skip the LLM
# ===========================================================================


class TestLowConfidenceGate:
    """Requirement 1.8 â€” empty / low-confidence transcripts re-prompt."""

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_backend_and_speaks_repeat(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        backend = _ScriptedBackend(rounds=[])
        manager, tts, memory, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        transcript = _good_transcript(text="   ", confidence=0.95)
        response = await manager.handle_turn(transcript, state)

        # Backend was never invoked.
        assert backend.call_count == 0
        # The repeat prompt was spoken.
        assert tts.spoken
        assert "repeat" in tts.spoken[0].lower()
        # Conversation state was NOT mutated.
        assert state.turns == []
        # Persistence was NOT attempted.
        assert memory.persisted_turns == []
        # Response carries the prompt for caller logging.
        assert isinstance(response, AssistantResponse)
        assert "repeat" in response.text.lower()
        assert response.audio_started_at is not None

    @pytest.mark.asyncio
    async def test_low_confidence_transcript_skips_backend(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        backend = _ScriptedBackend(rounds=[])
        manager, tts, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        # Confidence just below the 0.4 threshold.
        transcript = _good_transcript(text="hello", confidence=DEFAULT_MIN_CONFIDENCE - 0.01)
        await manager.handle_turn(transcript, state)

        assert backend.call_count == 0
        assert tts.spoken  # prompt was spoken


# ===========================================================================
# Property 11 / CP14: messages[0] is always the persona system prompt
# ===========================================================================


class TestPersonaInvariant:
    @pytest.mark.asyncio
    async def test_messages_zero_is_persona_system_prompt(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        rounds: list[list[LLMEvent]] = [
            [ContentDeltaEvent(text="Hello, sir.")]
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )
        persona = default_jarvis_persona()

        await manager.handle_turn(_good_transcript(), state)

        assert backend.call_count == 1
        sent_messages, _, _ = backend.calls[0]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == persona.system_prompt

    @pytest.mark.asyncio
    async def test_messages_zero_is_persona_in_every_loop_round(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        # Two rounds: first emits a tool call, second is the synthesis.
        skill = _ImmediateSkill(
            name="echo",
            result=SkillResult.success({"answer": 42}),
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="echo",
            arguments={"q": "x"},
            raw_arguments='{"q":"x"}',
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="The answer is 42.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )
        persona = default_jarvis_persona()

        await manager.handle_turn(_good_transcript(), state)

        assert backend.call_count == 2
        for round_messages, _, _ in backend.calls:
            assert round_messages[0]["role"] == "system"
            assert round_messages[0]["content"] == persona.system_prompt


# ===========================================================================
# Memory section embedding
# ===========================================================================


class TestMemoryEmbedding:
    @pytest.mark.asyncio
    async def test_memories_appear_under_delimited_section(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        memories = [
            MemoryRecord(
                record_id="r1",
                content="User prefers tea.",
                embedding=[0.0, 0.1],
                timestamp=_aware_now(),
                category="preference",
            ),
            MemoryRecord(
                record_id="r2",
                content="User: hi\nAssistant: hello",
                embedding=[0.1, 0.0],
                timestamp=_aware_now(),
                category="chat",
            ),
        ]
        memory = _StubMemoryStore(retrieved=memories)
        rounds: list[list[LLMEvent]] = [
            [ContentDeltaEvent(text="Of course, sir.")]
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        await manager.handle_turn(_good_transcript(text="What do I like?"), state)

        # Memory was queried with the user text and configured top-K.
        assert memory.retrieve_calls == [("What do I like?", DEFAULT_MEMORY_K)]
        # The second message is a system message containing the memory
        # delimiters and the record contents.
        sent_messages, _, _ = backend.calls[0]
        assert sent_messages[1]["role"] == "system"
        body = sent_messages[1]["content"]
        assert "# Memory" in body
        assert "# End memory" in body
        assert "User prefers tea." in body
        assert "User: hi" in body
        assert "preference" in body

    @pytest.mark.asyncio
    async def test_no_memories_omits_secondary_system_message(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        memory = _StubMemoryStore(retrieved=[])
        rounds = [[ContentDeltaEvent(text="OK.")]]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        await manager.handle_turn(_good_transcript(), state)

        sent_messages, _, _ = backend.calls[0]
        # Only persona + user message â€” no memory section.
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"


# ===========================================================================
# Sentence streaming to TTS (Requirement 12.2 / 19.5)
# ===========================================================================


class TestSentenceStreaming:
    @pytest.mark.asyncio
    async def test_tts_speaks_sentences_at_boundaries(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        # The accumulator emits a sentence the moment ".? " arrives, so
        # split the stream across multiple deltas to exercise that path.
        rounds: list[list[LLMEvent]] = [
            [
                ContentDeltaEvent(text="Hello there. "),
                ContentDeltaEvent(text="How are you today?"),
                ContentDeltaEvent(text=" Goodbye."),
            ]
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, tts, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        response = await manager.handle_turn(_good_transcript(), state)

        # First two sentences emit via the accumulator's boundary path;
        # the trailing "Goodbye." flushes after the stream ends.
        assert tts.spoken == ["Hello there.", "How are you today?", "Goodbye."]
        # Final text is the joined stream.
        assert response.text == "Hello there. How are you today? Goodbye."

    @pytest.mark.asyncio
    async def test_audio_started_at_records_first_speak(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        rounds = [[ContentDeltaEvent(text="Hello world.")]]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        response = await manager.handle_turn(_good_transcript(), state)

        assert response.audio_started_at is not None


# ===========================================================================
# Tool dispatch: classification, confirmation, audit pair
# ===========================================================================


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_safe_tool_call_dispatches_without_confirmation(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        skill = _ImmediateSkill(
            name="echo",
            result=SkillResult.success({"value": "ok"}),
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="echo",
            arguments={"q": "x"},
            raw_arguments='{"q":"x"}',
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="Done.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        confirmation = _StubConfirmationDialog(response="yes")
        manager, _, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            confirmation_dialog=confirmation,
        )

        response = await manager.handle_turn(_good_transcript(), state)

        # Skill was dispatched.
        assert skill.observed_calls == [{"q": "x"}]
        # No confirmation prompt was issued (safe call).
        assert confirmation.prompts == []
        # No destructive audit rows emitted (safe path).
        kinds = [e.kind for e in audit_log.entries()]
        assert "confirmation_requested" not in kinds
        # Tool call survives on the response.
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].skill_name == "echo"

    @pytest.mark.asyncio
    async def test_destructive_tool_call_requires_confirmation_and_audits(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        skill = _ImmediateSkill(
            name="SendEmailSkill",  # hard-coded destructive
            result=SkillResult.success({"sent": True}),
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="SendEmailSkill",
            arguments={"recipient": "alex@example.invalid", "subject": "hi", "body": "hello"},
            raw_arguments='{"recipient":"alex@example.invalid"}',
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="Sent.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        confirmation = _StubConfirmationDialog(response="yes")
        manager, _, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            confirmation_dialog=confirmation,
        )

        await manager.handle_turn(_good_transcript(), state)

        # The user was prompted to confirm.
        assert len(confirmation.prompts) == 1
        # Skill ran AFTER confirmation.
        assert skill.observed_calls
        # CP9 ordering invariant: confirmation_requested before executed.
        kinds = [e.kind for e in audit_log.entries()]
        assert kinds.count("confirmation_requested") == 1
        assert kinds.count("executed") == 1
        ids = [e.id for e in audit_log.entries()]
        # Strict id ordering.
        assert ids == sorted(ids)
        idx_req = kinds.index("confirmation_requested")
        idx_exec = kinds.index("executed")
        assert idx_req < idx_exec

    @pytest.mark.asyncio
    async def test_destructive_tool_call_denied_skips_dispatch(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        skill = _ImmediateSkill(
            name="SendEmailSkill",
            result=SkillResult.success({"sent": True}),
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="SendEmailSkill",
            arguments={"recipient": "alex@example.invalid"},
            raw_arguments='{"recipient":"alex@example.invalid"}',
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="Cancelled.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        confirmation = _StubConfirmationDialog(response="no")
        manager, _, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            confirmation_dialog=confirmation,
        )

        await manager.handle_turn(_good_transcript(), state)

        # Skill was NEVER dispatched.
        assert skill.observed_calls == []
        # Audit recorded both confirmation_requested and denied.
        kinds = [e.kind for e in audit_log.entries()]
        assert "confirmation_requested" in kinds
        assert "denied" in kinds


# ===========================================================================
# Schema-violation retry cap (Requirement 14.5)
# ===========================================================================


class TestSchemaViolationRetryCap:
    @pytest.mark.asyncio
    async def test_retry_cap_terminates_loop(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        # Build a Skill whose schema requires the integer field "n".
        # The model emits string "n" repeatedly to trigger schema_violation.
        manifest_schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        }

        class _StrictSkill:
            def __init__(self) -> None:
                self.manifest = SkillManifest(
                    name="strict",
                    description="strict schema",
                    json_schema=manifest_schema,
                )
                self.calls: list[dict[str, Any]] = []

            async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
                self.calls.append(dict(args))
                return SkillResult.success({})

        skill = _StrictSkill()
        skills = SkillRegistry()
        skills.register(skill)

        bad_tc = ToolCall(
            id="call-1",
            skill_name="strict",
            arguments={"n": "not-an-integer"},
            raw_arguments='{"n":"not-an-integer"}',
        )
        # max_tool_retry=2 means we allow up to 2 retries â€” i.e., 3
        # total rounds before bailing. Each round emits one violating
        # tool call; the loop should terminate after the budget is
        # exhausted.
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=bad_tc)],
            [ToolCallEvent(tool_call=bad_tc)],
            [ToolCallEvent(tool_call=bad_tc)],
            # Should never reach here.
            [ContentDeltaEvent(text="should never run")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            config=DialogConfig(max_tool_retry=2),
        )

        await manager.handle_turn(_good_transcript(), state)

        # Backend was called 3 times (initial + 2 retries) then loop bailed.
        assert backend.call_count == 3
        # Skill executor was never invoked because all calls were
        # schema_violations.
        assert skill.calls == []


# ===========================================================================
# Acknowledgement timer (Requirement 12.3)
# ===========================================================================


class TestAcknowledgementTimer:
    @pytest.mark.asyncio
    async def test_long_dispatch_emits_acknowledgement(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        # Skill sleeps for longer than the configured ack threshold so
        # the timer fires.
        skill = _ImmediateSkill(
            name="slow",
            result=SkillResult.success({}),
            delay=0.05,
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="slow",
            arguments={},
            raw_arguments="{}",
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="Done.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        # 10 ms threshold so the 50 ms skill triggers the ack.
        manager, tts, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            config=DialogConfig(acknowledge_after_ms=10),
        )

        await manager.handle_turn(_good_transcript(), state)

        # The "One moment, sir." utterance was spoken before "Done.".
        ack_messages = [s for s in tts.spoken if "moment" in s.lower()]
        assert ack_messages, f"expected acknowledgement utterance, got {tts.spoken!r}"
        # And the persona honorific is included.
        assert "sir" in ack_messages[0]

    @pytest.mark.asyncio
    async def test_fast_dispatch_does_not_emit_acknowledgement(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        skill = _ImmediateSkill(
            name="fast",
            result=SkillResult.success({}),
        )
        skills = SkillRegistry()
        skills.register(skill)

        tc = ToolCall(
            id="call-1",
            skill_name="fast",
            arguments={},
            raw_arguments="{}",
        )
        rounds: list[list[LLMEvent]] = [
            [ToolCallEvent(tool_call=tc)],
            [ContentDeltaEvent(text="Done.")],
        ]
        backend = _ScriptedBackend(rounds=rounds)
        # Long threshold relative to the fast skill.
        manager, tts, _, _ = _build_manager(
            backend=backend,
            skills=skills,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
            config=DialogConfig(acknowledge_after_ms=5_000),
        )

        await manager.handle_turn(_good_transcript(), state)

        ack_messages = [s for s in tts.spoken if "moment" in s.lower()]
        assert ack_messages == []


# ===========================================================================
# Memory persistence + incognito gating (Requirement 13.3)
# ===========================================================================


class TestPersistenceAndIncognito:
    @pytest.mark.asyncio
    async def test_persist_turn_called_in_normal_mode(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        memory = _StubMemoryStore()
        rounds = [[ContentDeltaEvent(text="Hi.")]]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        await manager.handle_turn(_good_transcript(), state)

        assert len(memory.persisted_turns) == 1

    @pytest.mark.asyncio
    async def test_incognito_state_skips_persistence(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        state.incognito = True
        memory = _StubMemoryStore()
        rounds = [[ContentDeltaEvent(text="Hi.")]]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        await manager.handle_turn(_good_transcript(), state)

        assert memory.persisted_turns == []


# ===========================================================================
# Persona guard rewrites forbidden self-references (Requirement 11.5)
# ===========================================================================


class TestPersonaGuardRewrite:
    @pytest.mark.asyncio
    async def test_chatgpt_self_reference_is_rewritten(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        # Single-delta stream so the offending phrase is delivered as
        # one chunk and reaches the post-loop guard.
        rounds: list[list[LLMEvent]] = [
            [ContentDeltaEvent(text="As an AI language model, I cannot help.")]
        ]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        response = await manager.handle_turn(_good_transcript(), state)

        # Final text rewrites the disclaimer to "As JARVIS, ...".
        assert "as JARVIS" in response.text
        assert "AI language model" not in response.text


# ===========================================================================
# State mutation: ConversationState reflects the turn
# ===========================================================================


class TestStateMutation:
    @pytest.mark.asyncio
    async def test_completed_turn_appears_in_state_turns(
        self,
        time_source: FakeTimeSource,
        audit_log: AuditLog,
        policy: AuthorizationPolicy,
        state: ConversationState,
    ) -> None:
        rounds = [[ContentDeltaEvent(text="Hi there.")]]
        backend = _ScriptedBackend(rounds=rounds)
        manager, _, _, _ = _build_manager(
            backend=backend,
            policy=policy,
            audit_log=audit_log,
            time_source=time_source,
        )

        await manager.handle_turn(_good_transcript("hello"), state)

        assert len(state.turns) == 1
        turn = state.turns[0]
        assert turn.user == "hello"
        assert turn.assistant == "Hi there."
