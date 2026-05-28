"""End-to-end integration tests for :class:`jarvis.dialog.manager.DialogManager`.

Task 22.4 of the JARVIS implementation plan asks for end-to-end
``Dialog_Manager`` flow integration tests covering five scenarios:

1. **Simple turn** — Mistral returns a content-only stream; the manager
   forwards sentences to the TTS engine and persists the turn.
2. **Tool-call turn with confirmation** — Mistral first returns a tool
   call for a destructive Skill, the user confirms, the registry
   dispatches, and a second round returns the final spoken text.
3. **Tool-call turn with denial** — Same as (2) but the user denies; the
   Skill is never invoked, an ``access_denied`` audit row is written,
   and the second-round narration completes the turn.
4. **Retry-on-schema-violation cap** — Mistral repeatedly returns a tool
   call whose arguments fail schema validation; the dialog loop bails
   once the configured ``max_tool_retry`` budget is exhausted.
5. **Fallback-to-Ollama after circuit opens** — The Mistral cloud is
   slow enough to trigger the :class:`BackendSelector` 3 s timeout
   (here scaled down for test speed); the selector routes the call to
   the local Ollama backend, which streams its own content and
   completes the turn.

All five tests instantiate the *real* :class:`DialogManager`,
:class:`BackendSelector`, :class:`MistralBackend`, :class:`OllamaBackend`,
:class:`SkillRegistry`, :class:`AuthorizationPolicy`, and
:class:`AuditLog`. Only the wire endpoints are stubbed:

* The cloud Mistral endpoint is replaced by :class:`FakeMistralServer`
  (``tests/fakes/fake_mistral_server.py``) — an in-process aiohttp
  server that replays canned SSE event sequences.
* The local Ollama endpoint is replaced by an :class:`httpx.MockTransport`
  that returns canned NDJSON.
* The TTS engine is a recording stub (no audio device is touched), and
  the confirmation dialog is a recording stub returning a configured
  affirmative/negative answer.
* The :class:`MemoryStore` is a stub recording ``persist_turn`` calls
  with no real ChromaDB / DPAPI involvement.

Validates: Requirements 1.4, 1.6, 12.4, 14.5, 16.2, 16.4, 19.4, 19.5
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from tests.fakes.fake_mistral_server import (
    FakeMistralServer,
    content_delta_event,
    finish_event,
    tool_call_delta_event,
)

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import AssistantResponse, DialogManager
from jarvis.dialog.persona import default_jarvis_persona
from jarvis.llm.mistral_backend import MistralBackend
from jarvis.llm.ollama_backend import OllamaBackend
from jarvis.llm.selector import BackendSelector
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
# Scripted FakeMistralServer — pops a scenario name per request
# ---------------------------------------------------------------------------


class _ScriptedFakeMistralServer(FakeMistralServer):
    """A :class:`FakeMistralServer` whose active scenario advances per request.

    The base class is single-active-scenario per :meth:`set_active`. The
    Dialog_Manager's tool dispatch loop opens **multiple** ``stream``
    contexts in a single :meth:`handle_turn` call — once per round — so
    we need a way to swap the canned response between rounds without
    racing the dialog loop on the test side.

    :meth:`queue` accepts the ordered list of scenario names. Each
    incoming POST pops the next entry off the queue and selects it as
    the active scenario before delegating to the parent handler. When
    the queue is empty the previously-set active scenario sticks (this
    keeps simple single-round tests working without queueing).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._scenario_queue: list[str] = []

    def queue(self, *names: str) -> None:
        """Append ``names`` to the per-request scenario queue.

        Raises :class:`KeyError` for unknown names so a typo fails loudly
        instead of silently falling back to the previous scenario.
        """
        for name in names:
            if name not in self._scenarios:
                raise KeyError(
                    f"unknown scenario: {name!r}; "
                    f"registered: {list(self._scenarios)!r}"
                )
        self._scenario_queue.extend(names)

    @property
    def remaining(self) -> tuple[str, ...]:
        """Snapshot of scenarios still queued, in service order."""
        return tuple(self._scenario_queue)

    async def _handle_chat_completions(self, request: Any) -> Any:  # type: ignore[override]
        if self._scenario_queue:
            self.set_active(self._scenario_queue.pop(0))
        return await super()._handle_chat_completions(request)


@pytest_asyncio.fixture
async def scripted_server() -> AsyncIterator[_ScriptedFakeMistralServer]:
    """Yield a started :class:`_ScriptedFakeMistralServer` for one test."""
    server = _ScriptedFakeMistralServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Recording stubs for the dialog dependencies
# ---------------------------------------------------------------------------


class _RecordingTTS:
    """Records every sentence handed to :meth:`speak`; never blocks."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stop_calls: int = 0

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:
        self.stop_calls += 1

    def is_playing(self) -> bool:
        return False

    async def aclose(self) -> None:  # pragma: no cover - convenience
        return None


class _RecordingConfirmation:
    """Confirmation dialog stub returning a canned response per call."""

    def __init__(self, response: str = "yes") -> None:
        self.response = response
        self.prompts: list[str] = []

    async def ask_user(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class _StubMemoryStore:
    """In-memory MemoryStore stand-in for the dialog tests."""

    def __init__(self, retrieved: list[MemoryRecord] | None = None) -> None:
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


class _RecordingSkill:
    """Minimal Skill stand-in returning a canned :class:`SkillResult`."""

    def __init__(
        self,
        *,
        name: str,
        result: SkillResult | None = None,
        destructive: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        if json_schema is None:
            json_schema = {"type": "object", "additionalProperties": True}
        self.manifest = SkillManifest(
            name=name,
            description=f"{name} (test fixture)",
            json_schema=json_schema,
            destructive=destructive,
        )
        self._result = result if result is not None else SkillResult.success({"ok": True})
        self.observed_calls: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        self.observed_calls.append(dict(args))
        return self._result


# ---------------------------------------------------------------------------
# Ollama fake — httpx.MockTransport returning canned NDJSON
# ---------------------------------------------------------------------------


def _ollama_ndjson(content_text: str) -> bytes:
    """Build a two-line NDJSON Ollama ``/api/chat`` stream.

    First line carries an assistant content delta; second line carries
    ``done: true`` so :meth:`OllamaBackend._iter_events` terminates.
    """
    lines = [
        json.dumps(
            {
                "model": "mistral",
                "message": {"role": "assistant", "content": content_text},
                "done": False,
            }
        ),
        json.dumps({"model": "mistral", "done": True, "done_reason": "stop"}),
    ]
    return ("\n".join(lines) + "\n").encode()


def _build_ollama_client(content_text: str) -> httpx.AsyncClient:
    """Return an :class:`httpx.AsyncClient` whose only response is canned NDJSON."""

    body = _ollama_ndjson(content_text)

    def _handler(request: httpx.Request) -> httpx.Response:
        # Echo the request body as a header for after-the-fact assertions.
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


def _aware_now() -> datetime:
    return datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def time_source() -> FakeTimeSource:
    return FakeTimeSource(now=_aware_now())


@pytest.fixture
def audit_log(tmp_path: Path, time_source: FakeTimeSource) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=time_source,
        run_id="test-run-22-4",
    )
    yield log
    log.close()


@pytest.fixture
def policy(audit_log: AuditLog) -> AuthorizationPolicy:
    return AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )


@pytest.fixture
def state(time_source: FakeTimeSource) -> ConversationState:
    return ConversationState(
        session_id="dialog-flow-session",
        started_at=time_source.now(),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_selector(
    *,
    server: FakeMistralServer,
    fallback_content: str = "Local fallback active, sir.",
    timeout_seconds: float = 10.0,
    cool_down_seconds: float = 30.0,
    time_source: FakeTimeSource | None = None,
) -> tuple[BackendSelector, MistralBackend, OllamaBackend, httpx.AsyncClient, list[int]]:
    """Build a real :class:`BackendSelector` over Mistral primary + Ollama fallback.

    Returns the selector, the underlying backends, the owned httpx client
    (so the caller can ``aclose`` it on teardown), and a one-element list
    holding the on-flip counter (so tests can assert the user-notification
    callback fired exactly once).
    """
    primary = MistralBackend(
        api_key="test-fake-key",
        endpoint=server.url,
        model="mistral-test",
        # Eliminate retries to keep tests deterministic.
        max_retries=0,
        retry_backoff_initial_ms=1,
        request_timeout_ms=5_000,
    )
    ollama_client = _build_ollama_client(fallback_content)
    fallback = OllamaBackend(client=ollama_client, model="mistral")

    flip_count = [0]

    def _on_flip() -> None:
        flip_count[0] += 1

    selector = BackendSelector(
        primary,
        fallback,
        timeout_seconds=timeout_seconds,
        cool_down_seconds=cool_down_seconds,
        time_source=time_source,
        on_flip=_on_flip,
    )
    return selector, primary, fallback, ollama_client, flip_count


def _build_manager(
    *,
    backend: BackendSelector,
    skills: SkillRegistry,
    memory: _StubMemoryStore,
    policy: AuthorizationPolicy,
    audit_log: AuditLog,
    tts: _RecordingTTS,
    time_source: FakeTimeSource,
    confirmation_dialog: _RecordingConfirmation | None = None,
    config: DialogConfig | None = None,
) -> DialogManager:
    persona = default_jarvis_persona()
    return DialogManager(
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


def _good_transcript(text: str = "Hello there", *, confidence: float = 0.95) -> Transcript:
    return Transcript(
        text=text,
        confidence=confidence,
        started_at=_aware_now(),
        duration_ms=500,
        language="en",
    )


# ---------------------------------------------------------------------------
# Scenario registration helpers
# ---------------------------------------------------------------------------


def _register_simple_text_scenario(
    server: _ScriptedFakeMistralServer,
    *,
    name: str,
    text: str,
) -> None:
    """One-event content stream: ``text`` then ``stop``."""
    server.add_scenario(
        name,
        events=[
            content_delta_event(text, role="assistant"),
            finish_event(finish_reason="stop"),
        ],
    )


def _register_tool_call_scenario(
    server: _ScriptedFakeMistralServer,
    *,
    name: str,
    skill_name: str,
    arguments_json: str,
    call_id: str = "call-fixture-1",
) -> None:
    """Streamed tool-call: a single fragment carrying name + arguments + finish.

    The Mistral SDK 1.x's pydantic models require ``function.arguments``
    to be present on every fragment, so we emit a single combined
    fragment rather than the usual name-then-args split. The backend
    still exercises its reassembly path (Mistral's contract permits a
    one-fragment tool call) and the scenario is canonical wire-shape.
    """
    server.add_scenario(
        name,
        events=[
            tool_call_delta_event(
                tool_index=0,
                call_id=call_id,
                function_name=skill_name,
                arguments=arguments_json,
            ),
            finish_event(finish_reason="tool_calls"),
        ],
    )


# ===========================================================================
# 1. Simple turn
# ===========================================================================


@pytest.mark.asyncio
async def test_simple_turn_streams_text_through_tts(
    scripted_server: _ScriptedFakeMistralServer,
    time_source: FakeTimeSource,
    audit_log: AuditLog,
    policy: AuthorizationPolicy,
    state: ConversationState,
) -> None:
    """A pure-text response is streamed to TTS and persisted to memory.

    Validates Requirements 1.4 (turn lifecycle), 1.6 (assistant text on
    state), 19.4 / 19.5 (Mistral streaming end-to-end through the
    selector).
    """
    _register_simple_text_scenario(
        scripted_server,
        name="simple_text",
        text="Good afternoon, sir. All systems are nominal.",
    )
    scripted_server.queue("simple_text")

    selector, _primary, _fallback, ollama_client, flip_count = _build_selector(
        server=scripted_server,
        time_source=time_source,
    )
    skills = SkillRegistry()
    memory = _StubMemoryStore()
    tts = _RecordingTTS()
    try:
        manager = _build_manager(
            backend=selector,
            skills=skills,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            tts=tts,
            time_source=time_source,
        )
        response = await manager.handle_turn(_good_transcript("hello"), state)
    finally:
        await ollama_client.aclose()

    assert isinstance(response, AssistantResponse)
    # The Mistral cloud answered, so the breaker never flipped.
    assert flip_count == [0]
    # The dialog loop produced exactly one streaming round.
    assert len(scripted_server.captured) == 1
    # The TTS engine spoke the two sentences at the boundary.
    assert tts.spoken == [
        "Good afternoon, sir.",
        "All systems are nominal.",
    ]
    # Final assistant text is the joined stream.
    assert response.text == "Good afternoon, sir. All systems are nominal."
    # No tool calls on a pure-text turn.
    assert response.tool_calls == ()
    # Conversation state recorded the turn end-to-end.
    assert len(state.turns) == 1
    turn = state.turns[0]
    assert turn.user == "hello"
    assert turn.assistant == "Good afternoon, sir. All systems are nominal."
    # And memory.persist_turn was invoked exactly once (non-incognito).
    assert len(memory.persisted_turns) == 1


# ===========================================================================
# 2. Tool-call turn with confirmation
# ===========================================================================


@pytest.mark.asyncio
async def test_tool_call_turn_with_confirmation_executes_skill(
    scripted_server: _ScriptedFakeMistralServer,
    time_source: FakeTimeSource,
    audit_log: AuditLog,
    policy: AuthorizationPolicy,
    state: ConversationState,
) -> None:
    """Round 1 emits a destructive tool call; user confirms; round 2 narrates.

    Validates Requirements 14.5 (registry dispatch), 16.2 (confirmation
    before destructive dispatch), and 19.4 (tool-call event reassembly
    over the wire).
    """
    skill = _RecordingSkill(
        name="SendEmailSkill",  # hard-coded destructive in the policy
        result=SkillResult.success({"sent": True, "id": "msg-42"}),
    )
    skills = SkillRegistry()
    skills.register(skill)

    _register_tool_call_scenario(
        scripted_server,
        name="round_email_call",
        skill_name="SendEmailSkill",
        arguments_json='{"recipient":"alex@example.invalid","subject":"hi","body":"hello"}',
    )
    _register_simple_text_scenario(
        scripted_server,
        name="round_email_done",
        text="Email dispatched, sir.",
    )
    scripted_server.queue("round_email_call", "round_email_done")

    selector, _primary, _fallback, ollama_client, flip_count = _build_selector(
        server=scripted_server,
        time_source=time_source,
    )
    confirmation = _RecordingConfirmation(response="yes")
    memory = _StubMemoryStore()
    tts = _RecordingTTS()

    try:
        manager = _build_manager(
            backend=selector,
            skills=skills,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            tts=tts,
            time_source=time_source,
            confirmation_dialog=confirmation,
        )
        response = await manager.handle_turn(
            _good_transcript("send Alex an email"), state
        )
    finally:
        await ollama_client.aclose()

    assert flip_count == [0]
    # Two rounds went over the wire (tool call, then narration).
    assert len(scripted_server.captured) == 2

    # The confirmation dialog was prompted exactly once.
    assert len(confirmation.prompts) == 1
    assert "SendEmailSkill" in confirmation.prompts[0]

    # The skill executed AFTER confirmation with the model's args.
    assert skill.observed_calls == [
        {
            "recipient": "alex@example.invalid",
            "subject": "hi",
            "body": "hello",
        }
    ]

    # Audit pair: confirmation_requested precedes executed (CP9).
    entries = audit_log.entries()
    kinds = [e.kind for e in entries]
    assert "confirmation_requested" in kinds
    assert "executed" in kinds
    req_idx = kinds.index("confirmation_requested")
    exec_idx = kinds.index("executed")
    assert req_idx < exec_idx
    assert entries[req_idx].id < entries[exec_idx].id
    assert entries[req_idx].skill == "SendEmailSkill"
    assert entries[exec_idx].skill == "SendEmailSkill"

    # The narration round produced the spoken closing sentence.
    assert tts.spoken == ["Email dispatched, sir."]
    assert response.text == "Email dispatched, sir."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].skill_name == "SendEmailSkill"


# ===========================================================================
# 3. Tool-call turn with denial
# ===========================================================================


@pytest.mark.asyncio
async def test_tool_call_turn_with_denial_skips_skill(
    scripted_server: _ScriptedFakeMistralServer,
    time_source: FakeTimeSource,
    audit_log: AuditLog,
    policy: AuthorizationPolicy,
    state: ConversationState,
) -> None:
    """The user denies; the skill is never invoked and a ``denied`` row is written.

    Validates Requirements 16.2 (confirmation gate) and 16.4 (denial
    audited and surfaced to the model on the next round).
    """
    skill = _RecordingSkill(
        name="SendEmailSkill",
        result=SkillResult.success({"sent": True}),
    )
    skills = SkillRegistry()
    skills.register(skill)

    _register_tool_call_scenario(
        scripted_server,
        name="round_email_call",
        skill_name="SendEmailSkill",
        arguments_json='{"recipient":"alex@example.invalid","subject":"hi","body":"hello"}',
    )
    _register_simple_text_scenario(
        scripted_server,
        name="round_email_cancelled",
        text="Understood, sir. Cancelled.",
    )
    scripted_server.queue("round_email_call", "round_email_cancelled")

    selector, _primary, _fallback, ollama_client, _flip_count = _build_selector(
        server=scripted_server,
        time_source=time_source,
    )
    confirmation = _RecordingConfirmation(response="no, cancel that")
    memory = _StubMemoryStore()
    tts = _RecordingTTS()

    try:
        manager = _build_manager(
            backend=selector,
            skills=skills,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            tts=tts,
            time_source=time_source,
            confirmation_dialog=confirmation,
        )
        response = await manager.handle_turn(
            _good_transcript("scratch that email"), state
        )
    finally:
        await ollama_client.aclose()

    # Two rounds: tool-call + narration after denial.
    assert len(scripted_server.captured) == 2
    # Confirmation was issued and answered negatively.
    assert len(confirmation.prompts) == 1
    # Skill must NEVER have run.
    assert skill.observed_calls == []
    # Audit log carries the denial entry instead of an executed row.
    entries = audit_log.entries()
    kinds = [e.kind for e in entries]
    assert "confirmation_requested" in kinds
    assert "denied" in kinds
    assert "executed" not in kinds
    req_idx = kinds.index("confirmation_requested")
    den_idx = kinds.index("denied")
    assert req_idx < den_idx
    assert entries[req_idx].id < entries[den_idx].id

    # The Dialog_Manager still narrated the cancellation in round 2.
    # SentenceAccumulator splits on ". " boundaries.
    assert tts.spoken == ["Understood, sir.", "Cancelled."]
    assert response.text == "Understood, sir. Cancelled."


# ===========================================================================
# 4. Schema-violation retry cap (Requirement 14.5)
# ===========================================================================


@pytest.mark.asyncio
async def test_schema_violation_retries_capped_at_max_tool_retry(
    scripted_server: _ScriptedFakeMistralServer,
    time_source: FakeTimeSource,
    audit_log: AuditLog,
    policy: AuthorizationPolicy,
    state: ConversationState,
) -> None:
    """Repeated schema-violating tool calls bail after ``max_tool_retry``.

    Validates Requirement 14.5: the dispatch loop SHALL cap retries on
    ``schema_violation`` at the configured maximum (here ``2``). With
    ``max_tool_retry=2``, the loop allows the initial call plus two
    retries — three rounds total — before giving up.
    """
    strict_schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    skill = _RecordingSkill(
        name="StrictSkill",
        result=SkillResult.success({"value": 1}),
        json_schema=strict_schema,
    )
    skills = SkillRegistry()
    skills.register(skill)

    # Each scenario emits the same schema-violating call so the registry
    # short-circuits with ``schema_violation`` every round.
    bad_args = '{"n":"not-an-integer"}'
    for i in range(3):
        _register_tool_call_scenario(
            scripted_server,
            name=f"bad_round_{i}",
            skill_name="StrictSkill",
            arguments_json=bad_args,
            call_id=f"call-bad-{i}",
        )
    # A "should never run" follow-up so we can prove the loop stops.
    _register_simple_text_scenario(
        scripted_server,
        name="never_reached",
        text="should never run",
    )
    scripted_server.queue("bad_round_0", "bad_round_1", "bad_round_2", "never_reached")

    selector, _primary, _fallback, ollama_client, _flip_count = _build_selector(
        server=scripted_server,
        time_source=time_source,
    )
    memory = _StubMemoryStore()
    tts = _RecordingTTS()

    try:
        manager = _build_manager(
            backend=selector,
            skills=skills,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            tts=tts,
            time_source=time_source,
            confirmation_dialog=_RecordingConfirmation(response="yes"),
            config=DialogConfig(max_tool_retry=2),
        )
        await manager.handle_turn(_good_transcript("compute n"), state)
    finally:
        await ollama_client.aclose()

    # Three rounds went over the wire (1 + 2 retries). The fourth
    # ("never_reached") MUST still be queued.
    assert len(scripted_server.captured) == 3
    assert "never_reached" in scripted_server.remaining
    # Skill executor was never invoked because every call failed schema
    # validation upstream of dispatch.
    assert skill.observed_calls == []


# ===========================================================================
# 5. Fallback to Ollama after the BackendSelector circuit opens
# ===========================================================================


@pytest.mark.asyncio
async def test_fallback_to_ollama_after_circuit_opens(
    scripted_server: _ScriptedFakeMistralServer,
    time_source: FakeTimeSource,
    audit_log: AuditLog,
    policy: AuthorizationPolicy,
    state: ConversationState,
) -> None:
    """A slow Mistral primary trips the breaker; Ollama serves the turn.

    Validates Requirement 12.4: when the cloud Mistral endpoint is
    unhealthy (here, slow past the configured timeout), the
    :class:`BackendSelector` opens its circuit, fires the ``on_flip``
    notification once, and routes the call to the local Ollama
    fallback. The fallback's content stream completes the turn.
    """
    # The "slow_for_test" scenario delays beyond the selector timeout.
    # We configure a tight selector timeout (50 ms) so the test is fast.
    scripted_server.add_scenario(
        "slow_for_test",
        events=[
            content_delta_event("Cloud reply that should not arrive.", role="assistant"),
            finish_event(finish_reason="stop"),
        ],
        delay_seconds=0.5,
    )
    scripted_server.queue("slow_for_test")

    fallback_text = "Local fallback responding, sir."
    selector, _primary, _fallback, ollama_client, flip_count = _build_selector(
        server=scripted_server,
        fallback_content=fallback_text,
        timeout_seconds=0.05,
        time_source=time_source,
    )
    memory = _StubMemoryStore()
    tts = _RecordingTTS()
    skills = SkillRegistry()

    try:
        manager = _build_manager(
            backend=selector,
            skills=skills,
            memory=memory,
            policy=policy,
            audit_log=audit_log,
            tts=tts,
            time_source=time_source,
        )
        response = await manager.handle_turn(
            _good_transcript("status report"), state
        )
    finally:
        await ollama_client.aclose()

    # The breaker tripped exactly once and is currently open.
    assert flip_count == [1]
    assert selector.is_open
    # The fallback's content reached the TTS engine.
    assert tts.spoken == [fallback_text]
    assert response.text == fallback_text
    # The conversation state recorded the assistant turn from the fallback.
    assert len(state.turns) == 1
    assert state.turns[0].assistant == fallback_text


# ===========================================================================
# Sanity: the queue mechanism rejects unknown scenarios
# ===========================================================================


@pytest.mark.asyncio
async def test_scripted_server_queue_rejects_unknown_scenario(
    scripted_server: _ScriptedFakeMistralServer,
) -> None:
    """``queue`` raises :class:`KeyError` for unknown scenario names.

    Sanity check on the test harness itself so a typo in a scenario
    name fails loudly rather than silently sticking with the previous
    active scenario.
    """
    with pytest.raises(KeyError, match="does-not-exist"):
        scripted_server.queue("does-not-exist")
