"""Dialog_Manager: the orchestration core of a conversation turn.

Implements the :class:`DialogManager` component sketched in
``design.md §Dialog_Manager`` and the corresponding pseudo-code in the
"Components and Interfaces" section. Given an STT
:class:`~jarvis.voice.stt.base.Transcript` and the in-flight
:class:`~jarvis.dialog.conversation_state.ConversationState`, the manager:

1. Gates empty / low-confidence transcripts away from the LLM (Requirement
   1.8 / Property 13).
2. Renders the message list with the persona system prompt at
   ``messages[0]`` (Requirement 11 / Property 11 / CP14) and embeds the
   top-K retrieved memories under a delimited "memory" section
   (Requirement 10.4).
3. Streams tokens through :class:`SentenceAccumulator` to the
   :class:`TTSEngine`, beginning synthesis at the first sentence boundary
   (Requirement 12.2 / 19.5).
4. Reassembles tool calls, classifies each via
   :class:`AuthorizationPolicy`, asks the user for confirmation on
   destructive Tool_Calls (Requirements 16.1-16.5), dispatches via
   :class:`SkillRegistry`, appends the result, and re-enters the LLM loop
   until no further tool calls are emitted. Caps schema-violation retries
   at the user-configured maximum (Requirement 14.5).
5. Schedules a "One moment, <honorific>." acknowledgement utterance when
   tool dispatch wall-time exceeds the configured threshold
   (Requirement 12.3).
6. Runs the post-generation persona guard
   (:class:`~jarvis.dialog.persona_guard.PersonaGuard`) over the
   assembled text (Requirement 11.5).
7. Persists the completed turn through :class:`MemoryStore` (skipping
   when :attr:`ConversationState.incognito` is set, Requirement 13.3),
   and returns an :class:`AssistantResponse`.

The manager is intentionally asynchronous: the audio I/O loop, the LLM
streaming task, and any pending ``tts.speak`` enqueues all live on the
same asyncio loop and cooperate over bounded queues. Concrete dependencies
(:class:`LLMBackend`, :class:`SkillRegistry`, :class:`MemoryStore`,
:class:`AuthorizationPolicy`, :class:`PersonaProfile`, :class:`TTSEngine`,
:class:`AuditLog`) are injected so tests can stand up a manager with
deterministic fakes.

Validates: Requirements 1.4, 1.6, 1.8, 10.1, 10.3, 10.4, 11.1, 11.3,
12.2, 12.3, 13.3, 14.5, 16.2, 16.3, 17.1, 17.2, 19.4, 19.5
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import Any, Final, cast

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.persona import PersonaProfile
from jarvis.dialog.persona_guard import PersonaGuard, PersonaLike
from jarvis.llm.base import (
    AssistantMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    LLMBackend,
    Message,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)
from jarvis.memory.store import MemoryRecord, MemoryStore
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    DESTRUCTIVE,
    AuthorizationPolicy,
    ConfirmationDialog,
)
from jarvis.skills.base import SkillContext, SkillManifest, SkillResult
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import SystemTimeSource, TimeSource
from jarvis.voice.stt.base import Transcript
from jarvis.voice.tts.base import SentenceAccumulator, TTSEngine

logger = logging.getLogger(__name__)

__all__ = ["AssistantResponse", "DialogManager"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default confidence threshold below which a :class:`Transcript` is
#: routed to the "please repeat" path instead of the LLM
#: (Requirement 1.8). Mirrors the default in
#: :attr:`jarvis.config.schema.VoiceSttConfig.min_confidence`.
DEFAULT_MIN_CONFIDENCE: Final[float] = 0.4

#: Default top-K memory retrieval depth. Mirrors
#: :attr:`jarvis.config.schema.MemoryConfig.top_k`.
DEFAULT_MEMORY_K: Final[int] = 5

#: Marker strings that wrap the retrieved memory snippets inside the
#: secondary system message. Matched verbatim by tests so the prompt
#: format is stable across releases.
_MEMORY_SECTION_HEADER: Final[str] = "# Memory"
_MEMORY_SECTION_FOOTER: Final[str] = "# End memory"


# ---------------------------------------------------------------------------
# AssistantResponse data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantResponse:
    """The structured outcome of one :meth:`DialogManager.handle_turn` call.

    Mirrors ``design.md §Data Models``. The frozen dataclass lets callers
    cache the response for diagnostics or pass it through asyncio queues
    without defensive copies.

    Attributes
    ----------
    text:
        The final assistant-facing text, after the persona guard has
        rewritten any forbidden self-references (Requirement 11.5).
        May be empty when the turn was gated (low-confidence transcript)
        and only the "please repeat" prompt was spoken — in that case
        :attr:`text` carries the prompt itself so callers can log it.
    audio_started_at:
        Wall-clock timestamp captured immediately before the first
        :meth:`TTSEngine.speak` of the turn. ``None`` when no speech was
        emitted (e.g., the LLM produced empty content and only emitted
        tool calls). Used by latency dashboards and Requirement 12.1
        diagnostics.
    cited_urls:
        Tuple of source URLs cited by the assistant during the turn.
        Today this is always empty; future Skills (web search) populate
        it via the SkillContext extras channel. Kept on the response
        shape for forward-compatibility with the design.
    tool_calls:
        Every Tool_Call dispatched during the turn, in order of issue.
        Empty when the model produced a pure-text response.
    """

    text: str
    audio_started_at: datetime | None
    cited_urls: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()


# ---------------------------------------------------------------------------
# DialogManager
# ---------------------------------------------------------------------------


class DialogManager:
    """Orchestrates a single conversational turn end-to-end.

    Parameters
    ----------
    backend:
        Active :class:`LLMBackend` (typically a :class:`BackendSelector`
        wrapping Mistral primary + Ollama fallback). The manager only
        ever calls ``backend.stream(messages, tools=tools)``.
    skills:
        :class:`SkillRegistry` used to dispatch Tool_Calls and to project
        every registered Skill into Mistral function definitions for
        ``backend.stream``.
    memory:
        :class:`MemoryStore` consulted for top-K context retrieval at
        turn start and persisted to at turn end (skipped when
        :attr:`ConversationState.incognito` is True).
    policy:
        :class:`AuthorizationPolicy` that classifies each Tool_Call and
        gates destructive ones behind the user confirmation flow. The
        manager threads the matching audit pair via
        :meth:`AuthorizationPolicy.confirm` and
        :meth:`AuthorizationPolicy.record_executed`.
    persona:
        :class:`PersonaProfile` whose ``system_prompt`` is rendered as
        ``messages[0]`` for every backend invocation. The manager never
        mutates the persona.
    tts:
        :class:`TTSEngine` to which finalised sentences are forwarded
        for streaming synthesis.
    audit_log:
        Append-only :class:`AuditLog` used for ``error`` and
        ``policy_violation`` rows. The Authorization_Policy and the
        Skill_Registry already write the bulk of the audit trail; the
        manager's own writes are limited to error / dialog-level events.
    config:
        Validated :class:`DialogConfig`. The manager reads
        ``acknowledge_after_ms``, ``max_tool_retry``, and the optional
        ``honorific`` override.
    confirmation_dialog:
        Implementation of :class:`ConfirmationDialog` that the manager
        forwards to :meth:`AuthorizationPolicy.confirm` when a
        destructive Tool_Call needs the user's "yes". When ``None``,
        every destructive Tool_Call that is not on the trusted-action
        allowlist is treated as denied (safety default — the cinematic
        JARVIS does not silently send email behind the user's back).
    persona_guard:
        Optional :class:`PersonaGuard` instance. The manager constructs
        a default one when omitted.
    time_source:
        Injectable :class:`TimeSource`. Defaults to
        :class:`SystemTimeSource`. Tests pass a :class:`FakeTimeSource`
        for deterministic timestamps.
    memory_k:
        Override for the top-K memory retrieval depth. Defaults to
        :data:`DEFAULT_MEMORY_K`. The :class:`MemoryConfig` value is
        the right production source; this kwarg is exposed so unit
        tests do not need a full :class:`Config` to construct a
        manager.
    min_confidence:
        Override for the empty/low-confidence transcript gate. Defaults
        to :data:`DEFAULT_MIN_CONFIDENCE`. Mirrors
        :attr:`VoiceSttConfig.min_confidence`.
    repeat_prompt:
        Text spoken when a transcript is gated (Requirement 1.8 /
        Property 13). Defaults to a short JARVIS-style apology rendered
        with the persona's honorific. Override for localisation tests.
    run_id:
        Run identifier propagated into the :class:`SkillContext` for
        audit attribution. ``None`` falls back to ``audit_log.run_id``.
    """

    def __init__(
        self,
        *,
        backend: LLMBackend,
        skills: SkillRegistry,
        memory: MemoryStore,
        policy: AuthorizationPolicy,
        persona: PersonaProfile,
        tts: TTSEngine,
        audit_log: AuditLog,
        config: DialogConfig | None = None,
        confirmation_dialog: ConfirmationDialog | None = None,
        persona_guard: PersonaGuard | None = None,
        time_source: TimeSource | None = None,
        memory_k: int = DEFAULT_MEMORY_K,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        repeat_prompt: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._backend: LLMBackend = backend
        self._skills: SkillRegistry = skills
        self._memory: MemoryStore = memory
        self._policy: AuthorizationPolicy = policy
        self._persona: PersonaProfile = persona
        self._tts: TTSEngine = tts
        self._audit: AuditLog = audit_log
        # Pydantic v2 ``model_validate({})`` is the closest equivalent to
        # "all defaults"; mypy treats the no-arg call as missing required
        # fields because pydantic computes defaults at runtime via
        # field metadata.
        self._config: DialogConfig = (
            config if config is not None else DialogConfig.model_validate({})
        )
        self._confirmation_dialog: ConfirmationDialog | None = confirmation_dialog
        self._persona_guard: PersonaGuard = persona_guard or PersonaGuard()
        self._time: TimeSource = time_source or SystemTimeSource()

        if memory_k < 0:
            raise ValueError("memory_k must be non-negative")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.0, 1.0]")

        self._memory_k: int = memory_k
        self._min_confidence: float = min_confidence
        # Build the repeat prompt once so the persona's honorific is
        # baked in. Callers can still override the whole string.
        self._repeat_prompt: str = repeat_prompt or (
            f"I'm sorry, {self._persona.honorific}; could you repeat that?"
        )
        self._run_id: str = run_id or audit_log.run_id

        # Pre-resolve numeric thresholds so the hot path doesn't
        # repeatedly cross attribute lookups.
        self._ack_delay_seconds: float = max(
            0.0, self._config.acknowledge_after_ms / 1000.0
        )
        self._max_schema_violation_retries: int = self._config.max_tool_retry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_turn(
        self,
        transcript: Transcript,
        state: ConversationState,
    ) -> AssistantResponse:
        """Execute one full conversational turn.

        See module docstring for the high-level steps. The method is
        idempotent only with respect to ``transcript``; ``state`` is
        mutated to record the new turn (user side at the start,
        assistant side at the end). When the transcript is gated the
        method short-circuits *before* mutating ``state`` so a low-
        confidence reading never corrupts the conversation history.
        """
        # ------------------ 1. Empty / low-confidence gate ----------------
        # Requirement 1.8 / Property 13: an empty transcript or
        # confidence below the threshold MUST NOT call backend.stream.
        if not transcript.text.strip() or transcript.confidence < self._min_confidence:
            return await self._handle_low_confidence(transcript)

        # ------------------ 2. Mutate state with the user side ------------
        # We capture ``now()`` once so the started/finished markers on
        # this turn share a single clock reading on tests with a
        # frozen :class:`FakeTimeSource`.
        started_at = self._time.now()
        state.append_user(transcript.text, at=started_at)

        # ------------------ 3. Retrieve memories --------------------------
        memories = await self._safe_retrieve_memories(transcript.text)

        # ------------------ 4. Build messages and tools -------------------
        messages = self._render_messages(state, memories)
        # The registry returns ``list[dict[str, Any]]``; the LLMBackend
        # protocol expects ``list[ToolDefinition]`` (a TypedDict subset of
        # the same shape). We cast for the type-checker — at runtime they
        # are interchangeable plain dicts.
        tools_raw = self._skills.mistral_tool_definitions()
        tools = cast(list[ToolDefinition], tools_raw)

        # ------------------ 5. LLM/tool loop ------------------------------
        loop_state = _TurnLoopState()
        try:
            await self._run_llm_tool_loop(
                messages=messages,
                tools=tools,
                state=state,
                loop_state=loop_state,
            )
        finally:
            # Always cancel any outstanding acknowledgement timer; we
            # don't want a stray "One moment, sir." after the turn is
            # already over.
            loop_state.cancel_pending_ack()

        # ------------------ 6. Persona guard on final text ----------------
        # Requirement 11.5: scan for forbidden self-references and rewrite
        # them. The streaming path already runs the guard sentence-by-
        # sentence; running it once more on the joined text catches the
        # case where the offending phrase straddled a chunk boundary the
        # accumulator never observed (for example, a phrase that sat
        # entirely inside the buffered tail).
        final_text_raw = "".join(loop_state.text_chunks)
        final_text, _violated = self._persona_guard.check(
            final_text_raw, cast(PersonaLike, self._persona)
        )

        # ------------------ 7. Finalise turn ------------------------------
        finished_at = self._time.now()
        state.append_assistant(
            final_text,
            at=finished_at,
            tool_calls=list(loop_state.all_tool_calls),
        )

        if not state.incognito:
            await self._safe_persist_turn(state)

        return AssistantResponse(
            text=final_text,
            audio_started_at=loop_state.audio_started_at,
            cited_urls=tuple(loop_state.cited_urls),
            tool_calls=tuple(loop_state.all_tool_calls),
        )

    # ------------------------------------------------------------------
    # Step 1 — low-confidence gate
    # ------------------------------------------------------------------

    async def _handle_low_confidence(
        self, transcript: Transcript
    ) -> AssistantResponse:
        """Speak the "please repeat" prompt without invoking the backend.

        Property 13 explicitly forbids calling ``backend.stream`` here,
        so we route the whole response through the TTS path and return
        an :class:`AssistantResponse` carrying the prompt as text. We
        deliberately do NOT mutate ``state`` — a low-confidence reading
        is treated as a non-event by the conversation history.
        """
        del transcript  # unused; the prompt is fixed per persona.
        audio_started_at = self._time.now()
        await self._safe_speak(self._repeat_prompt)
        return AssistantResponse(
            text=self._repeat_prompt,
            audio_started_at=audio_started_at,
            cited_urls=(),
            tool_calls=(),
        )

    # ------------------------------------------------------------------
    # Step 3 — memory retrieval
    # ------------------------------------------------------------------

    async def _safe_retrieve_memories(self, query: str) -> list[MemoryRecord]:
        """Retrieve top-K memories, swallowing failures.

        Memory retrieval is best-effort: a misbehaving vector store must
        not break the conversation (Requirement 17.1 / Property 7). We
        log the failure and proceed with an empty memory list so the
        LLM still answers, just without retrieved context.
        """
        if self._memory_k == 0:
            return []
        try:
            return await self._memory.retrieve(query, k=self._memory_k)
        except Exception:  # pragma: no cover - logged for diagnostics
            logger.exception(
                "MemoryStore.retrieve raised during handle_turn; "
                "continuing without retrieved context"
            )
            return []

    # ------------------------------------------------------------------
    # Step 4 — message rendering
    # ------------------------------------------------------------------

    def _render_messages(
        self,
        state: ConversationState,
        memories: list[MemoryRecord],
    ) -> list[Message]:
        """Build the message list for ``backend.stream``.

        Layout (Property 11 / CP14 invariant: ``messages[0]`` is the
        persona system prompt verbatim):

        * ``messages[0]`` — system, persona system prompt.
        * ``messages[1]`` (optional) — system, retrieved memories under a
          delimited "memory" section. Omitted when no memories were
          returned, so the message list shape stays minimal.
        * ``messages[2..]`` — alternating user / assistant turns from
          ``state.turns``. Past turns appear in original order; the
          *current* turn (the one ``append_user`` just added) contributes
          only its user side because the assistant side has not been
          generated yet.
        """
        messages: list[Message] = []

        # ``messages[0]`` MUST be the persona system prompt verbatim
        # (Property 11). We hand the same string through every turn —
        # the persona is frozen on construction so it is safe to share
        # the reference.
        persona_system: SystemMessage = {
            "role": "system",
            "content": self._persona.system_prompt,
        }
        messages.append(persona_system)

        if memories:
            memory_system: SystemMessage = {
                "role": "system",
                "content": _format_memory_section(memories),
            }
            messages.append(memory_system)

        # Replay history. The current (in-progress) turn is the last
        # entry in ``state.turns``; its assistant side is empty, so we
        # only emit a user message for it.
        last_index = len(state.turns) - 1
        for idx, turn in enumerate(state.turns):
            user_msg: UserMessage = {"role": "user", "content": turn.user}
            messages.append(user_msg)
            if idx == last_index:
                # Current turn — assistant side comes from the LLM next.
                continue
            assistant_msg: AssistantMessage = {
                "role": "assistant",
                "content": turn.assistant,
            }
            if turn.tool_calls:
                assistant_msg["tool_calls"] = [
                    _tool_call_to_assistant(tc) for tc in turn.tool_calls
                ]
            messages.append(assistant_msg)

        return messages

    # ------------------------------------------------------------------
    # Step 5 — LLM / tool loop
    # ------------------------------------------------------------------

    async def _run_llm_tool_loop(
        self,
        *,
        messages: list[Message],
        tools: list[ToolDefinition],
        state: ConversationState,
        loop_state: _TurnLoopState,
    ) -> None:
        """Drive the streaming LLM/tool dispatch cycle until it terminates.

        Loops until the LLM emits a content-only response, or until the
        per-turn schema-violation retry budget is exhausted. Each
        iteration:

        1. Opens a fresh stream via ``backend.stream``.
        2. Forwards content deltas through :class:`SentenceAccumulator`
           to the TTS engine, sentence-by-sentence (Requirement 12.2).
        3. Reassembles tool calls into :class:`ToolCall` objects via
           the backend's own assembly (the manager only consumes
           :class:`~jarvis.llm.base.ToolCallEvent`, never raw fragments).
        4. After the stream closes, if there are tool calls, dispatches
           each via :meth:`_dispatch_tool_call` while a single
           "One moment, …" acknowledgement timer runs in parallel.
        """
        schema_violation_retries = 0

        while True:
            accumulator = SentenceAccumulator()
            tool_calls_this_round: list[ToolCall] = []
            content_chunks_this_round: list[str] = []

            async with self._backend.stream(messages, tools=tools) as stream:
                async for event in stream:
                    if event.type == "content_delta":
                        if event.text:
                            content_chunks_this_round.append(event.text)
                            for sentence in accumulator.feed(event.text):
                                await self._speak_sentence(sentence, loop_state)
                    elif event.type == "tool_call":
                        tool_calls_this_round.append(event.tool_call)
                    # Defensive: unknown event types are ignored. A future
                    # backend may emit extra event variants that this
                    # manager does not care about (e.g., usage metering);
                    # treating them as no-ops keeps forward compatibility.

            tail = accumulator.flush()
            if tail and not tool_calls_this_round:
                # Only flush the tail when the turn ends here; otherwise
                # the partial sentence is part of an in-progress tool
                # narration that the next iteration will continue.
                await self._speak_sentence(tail, loop_state)

            # Append the assistant message for this round to ``messages``
            # so the next iteration sees the conversation history.
            assistant_text = "".join(content_chunks_this_round)
            loop_state.text_chunks.extend(content_chunks_this_round)
            messages.append(
                _make_assistant_message(assistant_text, tool_calls_this_round)
            )

            if not tool_calls_this_round:
                # Pure-text turn — we are done.
                return

            loop_state.all_tool_calls.extend(tool_calls_this_round)

            had_schema_violation = await self._dispatch_tool_calls(
                tool_calls_this_round,
                messages=messages,
                state=state,
                loop_state=loop_state,
            )

            if had_schema_violation:
                schema_violation_retries += 1
                if schema_violation_retries > self._max_schema_violation_retries:
                    # Requirement 14.5: cap retries on schema_violation
                    # at the configured maximum. We log so operators can
                    # spot a misbehaving Skill schema.
                    logger.warning(
                        "DialogManager: schema_violation retry budget "
                        "exhausted (%d retries); ending turn",
                        schema_violation_retries - 1,
                    )
                    return

    async def _speak_sentence(
        self, sentence: str, loop_state: _TurnLoopState
    ) -> None:
        """Apply persona-guard rewrite and forward ``sentence`` to TTS.

        We run the guard's *rewrite* path (the synchronous, pure-string
        substitution) on every sentence so users never hear the LLM
        identify itself as ChatGPT / Claude / "as an AI language model"
        (Requirement 11.5). The "trigger one stricter regeneration" path
        from the design is incompatible with sentence-by-sentence
        streaming; the post-loop guard pass at the end of
        :meth:`handle_turn` covers the remaining rewrite needs over the
        complete text.
        """
        rewritten, _ = self._persona_guard.check(
            sentence, cast(PersonaLike, self._persona)
        )
        text = rewritten.strip()
        if not text:
            return
        if loop_state.audio_started_at is None:
            loop_state.audio_started_at = self._time.now()
        await self._safe_speak(text)

    async def _safe_speak(self, text: str) -> None:
        """Wrap :meth:`TTSEngine.speak` with exception isolation.

        TTS errors are non-fatal (Requirement 17.1 / Property 7). We log
        and continue so a hiccup in the audio device cannot abort the
        whole turn — the user still gets the textual transcript via the
        UI log path.
        """
        try:
            await self._tts.speak(text)
        except Exception:  # pragma: no cover - exercised via fakes
            logger.exception(
                "TTSEngine.speak raised; dropping sentence and continuing"
            )

    # ------------------------------------------------------------------
    # Step 5b — tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        messages: list[Message],
        state: ConversationState,
        loop_state: _TurnLoopState,
    ) -> bool:
        """Dispatch each Tool_Call, append results, return schema_violation flag.

        Schedules a single "One moment, …" acknowledgement timer that
        fires when the *combined* dispatch wall-time exceeds
        ``acknowledge_after_ms`` (Requirement 12.3). The timer is
        cancelled as soon as dispatch completes — even if it has already
        fired, the cancellation is a no-op.
        """
        had_schema_violation = False

        ack_task = self._schedule_acknowledgement_timer(loop_state)
        try:
            for tc in tool_calls:
                result = await self._dispatch_tool_call(tc, state)
                messages.append(_tool_result_message(tc, result))
                if result.error_code == "schema_violation":
                    had_schema_violation = True
        finally:
            if ack_task is not None and not ack_task.done():
                ack_task.cancel()
                # Await the cancellation so we don't leak a "Task was
                # destroyed but it is pending!" warning at shutdown.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await ack_task

        return had_schema_violation

    async def _dispatch_tool_call(
        self,
        tool_call: ToolCall,
        state: ConversationState,
    ) -> SkillResult:
        """Authorize, dispatch, and audit-close a single Tool_Call.

        Returns the :class:`SkillResult` produced by the registry (or a
        synthetic ``access_denied`` result when the user denies a
        destructive call). Every successful destructive dispatch emits
        the matching ``executed`` audit row via
        :meth:`AuthorizationPolicy.record_executed`; failures in the
        registry surface as an ``error`` row via
        :meth:`AuthorizationPolicy.record_error_after_confirmation`.
        """
        manifest = self._lookup_manifest(tool_call.skill_name)

        is_destructive = False
        allowlist_bypass = False

        if manifest is not None:
            classification = self._policy.classify(tool_call, manifest)
            is_destructive = classification == DESTRUCTIVE
            if is_destructive:
                allowlist_bypass = (
                    self._policy.match_allowlist(tool_call) is not None
                )
                consented = await self._policy.confirm(
                    tool_call,
                    self._confirmation_dialog or _DenyAllDialog(),
                )
                if not consented:
                    # The policy already wrote a ``denied`` audit entry;
                    # produce a synthetic result so the LLM sees the
                    # cancellation in its tool stream.
                    return SkillResult.error(
                        "access_denied",
                        "User declined to authorise this Destructive_Action.",
                    )

        ctx = self._build_skill_context(state)
        result = await self._skills.dispatch(
            tool_call.skill_name,
            dict(tool_call.arguments),
            ctx,
        )

        if is_destructive:
            await self._close_destructive_audit_pair(
                tool_call=tool_call,
                result=result,
                allowlist_bypass=allowlist_bypass,
            )

        return result

    async def _close_destructive_audit_pair(
        self,
        *,
        tool_call: ToolCall,
        result: SkillResult,
        allowlist_bypass: bool,
    ) -> None:
        """Record the closing audit row for a destructive dispatch.

        Either ``executed`` (success) or ``error`` (registry returned a
        structured failure) — both close the
        ``confirmation_requested`` row that was emitted before the
        Skill ran (CP9).
        """
        if result.ok:
            await self._policy.record_executed(
                tool_call,
                outcome="ok",
                allowlist_bypass=allowlist_bypass,
            )
            return

        # Failed dispatch after confirmation → ``error`` audit row.
        # ``error_code`` is a Literal of the SkillResult error taxonomy;
        # we widen to plain str when prefixing with the bypass marker so
        # mypy doesn't complain about the f-string concatenation.
        base_outcome: str = result.error_code or "internal_error"
        outcome = (
            f"{base_outcome}:allowlist_bypass" if allowlist_bypass else base_outcome
        )
        await self._policy.record_error_after_confirmation(
            tool_call,
            outcome=outcome,
            justification=result.error_message,
        )

    def _lookup_manifest(self, skill_name: str) -> SkillManifest | None:
        """Return the manifest of a registered Skill, or ``None``.

        ``None`` means the LLM hallucinated a Skill name. The registry
        will surface ``internal_error`` on dispatch; classifying the
        Tool_Call as ``Safe`` here is the right call — there is no
        side effect to confirm because dispatch will never run.
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            return None
        return skill.manifest

    def _build_skill_context(self, state: ConversationState) -> SkillContext:
        """Assemble a :class:`SkillContext` for one Skill dispatch.

        Today the manager only forwards the bits the registry / built-in
        Skills actually consult: the audit log, time source, incognito
        flag, and run id. Application wiring (task 19.x) will extend
        this builder to pass through the platform adapter, credential
        store, LLM backend, and provider clients.
        """
        return SkillContext(
            audit_log=self._audit,
            time_source=self._time,
            incognito=state.incognito,
            run_id=self._run_id,
        )

    # ------------------------------------------------------------------
    # Step 5c — acknowledgement timer
    # ------------------------------------------------------------------

    def _schedule_acknowledgement_timer(
        self, loop_state: _TurnLoopState
    ) -> asyncio.Task[None] | None:
        """Schedule the "One moment, …" utterance.

        Returns the scheduled :class:`asyncio.Task`, or ``None`` when
        the configured threshold disables the feature
        (``acknowledge_after_ms == 0``). The task awaits
        :func:`asyncio.sleep` for the threshold and then forwards the
        prompt through :meth:`_safe_speak`. Cancellation before fire
        produces a clean no-op; cancellation after fire is also a
        no-op because the TTS enqueue has already returned.
        """
        if self._ack_delay_seconds <= 0:
            return None
        # Multiple rounds of tool dispatch in one turn each schedule
        # their own timer. We DO NOT keep a per-turn singleton because
        # the user benefit of "another quiet pause? speak again" is
        # marginal and the bookkeeping adds risk. The previous timer is
        # always cancelled before the next round begins.
        ack_text = f"One moment, {self._persona.honorific}."
        task = asyncio.create_task(
            self._fire_acknowledgement(ack_text, loop_state),
            name="dialog-ack-timer",
        )
        loop_state.pending_ack = task
        return task

    async def _fire_acknowledgement(
        self,
        text: str,
        loop_state: _TurnLoopState,
    ) -> None:
        """Sleep, then speak the acknowledgement.

        Wrapped as a separate coroutine so cancellation cleanly aborts
        either the sleep or the in-flight speak — both will surface as
        :class:`asyncio.CancelledError` and be swallowed by the caller's
        ``try/except`` in :meth:`_dispatch_tool_calls`.
        """
        try:
            await asyncio.sleep(self._ack_delay_seconds)
        except asyncio.CancelledError:
            return
        # Track that we did emit an acknowledgement so tests can assert
        # the design's "speaks within 1.5 s" behaviour.
        loop_state.acknowledgement_spoken = True
        if loop_state.audio_started_at is None:
            loop_state.audio_started_at = self._time.now()
        await self._safe_speak(text)

    # ------------------------------------------------------------------
    # Step 7 — persistence
    # ------------------------------------------------------------------

    async def _safe_persist_turn(self, state: ConversationState) -> None:
        """Persist the latest turn through :class:`MemoryStore`.

        Wrapped in exception isolation: a failing memory write must
        never propagate into the user-visible response (Requirement
        17.1 / Property 7). Incognito gating happens in
        :meth:`handle_turn` before this method is called, but
        :class:`MemoryStore` itself also short-circuits on the
        ``incognito`` flag.
        """
        last = state.last_turn()
        if last is None:  # pragma: no cover - guarded by handle_turn
            return
        try:
            await self._memory.persist_turn(last, persona=self._persona)
        except Exception:  # pragma: no cover - logged for diagnostics
            logger.exception(
                "MemoryStore.persist_turn raised at end of handle_turn; "
                "continuing"
            )


# ---------------------------------------------------------------------------
# Internal turn-state container
# ---------------------------------------------------------------------------


@dataclass
class _TurnLoopState:
    """Mutable scratch space shared across a single ``handle_turn`` call.

    Pulled into its own dataclass so the helper methods can mutate the
    same store without needing a host of out-parameters. Not part of the
    public API.
    """

    text_chunks: list[str] = field(default_factory=list)
    all_tool_calls: list[ToolCall] = field(default_factory=list)
    audio_started_at: datetime | None = None
    cited_urls: list[str] = field(default_factory=list)
    acknowledgement_spoken: bool = False
    pending_ack: asyncio.Task[None] | None = None

    def cancel_pending_ack(self) -> None:
        """Best-effort cancel of any acknowledgement timer still running.

        Called from the ``finally`` clause in :meth:`handle_turn` so that
        an unexpected error on the LLM side does not strand a timer
        that would speak after the turn has already been finalised.
        """
        if self.pending_ack is None:
            return
        if not self.pending_ack.done():
            self.pending_ack.cancel()
        self.pending_ack = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_memory_section(memories: list[MemoryRecord]) -> str:
    """Render retrieved memories as the secondary system message body.

    Format::

        # Memory
        [memory 1] (chat, 2024-01-01T12:00:00+00:00)
        User: hi
        Assistant: hello there

        [memory 2] (preference)
        ...
        # End memory

    The header / footer markers are matched verbatim by tests
    (Requirement 10.4 — "clearly delimited 'memory' section").
    """
    lines: list[str] = [_MEMORY_SECTION_HEADER, ""]
    for index, record in enumerate(memories, start=1):
        # Annotate as ``list[str]`` because ``record.category`` is a
        # Literal and mypy would otherwise infer the list element type
        # from the first append, refusing the ISO-8601 string later on.
        meta_parts: list[str] = [record.category]
        with contextlib.suppress(Exception):
            # Defensive: a corrupted timestamp shouldn't crash rendering.
            meta_parts.append(record.timestamp.isoformat())
        header = f"[memory {index}] ({', '.join(meta_parts)})"
        lines.append(header)
        # Memory content is plaintext at runtime; the MemoryStore has
        # already decrypted and (optionally) redacted it.
        lines.append(record.content.rstrip())
        lines.append("")
    lines.append(_MEMORY_SECTION_FOOTER)
    return "\n".join(lines)


def _tool_call_to_assistant(tc: ToolCall) -> AssistantToolCall:
    """Project a :class:`ToolCall` to the assistant-message replay shape.

    Used when re-rendering historical turns whose assistant side carried
    tool calls. Both Mistral and the Ollama OpenAI-compatible API expect
    the ``arguments`` field to be a JSON *string*; we forward the
    Tool_Call's preserved ``raw_arguments`` so the byte-equal payload
    survives through the next backend invocation.
    """
    function: AssistantToolCallFunction = {
        "name": tc.skill_name,
        "arguments": tc.raw_arguments,
    }
    return {"id": tc.id, "type": "function", "function": function}


def _make_assistant_message(
    content: str, tool_calls: list[ToolCall]
) -> AssistantMessage:
    """Build the assistant message appended after one stream completes.

    Used as the "history record" the model sees when the dispatch loop
    re-enters. The message includes the streamed content (possibly empty
    when the round was tool-calls-only) and the matching ``tool_calls``
    array.
    """
    msg: AssistantMessage = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [_tool_call_to_assistant(tc) for tc in tool_calls]
    return msg


def _tool_result_message(tool_call: ToolCall, result: SkillResult) -> ToolMessage:
    """Render a :class:`SkillResult` as the next tool message.

    The body is JSON-serialised with stable key ordering so the LLM sees
    a deterministic payload. We include both the structured ``value``
    (or ``error_code`` / ``error_message``) and the ``ok`` flag so the
    model can distinguish "tool ran, no useful payload" from "tool
    failed".
    """
    if result.ok:
        body: dict[str, Any] = {"ok": True, "value": result.value}
    else:
        body = {
            "ok": False,
            "error_code": result.error_code,
            "error_message": result.error_message,
        }
        if result.value is not None:
            body["value"] = result.value
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "role": "tool",
        "content": payload,
        "tool_call_id": tool_call.id,
        "name": tool_call.skill_name,
    }


# ---------------------------------------------------------------------------
# Safety default: the no-op ConfirmationDialog
# ---------------------------------------------------------------------------


class _DenyAllDialog:
    """Confirmation dialog used when no real one is wired up.

    The dialog returns the empty string, which the
    :class:`AuthorizationPolicy`'s affirmative-response parser treats as
    denial. This is the safety default required for a destructive
    Tool_Call whose context cannot ask the user for explicit consent —
    the cinematic JARVIS does not silently send email behind the user's
    back.
    """

    async def ask_user(self, prompt: str) -> str:
        del prompt  # unused; the answer is always denial.
        logger.warning(
            "DialogManager has no ConfirmationDialog wired up; treating "
            "destructive Tool_Call as denied."
        )
        return ""
