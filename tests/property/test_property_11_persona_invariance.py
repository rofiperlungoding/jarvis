"""Property 11 — Persona invariance.

From ``design.md §Correctness Properties``:

    *For every* ``LLMBackend.stream`` invocation issued by
    :meth:`DialogManager.handle_turn`, the ``messages`` argument SHALL
    have ``messages[0]`` equal to a ``system`` message whose
    ``content`` is byte-identical to the active
    :attr:`PersonaProfile.system_prompt`.

This property encodes Requirement 11.1 ("a system prompt … on every
LLM_Backend invocation"), Requirement 11.3 ("MAINTAIN consistent
persona tone across turns by including the persona system prompt in
every LLM_Backend invocation"), and Requirement 11.4 ("WHERE the user
has configured a custom persona profile, THE Dialog_Manager SHALL
apply that profile in place of the default JARVIS profile") together:
the same ``system_prompt`` string MUST appear at index 0 of every
``stream`` call's ``messages`` argument, including the intermediate
calls a multi-round tool-dispatch loop produces.

What the test exercises
-----------------------

The :class:`DialogManager`'s LLM/tool loop opens a fresh
``backend.stream`` per dispatch round (see
``_run_llm_tool_loop``) — one call per tool round plus one final
content-only round to terminate the loop. Property 11 therefore needs
multiple rounds to be meaningful: a single-round test would only
verify the persona on the *first* call, leaving the regression "the
manager appends to ``messages`` and the persona slot drifts when
history grows" undetected.

We drive the manager with:

* **Hypothesis-generated transcripts** (``transcripts()`` from
  :mod:`tests.strategies`) constrained to bypass the low-confidence
  gate (Requirement 1.8 / Property 13) so the manager actually
  invokes the backend.
* **Hypothesis-generated tool-call sequences** — a list of per-round
  tool-call counts that drives the multi-round loop. The stub backend
  replays one round per entry plus one final content-only round; the
  list may be empty (single-round, content-only conversation) or up to
  three tool rounds (multi-round multi-call conversation).

The recording stub backend (:class:`_RecordingBackend`) snapshots the
``messages`` argument on every ``stream`` invocation and exposes the
recorded list to the test body. The assertion battery walks each
recording and checks the three byte-equal clauses Property 11
quantifies over.

Closed-taxonomy companions
--------------------------

Two example tests pin specific corner cases that are valuable to keep
visible in the regression set:

* ``test_persona_invariance_for_default_persona_baseline`` — a smoke
  test using the shipped JARVIS persona and a single-round
  conversation. Guards against a regression where the property test
  passes only because the strategy happens to skip every example.
* ``test_persona_invariance_for_custom_persona`` — exercises
  Requirement 11.4 by registering a custom :class:`PersonaProfile`
  with a deliberately distinct ``system_prompt`` and asserting the
  manager forwards *that* prompt verbatim — not the default JARVIS
  one — on every call.

Validates: Requirements 11.1, 11.3, 11.4 (CP14)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
from tests.strategies import transcripts

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import DialogManager
from jarvis.dialog.persona import PersonaProfile, default_jarvis_persona
from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.base import SkillContext, SkillManifest, SkillResult
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: A fixed, timezone-aware reference instant for every
#: :class:`FakeTimeSource` constructed in this module. The exact
#: instant does not matter for Property 11 — we only need the clock to
#: be aware so :class:`Transcript` and :class:`ConversationState`
#: dataclasses accept it.
_FROZEN_INSTANT: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

#: The fixed text the recording backend emits in the *final* content-
#: only round. The string contains no forbidden self-references, so
#: the persona guard leaves it alone — a regression in the guard
#: would otherwise be visible as a mismatched ``response.text``
#: assertion in our companion tests.
_FINAL_RESPONSE_TEXT: str = "Very well, sir; the matter is settled."

#: Cap on the total number of tool rounds Hypothesis generates per
#: example. The Dialog_Manager opens one ``backend.stream`` per round
#: plus one final content-only round, so a cap of 3 produces up to 4
#: ``stream`` invocations per example. This is enough to exercise
#: persona invariance across "first call" / "intermediate call" /
#: "final call" without making the test unbearably slow on CI.
_MAX_TOOL_ROUNDS: int = 3

#: Cap on tool calls per round. Multiple calls per round share the
#: same ``messages`` argument (the round's ``stream`` is opened once),
#: so the cap exists primarily to stress the dispatch loop's
#: per-round bookkeeping rather than to widen Property 11's coverage.
_MAX_CALLS_PER_ROUND: int = 2


# ---------------------------------------------------------------------------
# Tolerant skill — passes any Tool_Call argument shape
# ---------------------------------------------------------------------------


class _TolerantSkill:
    """Skill that accepts any object as arguments and always succeeds.

    The :class:`SkillRegistry` validates Tool_Call arguments against the
    Skill's JSON Schema before dispatch (Requirement 14.4 / Property 2).
    Property 11 is about persona placement in the LLM prompt, *not*
    about the schema validator, so we register a skill whose schema
    accepts every dict — ``{"type": "object", "additionalProperties":
    True}`` with no ``required`` list. The recording stub backend can
    then emit bare ``ToolCallEvent``s with empty ``arguments`` and the
    registry will dispatch every one of them, producing the multi-
    round LLM loop the property needs.

    The executor records its argument count for diagnostic assertions
    in the companion tests; Property 11 itself does not consult it.
    """

    def __init__(self) -> None:
        self.manifest: SkillManifest = SkillManifest(
            name="tolerant_skill",
            description=(
                "Test fixture: accepts any arguments and returns success. "
                "Used by Property 11 (persona invariance) to drive the "
                "multi-round LLM dispatch loop."
            ),
            json_schema={"type": "object", "additionalProperties": True},
            destructive=False,
        )
        self.execute_calls: int = 0

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        del args, ctx
        self.execute_calls += 1
        return SkillResult.success(value={"ok": True})


# ---------------------------------------------------------------------------
# Recording stub backend
# ---------------------------------------------------------------------------


class _RecordingBackend:
    """LLM backend that records every ``messages`` argument it receives.

    Property 11's universal quantification reduces to "snapshot the
    ``messages`` list passed to every ``stream`` call and check
    ``messages[0]``". The stub:

    * Records a defensive copy of the ``messages`` list on every call
      so subsequent mutations by the dispatch loop (which appends
      assistant / tool messages between rounds) do not corrupt the
      snapshot.
    * Replays a Hypothesis-generated script of per-round events. Each
      script entry is the list of :class:`LLMEvent` values to yield
      for that round. The final entry is the content-only round that
      terminates the loop (a single :class:`ContentDeltaEvent` with
      :data:`_FINAL_RESPONSE_TEXT`); all preceding entries emit one
      or more :class:`ToolCallEvent` values referencing the
      ``tolerant_skill`` Skill.
    * Raises :class:`AssertionError` when the manager opens *more*
      streams than the script provides — a regression that loops
      indefinitely would surface as a clear test failure rather than
      a hang.
    """

    def __init__(self, *, rounds: list[list[LLMEvent]]) -> None:
        # Defensive copy of the script so mutations on the caller's
        # list (post-construction) do not affect replay.
        self._rounds: list[list[LLMEvent]] = [list(r) for r in rounds]
        # Each entry is a snapshot of the ``messages`` list as the
        # manager passed it. We freeze each slot to a plain ``dict``
        # because the backend's ``Message`` typeddict union widens to
        # ``object`` when copied; the recorded data is for read-only
        # assertions.
        self.calls: list[list[dict[str, Any]]] = []

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[LLMEvent]]:
        del tools, kwargs  # unused — Property 11 only inspects messages.
        # Snapshot the messages list. ``dict(m)`` copies each
        # TypedDict instance into a plain dict so a later
        # ``messages.append`` by the dispatch loop cannot mutate the
        # recorded snapshot. The recorded slot is a list-of-dict, not
        # a list-of-TypedDict, but we narrow back to ``dict`` so test
        # assertions can subscript freely.
        self.calls.append([dict(m) for m in messages])
        if not self._rounds:
            raise AssertionError(
                "_RecordingBackend ran out of scripted rounds; the "
                "Dialog_Manager opened more streams than expected."
            )
        events = self._rounds.pop(0)
        return _events_cm(events)


@contextlib.asynccontextmanager
async def _events_cm(
    events: list[LLMEvent],
) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    """Async-context-manager wrapper around a fixed event list."""
    yield _aiter(events)


async def _aiter(events: list[LLMEvent]) -> AsyncIterator[LLMEvent]:
    for e in events:
        yield e


# ---------------------------------------------------------------------------
# No-op TTS / memory fakes
# ---------------------------------------------------------------------------


class _NoopTTS:
    """No-op :class:`TTSEngine` stand-in.

    The dispatch loop streams sentences through TTS at sentence
    boundaries. Property 11 is indifferent to the spoken output; a
    recording-only fake keeps the test free of audio device
    dependencies and removes a source of nondeterministic timing.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:  # pragma: no cover - never invoked
        return None

    def is_playing(self) -> bool:  # pragma: no cover - never invoked
        return False

    async def aclose(self) -> None:  # pragma: no cover - never invoked
        return None


class _StubMemoryStore:
    """Empty-retrieval :class:`MemoryStore` stand-in.

    ``retrieve`` always returns ``[]`` so the *secondary* (memory)
    system message is never emitted — this keeps the prompt shape
    minimal and ensures ``messages[0]`` is exclusively the persona
    system message. ``persist_turn`` records the turn it was given
    for completeness; the property does not consult the recording.
    """

    def __init__(self) -> None:
        self.persisted_turns: list[Any] = []

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        del query, k
        return []

    async def persist_turn(
        self, turn: Any, persona: Any | None = None
    ) -> list[Any]:
        del persona
        self.persisted_turns.append(turn)
        return []


# ---------------------------------------------------------------------------
# Script builder
# ---------------------------------------------------------------------------


def _build_script(tool_call_counts: list[int]) -> list[list[LLMEvent]]:
    """Translate a per-round count list into a backend script.

    Each entry of ``tool_call_counts`` is the number of tool calls the
    corresponding round emits. We append one final content-only round
    whose single :class:`ContentDeltaEvent` carries
    :data:`_FINAL_RESPONSE_TEXT` so the dispatch loop terminates.

    Tool-call ids are made unique with a ``round-call`` template
    (``"r{round}-c{call}"``) because the registry / audit log key on
    them; collisions would interact with the audit log's uniqueness
    constraints in confusing ways unrelated to Property 11.
    """
    rounds: list[list[LLMEvent]] = []
    for round_idx, count in enumerate(tool_call_counts):
        events: list[LLMEvent] = []
        for call_idx in range(count):
            tc = ToolCall(
                id=f"r{round_idx}-c{call_idx}",
                skill_name="tolerant_skill",
                arguments={},
                raw_arguments="{}",
            )
            events.append(ToolCallEvent(tool_call=tc))
        rounds.append(events)
    rounds.append([ContentDeltaEvent(text=_FINAL_RESPONSE_TEXT)])
    return rounds


# ---------------------------------------------------------------------------
# Manager wiring helper
# ---------------------------------------------------------------------------


def _build_manager(
    *,
    persona: PersonaProfile,
    backend: _RecordingBackend,
    audit_path: Path,
) -> tuple[DialogManager, _NoopTTS, _StubMemoryStore, AuditLog, _TolerantSkill]:
    """Assemble a :class:`DialogManager` for one Hypothesis example.

    The wiring is intentionally end-to-end: real :class:`SkillRegistry`,
    :class:`AuthorizationPolicy`, :class:`AuditLog`, and
    :class:`PersonaGuard` instances live in the dependency graph so
    Property 11's invariant is verified against the *full* prompt-
    rendering path, not a stripped-down harness.

    The dispatch loop's ``acknowledge_after_ms`` timer is disabled
    (``acknowledge_after_ms=0``) because the timer's wall-clock
    behaviour is irrelevant to persona placement — leaving it on would
    only add asyncio scheduling noise to the test.
    """
    skills = SkillRegistry()
    tolerant_skill = _TolerantSkill()
    skills.register(tolerant_skill)

    tts = _NoopTTS()
    memory = _StubMemoryStore()
    time_source = FakeTimeSource(now=_FROZEN_INSTANT)
    audit_log = AuditLog(
        audit_path,
        time_source=time_source,
        run_id="prop11-run",
    )
    policy = AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )
    config = DialogConfig.model_validate({"acknowledge_after_ms": 0})

    manager = DialogManager(
        backend=backend,
        skills=skills,
        memory=memory,  # type: ignore[arg-type]
        policy=policy,
        persona=persona,
        tts=tts,
        audit_log=audit_log,
        config=config,
        time_source=time_source,
    )
    return manager, tts, memory, audit_log, tolerant_skill


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _tool_call_counts() -> st.SearchStrategy[list[int]]:
    """Generate the per-round tool-call counts for the backend script.

    The list may be empty (single-round, content-only conversation —
    the "no tools needed" baseline) or up to :data:`_MAX_TOOL_ROUNDS`
    entries long (the multi-round multi-call worst case). Each entry
    is the number of tool calls that round emits, in the inclusive
    range ``[1, _MAX_CALLS_PER_ROUND]``.

    The lower bound of 1 is structural rather than aesthetic: the
    :class:`DialogManager`'s ``_run_llm_tool_loop`` terminates as
    soon as a round emits zero tool calls (the content-only
    termination signal). A zero-count entry would therefore make the
    loop return *before* the final content-only round we always
    append, and the call-count post-condition below would see one
    fewer call than the script length predicts. Constraining each
    entry to at least one tool call keeps the script-length /
    call-count relationship deterministic; the empty list still
    covers the "no tools at all" baseline.
    """
    return st.lists(
        st.integers(min_value=1, max_value=_MAX_CALLS_PER_ROUND),
        min_size=0,
        max_size=_MAX_TOOL_ROUNDS,
    )


def _persona_safe_transcripts() -> st.SearchStrategy[Transcript]:
    """Wrapper around :func:`transcripts` that bypasses the gate.

    The :class:`DialogManager` short-circuits empty / low-confidence
    transcripts (Requirement 1.8 / Property 13) *before* invoking the
    backend. Property 11 quantifies over ``stream`` invocations, so a
    gated transcript would make the property vacuously true on that
    example. We constrain ``confidence >= 0.5`` (well above the 0.4
    gate) and force non-empty text so the dispatch loop actually
    opens at least one stream per example.

    The gate's empty-text clause uses ``transcript.text.strip()``, so
    a whitespace-only transcript still fires the gate even with
    ``allow_empty=False``. We additionally ``.filter`` for
    ``text.strip()`` to ensure every example carries genuine content.
    """
    return transcripts(allow_empty=False, min_confidence=0.5).filter(
        lambda t: bool(t.text.strip())
    )


# ---------------------------------------------------------------------------
# Property 11 — universal quantification over all stream invocations
# ---------------------------------------------------------------------------


@given(
    transcript=_persona_safe_transcripts(),
    tool_call_counts=_tool_call_counts(),
)
@settings(
    suppress_health_check=(
        # ``tmp_path_factory`` is per-test and shared across examples;
        # each example's :func:`tmp_path_factory.mktemp` call gives us
        # an isolated subdirectory so SQLite file locks do not
        # collide on Windows.
        HealthCheck.function_scoped_fixture,
        # Wiring a fresh manager + audit log + persona-guard chain
        # per example lands above Hypothesis's default 200 ms budget
        # on slower runners; the actual work is bounded and benign.
        HealthCheck.too_slow,
    ),
)
def test_property_11_persona_at_messages_zero(
    tmp_path_factory: Any,
    transcript: Transcript,
    tool_call_counts: list[int],
) -> None:
    """Every ``backend.stream`` invocation has the persona at ``messages[0]``.

    For each Hypothesis example we wire a fresh :class:`DialogManager`
    around a :class:`_RecordingBackend`, drive a single
    :meth:`DialogManager.handle_turn` call with the generated
    ``transcript`` and the generated ``tool_call_counts`` script, and
    then walk the recorded ``messages`` snapshots. Three byte-equal
    clauses MUST hold for every recorded call:

    1. ``messages`` is non-empty (the persona slot exists).
    2. ``messages[0]['role'] == 'system'``.
    3. ``messages[0]['content'] == persona.system_prompt`` byte-for-
       byte (no truncation, no honorific drift, no UTF-8 round-trip
       glitch).

    A fourth post-condition asserts that the recording captured the
    expected number of calls (one per script entry), guarding against
    the "the manager never invoked the backend" vacuity case.

    **Validates: Requirements 11.1, 11.3, 11.4 (CP14)**
    """

    audit_path: Path = tmp_path_factory.mktemp("prop11-audit") / "audit.sqlite"

    persona = default_jarvis_persona()
    rounds = _build_script(tool_call_counts)
    backend = _RecordingBackend(rounds=rounds)

    manager, _tts, _memory, audit_log, _tolerant_skill = _build_manager(
        persona=persona,
        backend=backend,
        audit_path=audit_path,
    )

    state = ConversationState(
        session_id="prop11-session",
        started_at=_FROZEN_INSTANT,
    )

    try:
        asyncio.run(manager.handle_turn(transcript, state))
    finally:
        # Closing the audit log here keeps the SQLite file handle
        # from leaking across Hypothesis examples on Windows, where
        # deferred close can collide with the next example's
        # ``mktemp`` + open.
        audit_log.close()

    # ---- Post-condition 0: backend was invoked at least once --------
    # Property 11 quantifies over backend.stream invocations. If the
    # gate short-circuited (it should not, given the strategy bounds),
    # the property would be vacuously true; assert non-vacuity so a
    # regression in the strategy or the gate surfaces here rather than
    # silently passing.
    assert backend.calls, (
        "Property 11 requires at least one backend.stream call per "
        "example, but the manager never invoked the backend "
        f"(transcript={transcript.text!r}, counts={tool_call_counts!r})"
    )

    # ---- Post-condition 1..3: persona at messages[0] for every call -
    expected_prompt = persona.system_prompt
    for index, recorded_messages in enumerate(backend.calls):
        # 1. messages is non-empty — the persona slot exists.
        assert recorded_messages, (
            f"backend.stream call #{index} received an empty messages "
            f"list; persona slot missing"
        )
        first = recorded_messages[0]
        # 2. role is 'system' — the persona slot is a system message.
        assert first.get("role") == "system", (
            f"backend.stream call #{index} messages[0].role was "
            f"{first.get('role')!r}; expected 'system' "
            f"(transcript={transcript.text!r}, counts={tool_call_counts!r})"
        )
        # 3. content is byte-identical to PersonaProfile.system_prompt.
        # ``content`` may be any JSON-serialisable string; we compare
        # by Python equality first (cheap) and, if that fails, also
        # render the comparison's UTF-8 byte view in the assertion
        # message so a Unicode-normalisation regression is easy to
        # diagnose.
        actual = first.get("content")
        assert actual == expected_prompt, (
            f"backend.stream call #{index} messages[0].content drifted "
            f"from PersonaProfile.system_prompt:\n"
            f"  expected: {expected_prompt!r}\n"
            f"  actual:   {actual!r}\n"
            f"(transcript={transcript.text!r}, counts={tool_call_counts!r})"
        )
        # The system message MUST NOT carry any other keys (e.g.,
        # ``tool_call_id`` would be an obvious bug). The TypedDict
        # for :class:`SystemMessage` defines exactly two keys; assert
        # the recorded snapshot matches that shape.
        assert set(first.keys()) == {"role", "content"}, (
            f"backend.stream call #{index} messages[0] has unexpected "
            f"keys {sorted(first.keys())!r}; expected {{'role', 'content'}}"
        )

    # ---- Post-condition 4: call count matches the script length -----
    # The script feeds the dispatch loop ``len(tool_call_counts) + 1``
    # rounds (the trailing content-only round). The manager opens
    # exactly one ``backend.stream`` per round, so this equality is
    # a regression guard against a malformed script or a schema
    # violation that prematurely terminated the loop.
    expected_call_count = len(tool_call_counts) + 1
    assert len(backend.calls) == expected_call_count, (
        f"backend.stream invoked {len(backend.calls)} times; expected "
        f"{expected_call_count} (one per script round + final content-only) "
        f"(counts={tool_call_counts!r})"
    )


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: default JARVIS persona baseline
# ---------------------------------------------------------------------------


def test_persona_invariance_for_default_persona_baseline(
    tmp_path: Path,
) -> None:
    """Smoke test: the default persona is forwarded verbatim on a single round.

    A single content-only round with a hand-built transcript exercises
    the simplest path through the dispatch loop. Pinning it as a
    dedicated example test means a regression in the strategy
    (``transcripts()`` accidentally generating only gated transcripts,
    say) cannot silently turn the property test into a no-op.

    **Validates: Requirements 11.1, 11.3, 11.4 (CP14)**
    """
    persona = default_jarvis_persona()
    rounds = _build_script([])  # no tool rounds — just the final content
    backend = _RecordingBackend(rounds=rounds)
    manager, _tts, _memory, audit_log, _tolerant_skill = _build_manager(
        persona=persona,
        backend=backend,
        audit_path=tmp_path / "audit.sqlite",
    )
    transcript = Transcript(
        text="how are you, JARVIS?",
        confidence=0.95,
        started_at=_FROZEN_INSTANT,
        duration_ms=500,
        language="en",
    )
    state = ConversationState(
        session_id="prop11-baseline",
        started_at=_FROZEN_INSTANT,
    )
    try:
        asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    assert len(backend.calls) == 1
    first = backend.calls[0][0]
    assert first["role"] == "system"
    assert first["content"] == persona.system_prompt


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: custom persona profile (Requirement 11.4)
# ---------------------------------------------------------------------------


def test_persona_invariance_for_custom_persona(
    tmp_path: Path,
) -> None:
    """A custom :class:`PersonaProfile` is forwarded verbatim, not the default.

    Requirement 11.4 states that a user-configured custom persona
    replaces the default JARVIS profile. This test constructs a
    persona with a deliberately distinct ``system_prompt`` (and a
    different ``name`` / ``honorific`` so the two prompts cannot
    accidentally collide) and verifies that *that* prompt — not the
    JARVIS default — appears at ``messages[0]`` on every round of a
    multi-round dispatch. Two tool rounds are scripted so the test
    also covers Requirement 11.3 ("consistent persona tone across
    turns by including the persona system prompt in every
    LLM_Backend invocation") explicitly.

    **Validates: Requirement 11.4 (CP14)**
    """
    custom_prompt = (
        "You are FRIDAY, a private AI assistant. Address the user as "
        '"boss". Be friendly, modern, and direct.'
    )
    custom_persona = PersonaProfile(
        name="FRIDAY",
        honorific="boss",
        system_prompt=custom_prompt,
        tts_voice="en_US-ryan-medium",
        # Trim the forbidden-self-refs to a single distinct string so
        # the persona guard does not rewrite our final-response text
        # by accident.
        forbidden_self_refs=("ChatGPT",),
    )
    rounds = _build_script([1, 1])  # two tool rounds + one content round
    backend = _RecordingBackend(rounds=rounds)
    manager, _tts, _memory, audit_log, tolerant_skill = _build_manager(
        persona=custom_persona,
        backend=backend,
        audit_path=tmp_path / "audit.sqlite",
    )
    transcript = Transcript(
        text="run the tolerant skill twice please",
        confidence=0.95,
        started_at=_FROZEN_INSTANT,
        duration_ms=500,
        language="en",
    )
    state = ConversationState(
        session_id="prop11-custom",
        started_at=_FROZEN_INSTANT,
    )
    try:
        asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    # Three rounds total (two tool + one content) → three stream calls.
    assert len(backend.calls) == 3
    # The skill was dispatched once per tool round.
    assert tolerant_skill.execute_calls == 2
    # Every call must carry the *custom* prompt, byte-for-byte. The
    # default JARVIS prompt opens with ``"You are JARVIS"``; the
    # custom prompt opens with ``"You are FRIDAY"``. The negative
    # check below catches a regression where the manager forgot to
    # honour the configured persona.
    default_prompt = default_jarvis_persona().system_prompt
    assert custom_prompt != default_prompt, (
        "test invariant: the custom prompt must differ from the default"
    )
    for index, recorded_messages in enumerate(backend.calls):
        first = recorded_messages[0]
        assert first["role"] == "system"
        assert first["content"] == custom_prompt, (
            f"backend.stream call #{index} forwarded the default "
            "JARVIS prompt instead of the configured custom persona"
        )
        assert first["content"] != default_prompt
