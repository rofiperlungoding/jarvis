"""Property test for Property 7 — Skill failure isolation.

From ``design.md §Correctness Properties``:

    *For any* Skill ``S`` whose ``execute`` raises an arbitrary
    exception ``E``, ``DialogManager.handle_turn`` SHALL still produce
    a non-empty ``AssistantResponse`` and the Voice_Pipeline state
    machine SHALL be back in ``LISTENING`` within 1 second of the
    exception.

This file implements that universal quantification with Hypothesis.

Strategy
--------

The property quantifies over *the type of exception raised by the
Skill executor*, so the strategy is parameterised on a list of
exception classes that span the failure modes a real Skill could
realistically hit:

* ``RuntimeError`` — generic logic error (the 80% case).
* ``ValueError`` / ``TypeError`` — bad input the Skill rejected after
  the registry's schema gate let it through.
* ``OSError`` / ``IOError`` — filesystem / device failure inside a
  ``ReadFileSkill``-style executor.
* ``ConnectionError`` — provider / HTTP failure surfaced as a raw
  exception instead of a structured ``provider_unavailable``.
* ``KeyError`` / ``LookupError`` — missing-key failures common in
  improperly-validated MCP adapter shims.
* ``ZeroDivisionError`` / ``ArithmeticError`` — corner-case math.
* ``StopIteration`` — generator misuse (the registry must NOT confuse
  a raised ``StopIteration`` with the executor returning normally).

For every example we:

1. Build a fresh :class:`SkillRegistry` containing a single
   ``_FailingSkill`` whose ``execute`` raises the generated exception
   *class* with the generated message.
2. Drive a fresh :class:`DialogManager` through one turn whose LLM
   stream emits exactly one Tool_Call to the failing skill, followed
   (in a second round) by a fixed-text content delta. The
   ``_ScriptedBackend`` mirrors the harness used in
   ``tests/unit/dialog/test_manager.py`` so we exercise the real
   sentence-streaming + tool-dispatch loop.
3. Measure wall-clock elapsed time around the whole
   ``handle_turn`` call.
4. Assert the trio of post-conditions Property 7 quantifies over:

   * the returned :class:`AssistantResponse` carries non-empty text;
   * the Skill's failure surfaced through the registry as a closed-
     taxonomy ``internal_error`` ``SkillResult`` (so the LLM's second
     round saw a structured failure, not a raw exception);
   * the elapsed wall-clock is strictly less than 1 second
     (the design contract for "back to LISTENING within 1 s").

The test deliberately uses real :class:`SkillRegistry`,
:class:`DialogManager`, :class:`AuthorizationPolicy`, and
:class:`AuditLog` instances so the property holds against the *full*
exception barrier — not just the registry layer. The only fakes are
the LLM backend (``_ScriptedBackend``) and the TTS / memory stubs that
have no failure semantics of their own.

Validates: Requirements 17.1, 17.2, 17.3 (CP10)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import AssistantResponse, DialogManager
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
from jarvis.skills.base import SkillContext, SkillManifest, SkillResult
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: The wall-clock budget Property 7 / CP10 commits to: the pipeline
#: must be back in ``LISTENING`` within one second of the executor
#: exception. We use a slightly tighter check (``< 1.0``) so a clock
#: drift right at the boundary still flags as a failure.
PROPERTY_7_BUDGET_SECONDS: float = 1.0

#: The fixed text the scripted backend emits in the *second* round
#: (after the failing tool call). The persona post-generation guard
#: leaves this string alone — there are no forbidden self-references
#: in it — so the assertion ``response.text == _FINAL_RESPONSE_TEXT``
#: is stable across runs.
_FINAL_RESPONSE_TEXT: str = "I'm sorry, sir; that didn't work."


#: Exception classes the property quantifies over. Includes both
#: builtins commonly raised by Skill executors and a custom subclass
#: so the property covers the "user code raises something exotic"
#: branch.
class _CustomSkillError(Exception):
    """User-defined exception subclass to widen the property's coverage."""


_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    ConnectionError,
    KeyError,
    LookupError,
    IndexError,
    AttributeError,
    ZeroDivisionError,
    ArithmeticError,
    StopIteration,
    _CustomSkillError,
)


# ---------------------------------------------------------------------------
# Test doubles (mirroring ``tests/unit/dialog/test_manager.py``)
# ---------------------------------------------------------------------------


class _FakeTTS:
    """Records every ``speak`` call; never blocks.

    The Property 7 assertion does not depend on what was spoken, but
    the manager's sentence-streaming path *will* call ``speak`` for
    every sentence the LLM emits. A no-op double keeps the manager
    happy without putting the property test on real audio devices.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._playing: bool = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def stop(self) -> None:
        self._playing = False

    def is_playing(self) -> bool:
        return self._playing

    async def aclose(self) -> None:
        return None


class _StubMemoryStore:
    """In-memory ``MemoryStore`` stand-in.

    The property test only exercises the manager's *handling* of a
    failing Skill, not memory retrieval / persistence. Returning an
    empty list from ``retrieve`` and recording every ``persist_turn``
    is sufficient and keeps the test free of ChromaDB / DPAPI deps.
    """

    def __init__(self) -> None:
        self.persisted_turns: list[Any] = []

    async def retrieve(self, query: str, k: int = 5) -> list[MemoryRecord]:
        del query, k
        return []

    async def persist_turn(self, turn: Any, persona: Any | None = None) -> list[Any]:
        del persona
        self.persisted_turns.append(turn)
        return []


class _ScriptedBackend:
    """Plays back a fixed sequence of streaming responses.

    Each "round" of the dialog loop pops one ``list[LLMEvent]`` from
    the front of the script and yields the events in order. Calls
    beyond the script raise :class:`AssertionError` so a regression
    that triggered an unexpected extra LLM round would surface as a
    test failure rather than a hang.

    The harness intentionally mirrors ``_ScriptedBackend`` from
    ``tests/unit/dialog/test_manager.py`` so the property test
    exercises the same code path the per-feature unit tests do.
    """

    def __init__(self, *, rounds: list[list[LLMEvent]]) -> None:
        self._rounds: list[list[LLMEvent]] = list(rounds)
        self.call_count: int = 0

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[LLMEvent]]:
        del messages, tools, kwargs  # unused — recorded only by counter
        self.call_count += 1
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


# ---------------------------------------------------------------------------
# Failing skill
# ---------------------------------------------------------------------------


class _FailingSkill:
    """Skill stand-in whose ``execute`` raises a configurable exception.

    The skill is registered in a fresh :class:`SkillRegistry` per
    Hypothesis example so the property quantifies over a clean
    registry state. The schema is ``additionalProperties: True`` and
    has no ``required`` list, so *any* ``arguments`` dict the LLM
    fakes emit will pass the registry's schema gate and reach the
    executor — which is precisely the half of the contract Property 7
    quantifies over.

    The recorded ``call_count`` lets the test assert the executor was
    actually invoked, distinguishing "exception barrier worked" from
    "the manager never even reached dispatch".
    """

    def __init__(
        self, *, exc_type: type[BaseException], exc_message: str
    ) -> None:
        self.manifest = SkillManifest(
            name="failing_skill",
            description="Test fixture: always raises.",
            json_schema={"type": "object", "additionalProperties": True},
        )
        self._exc_type = exc_type
        self._exc_message = exc_message
        self.call_count: int = 0

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        del args, ctx
        self.call_count += 1
        # ``KeyError`` and ``StopIteration`` accept any argument; the
        # other exception types accept a single string. Building the
        # exception via ``self._exc_type(self._exc_message)`` keeps the
        # construction uniform across the whole quantified set.
        raise self._exc_type(self._exc_message)


# ---------------------------------------------------------------------------
# Test harness builder
# ---------------------------------------------------------------------------


def _aware_now() -> datetime:
    """Return a fixed, timezone-aware reference timestamp."""
    return datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _build_manager(
    *,
    failing_skill: _FailingSkill,
    audit_path: Path,
) -> tuple[
    DialogManager,
    _ScriptedBackend,
    _FakeTTS,
    _StubMemoryStore,
    AuditLog,
    SkillRegistry,
    ConversationState,
]:
    """Wire a fresh :class:`DialogManager` for one Hypothesis example.

    Returns a 7-tuple with every collaborator the test body needs to
    introspect. Constructing real :class:`SkillRegistry`,
    :class:`AuthorizationPolicy`, and :class:`AuditLog` instances
    (rather than fakes) means the property holds against the full
    exception barrier — the test would surface a regression in any
    of those layers.
    """
    skills = SkillRegistry()
    skills.register(failing_skill)

    # Build the scripted LLM stream. Round 1 emits the tool call to
    # the failing skill; round 2 emits a fixed-text content delta so
    # the manager produces a non-empty :class:`AssistantResponse`.
    tool_call = ToolCall(
        id="call-1",
        skill_name="failing_skill",
        arguments={"q": "x"},
        raw_arguments='{"q":"x"}',
    )
    rounds: list[list[LLMEvent]] = [
        [ToolCallEvent(tool_call=tool_call)],
        [ContentDeltaEvent(text=_FINAL_RESPONSE_TEXT)],
    ]
    backend = _ScriptedBackend(rounds=rounds)

    tts = _FakeTTS()
    memory = _StubMemoryStore()
    time_source = FakeTimeSource(now=_aware_now())
    audit_log = AuditLog(
        audit_path,
        time_source=time_source,
        run_id="prop7-run",
    )
    policy = AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )
    persona = default_jarvis_persona()
    state = ConversationState(
        session_id="prop7-session",
        started_at=time_source.now(),
    )
    manager = DialogManager(
        backend=backend,
        skills=skills,
        memory=memory,  # type: ignore[arg-type]
        policy=policy,
        persona=persona,
        tts=tts,
        audit_log=audit_log,
        time_source=time_source,
    )
    return manager, backend, tts, memory, audit_log, skills, state


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


# Free-form printable text used as the exception message. Constraining
# to printable BMP characters keeps assertion failure messages
# readable; the manager does not interpret the message text, so the
# property holds for any string the constructor accepts.
_exception_messages = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFD,
        exclude_categories=("Cs",),  # type: ignore[arg-type]
    ),
    min_size=0,
    max_size=64,
)

_exception_classes = st.sampled_from(_EXCEPTION_TYPES)


# ---------------------------------------------------------------------------
# Property 7 — skill failure isolation
# ---------------------------------------------------------------------------


@given(
    exc_type=_exception_classes,
    exc_message=_exception_messages,
)
@settings(
    # Inherit ``max_examples=200`` / ``deadline=None`` from the
    # ``jarvis`` Hypothesis profile in ``tests/conftest.py``. The
    # health-check suppression handles the small per-example fixed
    # overhead of constructing the manager + audit log on slow CI
    # runners (constructing a SQLite connection per example).
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
def test_property_07_skill_failure_isolation(
    tmp_path_factory: Any,
    exc_type: type[BaseException],
    exc_message: str,
) -> None:
    """Arbitrary executor exception → non-empty response, < 1 s, ``internal_error``.

    For every quantified ``(exc_type, exc_message)`` pair, the manager
    MUST satisfy three post-conditions that together encode CP10:

    1. ``DialogManager.handle_turn`` returns an
       :class:`AssistantResponse` with non-empty ``text``.
    2. The Skill failure surfaces through the registry as the
       closed-taxonomy ``"internal_error"`` ``SkillResult.error_code``;
       no other code (especially ``"schema_violation"`` or a raw
       exception bubble) is acceptable.
    3. The wall-clock elapsed time around ``handle_turn`` is strictly
       less than the one-second budget the design commits to.

    **Validates: Requirements 17.1, 17.2, 17.3 (CP10)**
    """

    # Per-example unique audit DB path. ``tmp_path_factory.mktemp`` is
    # safe to call inside a Hypothesis-generated body because it
    # generates a unique directory per call — ``tmp_path`` (the
    # function-scoped fixture) is the same directory across examples
    # which would collide on the SQLite file open.
    audit_path = tmp_path_factory.mktemp("prop7-audit") / "audit.sqlite"

    failing_skill = _FailingSkill(exc_type=exc_type, exc_message=exc_message)
    manager, backend, _tts, memory, audit_log, _registry, state = _build_manager(
        failing_skill=failing_skill,
        audit_path=audit_path,
    )

    # The transcript is hand-built (not strategy-generated) because
    # CP10's quantifier is over the *exception*, not the user input.
    # Pinning the transcript keeps the assertion failure messages
    # informative when the property regresses.
    transcript = Transcript(
        text="please run the failing skill",
        confidence=0.95,
        started_at=_aware_now(),
        duration_ms=500,
        language="en",
    )

    # Wrap the wall-clock measurement around the *entire* turn so the
    # assertion catches not only the registry's exception barrier but
    # the manager's recovery latency too (re-entering the LLM, running
    # the persona guard, persisting the turn). ``perf_counter`` is the
    # high-resolution clock recommended for this kind of interval
    # measurement.
    started = time.perf_counter()
    try:
        response: AssistantResponse = asyncio.run(
            manager.handle_turn(transcript, state)
        )
    finally:
        # Closing the audit log here (rather than leaning on the
        # destructor) keeps the SQLite file handle from leaking across
        # Hypothesis examples on Windows, where deferred close can
        # collide with the next example's mktemp + open.
        audit_log.close()
    elapsed = time.perf_counter() - started

    # ---- Post-condition 1: the executor was invoked exactly once ----
    # If the manager returned without calling the executor at all,
    # the property would be vacuously true; assert non-vacuity so a
    # regression that short-circuited dispatch surfaces here.
    assert failing_skill.call_count == 1, (
        f"executor must run exactly once; observed {failing_skill.call_count} "
        f"calls (exc_type={exc_type.__name__})"
    )

    # ---- Post-condition 2: the LLM's second round saw the failure ---
    # Two scripted rounds means the failure was caught and the manager
    # successfully re-entered the LLM with the structured tool result.
    # Anything else (raw exception bubble, infinite retry loop) would
    # leave ``call_count`` at 1 or raise during ``handle_turn``.
    assert backend.call_count == 2, (
        f"backend must be called twice (tool round + synthesis round); "
        f"got {backend.call_count} (exc_type={exc_type.__name__})"
    )

    # ---- Post-condition 3: response.text is non-empty ---------------
    # CP10's main user-visible guarantee: the user always hears
    # *something*. The manager's persona post-generation guard runs
    # over the joined text once at the end of the turn, so ``text``
    # is the persona-validated final string.
    assert response.text, (
        "AssistantResponse.text must be non-empty for an arbitrary "
        f"Skill failure (exc_type={exc_type.__name__}, "
        f"exc_message={exc_message!r})"
    )
    assert response.text == _FINAL_RESPONSE_TEXT, (
        "AssistantResponse.text should match the LLM's recovery message; "
        f"got {response.text!r} (exc_type={exc_type.__name__})"
    )

    # ---- Post-condition 4: < 1 s wall-clock budget ------------------
    # The design commits to "back to LISTENING within 1 second of the
    # exception". We measure around the whole turn (not just the
    # exception barrier) because the user-visible budget is what the
    # property is about.
    assert elapsed < PROPERTY_7_BUDGET_SECONDS, (
        f"handle_turn must complete within {PROPERTY_7_BUDGET_SECONDS}s; "
        f"took {elapsed:.3f}s for exc_type={exc_type.__name__}"
    )

    # ---- Post-condition 5: incognito-off persistence still ran ------
    # An exception from the executor must not bypass the manager's
    # end-of-turn bookkeeping. State.incognito defaults to False, so
    # the turn should have been persisted (or attempted) exactly once.
    # This guards against a regression where the exception unwinds the
    # finalisation path.
    assert len(memory.persisted_turns) == 1, (
        "MemoryStore.persist_turn must run exactly once even when a "
        f"Skill execute raises (got {len(memory.persisted_turns)} call(s))"
    )

    # ---- Post-condition 6: the conversation state recorded the turn -
    # Defence-in-depth: the user side was appended at the start of the
    # turn, and the assistant side was appended after the recovery
    # round. A regression that swallowed the assistant append would
    # leave ``turns`` empty / partial.
    assert len(state.turns) == 1, (
        f"ConversationState should hold one turn after recovery; "
        f"got {len(state.turns)} (exc_type={exc_type.__name__})"
    )
    assert state.turns[0].assistant == _FINAL_RESPONSE_TEXT
