"""Property 13 — STT empty / low-confidence gating.

From ``design.md §Correctness Properties`` and Requirement 1.8:

    *For every* :class:`Transcript` ``T`` whose ``T.text == ""`` (after
    whitespace stripping) or whose ``T.confidence < 0.4``,
    :meth:`DialogManager.handle_turn` SHALL NOT call
    :meth:`LLMBackend.stream`. Instead the manager prompts the user to
    repeat (Requirement 1.8) and SHALL NOT mutate the
    :class:`ConversationState` to record an assistant turn.

This is the gating contract the rest of the dialog pipeline relies on:
once the gate fires, no LLM tokens are spent, no memory is persisted,
no tool is dispatched, and the conversation history stays exactly where
it was before the spurious utterance arrived. The corresponding
implementation lives in :meth:`DialogManager._handle_low_confidence`.

What the property quantifies over
---------------------------------

The Hypothesis strategy ``transcripts(...)`` (from
:mod:`tests.strategies`) is constrained per-example to land on *one*
of the gate's two trigger conditions:

* **Empty text branch** — ``text`` is the empty string or pure
  whitespace; ``confidence`` ranges over ``[0.0, 1.0]``. The gate's
  first clause (``not transcript.text.strip()``) fires regardless of
  the confidence value.
* **Low-confidence branch** — ``text`` is non-empty; ``confidence``
  ranges over ``[0.0, 0.4)``. The gate's second clause
  (``transcript.confidence < self._min_confidence``) fires regardless
  of the text content.

Per-example we draw a boolean to pick the branch, then draw a
correspondingly-constrained transcript from the strategy. This keeps
shrinkage well-defined (a failing example collapses cleanly to either
``Transcript(text="", confidence=0)`` or ``Transcript(text="x",
confidence=0)``) and stops Hypothesis from wasting examples on the
post-gate path Property 5 / Property 11 already cover.

Three post-conditions form the property
---------------------------------------

For every drawn :class:`Transcript`, after exactly one call to
:meth:`DialogManager.handle_turn`:

1. The recording :class:`_RecordingBackend`'s ``stream`` was invoked
   zero times. The backend's :meth:`stream` raises
   :class:`AssertionError` when called, so any regression that lets a
   gated transcript through to the LLM surfaces inside the
   :func:`asyncio.run` boundary as a failing example.
2. The :class:`AssistantResponse` carries a "please repeat"-style
   clarification phrase. The default :class:`PersonaProfile` renders
   the phrase ``"I'm sorry, sir; could you repeat that?"``; we accept
   any string containing the case-insensitive word ``"repeat"`` so a
   future localisation pass that keeps the same intent does not break
   the test. The TTS engine receives the same phrase via
   :meth:`TTSEngine.speak`, which we cross-check.
3. The :class:`ConversationState` has *not* gained an assistant turn —
   ``state.turns`` is still the empty list, and
   ``state.pending_confirmation`` is still ``None``. This guards
   against a regression where the manager appends a partial turn
   before short-circuiting.

Closed-taxonomy companions pin three corner cases that survive
shrinkage poorly when left only to Hypothesis:

* The empty-string transcript (``text=""``).
* The whitespace-only transcript (``text="\\t   \\n"``).
* A non-empty transcript at exactly ``confidence == 0.39`` — the
  largest float strictly below the 0.4 floor. Hypothesis's float
  shrinker tends to land on values much smaller than the boundary,
  so a dedicated example test is the easiest way to keep the
  boundary covered.

Validates: Requirement 1.8
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
from tests.strategies import transcripts

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import DEFAULT_MIN_CONFIDENCE, DialogManager
from jarvis.dialog.persona import default_jarvis_persona
from jarvis.llm.base import LLMEvent, Message, ToolDefinition
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: A fixed, timezone-aware reference instant for every
#: :class:`FakeTimeSource` constructed in this module. The exact value
#: is irrelevant to Property 13 — the gate path never reads the clock
#: in a way the assertions depend on. Pinning it makes failing
#: examples reproducible at a glance.
_FROZEN_INSTANT: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

#: Substring the gate's spoken / returned text must contain. The
#: default JARVIS prompt is ``"I'm sorry, sir; could you repeat
#: that?"``; we match on the case-insensitive word ``"repeat"`` so a
#: future localisation pass (Requirement 1.8 implies the *intent* of
#: a clarification request, not a fixed wording) does not regress
#: this property.
_REPEAT_TOKEN: str = "repeat"


# ---------------------------------------------------------------------------
# Recording stub backend — calling stream is a test failure
# ---------------------------------------------------------------------------


class _RecordingBackend:
    """LLM backend that records call attempts and refuses to stream.

    Property 13's universal quantification reduces to "the recording
    backend's :meth:`stream` MUST never be called when the transcript
    is gated." We implement the contract by:

    * Bumping a counter and snapshotting the ``messages`` argument
      every time :meth:`stream` is entered. The snapshot is for
      diagnostic richness — failing examples can show what prompt
      slipped through the gate.
    * Raising :class:`AssertionError` *before* yielding the async
      context manager. The error propagates out of the
      :func:`asyncio.run` boundary in the test body and surfaces as
      a Hypothesis-shrunk counter-example. We deliberately raise
      *eagerly* (not after entering the context) because
      :class:`DialogManager` invokes the context-manager factory
      synchronously inside its dispatch loop; raising here aborts
      the loop before any post-gate side effects can run.

    The class is intentionally *not* a context-managed iterator on
    its own — the type signature matches the :class:`LLMBackend`
    protocol so :func:`isinstance(_RecordingBackend(), LLMBackend)`
    holds without extra ceremony.
    """

    def __init__(self) -> None:
        # Per-call message snapshots, retained for diagnostic
        # richness in failing examples. ``call_count`` is the
        # quantity Property 13 actually checks.
        self.calls: list[list[dict[str, Any]]] = []

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
        del tools, kwargs  # property does not consult these.
        # Snapshot first so the diagnostic message in the assertion
        # below carries the offending prompt.
        self.calls.append([dict(m) for m in messages])
        raise AssertionError(
            "Property 13 violation: DialogManager.handle_turn invoked "
            "LLMBackend.stream on a gated transcript. messages[0] = "
            f"{self.calls[-1][0]!r}"
        )


# ---------------------------------------------------------------------------
# No-op TTS / memory fakes
# ---------------------------------------------------------------------------


class _RecordingTTS:
    """Records every spoken sentence; never blocks.

    The gate path goes through :meth:`DialogManager._safe_speak` to
    deliver the "please repeat" prompt to the TTS engine. The
    property only requires that *no LLM call* happens — the spoken
    text is companion evidence the gate fired through the documented
    branch (rather than, say, a silent return that left the user
    confused).
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

    The gate path short-circuits before memory retrieval. We still
    record :meth:`persist_turn` calls so the test body can assert
    persistence was *not* attempted — a regression that mutated
    state would otherwise also persist a phantom turn here.
    """

    def __init__(self) -> None:
        self.persisted_turns: list[Any] = []
        self.retrieve_calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        self.retrieve_calls.append((query, k))
        return []

    async def persist_turn(
        self, turn: Any, persona: Any | None = None
    ) -> list[Any]:
        del persona
        self.persisted_turns.append(turn)
        return []


# ---------------------------------------------------------------------------
# Manager wiring helper
# ---------------------------------------------------------------------------


def _build_manager(
    *,
    audit_path: Path,
) -> tuple[
    DialogManager,
    _RecordingBackend,
    _RecordingTTS,
    _StubMemoryStore,
    AuditLog,
]:
    """Assemble a :class:`DialogManager` for one Hypothesis example.

    Every dependency is real except the LLM backend, TTS, and memory
    store: those are the seams Property 13 is quantifying over. The
    :class:`SkillRegistry` is empty (the gate path never dispatches
    any skill); the :class:`AuthorizationPolicy` is fully wired so
    its constructor checks fire if the gate path ever tries to
    confirm an action; ``acknowledge_after_ms=0`` disables the
    asyncio.sleep timer so the test does not wait on it.
    """
    persona = default_jarvis_persona()
    backend = _RecordingBackend()
    tts = _RecordingTTS()
    memory = _StubMemoryStore()
    skills = SkillRegistry()
    time_source = FakeTimeSource(now=_FROZEN_INSTANT)
    audit_log = AuditLog(
        audit_path,
        time_source=time_source,
        run_id="prop13-run",
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
    return manager, backend, tts, memory, audit_log


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _gated_transcripts() -> st.SearchStrategy[Transcript]:
    """Generate :class:`Transcript` values that MUST trigger the gate.

    The gate fires when *either* ``text.strip() == ""`` *or*
    ``confidence < 0.4``. We pick a branch per example with a fair
    coin flip (``st.booleans``) and then pin the unrelated dimension
    over its full legal range — the gate's logical disjunction makes
    *both* branches genuine triggers, regardless of the other axis.

    The empty-text branch's text strategy is constrained to either
    the bare empty string or whitespace-only strings, so the
    :class:`Transcript` constructor (which accepts both) sees
    realistic gate-eligible inputs. The low-confidence branch keeps
    ``text`` non-empty to ensure the *low-confidence* clause is the
    one that fires — a regression that swapped the disjunction for a
    conjunction would let those examples through.
    """

    # Whitespace-only strings — pure spaces / tabs / newlines.
    _whitespace_text = st.text(
        alphabet=st.sampled_from((" ", "\t", "\n", "\r")),
        min_size=0,
        max_size=8,
    )

    # ``transcripts(allow_empty=True)`` already accepts empty text;
    # we constrain text via ``.filter`` because the strategy does not
    # expose a "force empty" knob and rebuilding it would duplicate
    # the BMP-printable alphabet logic. ``.map`` would also work, but
    # ``.filter`` keeps shrinkage well-defined: Hypothesis prefers
    # smaller inputs, so the empty string is the natural shrink target.
    empty_text_branch = transcripts(allow_empty=True).map(
        lambda t: Transcript(
            text="",  # force the gate's empty-text clause.
            confidence=t.confidence,
            started_at=t.started_at,
            duration_ms=t.duration_ms,
            language=t.language,
        )
    )

    whitespace_text_branch = st.builds(
        lambda ws, base: Transcript(
            text=ws,
            confidence=base.confidence,
            started_at=base.started_at,
            duration_ms=base.duration_ms,
            language=base.language,
        ),
        ws=_whitespace_text,
        base=transcripts(allow_empty=True),
    )

    # ``min_confidence=0.0, max_confidence=0.4 - epsilon`` keeps the
    # generated value strictly below the gate's 0.4 floor. We use
    # ``DEFAULT_MIN_CONFIDENCE`` (0.4) as the upper *exclusive*
    # boundary by passing ``DEFAULT_MIN_CONFIDENCE`` directly and
    # then ``.filter`` for strict inequality — Hypothesis's float
    # strategy is closed on both sides, so the filter is the
    # cheapest safe way to exclude exactly 0.4. Without the
    # exclusion an example at confidence==0.4 would NOT trigger the
    # gate (the manager's check is ``< self._min_confidence``).
    low_confidence_branch = transcripts(
        allow_empty=False,
        min_confidence=0.0,
        max_confidence=DEFAULT_MIN_CONFIDENCE,
    ).filter(lambda t: t.confidence < DEFAULT_MIN_CONFIDENCE)

    # Equal weight on each branch keeps the search budget evenly
    # spread between the two gate clauses. ``st.one_of`` re-exposes
    # the union without prematurely committing to one branch during
    # shrinkage.
    return st.one_of(
        empty_text_branch,
        whitespace_text_branch,
        low_confidence_branch,
    )


# ---------------------------------------------------------------------------
# Property 13 — main universally-quantified test
# ---------------------------------------------------------------------------


@given(transcript=_gated_transcripts())
@settings(
    suppress_health_check=(
        # ``tmp_path_factory`` is per-test and shared across examples;
        # each example's :func:`tmp_path_factory.mktemp` call gives us
        # an isolated subdirectory so SQLite file locks do not collide
        # on Windows.
        HealthCheck.function_scoped_fixture,
        # Wiring a fresh manager + audit log per example lands above
        # Hypothesis's default 200 ms budget on slower runners; the
        # actual work is bounded and benign.
        HealthCheck.too_slow,
    ),
)
def test_gated_transcript_skips_backend_and_does_not_mutate_state(
    tmp_path_factory: Any,
    transcript: Transcript,
) -> None:
    """The gate clauses fire universally without invoking the LLM.

    For every Hypothesis-drawn gated transcript, after one
    :meth:`DialogManager.handle_turn` call:

    1. ``backend.call_count == 0`` — :meth:`LLMBackend.stream` was
       never invoked.
    2. The returned :class:`AssistantResponse` and the spoken TTS
       text both contain the case-insensitive word ``"repeat"``.
    3. ``state.turns == []`` and
       ``state.pending_confirmation is None`` — the manager did not
       record an assistant (or even a partial user) turn for the
       gated transcript.
    4. ``memory.persisted_turns == []`` and
       ``memory.retrieve_calls == []`` — the manager did not
       attempt memory retrieval or persistence for the gated turn.

    **Validates: Requirement 1.8**
    """

    audit_path: Path = (
        tmp_path_factory.mktemp("prop13-audit") / "audit.sqlite"
    )

    manager, backend, tts, memory, audit_log = _build_manager(
        audit_path=audit_path,
    )

    state = ConversationState(
        session_id="prop13-session",
        started_at=_FROZEN_INSTANT,
    )

    try:
        response = asyncio.run(manager.handle_turn(transcript, state))
    finally:
        # Closing here keeps the SQLite file handle from leaking
        # across Hypothesis examples on Windows.
        audit_log.close()

    # ---- Post-condition 1: backend was never invoked ---------------
    assert backend.call_count == 0, (
        "Property 13 violation: backend.stream was called "
        f"{backend.call_count} times on a gated transcript "
        f"(text={transcript.text!r}, "
        f"confidence={transcript.confidence!r})"
    )

    # ---- Post-condition 2: response and TTS carry "repeat" ---------
    assert _REPEAT_TOKEN in response.text.lower(), (
        f"AssistantResponse.text {response.text!r} did not contain a "
        f"clarification cue ({_REPEAT_TOKEN!r}); the gate path is "
        f"expected to ask the user to repeat (Req 1.8)"
    )
    assert tts.spoken, (
        "Property 13 expects the gate path to speak the repeat "
        f"prompt; tts.spoken is empty (transcript={transcript!r})"
    )
    # Every spoken phrase MUST contain the cue — the gate path only
    # emits the repeat prompt and nothing else.
    for spoken in tts.spoken:
        assert _REPEAT_TOKEN in spoken.lower(), (
            f"TTS.speak received an unexpected payload {spoken!r} on "
            f"the gate path (transcript={transcript!r})"
        )

    # ---- Post-condition 3: state was not mutated -------------------
    assert state.turns == [], (
        "Property 13 violation: ConversationState.turns gained "
        f"{len(state.turns)} entries on a gated transcript "
        f"(transcript={transcript!r}, turns={state.turns!r})"
    )
    assert state.pending_confirmation is None, (
        "Property 13 violation: ConversationState.pending_confirmation "
        f"was set to {state.pending_confirmation!r} on a gated "
        f"transcript (transcript={transcript!r})"
    )

    # ---- Post-condition 4: memory was not touched ------------------
    assert memory.persisted_turns == [], (
        "Property 13 violation: MemoryStore.persist_turn was called "
        f"{len(memory.persisted_turns)} times on a gated transcript "
        f"(transcript={transcript!r})"
    )
    assert memory.retrieve_calls == [], (
        "Property 13 violation: MemoryStore.retrieve was called "
        f"{len(memory.retrieve_calls)} times on a gated transcript "
        f"(transcript={transcript!r})"
    )


# ---------------------------------------------------------------------------
# Closed-taxonomy companions — boundary corners
# ---------------------------------------------------------------------------


def test_gate_fires_on_empty_string_transcript(tmp_path: Path) -> None:
    """Pinpoint corner: bare empty-string transcript triggers the gate.

    Hypothesis's ``empty_text_branch`` shrinks to this case, but
    pinning it as a dedicated example test guards against a
    regression in the strategy (e.g., a future ``transcripts()``
    variant that disallows empty text by default) silently turning
    the property test into a no-op.

    **Validates: Requirement 1.8**
    """
    manager, backend, tts, memory, audit_log = _build_manager(
        audit_path=tmp_path / "audit.sqlite",
    )
    state = ConversationState(
        session_id="prop13-empty",
        started_at=_FROZEN_INSTANT,
    )
    transcript = Transcript(
        text="",
        confidence=0.95,  # high-confidence empty — the text clause fires
        started_at=_FROZEN_INSTANT,
        duration_ms=300,
        language="en",
    )
    try:
        response = asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    assert backend.call_count == 0
    assert _REPEAT_TOKEN in response.text.lower()
    assert tts.spoken and _REPEAT_TOKEN in tts.spoken[0].lower()
    assert state.turns == []
    assert memory.persisted_turns == []


def test_gate_fires_on_whitespace_only_transcript(tmp_path: Path) -> None:
    """Pinpoint corner: whitespace-only transcripts trigger the gate.

    Requirement 1.8's *spirit* is "the user said nothing useful";
    a transcript whose ``.strip()`` returns the empty string is the
    same situation as bare ``""`` from the user's perspective. The
    manager's gate (``not transcript.text.strip()``) honours that
    intent. Pinning a tab-and-newline example here keeps the
    invariant visible in the regression set.

    **Validates: Requirement 1.8**
    """
    manager, backend, tts, _memory, audit_log = _build_manager(
        audit_path=tmp_path / "audit.sqlite",
    )
    state = ConversationState(
        session_id="prop13-ws",
        started_at=_FROZEN_INSTANT,
    )
    transcript = Transcript(
        text="\t  \n  ",
        confidence=0.95,
        started_at=_FROZEN_INSTANT,
        duration_ms=300,
        language="en",
    )
    try:
        response = asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    assert backend.call_count == 0
    assert _REPEAT_TOKEN in response.text.lower()
    assert tts.spoken
    assert state.turns == []


def test_gate_fires_just_below_confidence_floor(tmp_path: Path) -> None:
    """Pinpoint corner: confidence one tick below the 0.4 floor triggers the gate.

    The gate uses strict inequality ``confidence < 0.4``. The largest
    representable float strictly below 0.4 is what a regression to
    ``confidence <= 0.4`` (or ``< 0.5``, etc.) would mis-handle in
    the most subtle way. Hypothesis's float shrinker tends to land
    on values much smaller than the boundary, so a dedicated
    example test is the easiest way to keep the boundary covered.

    **Validates: Requirement 1.8**
    """
    manager, backend, tts, _memory, audit_log = _build_manager(
        audit_path=tmp_path / "audit.sqlite",
    )
    state = ConversationState(
        session_id="prop13-boundary",
        started_at=_FROZEN_INSTANT,
    )
    # ``DEFAULT_MIN_CONFIDENCE`` is 0.4. ``math.nextafter(0.4, 0.0)``
    # yields the largest IEEE-754 double strictly below 0.4 — the
    # tightest regression-trap value.
    just_below = math.nextafter(DEFAULT_MIN_CONFIDENCE, 0.0)
    assert just_below < DEFAULT_MIN_CONFIDENCE, (
        "test invariant: nextafter(0.4, 0.0) must be strictly less "
        "than 0.4 — if this fails, the platform's float semantics "
        "are not IEEE-754"
    )
    transcript = Transcript(
        text="hello",  # non-empty — the confidence clause fires
        confidence=just_below,
        started_at=_FROZEN_INSTANT,
        duration_ms=300,
        language="en",
    )
    try:
        response = asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    assert backend.call_count == 0
    assert _REPEAT_TOKEN in response.text.lower()
    assert tts.spoken
    assert state.turns == []


def test_gate_does_not_fire_at_confidence_floor(tmp_path: Path) -> None:
    """Negative companion: confidence exactly at the floor is NOT gated.

    The gate is strict-less-than, so ``confidence == 0.4`` MUST go
    through to the LLM. We verify the property's *converse*: a
    transcript at the boundary value with non-empty text reaches
    the backend (the recording backend's eager assertion fires when
    ``stream`` is called — we *want* it to fire here, so we catch
    it and inspect the recorded call).

    Pinning this case prevents a future regression that "fixes" a
    perceived off-by-one by widening the gate to ``<=`` and
    silently swallowing every utterance at exactly the floor.

    **Validates: Requirement 1.8 (negative-direction sanity check)**
    """
    manager, backend, _tts, _memory, audit_log = _build_manager(
        audit_path=tmp_path / "audit.sqlite",
    )
    state = ConversationState(
        session_id="prop13-floor",
        started_at=_FROZEN_INSTANT,
    )
    transcript = Transcript(
        text="hello",
        confidence=DEFAULT_MIN_CONFIDENCE,  # exactly 0.4 — NOT gated
        started_at=_FROZEN_INSTANT,
        duration_ms=300,
        language="en",
    )
    try:
        # The recording backend raises AssertionError eagerly inside
        # ``stream``; we *expect* that here because the gate must
        # NOT fire at exactly 0.4. Catching it confirms the gate let
        # the transcript through.
        with contextlib.suppress(AssertionError):
            asyncio.run(manager.handle_turn(transcript, state))
    finally:
        audit_log.close()

    assert backend.call_count == 1, (
        "At confidence == 0.4 (the gate's strict-lt boundary), the "
        "manager MUST forward the transcript to the LLM. Recorded "
        f"call_count={backend.call_count}; transcript={transcript!r}"
    )
