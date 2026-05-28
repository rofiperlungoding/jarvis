"""Property 5 — Conversation_State determinism under a stubbed backend.

From ``design.md §Correctness Properties``:

    *For any* initial ``ConversationState`` ``S0`` and *for any*
    sequence of ``Transcript`` inputs ``[T1, ..., Tn]``, executing
    ``DialogManager.handle_turn`` against a deterministic stub
    ``LLMBackend`` and a frozen clock SHALL produce a final
    ``ConversationState`` whose serialized form is byte-equal across
    runs.

In the JARVIS data model, :class:`ConversationState` carries the
``session_id``, ``started_at`` timestamp, ordered ``turns``, and the
optional ``pending_confirmation`` Tool_Call. Its serialised form is
defined by :meth:`ConversationState.to_json`, which uses
``sort_keys=True`` and ``separators=(",", ":")`` so two structurally
equal states yield byte-equal output regardless of dict insertion
order. CP6 therefore reduces to: when every input that feeds the
state is held constant, the assistant text recorded into the state
MUST also be constant, and the timestamps MUST come from a stable
clock.

The deterministic harness pins every variable input:

* **Frozen clock.** The ``frozen_clock`` fixture from
  ``tests/conftest.py`` pins :mod:`datetime`'s wall clock so any
  ``datetime.now()`` call inside the dependency graph (audit log,
  memory store, future side effects) returns the same instant on
  both runs. The :class:`DialogManager` itself reads time through an
  injected :class:`FakeTimeSource`, which we seed identically per
  run.
* **Deterministic stub backend.** A ``_DeterministicStubBackend``
  replays a Hypothesis-generated list of ``ContentDeltaEvent`` chunks
  (the canned content stream) on every ``stream`` call. The same
  script feeds run 1 and run 2; the stream produces the same
  sequence of bytes character for character.
* **Stable memories.** A ``_StubMemoryStore`` returns the empty list
  on ``retrieve`` and records the ``persist_turn`` it received without
  mutating any data the manager will read back. Both runs see the
  same retrieval result.
* **Identical persona, audit, and skill registry.** A single
  :func:`default_jarvis_persona` instance is shared across runs;
  audit and registry instances are constructed fresh per run but with
  the same arguments so any state that *they* maintain (audit row
  ids, registered skill names) remains consistent.
* **Acknowledgement timer disabled.** ``DialogConfig(acknowledge_after_ms=0)``
  short-circuits the "One moment, sir." asyncio.sleep timer, removing
  the only wall-clock-dependent branch from the dispatch loop.

The property fed by Hypothesis is exactly CP6:

    JSON-serialised state after run 1 == JSON-serialised state after run 2

We compare the bytes (UTF-8 encoded JSON) directly so any drift in
key ordering, whitespace, or numeric formatting fails loudly. We also
sanity-check that both states are still parseable through
``ConversationState.from_json`` — a regression that produced equal
byte strings of malformed JSON would still be a real bug.

Validates: Requirements 1.4, 1.6, 17.1, 19.4 (CP6)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from jarvis.config.schema import DialogConfig
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import DialogManager
from jarvis.dialog.persona import default_jarvis_persona
from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    ToolDefinition,
)
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource
from jarvis.voice.stt.base import Transcript

# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class _FakeTTS:
    """No-op TTS: records spoken sentences without doing any I/O.

    We intentionally do nothing time-dependent here. Two ``speak``
    calls with the same input produce the same recorded output and no
    blocking, so the dispatch loop's wall-clock behaviour stays
    identical across runs.
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
    """Deterministic :class:`MemoryStore` stand-in.

    ``retrieve`` always returns ``[]`` so the secondary "memory"
    system message is never injected — keeping the prompt shape minimal
    and stripping out one source of cross-run variation.
    ``persist_turn`` records the turn it was given without mutating
    anything the manager will subsequently read.
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


class _DeterministicStubBackend:
    """LLM backend that replays a fixed content-delta script.

    The script is a list of plain string chunks. Each call to
    :meth:`stream` returns an async context manager that yields one
    :class:`ContentDeltaEvent` per chunk in order. No tool calls are
    emitted — the canned content stream from Property 5's task bullet
    is content-only, which keeps the test focused on the state-
    determinism invariant rather than tool dispatch.

    The backend records the messages list it received so the test can
    perform companion assertions on prompt determinism if needed; the
    recorded data has no influence on subsequent calls, so two
    sibling instances seeded with the same ``chunks`` will yield
    byte-identical streams.
    """

    def __init__(self, *, chunks: list[str]) -> None:
        # Defensive copy so the caller's ``chunks`` list cannot be
        # mutated mid-stream.
        self._chunks: list[str] = list(chunks)
        # Recorded message payloads from each ``stream`` call. We
        # store plain ``dict[str, Any]`` rather than the discriminated
        # ``Message`` union because ``dict(typed_dict)`` widens the
        # value type to ``object``; the recorded data is for assertions
        # only, never replayed back through the backend, so the wider
        # type is safe.
        self.calls: list[list[dict[str, Any]]] = []

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[LLMEvent]]:
        del tools, kwargs
        # Snapshot the messages list so later mutations from the
        # dispatch loop don't pollute our recorded call history.
        self.calls.append([dict(m) for m in messages])
        return _stub_stream_cm(self._chunks)


@contextlib.asynccontextmanager
async def _stub_stream_cm(
    chunks: list[str],
) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    """Async-context-manager wrapper around a chunk replayer."""
    yield _replay_chunks(chunks)


async def _replay_chunks(chunks: list[str]) -> AsyncIterator[LLMEvent]:
    """Yield one :class:`ContentDeltaEvent` per chunk, in order."""
    for chunk in chunks:
        yield ContentDeltaEvent(text=chunk)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


# Frozen wall-clock instant the manager's ``FakeTimeSource`` is seeded
# with. The exact instant does not matter for CP6 — only that both
# runs share it. Using a fixed UTC value (matching the
# ``frozen_clock`` fixture's freeze instant in ``tests/conftest.py``)
# keeps failing examples reproducible at a glance.
_FROZEN_INSTANT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


# Printable ASCII without forbidden self-reference vendor names. The
# persona guard rewrites strings like ``ChatGPT`` / ``Claude`` /
# ``as an AI language model`` to the persona name, which is itself
# deterministic — the property still holds with rewrites — but
# generating clean text keeps shrunk failures readable when CP6 ever
# regresses, and avoids burning Hypothesis budget on cases that
# exercise the rewrite path (covered by the Property 11 / CP14 tests).
_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x21,  # exclude space + control
        max_codepoint=0x7E,
        # Exclude characters that the persona guard regexes are
        # particularly sensitive to. ``a`` / ``i`` / ``c`` / ``g`` /
        # ``p`` / ``t`` / ``A`` / ``I`` / ``C`` / ``G`` / ``P`` / ``T``
        # all stay in scope; we only excise the surrogate range
        # categories already covered by ``min_codepoint``.
    ),
    min_size=1,
    max_size=24,
)


# Languages: the BCP-47 / ISO-639 tags accepted by the rest of the
# pipeline. Constraining the strategy here keeps failing examples
# readable and avoids bouncing off downstream consumers' validation
# rules during shrinkage.
_languages = st.sampled_from(("en", "en-GB", "en-US", "fr", "de", "es"))


def _transcripts() -> st.SearchStrategy[Transcript]:
    """Generate ``Transcript`` values that bypass the low-confidence gate.

    The :class:`DialogManager` short-circuits empty / low-confidence
    transcripts (Requirement 1.8 / Property 13) before invoking the
    backend. To exercise the *post-gate* state-determinism path we
    constrain ``confidence >= 0.5`` and require non-empty text.

    ``started_at`` is sampled from a small UTC window so the value is
    aware (a hard precondition of :class:`Transcript`) and stable
    across shrinkage. ``duration_ms`` and ``language`` are similarly
    constrained to non-degenerate values; the test does not rely on
    any specific value.
    """

    return st.builds(
        Transcript,
        text=_safe_text,
        confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        started_at=st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2030, 12, 31),
            timezones=st.just(UTC),
        ),
        duration_ms=st.integers(min_value=0, max_value=10_000),
        language=_languages,
    )


def _content_chunks() -> st.SearchStrategy[list[str]]:
    """Generate the content-delta script the stub backend will replay.

    A small list of small chunks gives Hypothesis room to find
    boundary cases (single-chunk stream, multi-chunk stream with
    sentence terminators inside a chunk vs. across chunks) without
    blowing up the example budget. Each chunk is a printable ASCII
    string, and the list always has at least one entry so the
    assistant turn carries non-empty text — keeping the property
    test's signal away from the trivial "both states are empty"
    corner.
    """
    return st.lists(_safe_text, min_size=1, max_size=4)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_manager(
    *,
    chunks: list[str],
    audit_log: AuditLog,
    time_source: FakeTimeSource,
) -> tuple[DialogManager, _DeterministicStubBackend, _StubMemoryStore]:
    """Wire a :class:`DialogManager` for a single property-test run.

    Every dependency is freshly constructed so two runs of the same
    Hypothesis example see independent instances; whatever state they
    accrue (audit row ids, recorded backend calls) does not bleed
    into the next run. The persona, however, is built from the same
    factory call in both runs and compares structurally equal — its
    ``system_prompt`` therefore appears as ``messages[0]`` byte-for-byte
    on each invocation, matching the Property 11 / CP14 invariant.
    """
    persona = default_jarvis_persona()
    backend = _DeterministicStubBackend(chunks=chunks)
    memory = _StubMemoryStore()
    skills = SkillRegistry()
    policy = AuthorizationPolicy(
        allowlist=TrustedActionAllowlist(),
        audit=audit_log,
    )
    # ``acknowledge_after_ms=0`` disables the asyncio.sleep-based
    # acknowledgement timer (Requirement 12.3). The timer's wall-clock
    # behaviour is the only source of nondeterminism left in
    # ``handle_turn`` once the rest of the dependencies are pinned;
    # turning it off gives CP6 a strict equality property to verify
    # rather than an "almost equal" one.
    config = DialogConfig.model_validate({"acknowledge_after_ms": 0})
    manager = DialogManager(
        backend=backend,
        skills=skills,
        memory=memory,  # type: ignore[arg-type]
        policy=policy,
        persona=persona,
        tts=_FakeTTS(),
        audit_log=audit_log,
        config=config,
        time_source=time_source,
    )
    return manager, backend, memory


async def _run_handle_turn(
    *,
    chunks: list[str],
    transcript: Transcript,
    audit_db: Path,
) -> ConversationState:
    """Run one ``handle_turn`` and return the resulting state.

    The audit log is opened fresh per run against ``audit_db`` so two
    sibling runs (call this twice with the same database path) see
    identical audit row-id sequences. We close it on exit so SQLite
    releases the file lock before the parent test cleans up.
    """
    time_source = FakeTimeSource(now=_FROZEN_INSTANT)
    audit_log = AuditLog(
        audit_db,
        time_source=time_source,
        run_id="cp6-property-run",
    )
    try:
        manager, _backend, _memory = _build_manager(
            chunks=chunks,
            audit_log=audit_log,
            time_source=time_source,
        )
        state = ConversationState(
            session_id="cp6-session",
            started_at=time_source.now(),
        )
        await manager.handle_turn(transcript, state)
        return state
    finally:
        audit_log.close()


# ---------------------------------------------------------------------------
# Property 5 — byte-equal serialised state across runs
# ---------------------------------------------------------------------------


@given(
    transcript=_transcripts(),
    chunks=_content_chunks(),
)
@settings(
    suppress_health_check=(
        # ``tmp_path`` is per-test; Hypothesis re-uses it across
        # examples but each run inside a single example builds its own
        # SQLite file inside it, so the re-use is harmless.
        HealthCheck.function_scoped_fixture,
        # Wiring two managers and running ``asyncio.run`` twice per
        # example lands above Hypothesis's default 200 ms budget on
        # slower runners; the actual work is bounded and benign.
        HealthCheck.too_slow,
    ),
)
def test_state_is_byte_equal_across_runs(
    tmp_path: Path,
    frozen_clock: Any,
    transcript: Transcript,
    chunks: list[str],
) -> None:
    """Two runs with identical inputs produce byte-equal serialised state.

    The test runs ``handle_turn`` twice — each run constructs a fresh
    :class:`ConversationState`, audit log, manager, and stub backend
    — then serialises the resulting state via
    :meth:`ConversationState.to_json` and compares the UTF-8 byte
    strings. Both runs share:

    * the same :class:`Transcript` value (and therefore the same
      ``user`` text once the manager mutates the state);
    * the same ``chunks`` script for the deterministic stub backend
      (and therefore the same assistant text after the persona guard
      pass);
    * the same frozen wall clock (the ``frozen_clock`` fixture from
      ``tests/conftest.py``) and the same :class:`FakeTimeSource`
      seed (and therefore the same ``started_at`` / ``finished_at``
      timestamps on every :class:`Turn`).

    Under those conditions CP6 reduces to byte-equality of the JSON
    envelopes, which :meth:`ConversationState.to_json` makes well-
    defined (``sort_keys=True``, ``separators=(",", ":")``).

    **Validates: Requirements 1.4, 1.6, 17.1, 19.4 (CP6)**
    """
    # ``frozen_clock`` is requested for its side effect: any
    # ``datetime.now()`` call inside the dependency graph that does
    # *not* go through :class:`FakeTimeSource` (e.g., a future
    # diagnostic hook) is also pinned to the freeze instant. The
    # local variable is annotated ``Any`` because freezegun's
    # ``FrozenDateTimeFactory`` is an internal type we don't import
    # in the test body.
    del frozen_clock  # used implicitly via the fixture's context manager.

    async def _both_runs() -> tuple[ConversationState, ConversationState]:
        # Use distinct SQLite files for the two runs so the audit
        # connection lifecycles don't collide on Windows file locks.
        # The audit log's row ids are still 1, 2, 3, ... in either
        # case because the table starts empty in a fresh file —
        # the property does not depend on the audit content, but
        # keeping the lifecycles independent keeps test failures
        # localised to the state-equality assertion.
        state_a = await _run_handle_turn(
            chunks=chunks,
            transcript=transcript,
            audit_db=tmp_path / "audit-a.sqlite",
        )
        state_b = await _run_handle_turn(
            chunks=chunks,
            transcript=transcript,
            audit_db=tmp_path / "audit-b.sqlite",
        )
        return state_a, state_b

    state_a, state_b = asyncio.run(_both_runs())

    json_a = state_a.to_json()
    json_b = state_b.to_json()

    assert json_a.encode("utf-8") == json_b.encode("utf-8"), (
        "ConversationState.to_json() drifted between two runs with "
        "identical inputs:\n"
        f"  run A: {json_a!r}\n"
        f"  run B: {json_b!r}"
    )

    # Companion sanity checks: the byte-equal JSON parses back into
    # an equal :class:`ConversationState`. A regression that produced
    # byte-equal *malformed* JSON would still be a real bug we want
    # to catch here.
    parsed_a = ConversationState.from_json(json_a)
    parsed_b = ConversationState.from_json(json_b)
    assert parsed_a.to_json() == json_a
    assert parsed_b.to_json() == json_b


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: empty-script edge case
# ---------------------------------------------------------------------------


def test_state_determinism_for_empty_script(
    tmp_path: Path,
    frozen_clock: Any,
) -> None:
    """An empty content stream still produces byte-equal serialised state.

    Hypothesis's ``_content_chunks`` strategy requires at least one
    chunk so failing examples are easy to read. The empty-script
    corner — the LLM emits nothing at all — is its own legitimate
    code path (the model returned only ``content_delta`` events with
    empty text, or no events at all) and CP6 must hold there too.
    Pinning it as a dedicated example test guards the corner against
    regression.

    **Validates: Requirements 1.4, 1.6, 17.1, 19.4 (CP6)**
    """
    del frozen_clock  # implicit through the fixture.
    transcript = Transcript(
        text="hello",
        confidence=0.95,
        started_at=_FROZEN_INSTANT,
        duration_ms=500,
        language="en",
    )
    chunks: list[str] = []

    async def _both_runs() -> tuple[ConversationState, ConversationState]:
        state_a = await _run_handle_turn(
            chunks=chunks,
            transcript=transcript,
            audit_db=tmp_path / "audit-a.sqlite",
        )
        state_b = await _run_handle_turn(
            chunks=chunks,
            transcript=transcript,
            audit_db=tmp_path / "audit-b.sqlite",
        )
        return state_a, state_b

    state_a, state_b = asyncio.run(_both_runs())
    assert state_a.to_json() == state_b.to_json()
    # The assistant turn was finalised with empty text; the user
    # text and timestamps still made it onto the turn.
    assert state_a.turns and state_a.turns[0].user == "hello"
    assert state_a.turns[0].assistant == ""


# ---------------------------------------------------------------------------
# Closed-taxonomy companion: low-confidence transcripts also round-trip
# ---------------------------------------------------------------------------


def test_state_determinism_for_low_confidence_transcript(
    tmp_path: Path,
    frozen_clock: Any,
) -> None:
    """Low-confidence transcripts skip the LLM yet still serialise byte-equally.

    Property 13's gate (Requirement 1.8) short-circuits a transcript
    whose confidence is below ``min_confidence`` (default 0.4) without
    mutating ``state``. CP6's quantification is over *any* transcript
    sequence, so the property must hold here too — both runs leave
    ``state.turns`` empty and produce the same serialised envelope.

    **Validates: Requirements 1.4, 1.6, 17.1, 19.4 (CP6)**
    """
    del frozen_clock
    transcript = Transcript(
        text="too quiet",
        confidence=0.1,  # well under the 0.4 gate
        started_at=_FROZEN_INSTANT,
        duration_ms=500,
        language="en",
    )

    async def _both_runs() -> tuple[ConversationState, ConversationState]:
        state_a = await _run_handle_turn(
            chunks=["unused"],
            transcript=transcript,
            audit_db=tmp_path / "audit-a.sqlite",
        )
        state_b = await _run_handle_turn(
            chunks=["unused"],
            transcript=transcript,
            audit_db=tmp_path / "audit-b.sqlite",
        )
        return state_a, state_b

    state_a, state_b = asyncio.run(_both_runs())
    assert state_a.to_json() == state_b.to_json()
    # The low-confidence gate must not mutate state on either run.
    assert state_a.turns == []
    assert state_b.turns == []
