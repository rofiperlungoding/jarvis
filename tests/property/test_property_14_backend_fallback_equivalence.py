"""Property 14 — Backend fallback equivalence shape.

From ``design.md §Correctness Properties``:

    *For any* request ``R = (messages, tools)`` accepted by
    ``MistralBackend.stream``, if ``BackendSelector`` opens its
    circuit and routes ``R`` to ``OllamaBackend``, the request payload
    sent to Ollama SHALL contain the same ``messages`` and a ``tools``
    array of equal length whose entries have the same ``name`` and
    ``parameters`` keys as the Mistral payload.

This file implements that universal quantification with Hypothesis.

Strategy
--------

Two free generators feed the property:

* ``messages`` — a non-empty list whose first entry is the persona
  ``SystemMessage`` (Property 11 / CP14 already pins this for
  every Dialog_Manager call), followed by an arbitrary mix of
  ``UserMessage`` / ``AssistantMessage`` / ``ToolMessage`` shapes.
  The role/content shapes mirror the wire-level :mod:`Message`
  TypedDicts in :mod:`jarvis.llm.base` — exactly what every concrete
  backend (Mistral, Ollama, BackendSelector) consumes.
* ``tools`` — drawn from
  :func:`tests.strategies.mistral_tool_payloads`. That strategy is
  the catalogue's already-canonical Mistral function-definition shape;
  re-using it here keeps Property 14 consistent with the upstream
  Property 12 invariants (``parameters.type == "object"``, etc.).

Test wiring
-----------

The selector is built around two recording stub backends:

* a *primary* (Mistral-style) configured to raise on entry — first an
  HTTP 503 to trip the circuit, then permanently kept tripping so the
  property's ``selector.stream(...)`` call routes directly to the
  fallback every time.
* a *fallback* (Ollama-style) that yields a small content delta and
  records the exact ``messages``/``tools``/``kwargs`` it was handed.

A :class:`~jarvis.utils.time_source.FakeTimeSource` keeps the
cool-down deterministic without sleeping for real seconds.

Property assertions
-------------------

After tripping the breaker and re-issuing ``selector.stream(messages,
tools=tools)``, the test asserts:

1. The fallback received exactly one call (the property call),
   confirming the breaker is open and the routing decision worked.
2. The ``messages`` list seen by the fallback is *deeply equal* to
   the caller's input — same length, same entries, same keys, same
   values.
3. The ``tools`` list seen by the fallback has the same length as
   the caller's input.
4. For every index ``i``, the function ``name`` and the set of
   ``parameters`` top-level keys match between the caller's
   ``tools[i]`` and the fallback-observed ``tools[i]``.

Hash assertions (1)-(4) directly mirror the Property 14 wording in
``design.md``: "*the request payload sent to Ollama SHALL contain
the same `messages` and a `tools` array of equal length whose
entries have the same `name` and `parameters` keys as the Mistral
payload*".

Validates: Requirements 12.4
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from hypothesis import given, strategies as st
from tests.strategies import mistral_tool_payloads

from jarvis.llm.base import ContentDeltaEvent, LLMEvent, Message, ToolDefinition
from jarvis.llm.selector import BackendSelector
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Recording stub backends
# ---------------------------------------------------------------------------


@dataclass
class _RecordedCall:
    """One observed invocation of :meth:`_RecordingBackend.stream`.

    Stores the *exact* arguments handed to the backend so the
    equivalence assertions compare against the caller's original
    payload rather than against any later mutation. The selector
    snapshots ``messages`` / ``tools`` via ``list(...)`` before
    forwarding — recording the snapshotted lists is what Property 14
    is about.
    """

    messages: list[Message]
    tools: list[ToolDefinition]
    kwargs: dict[str, Any] = field(default_factory=dict)


class _RecordingBackend:
    """Configurable :class:`~jarvis.llm.base.LLMBackend` test double.

    Mirrors the shape of the production backends (a ``stream(...)``
    method returning an async context manager) without any of the
    network plumbing. Two knobs:

    * ``enter_exc`` — if set, the context manager raises on
      ``__aenter__``. Used to drive the primary into the circuit-trip
      path (HTTP 5xx).
    * ``events`` — the events the context manager yields once entered.
      Defaults to a single content delta so the fallback path streams
      something through the selector without forcing every test to
      hand-roll a list.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[_RecordedCall] = []
        self.enter_exc: BaseException | None = None
        self.events: list[LLMEvent] | None = None

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> Any:
        # Record BEFORE entering the context manager so a tripping
        # primary still leaves a trace, mirroring the existing
        # FakeBackend in ``tests/unit/llm/test_backend_selector.py``.
        self.calls.append(
            _RecordedCall(
                messages=messages,
                tools=tools,
                kwargs=dict(kwargs),
            )
        )
        snapshot_exc = self.enter_exc
        snapshot_events = (
            list(self.events)
            if self.events is not None
            else [ContentDeltaEvent(text=f"from-{self.name}")]
        )
        return _stream_cm(snapshot_exc, snapshot_events)


@asynccontextmanager
async def _stream_cm(
    enter_exc: BaseException | None,
    events: Sequence[LLMEvent],
) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    """Async context manager mirroring the real backends' shape."""

    if enter_exc is not None:
        raise enter_exc

    async def _gen() -> AsyncIterator[LLMEvent]:
        for ev in events:
            yield ev

    yield _gen()


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    """Construct an :class:`httpx.HTTPStatusError` carrying ``status``."""

    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies — wire-shape ``messages`` lists
# ---------------------------------------------------------------------------


# A short text alphabet for free-form fields. Identical-in-spirit to
# ``tests.strategies._safe_text`` but locally scoped so this property
# test stays self-contained — Property 14 only cares about deep
# equality of the forwarded lists, so the text content is irrelevant
# beyond "Hypothesis is allowed to vary it".
_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFD,
        exclude_categories=("Cs",),  # type: ignore[arg-type]
    ),
    min_size=0,
    max_size=32,
)

# Persona system prompt — non-empty so the resulting message satisfies
# the SystemMessage ``content: str`` shape used everywhere in the
# Dialog_Manager.
_persona_text = _safe_text.filter(bool)


def _system_message() -> st.SearchStrategy[Any]:
    """A ``SystemMessage`` carrying a non-empty persona prompt."""

    return st.fixed_dictionaries(
        {
            "role": st.just("system"),
            "content": _persona_text,
        }
    )


def _user_message() -> st.SearchStrategy[Any]:
    """A ``UserMessage`` carrying free-form transcribed text."""

    return st.fixed_dictionaries(
        {
            "role": st.just("user"),
            "content": _safe_text,
        }
    )


def _assistant_message() -> st.SearchStrategy[Any]:
    """An ``AssistantMessage`` without tool calls (the common case)."""

    return st.fixed_dictionaries(
        {
            "role": st.just("assistant"),
            "content": _safe_text,
        }
    )


def _tool_message() -> st.SearchStrategy[Any]:
    """A ``ToolMessage`` replaying a Skill execution result."""

    return st.fixed_dictionaries(
        {
            "role": st.just("tool"),
            "content": _safe_text,
            "tool_call_id": st.from_regex(
                r"call_[A-Za-z0-9]{1,16}", fullmatch=True
            ),
        }
    )


def _trailing_messages() -> st.SearchStrategy[list[Message]]:
    """Zero or more non-system messages following the persona slot.

    The four message shapes (user / assistant / tool) cover the
    discriminated union :class:`Message` from :mod:`jarvis.llm.base`
    minus the persona system message — that one is pinned at
    ``messages[0]`` by Property 11 / CP14 and is generated separately
    so the wire shape matches every real Dialog_Manager call.
    """

    return st.lists(
        st.one_of(_user_message(), _assistant_message(), _tool_message()),
        min_size=0,
        max_size=4,
    )


@st.composite
def _messages_lists(draw: st.DrawFn) -> list[Message]:
    """A non-empty messages list with a system prompt at index 0."""

    head = draw(_system_message())
    tail = draw(_trailing_messages())
    return [head, *tail]


def _tools_lists() -> st.SearchStrategy[list[ToolDefinition]]:
    """Zero or more Mistral-shaped tool definitions.

    Re-uses :func:`tests.strategies.mistral_tool_payloads` so every
    generated entry already satisfies Property 12 (CP15) — flat
    object schema, scalar property types, JSON round-trippable. The
    tools list itself may be empty; the design's
    :meth:`LLMBackend.stream` contract treats an empty tools array
    as "no function calling on this turn" (Requirement 19.4).
    """

    # ``mistral_tool_payloads`` returns ``dict[str, Any]`` whose runtime
    # shape matches the :class:`ToolDefinition` TypedDict; the cast
    # tells mypy what the strategy guarantees by construction.
    return cast(
        "st.SearchStrategy[list[ToolDefinition]]",
        st.lists(mistral_tool_payloads(), min_size=0, max_size=4),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drain(
    selector: BackendSelector,
    messages: list[Message],
    tools: list[ToolDefinition],
) -> list[LLMEvent]:
    """Open ``selector.stream`` and collect every yielded event."""

    out: list[LLMEvent] = []
    async with selector.stream(messages, tools=tools) as events:
        async for event in events:
            out.append(event)
    return out


async def _trip_circuit(
    selector: BackendSelector,
    primary: _RecordingBackend,
) -> None:
    """Force the breaker open by issuing a single tripping call.

    The selector treats an HTTP 5xx on primary entry as a trip
    condition (see :func:`jarvis.llm.selector._is_server_error`).
    After this call returns, ``selector.is_open`` is ``True`` and
    the next ``stream()`` call routes straight to the fallback.
    """

    primary.enter_exc = _http_status_error(503)
    # The 5xx trip is caught inside the selector's @asynccontextmanager
    # body and routed transparently to the fallback, so this call
    # completes normally.
    await _drain(selector, [{"role": "system", "content": "trip"}], [])
    primary.enter_exc = None  # subsequent calls are blocked by the open breaker


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


def _tool_name(tool: ToolDefinition) -> str:
    """Project a tool definition to its function name."""

    return tool["function"]["name"]


def _parameter_keys(tool: ToolDefinition) -> set[str]:
    """Project a tool definition to the top-level keys of ``parameters``."""

    return set(tool["function"]["parameters"].keys())


@given(messages=_messages_lists(), tools=_tools_lists())
def test_backend_fallback_equivalence_shape(
    messages: list[Message],
    tools: list[ToolDefinition],
) -> None:
    """Fallback receives ``messages`` deeply equal and ``tools`` shape-equal.

    Property 14: when ``BackendSelector`` routes from Mistral to
    Ollama after the circuit opens, the request payload sent to
    the fallback contains the same ``messages`` and a ``tools``
    array of equal length whose entries have matching ``name`` and
    ``parameters`` keys.

    Validates: Requirements 12.4
    """

    primary = _RecordingBackend("primary")
    fallback = _RecordingBackend("fallback")
    fake_time = FakeTimeSource()

    selector = BackendSelector(
        primary,
        fallback,
        # Small but positive timeout: the trip path here is an HTTP
        # 5xx, not a sleep, so the timeout value never matters — but
        # ``BackendSelector`` rejects zero/negative.
        timeout_seconds=0.05,
        # Generous cool-down so the breaker stays open across the
        # property call without us having to advance the fake clock.
        cool_down_seconds=60.0,
        time_source=fake_time,
    )

    async def _exercise() -> None:
        # Step 1: trip the circuit.
        await _trip_circuit(selector, primary)
        assert selector.is_open, (
            "BackendSelector failed to open its circuit after a 5xx trip"
        )

        # Snapshot the fallback's call log up to this point (the
        # tripping call routed through the fallback after the primary
        # raised on entry). The property call below will append
        # exactly one new entry — the one we'll assert against.
        fallback_calls_before = len(fallback.calls)

        # Step 2: issue the actual property call. The breaker is
        # open, so this routes straight to the fallback without
        # touching the primary.
        await _drain(selector, messages, tools)

        # ------------------------------------------------------------
        # Assertion 1 — exactly one new fallback call observed.
        # The breaker is open; the primary should not have been
        # invoked at all on this turn (its call count is whatever it
        # was after step 1).
        # ------------------------------------------------------------
        assert len(fallback.calls) == fallback_calls_before + 1, (
            "BackendSelector did not route the property call to the "
            "fallback while the circuit was open: "
            f"fallback.calls={fallback.calls}"
        )
        recorded = fallback.calls[-1]

        # ------------------------------------------------------------
        # Assertion 2 — messages deep-equality.
        # ``selector.stream`` defensively copies ``messages`` via
        # ``list(...)``, so the recorded list is a *new* list object
        # but its contents must be deeply equal to the caller's.
        # ------------------------------------------------------------
        assert recorded.messages == messages, (
            "Fallback received messages that differ from the caller's: "
            f"sent={messages!r} observed={recorded.messages!r}"
        )

        # ------------------------------------------------------------
        # Assertion 3 — tools length parity.
        # ------------------------------------------------------------
        assert len(recorded.tools) == len(tools), (
            "tools array length changed across the fallback boundary: "
            f"sent={len(tools)} observed={len(recorded.tools)}"
        )

        # ------------------------------------------------------------
        # Assertion 4 — for every index, function ``name`` matches and
        # the top-level ``parameters`` keys match. This is the exact
        # invariant Property 14 spells out and matches the wording in
        # the task description ("matching name/parameters keys").
        # ------------------------------------------------------------
        for index, (sent, observed) in enumerate(
            zip(tools, recorded.tools, strict=True)
        ):
            assert _tool_name(observed) == _tool_name(sent), (
                f"tools[{index}].function.name diverged across the "
                f"fallback boundary: sent={_tool_name(sent)!r} "
                f"observed={_tool_name(observed)!r}"
            )
            assert _parameter_keys(observed) == _parameter_keys(sent), (
                f"tools[{index}].function.parameters keys diverged "
                f"across the fallback boundary: sent="
                f"{sorted(_parameter_keys(sent))!r} observed="
                f"{sorted(_parameter_keys(observed))!r}"
            )

    asyncio.run(_exercise())
