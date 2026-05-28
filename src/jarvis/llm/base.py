"""Backend-agnostic LLM streaming protocol and shared message / event types.

This module defines the structural contract every concrete LLM backend
(``MistralBackend``, ``OllamaBackend``, ``BackendSelector``) MUST satisfy
so that :class:`jarvis.dialog.manager.DialogManager` can call them
interchangeably. The shape mirrors the Dialog_Manager pseudo-code in
``design.md`` and the data models in the same document, in particular:

* ``LLMBackend.stream(messages, *, tools, **kw)`` returns an
  *async* context manager whose body yields a :class:`Stream` —
  itself an async iterator of :class:`LLMEvent` values.
* The dialog loop dispatches events by their ``type`` discriminator
  (``"content_delta"`` for raw text deltas, ``"tool_call"`` for fully
  materialised function-call payloads), exactly as the design document's
  control loop does (``if event.type == "content_delta": ... elif
  event.type == "tool_call": ...``).
* :class:`ToolCall` matches the Data Models section verbatim — ``id``,
  ``skill_name``, ``arguments`` (parsed) and ``raw_arguments`` (the
  original JSON string from the model). Keeping both forms lets the
  Authorization_Policy and Skill_Registry validate against
  ``arguments`` while CP1 (intent serialization round-trip) still has
  the raw bytes available for byte-equal comparisons.
* The TypedDict-based message shapes match the JSON the Mistral SDK
  and the Ollama OpenAI-compatible ``/api/chat`` endpoint accept on
  the wire, so backends do not need to translate between an internal
  representation and the SDK's expectation.

Why a structural :class:`typing.Protocol` rather than an abstract base
class: the cloud SDK (``mistralai``) and the local fallback
(``ollama`` / OpenAI-compatible HTTP) come from independent vendors and
we never want to subclass through them. A ``Protocol`` lets each
backend be a plain class and still satisfy the type checker, which is
the explicit goal of task 8.1: *"Make these structural so MistralBackend,
OllamaBackend, and BackendSelector can all conform."*

Validates: Requirements 12.4, 19.4, 19.5
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import (
    Any,
    Final,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

__all__ = [
    "EVENT_TYPE_CONTENT_DELTA",
    "EVENT_TYPE_TOOL_CALL",
    "AssistantMessage",
    "AssistantToolCall",
    "AssistantToolCallFunction",
    "ContentDeltaEvent",
    "LLMBackend",
    "LLMEvent",
    "Message",
    "Role",
    "Stream",
    "SystemMessage",
    "ToolCall",
    "ToolCallEvent",
    "ToolDefinition",
    "ToolFunctionDefinition",
    "ToolMessage",
    "UserMessage",
]


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


# The closed set of message roles understood by both Mistral and the
# Ollama OpenAI-compatible chat endpoint. Centralising the literal makes
# the message TypedDicts below easy to discriminate in the type checker
# and prevents typos at call sites in the Dialog_Manager.
Role = Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Tool / function definitions
# ---------------------------------------------------------------------------


class ToolFunctionDefinition(TypedDict):
    """The ``function`` payload of a Mistral / OpenAI function tool.

    ``parameters`` is the Skill's JSON Schema as produced by
    ``SkillRegistry.mistral_tool_definitions()``. The schema is
    pre-validated by ``MistralSchemaValidator`` (task 9.1) before the
    Skill is registered, so backends can forward it verbatim.
    """

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinition(TypedDict):
    """One element of the ``tools`` argument passed to :meth:`LLMBackend.stream`.

    Mirrors the ``{"type": "function", "function": {...}}`` shape the
    Mistral SDK and OpenAI-compatible APIs accept. The Dialog_Manager
    builds the list once per turn from :class:`SkillRegistry`; backends
    forward it to their respective wire formats unchanged.
    """

    type: Literal["function"]
    function: ToolFunctionDefinition


# ---------------------------------------------------------------------------
# Tool call (model output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A function-call request emitted by the LLM backend.

    Exactly the data model declared in ``design.md``:

    * ``id`` — the opaque identifier the model assigned to this call.
      The Dialog_Manager threads it back into the matching ``tool``
      message so the model can correlate the result with the request.
    * ``skill_name`` — the registered Skill name selected by the model.
      The Skill_Registry uses this to look up the executor and JSON
      Schema.
    * ``arguments`` — the parsed JSON object. Stored as a plain
      :class:`dict` so :class:`jsonschema.validators.Draft202012Validator`
      can validate it directly and so the dataclass remains
      JSON-serialisable for CP1 (intent round-trip).
    * ``raw_arguments`` — the original JSON string. Preserved verbatim
      so the audit log (CP9) and CP1's serialise/parse round-trip can
      both reproduce byte-equal payloads even when the model emits a
      non-canonical JSON encoding (whitespace, key order, escape
      forms).

    The dataclass is frozen so :class:`ToolCall` instances are hashable
    and safe to use as keys in the trusted-action allowlist
    (Requirement 16.3) without further wrapping. ``arguments`` itself is
    *not* frozen — JSON Schema validation occasionally needs to read
    nested mutable values — but callers SHOULD treat it as
    read-only and copy it before mutating.
    """

    id: str
    skill_name: str
    arguments: dict[str, Any]
    raw_arguments: str

    def __post_init__(self) -> None:
        # Defensive validation: catching obviously broken payloads here
        # keeps the audit log and Skill_Registry simpler downstream.
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("ToolCall.id must be a non-empty string")
        if not isinstance(self.skill_name, str) or not self.skill_name:
            raise ValueError("ToolCall.skill_name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise TypeError("ToolCall.arguments must be a dict")
        if not isinstance(self.raw_arguments, str):
            raise TypeError("ToolCall.raw_arguments must be a string")


# ---------------------------------------------------------------------------
# Wire messages (dialog manager -> backend)
# ---------------------------------------------------------------------------


# We use ``TypedDict`` rather than dataclasses for messages because the
# concrete backends (mistralai, ollama, OpenAI-compatible HTTP) all expect
# JSON-shaped dicts on the wire. Modelling them as TypedDicts means we
# can hand the same value straight to the SDK without an intermediate
# conversion step, while still getting field-level type checking from
# mypy at every call site in the Dialog_Manager.


class SystemMessage(TypedDict):
    """The persona system prompt; MUST be ``messages[0]`` (CP14)."""

    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    """A transcribed user turn forwarded to the backend."""

    role: Literal["user"]
    content: str


class AssistantToolCallFunction(TypedDict):
    """The ``function`` payload of an :class:`AssistantToolCall`."""

    name: str
    arguments: str  # raw JSON string, matching ToolCall.raw_arguments


class AssistantToolCall(TypedDict):
    """Mistral / OpenAI ``tool_calls`` element on an assistant message.

    Used when replaying a previous assistant turn that emitted tool
    calls back to the model, so the model can correlate the upcoming
    ``tool`` messages with the original request.
    """

    id: str
    type: Literal["function"]
    function: AssistantToolCallFunction


class AssistantMessage(TypedDict):
    """A previous assistant turn replayed to the backend.

    ``content`` may be empty when the assistant turn consisted purely of
    tool calls; the SDK contracts for both Mistral and Ollama allow an
    empty string in that case.
    """

    role: Literal["assistant"]
    content: str
    tool_calls: NotRequired[list[AssistantToolCall]]


class ToolMessage(TypedDict):
    """A Tool_Call result fed back into the conversation.

    The ``tool_call_id`` MUST match the ``id`` of a prior
    :class:`AssistantToolCall` so the model can attribute the result.
    ``content`` is the JSON-serialised :class:`SkillResult`.
    """

    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: NotRequired[str]  # Skill name; some backends require it


# Discriminated union of all valid message shapes. Using ``Union`` rather
# than a single ``Message`` TypedDict with optional fields lets mypy
# narrow on ``role`` and reject, e.g., a ``tool_call_id`` on a system
# message at the call site.
Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# ---------------------------------------------------------------------------
# Streaming events (backend -> dialog manager)
# ---------------------------------------------------------------------------


# The event type discriminator literal values are intentionally string
# literals (not an :class:`enum.Enum`) so the Dialog_Manager can write
# the natural ``event.type == "content_delta"`` comparison shown in the
# design document and so JSON-serialised audit fixtures match the wire
# spelling.
EVENT_TYPE_CONTENT_DELTA: Final = "content_delta"
EVENT_TYPE_TOOL_CALL: Final = "tool_call"


@dataclass(frozen=True)
class ContentDeltaEvent:
    """An incremental token (or token chunk) of assistant text.

    ``text`` is the raw delta as the backend received it; the
    Dialog_Manager feeds these into a :class:`SentenceAccumulator` so
    TTS synthesis can begin at sentence boundaries (Requirement 12.2,
    Requirement 19.5). The string MAY be empty — Mistral occasionally
    emits zero-length deltas to signal a heartbeat — and consumers MUST
    handle that case by treating it as a no-op rather than as an end of
    stream.
    """

    text: str
    type: Literal["content_delta"] = field(default=EVENT_TYPE_CONTENT_DELTA, init=False)


@dataclass(frozen=True)
class ToolCallEvent:
    """A fully-materialised :class:`ToolCall` selected by the model.

    Backends are responsible for *re-assembling* the streamed function
    call fragments (Mistral and the OpenAI streaming protocol both ship
    a tool-call's name and arguments across multiple deltas) into a
    single :class:`ToolCall` before emitting this event. This contract
    keeps the Dialog_Manager simple: it never has to reason about
    partial tool calls, and it can dispatch each event against the
    Authorization_Policy as soon as it arrives.
    """

    tool_call: ToolCall
    type: Literal["tool_call"] = field(default=EVENT_TYPE_TOOL_CALL, init=False)


# Discriminated union; consumers ``match`` or ``if event.type == ...`` to
# narrow. Adding a new event variant is a deliberately breaking change
# because the Dialog_Manager exhaustively dispatches on ``type``.
LLMEvent = Union[ContentDeltaEvent, ToolCallEvent]


# ---------------------------------------------------------------------------
# Stream and backend protocols
# ---------------------------------------------------------------------------


# A :class:`Stream` is just an async iterator of :class:`LLMEvent` values.
# We expose it as a plain alias rather than a dedicated Protocol so that
# any object satisfying ``AsyncIterator[LLMEvent]`` — including bare
# ``async for`` generators in the test fakes — counts as a Stream.
Stream = AsyncIterator[LLMEvent]


@runtime_checkable
class LLMBackend(Protocol):
    """Structural contract for an LLM provider used by the Dialog_Manager.

    A backend's only required method is :meth:`stream`, which returns an
    *async context manager* whose body yields a :class:`Stream`. The
    context manager is responsible for the underlying network connection
    lifecycle: opening it on ``__aenter__``, closing it on ``__aexit__``,
    and aborting cleanly if the body raises (e.g., when the
    Voice_Pipeline tears down the dialog loop on barge-in).

    Concrete implementations:

    * :class:`jarvis.llm.mistral_backend.MistralBackend` — wraps the
      ``mistralai`` async client's ``chat.stream`` API and is the
      default backend when the Mistral API is reachable
      (Requirement 19.1, Requirement 19.5).
    * :class:`jarvis.llm.ollama_backend.OllamaBackend` — wraps the
      Ollama OpenAI-compatible ``/api/chat`` streaming endpoint and is
      the local fallback selected by the circuit breaker
      (Requirement 12.4).
    * :class:`jarvis.llm.backend_selector.BackendSelector` — composes
      the two with a circuit breaker, presenting the same
      :class:`LLMBackend` interface to the Dialog_Manager so the
      fallback is invisible above the protocol boundary.

    The :func:`runtime_checkable` decorator lets test fakes assert the
    fake conforms via ``isinstance(fake, LLMBackend)`` without paying
    for nominal subclassing.
    """

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[Stream]:
        """Open a streaming chat completion and return the event stream.

        Parameters
        ----------
        messages:
            The full message list the model should consume. The first
            element MUST be a :class:`SystemMessage` carrying the
            persona prompt — Property 11 (CP14) checks this invariant
            inside the Dialog_Manager and the backend SHOULD pass the
            list through unmodified.
        tools:
            The Skill_Registry's Mistral-format tool definitions. May
            be empty when no Skills are available; backends MUST treat
            an empty list as "no function calling on this turn"
            (Requirement 19.4).
        **kwargs:
            Backend-specific knobs such as ``model``, ``temperature``,
            or ``max_tokens``. Names that conflict across backends
            (e.g., Mistral's ``random_seed`` versus Ollama's ``seed``)
            are translated by each implementation; the Dialog_Manager
            only uses the keys documented in the configuration schema.

        Returns
        -------
        :class:`contextlib.AbstractAsyncContextManager` of
        :class:`Stream`.
            Use as ``async with backend.stream(...) as events: async
            for event in events: ...``. The context manager guarantees
            the underlying connection is released even if the consumer
            breaks out of the ``async for`` early.
        """
        ...
