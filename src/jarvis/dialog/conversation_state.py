"""Conversation_State and Turn data models.

This module materialises the two dialog data models declared in
``design.md §Data Models``:

* :class:`Turn` — one user/assistant exchange together with any tool
  calls the assistant emitted during that exchange and the wall-clock
  start/finish timestamps.
* :class:`ConversationState` — the serialisable session state the
  Dialog_Manager mutates across turns: a stable ``session_id``, the
  ordered list of completed :class:`Turn` records, an optional
  :class:`ToolCall` awaiting user confirmation, and the ``incognito``
  flag (Requirement 13.3) that the Memory_Store consults before
  persisting any record.

Why this lives in its own module
--------------------------------

The Dialog_Manager (task 13.4) imports :class:`ConversationState` and
:class:`Turn`, but so do a number of independent consumers — the
Memory_Store ``persist_turn`` call (task 14.3), the audit log fixtures,
Property 5 / CP6 round-trip tests, and any future debugging UI that
inspects a saved session. Keeping the data models in
``jarvis.dialog.conversation_state`` (rather than co-locating them with
:class:`DialogManager`) avoids importing the heavy dialog-manager module
just to construct a state object in a test or a memory persister.

API surface (matches the task acceptance bullet)
------------------------------------------------

* :meth:`ConversationState.append_user` — appends a new :class:`Turn`
  whose ``user`` field is filled and the ``assistant`` / ``tool_calls``
  fields are still empty. The Dialog_Manager calls this immediately
  after gating empty / low-confidence transcripts (Requirement 1.8).
* :meth:`ConversationState.append_assistant` — finalises the most
  recent :class:`Turn` with the assistant text, optional tool calls, and
  the finish timestamp. Mirrors the
  ``state.append_assistant(messages[-1].content)`` call in the design's
  Dialog_Manager pseudo-code.
* :meth:`ConversationState.last_turn` — returns the most recently
  *finalised* turn (the one ``persist_turn`` should consume) or ``None``
  when no completed turn exists yet. The Memory_Store relies on this
  signature (``state.last_turn()``).
* :meth:`ConversationState.to_json` and
  :classmethod:`ConversationState.from_json` — JSON serialisation used
  by Property 5 / CP6 (deterministic byte-equal round-trip across runs
  with a frozen clock and deterministic stub backend) and by the audit
  log when a turn fixture is captured for replay.

Determinism notes
-----------------

The JSON serialiser uses ``sort_keys=True`` and ``separators=(",", ":")``
so two structurally equal :class:`ConversationState` values produce
byte-equal strings regardless of key insertion order, the local default
encoder, or whitespace defaults. Datetimes are emitted in ISO-8601 form
via :meth:`datetime.isoformat`, which is stable across Python versions
and timezone conversions.

Validates: Requirements 1.4, 1.6, 13.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Any

from jarvis.llm.base import ToolCall

__all__ = ["ConversationState", "Turn"]


# ---------------------------------------------------------------------------
# ISO-8601 datetime helpers
# ---------------------------------------------------------------------------


def _ensure_aware(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes; the rest of the codebase assumes aware values.

    Mirrors the same defensive check that :class:`FakeTimeSource` performs
    on construction. Working exclusively with timezone-aware datetimes
    keeps :meth:`isoformat` round-trips lossless and matches the contract
    documented on :class:`jarvis.utils.time_source.TimeSource`.
    """

    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} must be a datetime instance "
            f"(got {type(value).__name__!r})"
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field_name} must be a timezone-aware datetime "
            "(use datetime.now(tz=timezone.utc) or a TimeSource)"
        )
    return value


def _datetime_to_iso(value: datetime) -> str:
    """Serialise a timezone-aware datetime to ISO-8601.

    ``datetime.isoformat`` already produces a deterministic string for
    aware values (e.g., ``"2024-01-01T00:00:00+00:00"``). Wrapping it in
    a small helper keeps the JSON encoder path readable and gives us a
    single place to add precision normalisation if a future requirement
    demands it.
    """

    return value.isoformat()


def _datetime_from_iso(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 string back into a timezone-aware datetime."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be an ISO-8601 string "
            f"(got {type(value).__name__!r})"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} is not a valid ISO-8601 datetime: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        # Default unannotated values to UTC. We prefer raising here, but
        # allowing UTC-as-default keeps the round-trip lossless for older
        # fixtures captured before the aware-only invariant was tightened.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# ---------------------------------------------------------------------------
# ToolCall <-> JSON
# ---------------------------------------------------------------------------


def _tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    """Convert a :class:`ToolCall` to a JSON-friendly mapping.

    We persist both ``arguments`` (the parsed dict) and ``raw_arguments``
    (the original JSON string from the model). Property 1 / CP1 requires
    the byte-equal raw payload to survive a serialise / parse round trip,
    so we cannot recompute it from ``arguments`` alone — the model may
    emit non-canonical JSON (whitespace, key order, escape forms) that
    ``json.dumps`` would not reproduce.
    """

    return {
        "id": tc.id,
        "skill_name": tc.skill_name,
        "arguments": dict(tc.arguments),
        "raw_arguments": tc.raw_arguments,
    }


def _tool_call_from_dict(data: Any, field_name: str) -> ToolCall:
    """Inverse of :func:`_tool_call_to_dict`; raises on malformed input."""

    if not isinstance(data, dict):
        raise TypeError(
            f"{field_name} must be an object " f"(got {type(data).__name__!r})"
        )
    missing = {"id", "skill_name", "arguments", "raw_arguments"} - data.keys()
    if missing:
        raise ValueError(f"{field_name} is missing required keys: {sorted(missing)!r}")
    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        raise TypeError(
            f"{field_name}.arguments must be an object "
            f"(got {type(arguments).__name__!r})"
        )
    return ToolCall(
        id=data["id"],
        skill_name=data["skill_name"],
        arguments=dict(arguments),
        raw_arguments=data["raw_arguments"],
    )


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One user/assistant exchange recorded by the Dialog_Manager.

    Mirrors ``design.md §Data Models``:

    * ``user`` — the transcribed user text. Set at the start of a turn
      via :meth:`ConversationState.append_user`.
    * ``assistant`` — the assistant's final spoken text. Empty until
      :meth:`ConversationState.append_assistant` finalises the turn;
      may legitimately remain empty when the assistant turn consists
      entirely of tool calls (e.g., a confirmation that is voiced via
      a separate path).
    * ``tool_calls`` — the function calls the LLM emitted during the
      turn, in arrival order, after re-assembly by the backend.
    * ``started_at`` / ``finished_at`` — wall-clock timestamps captured
      from the injected :class:`TimeSource`. Both are timezone-aware;
      ``finished_at >= started_at`` is enforced on construction.

    The dataclass is *not* frozen because the Dialog_Manager mutates
    ``assistant``, ``tool_calls``, and ``finished_at`` after the initial
    user-only construction (see :meth:`ConversationState.append_assistant`).
    Callers SHOULD treat finalised turns as immutable in spirit and
    avoid further mutation; the JSON round-trip and Property 5 / CP6 only
    care about *value* equality, not object identity.
    """

    user: str
    assistant: str
    tool_calls: list[ToolCall]
    started_at: datetime
    finished_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.user, str):
            raise TypeError("Turn.user must be a string")
        if not isinstance(self.assistant, str):
            raise TypeError("Turn.assistant must be a string")
        if not isinstance(self.tool_calls, list):
            raise TypeError("Turn.tool_calls must be a list")
        for i, tc in enumerate(self.tool_calls):
            if not isinstance(tc, ToolCall):
                raise TypeError(
                    f"Turn.tool_calls[{i}] must be a ToolCall "
                    f"(got {type(tc).__name__!r})"
                )
        self.started_at = _ensure_aware(self.started_at, "Turn.started_at")
        self.finished_at = _ensure_aware(self.finished_at, "Turn.finished_at")
        if self.finished_at < self.started_at:
            raise ValueError(
                "Turn.finished_at must be >= Turn.started_at "
                f"(started_at={self.started_at.isoformat()}, "
                f"finished_at={self.finished_at.isoformat()})"
            )

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render the turn as a JSON-friendly mapping.

        Used both directly (debugging dumps, audit fixtures) and as a
        building block of :meth:`ConversationState.to_json`. The shape
        exactly mirrors :meth:`from_dict`'s expectations so a
        ``Turn.from_dict(t.to_dict())`` round-trip is value-preserving.
        """

        return {
            "user": self.user,
            "assistant": self.assistant,
            "tool_calls": [_tool_call_to_dict(tc) for tc in self.tool_calls],
            "started_at": _datetime_to_iso(self.started_at),
            "finished_at": _datetime_to_iso(self.finished_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Turn:
        """Inverse of :meth:`to_dict`; raises on malformed input."""

        if not isinstance(data, dict):
            raise TypeError(
                f"Turn.from_dict expected a mapping (got {type(data).__name__!r})"
            )
        missing = {
            "user",
            "assistant",
            "tool_calls",
            "started_at",
            "finished_at",
        } - data.keys()
        if missing:
            raise ValueError(
                f"Turn payload is missing required keys: {sorted(missing)!r}"
            )
        raw_tool_calls = data["tool_calls"]
        if not isinstance(raw_tool_calls, list):
            raise TypeError("Turn.tool_calls must be a list")
        tool_calls = [
            _tool_call_from_dict(tc, f"Turn.tool_calls[{i}]")
            for i, tc in enumerate(raw_tool_calls)
        ]
        return cls(
            user=data["user"],
            assistant=data["assistant"],
            tool_calls=tool_calls,
            started_at=_datetime_from_iso(data["started_at"], "Turn.started_at"),
            finished_at=_datetime_from_iso(data["finished_at"], "Turn.finished_at"),
        )


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------


@dataclass
class ConversationState:
    """The mutable session state owned by the Dialog_Manager.

    Mirrors ``design.md §Data Models``:

    * ``session_id`` — opaque, caller-supplied identifier (typically a
      UUID4 string). Treated as a black box by this module; callers
      pick the format and uniqueness guarantee.
    * ``started_at`` — timezone-aware wall-clock timestamp of session
      creation, sourced from the injected :class:`TimeSource`.
    * ``turns`` — ordered list of completed :class:`Turn` records. The
      *current* (in-progress) turn lives at the end of the list while
      its ``assistant`` text is still empty; :meth:`append_assistant`
      finalises it in place.
    * ``pending_confirmation`` — optional :class:`ToolCall` awaiting an
      explicit user "yes" before the Authorization_Policy dispatches it
      (Requirement 16.2). ``None`` outside the confirmation window.
    * ``incognito`` — Requirement 13.3 flag; the Memory_Store skips
      persistence when this is ``True``. The Dialog_Manager also forwards
      it into :class:`SkillContext` so individual Skills can adjust
      behaviour (e.g., the audit log still records, but personal memory
      writes are suppressed).

    Construction
    ------------

    The dataclass is intentionally *not* frozen because the Dialog_Manager
    mutates the turn list and ``pending_confirmation`` across turns.
    Defaults are provided for everything except ``session_id`` and
    ``started_at`` so call sites in the manager and tests stay terse:

        >>> state = ConversationState(
        ...     session_id="abc",
        ...     started_at=datetime.now(tz=timezone.utc),
        ... )
    """

    session_id: str
    started_at: datetime
    turns: list[Turn] = field(default_factory=list)
    pending_confirmation: ToolCall | None = None
    incognito: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("ConversationState.session_id must be a non-empty string")
        self.started_at = _ensure_aware(self.started_at, "ConversationState.started_at")
        if not isinstance(self.turns, list):
            raise TypeError("ConversationState.turns must be a list")
        for i, t in enumerate(self.turns):
            if not isinstance(t, Turn):
                raise TypeError(
                    f"ConversationState.turns[{i}] must be a Turn "
                    f"(got {type(t).__name__!r})"
                )
        if self.pending_confirmation is not None and not isinstance(
            self.pending_confirmation, ToolCall
        ):
            raise TypeError(
                "ConversationState.pending_confirmation must be a ToolCall " "or None"
            )
        if not isinstance(self.incognito, bool):
            raise TypeError("ConversationState.incognito must be a bool")

    # -- Mutation -----------------------------------------------------------

    def append_user(self, text: str, *, at: datetime) -> Turn:
        """Begin a new turn with the user's transcribed text.

        Appends a fresh :class:`Turn` whose ``assistant`` is empty and
        ``tool_calls`` is empty. The same ``at`` timestamp seeds both
        ``started_at`` and ``finished_at`` — the latter will be rewritten
        by :meth:`append_assistant` when the assistant side of the turn
        completes. We initialise ``finished_at = started_at`` (rather
        than leaving it ``None``) so :class:`Turn`'s invariant
        ``finished_at >= started_at`` holds throughout the in-progress
        window and so a partially-built state is still serialisable.

        Parameters
        ----------
        text:
            Transcribed user utterance. Empty strings are accepted at
            this layer — the empty/low-confidence guard belongs to the
            Dialog_Manager (Requirement 1.8) — but the value MUST be a
            string.
        at:
            Timezone-aware wall-clock timestamp from the injected
            :class:`TimeSource`. Required as a keyword argument so call
            sites cannot accidentally rely on the wall clock implicitly
            (which would break Property 5 / CP6 determinism).

        Returns
        -------
        Turn
            The newly-appended turn. Returning it lets the
            Dialog_Manager mutate the same object in place if it needs
            to (e.g., attach metadata) without re-indexing the list.
        """

        if not isinstance(text, str):
            raise TypeError("append_user.text must be a string")
        started = _ensure_aware(at, "append_user.at")
        turn = Turn(
            user=text,
            assistant="",
            tool_calls=[],
            started_at=started,
            finished_at=started,
        )
        self.turns.append(turn)
        return turn

    def append_assistant(
        self,
        text: str,
        *,
        at: datetime,
        tool_calls: list[ToolCall] | None = None,
    ) -> Turn:
        """Finalise the most recent turn with the assistant's reply.

        Mirrors the design's
        ``state.append_assistant(messages[-1].content)`` call. The most
        recent :class:`Turn` (the one created by :meth:`append_user`)
        is mutated in place: ``assistant`` is set, ``tool_calls`` is
        replaced (defaulting to an empty list), and ``finished_at`` is
        advanced to ``at``.

        Parameters
        ----------
        text:
            The assistant's final text for the turn. May be empty when
            the turn consisted purely of tool calls (the SDK contracts
            for both Mistral and Ollama allow that).
        at:
            Timezone-aware wall-clock timestamp from the injected
            :class:`TimeSource`. MUST be ``>= started_at`` of the
            in-progress turn; the :class:`Turn` constructor enforces
            this on the resulting object.
        tool_calls:
            Tool calls the LLM emitted during the turn, in arrival
            order. ``None`` is treated as "no tool calls"; passing an
            empty list has the same effect.

        Returns
        -------
        Turn
            The finalised turn (same object as
            :meth:`last_turn` / ``self.turns[-1]``).

        Raises
        ------
        RuntimeError
            If there is no in-progress turn — i.e., :meth:`append_user`
            has not been called first. Mirrors the Dialog_Manager's
            own loop invariant: every assistant turn is preceded by a
            user turn within the same session.
        """

        if not isinstance(text, str):
            raise TypeError("append_assistant.text must be a string")
        if not self.turns:
            raise RuntimeError("append_assistant requires a prior append_user call")
        finished = _ensure_aware(at, "append_assistant.at")
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise TypeError(
                "append_assistant.tool_calls must be a list of ToolCall or None"
            )
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, ToolCall):
                raise TypeError(
                    f"append_assistant.tool_calls[{i}] must be a ToolCall "
                    f"(got {type(tc).__name__!r})"
                )
        current = self.turns[-1]
        if finished < current.started_at:
            raise ValueError(
                "append_assistant.at must be >= the in-progress turn's "
                f"started_at (started_at={current.started_at.isoformat()}, "
                f"at={finished.isoformat()})"
            )
        # Replace via direct attribute assignment; Turn is not frozen so
        # this is the natural in-place mutation path. We also re-validate
        # by constructing a fresh Turn so the invariants in
        # Turn.__post_init__ are re-checked (cheap: O(len(tool_calls))).
        updated = Turn(
            user=current.user,
            assistant=text,
            tool_calls=list(tool_calls),
            started_at=current.started_at,
            finished_at=finished,
        )
        self.turns[-1] = updated
        return updated

    def last_turn(self) -> Turn | None:
        """Return the most recent turn, or ``None`` if there is none.

        The Memory_Store calls this immediately before
        ``persist_turn`` (see ``design.md`` Dialog_Manager pseudo-code).
        We deliberately return the *most recent* turn — finalised or
        not — rather than only fully-finalised turns: the Dialog_Manager
        invokes ``last_turn`` after :meth:`append_assistant`, at which
        point the latest turn IS finalised. Surfacing the in-progress
        turn for diagnostics/inspection is therefore both useful and
        safe.
        """

        if not self.turns:
            return None
        return self.turns[-1]

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render the conversation state as a JSON-friendly mapping."""

        return {
            "session_id": self.session_id,
            "started_at": _datetime_to_iso(self.started_at),
            "turns": [turn.to_dict() for turn in self.turns],
            "pending_confirmation": (
                None
                if self.pending_confirmation is None
                else _tool_call_to_dict(self.pending_confirmation)
            ),
            "incognito": self.incognito,
        }

    def to_json(self) -> str:
        """Serialise to a deterministic JSON string.

        ``sort_keys=True`` and ``separators=(",", ":")`` ensure that two
        structurally equal :class:`ConversationState` values produce
        byte-equal output, satisfying Property 5 / CP6's byte-equal
        requirement when paired with a deterministic stub backend and a
        frozen clock.
        """

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: Any) -> ConversationState:
        """Inverse of :meth:`to_dict`; raises on malformed input."""

        if not isinstance(data, dict):
            raise TypeError(
                "ConversationState.from_dict expected a mapping "
                f"(got {type(data).__name__!r})"
            )
        missing = {
            "session_id",
            "started_at",
            "turns",
            "pending_confirmation",
            "incognito",
        } - data.keys()
        if missing:
            raise ValueError(
                "ConversationState payload is missing required keys: "
                f"{sorted(missing)!r}"
            )
        raw_turns = data["turns"]
        if not isinstance(raw_turns, list):
            raise TypeError("ConversationState.turns must be a list")
        turns = [Turn.from_dict(t) for t in raw_turns]
        pending_raw = data["pending_confirmation"]
        pending = (
            None
            if pending_raw is None
            else _tool_call_from_dict(
                pending_raw, "ConversationState.pending_confirmation"
            )
        )
        incognito = data["incognito"]
        if not isinstance(incognito, bool):
            raise TypeError("ConversationState.incognito must be a bool")
        return cls(
            session_id=data["session_id"],
            started_at=_datetime_from_iso(
                data["started_at"], "ConversationState.started_at"
            ),
            turns=turns,
            pending_confirmation=pending,
            incognito=incognito,
        )

    @classmethod
    def from_json(cls, payload: str) -> ConversationState:
        """Parse a JSON string previously produced by :meth:`to_json`."""

        if not isinstance(payload, str):
            raise TypeError(
                "ConversationState.from_json expected a string "
                f"(got {type(payload).__name__!r})"
            )
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ConversationState.from_json: invalid JSON ({exc.msg})"
            ) from exc
        return cls.from_dict(data)
