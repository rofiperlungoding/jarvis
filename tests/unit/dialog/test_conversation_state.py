"""Unit tests for ``jarvis.dialog.conversation_state``.

Covers the shape and invariants of :class:`Turn` and
:class:`ConversationState`, the ``append_user`` / ``append_assistant`` /
``last_turn`` mutation API, and the JSON round-trip required by Property
5 / CP6.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from jarvis.dialog.conversation_state import ConversationState, Turn
from jarvis.llm.base import ToolCall

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
T2 = T0 + timedelta(seconds=2)


def _tool_call(idx: int = 1) -> ToolCall:
    raw = '{"q":"hello"}'
    return ToolCall(
        id=f"call_{idx}",
        skill_name="echo",
        arguments={"q": "hello"},
        raw_arguments=raw,
    )


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------


def test_turn_basic_construction() -> None:
    turn = Turn(
        user="hi",
        assistant="hello",
        tool_calls=[],
        started_at=T0,
        finished_at=T1,
    )
    assert turn.user == "hi"
    assert turn.assistant == "hello"
    assert turn.tool_calls == []
    assert turn.started_at == T0
    assert turn.finished_at == T1


def test_turn_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        Turn(
            user="hi",
            assistant="hello",
            tool_calls=[],
            started_at=datetime(2024, 1, 1, 12, 0, 0),  # naive
            finished_at=T1,
        )


def test_turn_rejects_finished_before_started() -> None:
    with pytest.raises(ValueError):
        Turn(
            user="hi",
            assistant="hello",
            tool_calls=[],
            started_at=T1,
            finished_at=T0,
        )


def test_turn_rejects_non_toolcall_in_tool_calls() -> None:
    with pytest.raises(TypeError):
        Turn(
            user="hi",
            assistant="hello",
            tool_calls=["not a tool call"],  # type: ignore[list-item]
            started_at=T0,
            finished_at=T1,
        )


def test_turn_round_trip_via_dict() -> None:
    tc = _tool_call()
    turn = Turn(
        user="hi",
        assistant="hello",
        tool_calls=[tc],
        started_at=T0,
        finished_at=T1,
    )
    restored = Turn.from_dict(turn.to_dict())
    assert restored == turn
    # ToolCall equality: the parsed args and raw string both match.
    assert restored.tool_calls[0].raw_arguments == tc.raw_arguments


# ---------------------------------------------------------------------------
# ConversationState construction & validation
# ---------------------------------------------------------------------------


def test_state_default_construction() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    assert state.session_id == "abc"
    assert state.started_at == T0
    assert state.turns == []
    assert state.pending_confirmation is None
    assert state.incognito is False


def test_state_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError):
        ConversationState(session_id="", started_at=T0)


def test_state_rejects_naive_started_at() -> None:
    with pytest.raises(ValueError):
        ConversationState(
            session_id="abc",
            started_at=datetime(2024, 1, 1, 12, 0, 0),  # naive
        )


def test_state_rejects_non_bool_incognito() -> None:
    with pytest.raises(TypeError):
        ConversationState(
            session_id="abc",
            started_at=T0,
            incognito="yes",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# append_user / append_assistant / last_turn
# ---------------------------------------------------------------------------


def test_last_turn_returns_none_when_empty() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    assert state.last_turn() is None


def test_append_user_creates_in_progress_turn() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    turn = state.append_user("hi", at=T1)
    assert turn is state.turns[-1]
    assert turn.user == "hi"
    assert turn.assistant == ""
    assert turn.tool_calls == []
    assert turn.started_at == T1
    assert turn.finished_at == T1
    assert state.last_turn() is turn


def test_append_assistant_finalises_latest_turn() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    state.append_user("hi", at=T1)
    tc = _tool_call()
    finalised = state.append_assistant("hello", at=T2, tool_calls=[tc])
    assert finalised is state.turns[-1]
    assert finalised.user == "hi"
    assert finalised.assistant == "hello"
    assert finalised.tool_calls == [tc]
    assert finalised.started_at == T1
    assert finalised.finished_at == T2


def test_append_assistant_default_tool_calls_is_empty() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    state.append_user("hi", at=T1)
    finalised = state.append_assistant("hello", at=T2)
    assert finalised.tool_calls == []


def test_append_assistant_without_prior_user_raises() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    with pytest.raises(RuntimeError):
        state.append_assistant("hello", at=T1)


def test_append_assistant_rejects_finished_before_started() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    state.append_user("hi", at=T2)
    with pytest.raises(ValueError):
        state.append_assistant("hello", at=T1)


def test_append_user_rejects_naive_at() -> None:
    state = ConversationState(session_id="abc", started_at=T0)
    with pytest.raises(ValueError):
        state.append_user("hi", at=datetime(2024, 1, 1, 12, 0, 0))


# ---------------------------------------------------------------------------
# JSON round-trip and determinism (Property 5 / CP6)
# ---------------------------------------------------------------------------


def test_to_json_round_trip_with_pending_confirmation() -> None:
    state = ConversationState(
        session_id="abc",
        started_at=T0,
        pending_confirmation=_tool_call(),
        incognito=True,
    )
    state.append_user("hi", at=T1)
    state.append_assistant("hello", at=T2, tool_calls=[_tool_call(2)])

    payload = state.to_json()
    restored = ConversationState.from_json(payload)
    assert restored == state


def test_to_json_is_deterministic_byte_equal() -> None:
    """Two equal states MUST serialise to byte-equal JSON.

    Backs the byte-equal requirement of Property 5 / CP6.
    """

    def _build() -> ConversationState:
        s = ConversationState(session_id="abc", started_at=T0)
        s.append_user("hi", at=T1)
        s.append_assistant("hello", at=T2, tool_calls=[_tool_call()])
        return s

    a = _build().to_json()
    b = _build().to_json()
    assert a == b
    # Sanity: the encoded form is well-formed JSON.
    json.loads(a)


def test_from_json_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError):
        ConversationState.from_json("not-json")


def test_from_json_rejects_missing_keys() -> None:
    with pytest.raises(ValueError):
        ConversationState.from_json(json.dumps({"session_id": "abc"}))


def test_to_json_preserves_raw_arguments_byte_equal() -> None:
    """Property 1 / CP1: raw_arguments survives a serialise/parse round trip."""

    raw = '{"q":  "hello",\n  "n": 1}'  # non-canonical whitespace
    tc = ToolCall(
        id="call_1",
        skill_name="echo",
        arguments={"q": "hello", "n": 1},
        raw_arguments=raw,
    )
    state = ConversationState(session_id="abc", started_at=T0)
    state.append_user("hi", at=T1)
    state.append_assistant("hello", at=T2, tool_calls=[tc])

    restored = ConversationState.from_json(state.to_json())
    assert restored.turns[-1].tool_calls[0].raw_arguments == raw
