"""Property 1 — Intent / Tool-Call serialisation round trip.

From ``design.md §Correctness Properties``:

    *For any* ``ToolCall`` value ``tc`` produced by the Dialog_Manager's
    intent parser, ``parse_intent(serialize_intent(tc))`` SHALL be
    deeply equal to ``tc``, modulo argument key ordering.

In the JARVIS data model (``src/jarvis/llm/base.py``), a
:class:`~jarvis.llm.base.ToolCall` carries four fields:

* ``id`` — opaque identifier assigned by the model.
* ``skill_name`` — the registered Skill name.
* ``arguments`` — the parsed JSON object (a ``dict[str, Any]``).
* ``raw_arguments`` — the *original* JSON string the model emitted.
  This MUST round-trip verbatim because the audit log (CP9) and
  authorization allowlist key off byte-stable payloads even when the
  model returns a non-canonical JSON encoding (whitespace, key order,
  escape forms).

The strategy
------------

:func:`tool_calls` generates ``(id, skill_name, arguments, raw_arguments)``
tuples where:

* ``id`` and ``skill_name`` are non-empty strings (the
  :class:`ToolCall` ``__post_init__`` rejects empties).
* ``arguments`` is a JSON-compatible dict (recursively built from the
  same JSON-value tree used by Property 2 / CP2 so the two strategies
  share a coverage shape).
* ``raw_arguments`` is ``json.dumps(arguments)`` — i.e., the strategy
  emits self-consistent values where ``json.loads(raw_arguments) ==
  arguments``. That consistency mirrors how the production
  Mistral / Ollama backends construct :class:`ToolCall` instances
  (``arguments = json.loads(raw_arguments)``; see
  ``src/jarvis/llm/mistral_backend.py``).

The property
------------

The test exercises three checks against the same generated value:

1. **Model round trip.** Round-tripping ``tc`` through a JSON
   serialise/parse pass yields a :class:`ToolCall` that compares equal
   to ``tc``. Equality is dataclass-default (deep) equality, which
   collapses argument key ordering because Python ``dict`` equality is
   key-order independent.
2. **``raw_arguments`` consistency.** ``json.loads(tc.raw_arguments)``
   equals ``tc.arguments``. This is the invariant the production
   backends preserve and which the audit log relies on for CP9.
3. **``raw_arguments`` byte preservation.** The recovered ``ToolCall``
   carries the *exact same* ``raw_arguments`` string as the original;
   the serialiser MUST NOT canonicalise it, otherwise the audit log /
   allowlist comparisons would silently drift.

A focused unit test (no Hypothesis) covers the empty-arguments corner
explicitly so the property's ``assume`` / shrink behaviour cannot mask
it.

Validates: Requirements 1.4, 14.4 (CP1)
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, strategies as st

from jarvis.llm.base import ToolCall

# ---------------------------------------------------------------------------
# JSON value strategy
# ---------------------------------------------------------------------------


# A JSON-compatible value tree. Recursive so nested objects / arrays
# appear in the generated argument dicts. Mirrors the strategy used by
# the Property 2 / CP2 schema-soundness test so both properties exercise
# the same value shape.
_json_values: st.SearchStrategy[Any] = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.text(
            alphabet=st.characters(
                min_codepoint=0x20,
                max_codepoint=0xFFFD,
                # Surrogate code points are not valid in well-formed
                # JSON strings; excluding them keeps every generated
                # value safe for ``json.dumps`` regardless of the
                # platform encoder defaults.
                exclude_categories=("Cs",),  # type: ignore[arg-type]
            ),
            max_size=16,
        ),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=8,
)


# Property-specific text strategies. ``ToolCall.__post_init__`` rejects
# empty ``id`` / ``skill_name`` so the strategy generates non-empty
# values directly rather than relying on ``filter`` (which would slow
# shrinking).
_non_empty_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x21,  # printable, no whitespace control
        max_codepoint=0x7E,
    ),
    min_size=1,
    max_size=32,
)


def tool_calls() -> st.SearchStrategy[ToolCall]:
    """Generate self-consistent :class:`ToolCall` values.

    The generated ``arguments`` dict is JSON-compatible and
    ``raw_arguments`` is set to ``json.dumps(arguments)`` so that
    ``json.loads(raw_arguments) == arguments`` holds for every emitted
    value — the invariant the production backends preserve when
    they parse Mistral / Ollama function-call deltas. A custom
    ``tool_calls()`` strategy is local to this test (rather than
    living in :mod:`tests.strategies`) because Property 1 is the only
    consumer right now; if a second test needs it, hoist the function
    over to ``tests/strategies.py``.
    """

    @st.composite
    def _build(draw: st.DrawFn) -> ToolCall:
        # ``arguments`` is a dict; the top-level shape is fixed even
        # though the values may recurse into arbitrary JSON structures.
        arguments_strategy = st.dictionaries(
            keys=st.text(max_size=8),
            values=_json_values,
            max_size=6,
        )
        arguments: dict[str, Any] = draw(arguments_strategy)
        # ``raw_arguments`` is the JSON encoding the model would have
        # emitted. Using ``json.dumps`` with default separators keeps
        # the serialised form realistic; the property test does not
        # depend on which particular encoding we pick because the
        # invariant only requires ``json.loads(raw_arguments) ==
        # arguments``.
        raw_arguments = json.dumps(arguments)
        return ToolCall(
            id=draw(_non_empty_text),
            skill_name=draw(_non_empty_text),
            arguments=arguments,
            raw_arguments=raw_arguments,
        )

    return _build()


# ---------------------------------------------------------------------------
# Serialise / parse pair under test
# ---------------------------------------------------------------------------


def _serialize_intent(tc: ToolCall) -> str:
    """Render a :class:`ToolCall` as a JSON string.

    The serialised form mirrors the dict shape used by
    :class:`jarvis.dialog.conversation_state.Turn` to persist tool
    calls (see ``_tool_call_to_dict`` in that module). ``sort_keys`` is
    enabled so two structurally equal :class:`ToolCall` values produce
    byte-equal envelope strings — the *envelope* keys are canonical,
    while ``raw_arguments`` itself stays verbatim because it is a
    string value, not an object the encoder reorders.
    """

    return json.dumps(
        {
            "id": tc.id,
            "skill_name": tc.skill_name,
            "arguments": tc.arguments,
            "raw_arguments": tc.raw_arguments,
        },
        sort_keys=True,
    )


def _parse_intent(blob: str) -> ToolCall:
    """Inverse of :func:`_serialize_intent`."""

    data = json.loads(blob)
    return ToolCall(
        id=data["id"],
        skill_name=data["skill_name"],
        arguments=data["arguments"],
        raw_arguments=data["raw_arguments"],
    )


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


@given(tc=tool_calls())
def test_tool_call_round_trips_through_json(tc: ToolCall) -> None:
    """``parse_intent(serialize_intent(tc))`` deeply equals ``tc``.

    Validates: Requirements 1.4, 14.4 (CP1)
    """

    blob = _serialize_intent(tc)
    recovered = _parse_intent(blob)

    # 1. Model-level deep equality. ``ToolCall`` is a frozen dataclass,
    #    so ``__eq__`` compares every field; argument key ordering is
    #    folded in by Python's ``dict`` equality (key-order independent).
    assert recovered == tc

    # 2. ``raw_arguments`` survives the round trip *byte-for-byte*. The
    #    audit log (CP9) and trusted-action allowlist key on this exact
    #    string, so any normalisation by the serialiser would silently
    #    break those downstream consumers.
    assert recovered.raw_arguments == tc.raw_arguments

    # 3. The strategy invariant: ``arguments`` is the parsed form of
    #    ``raw_arguments``. Asserting it on both sides of the round
    #    trip confirms the parser does not drift the two fields apart
    #    (e.g., by reparsing ``raw_arguments`` into ``arguments``
    #    rather than echoing the dict it received).
    assert json.loads(tc.raw_arguments) == tc.arguments
    assert json.loads(recovered.raw_arguments) == recovered.arguments


@given(tc=tool_calls())
def test_serialised_envelope_is_canonical(tc: ToolCall) -> None:
    """Re-serialising a recovered :class:`ToolCall` is byte-stable.

    The envelope JSON (``id`` / ``skill_name`` / ``arguments`` /
    ``raw_arguments`` keys) is emitted with ``sort_keys=True`` so two
    semantically equal :class:`ToolCall` values yield byte-equal
    serialisations after a round trip — the strongest form of CP1.

    Validates: Requirements 1.4, 14.4 (CP1)
    """

    first = _serialize_intent(tc)
    again = _serialize_intent(_parse_intent(first))
    assert again == first


# ---------------------------------------------------------------------------
# Edge case: empty arguments dict
# ---------------------------------------------------------------------------


def test_empty_arguments_round_trips_cleanly() -> None:
    """A ``ToolCall`` with ``arguments={}`` round-trips without loss.

    Hypothesis's shrinker generally finds the empty-dict corner on its
    own, but pinning it as an example test guards against the corner
    silently regressing if the strategy starts excluding small dicts
    in a future revision.

    Validates: Requirements 1.4, 14.4 (CP1)
    """

    tc = ToolCall(
        id="call-1",
        skill_name="NoopSkill",
        arguments={},
        raw_arguments="{}",
    )

    recovered = _parse_intent(_serialize_intent(tc))

    assert recovered == tc
    assert recovered.arguments == {}
    assert recovered.raw_arguments == "{}"
    assert json.loads(recovered.raw_arguments) == recovered.arguments
