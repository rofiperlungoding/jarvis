"""Property 2 — Schema validation soundness.

For *any* registered Skill ``S`` and *any* JSON-compatible argument
object ``A``, :meth:`SkillRegistry.dispatch` SHALL return a
``schema_violation`` error iff
``jsonschema.Draft7Validator(S.json_schema).is_valid(A)`` is ``False``,
and SHALL invoke ``S.execute`` exactly once otherwise.

Strategy
--------

The test exercises three representative built-in Skills
(:class:`~jarvis.skills.builtin.launch_app.LaunchAppSkill`,
:class:`~jarvis.skills.builtin.media_control.MediaControlSkill`,
:class:`~jarvis.skills.builtin.volume.VolumeSkill`) — each carries a
distinct schema shape (free string + ``minLength``, ``enum``, and
``allOf``/``if``/``then`` respectively) so the property has good
coverage of the JSON Schema features Mistral function calling uses.

Each Skill is registered in a fresh :class:`SkillRegistry` *wrapped* by
a :class:`_RecordingSkill`. The wrapper preserves the original
:class:`SkillManifest` (so the registry's :class:`Draft7Validator` is
built from the genuine schema) but replaces the executor body with a
counter that returns a trivial success. Two consequences follow:

* the test does not need a fully-wired platform adapter / credential
  store / application registry — the recorder ignores them; and
* the only way to observe a ``schema_violation`` is via the
  *registry's* schema gate, which is exactly the surface CP2 covers.

The "valid args" branch generates arguments through the shared
:func:`tests.strategies.tool_call_arguments` helper (which delegates to
:func:`hypothesis_jsonschema.from_schema`) and asserts the executor was
invoked exactly once and the result's ``error_code`` is *not*
``"schema_violation"``. The "invalid args" branch generates arbitrary
JSON-compatible dicts, filters them through ``assume(not is_valid)``,
and asserts the inverse: the executor was NOT invoked and the result
carries ``error_code == "schema_violation"``.

Validates: Requirements 14.3, 14.4, 14.5 (CP2)
"""

from __future__ import annotations

import asyncio
from typing import Any

from hypothesis import assume, given, strategies as st
from jsonschema import Draft7Validator
import pytest
from tests.strategies import tool_call_arguments

from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.builtin.launch_app import LaunchAppSkill
from jarvis.skills.builtin.media_control import MediaControlSkill
from jarvis.skills.builtin.volume import VolumeSkill
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------


class _RecordingSkill:
    """Wrap a real :class:`Skill`, recording every ``execute`` invocation.

    The wrapper exposes the wrapped Skill's :class:`SkillManifest`
    verbatim so the :class:`SkillRegistry` builds its
    :class:`Draft7Validator` from the genuine schema. The executor body
    is replaced by a counter / arg-recorder that always returns a
    trivial :meth:`SkillResult.success`. Substituting the body keeps
    the test focused on the registry's schema gate (the surface
    Property 2 / CP2 quantifies over) and frees the harness from having
    to provide a platform adapter, credential store, or application
    registry that real built-in Skills depend on.
    """

    def __init__(self, wrapped: Skill) -> None:
        # Holding a reference to the wrapped Skill is unnecessary for
        # the property under test, but it keeps the wrapper honest:
        # static checkers can confirm the wrapped object satisfies the
        # :class:`Skill` Protocol at construction time.
        self._wrapped: Skill = wrapped
        self.manifest: SkillManifest = wrapped.manifest
        self.execute_calls: int = 0
        self.last_args: dict[str, Any] | None = None

    async def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> SkillResult:
        self.execute_calls += 1
        self.last_args = args
        # ``SkillResult.success`` returns ``error_code=None``, which is
        # the easy way to satisfy the "result.error_code != 'schema_violation'"
        # half of the property in the valid-args branch.
        return SkillResult.success(value={"recorded": True})


# ---------------------------------------------------------------------------
# Skill matrix
# ---------------------------------------------------------------------------


# Three representative built-in Skills covering distinct schema shapes:
# * LaunchAppSkill   — single string field with ``minLength: 1`` and
#                      ``additionalProperties: false``
# * MediaControlSkill — single ``enum`` field
# * VolumeSkill      — ``enum`` + integer + ``allOf``/``if``/``then``
SKILL_FACTORIES: tuple[type, ...] = (
    LaunchAppSkill,
    MediaControlSkill,
    VolumeSkill,
)


def _make_registry(skill_cls: type) -> tuple[SkillRegistry, _RecordingSkill]:
    """Build a fresh registry wrapping ``skill_cls()`` in a recorder.

    Returns the registry and the recorder so tests can introspect the
    invocation count after dispatch.
    """
    recorder = _RecordingSkill(skill_cls())
    registry = SkillRegistry()
    # The wrapper structurally satisfies the :class:`Skill` Protocol —
    # ``register`` performs both meta-schema validation and the Mistral
    # subset check on ``recorder.manifest.json_schema`` (i.e., the
    # actual production schema for the wrapped Skill).
    registry.register(recorder)
    return registry, recorder


# ---------------------------------------------------------------------------
# Strategies for the "invalid args" branch
# ---------------------------------------------------------------------------


# A JSON-compatible value tree. Recursive so nested objects / arrays
# show up in the generated dicts, exercising schemas that constrain
# nested structure (none of our representative Skills do, but keeping
# the strategy general means the same property test transparently
# covers any future Skill we add to ``SKILL_FACTORIES``).
_json_values: st.SearchStrategy[Any] = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.text(max_size=16),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=8,
)


def _arbitrary_dicts() -> st.SearchStrategy[dict[str, Any]]:
    """Free-form dict strategy used to seed schema-violation candidates.

    Combined with :func:`hypothesis.assume(not is_valid)` inside the
    test, this gives the property a broad catchment of malformed
    Tool_Calls (missing required fields, wrong types, extra properties,
    enum mismatches, ``set`` without ``level`` for VolumeSkill, etc.).
    """
    return st.dictionaries(
        keys=st.text(max_size=8),
        values=_json_values,
        max_size=6,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch(
    registry: SkillRegistry, name: str, args: dict[str, Any]
) -> SkillResult:
    """Run :meth:`SkillRegistry.dispatch` on a fresh asyncio loop.

    Mirrors the ``_run`` helper used in the registry's hand-shaped unit
    tests so we do not pull pytest-asyncio into a property-test module
    that would otherwise be fully synchronous.
    """
    return asyncio.run(registry.dispatch(name, args, SkillContext()))


# ---------------------------------------------------------------------------
# Property 2: valid branch — execute is invoked exactly once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_cls", SKILL_FACTORIES, ids=lambda c: c.__name__
)
def test_property_02_valid_args_invoke_execute_exactly_once(
    skill_cls: type,
) -> None:
    """Schema-valid args -> ``execute`` runs once, no ``schema_violation``.

    Validates: Requirements 14.3, 14.4, 14.5 (CP2)
    """

    registry, recorder = _make_registry(skill_cls)
    schema = recorder.manifest.json_schema
    validator = Draft7Validator(schema)
    skill_name = recorder.manifest.name

    @given(args=tool_call_arguments(recorder))
    def _check(args: dict[str, Any]) -> None:
        # Sanity check on the strategy: ``hypothesis_jsonschema.from_schema``
        # only emits schema-valid documents, so this assertion is a
        # belt-and-braces guard against accidental drift between the
        # strategy and the registry's validator (CP2 requires the two
        # to agree exactly).
        assert validator.is_valid(args), list(validator.iter_errors(args))

        before = recorder.execute_calls
        result = _dispatch(registry, skill_name, args)
        # The executor must run *exactly once* per dispatch: not zero
        # (which would mean the schema gate spuriously rejected valid
        # args) and not more than one (which would imply a hidden
        # retry / recursion the property explicitly forbids).
        assert recorder.execute_calls == before + 1, (
            "Skill.execute must be invoked exactly once for schema-valid args; "
            f"observed {recorder.execute_calls - before} call(s) for {args!r}"
        )
        # The other half of the iff: the result must NOT be a
        # schema_violation. We do not over-constrain to ``ok=True``
        # because the recorder *does* return success here — but in
        # principle a future Skill could legitimately fail with
        # another error code (e.g., ``provider_unavailable``) on
        # schema-valid input, and the property still holds as long as
        # the code is not ``schema_violation``.
        assert result.error_code != "schema_violation", (
            "schema-valid args must not yield schema_violation; got "
            f"{result.error_code!r} (message={result.error_message!r})"
        )

    _check()


# ---------------------------------------------------------------------------
# Property 2: invalid branch — schema_violation, executor NOT invoked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "skill_cls", SKILL_FACTORIES, ids=lambda c: c.__name__
)
def test_property_02_invalid_args_yield_schema_violation_without_executor(
    skill_cls: type,
) -> None:
    """Schema-invalid args -> ``schema_violation``, ``execute`` not called.

    Validates: Requirements 14.3, 14.4, 14.5 (CP2)
    """

    registry, recorder = _make_registry(skill_cls)
    schema = recorder.manifest.json_schema
    validator = Draft7Validator(schema)
    skill_name = recorder.manifest.name

    @given(args=_arbitrary_dicts())
    def _check(args: dict[str, Any]) -> None:
        # Restrict to the schema-invalid half-plane. ``assume`` lets
        # Hypothesis discard the (rare) schema-valid candidates without
        # counting them against the example budget.
        assume(not validator.is_valid(args))

        before = recorder.execute_calls
        result = _dispatch(registry, skill_name, args)
        # ``ok=False`` is implied by ``error_code='schema_violation'``
        # via the :class:`SkillResult` invariants, but checking it
        # explicitly produces a clearer failure message when the
        # registry contract regresses.
        assert result.ok is False
        assert result.error_code == "schema_violation", (
            "schema-invalid args must yield schema_violation; got "
            f"{result.error_code!r} (message={result.error_message!r})"
        )
        # The executor must NOT have been invoked. This is the strict
        # half of the iff: the registry's schema gate short-circuits
        # *before* dispatch, so the recorder's counter must not move.
        assert recorder.execute_calls == before, (
            "Skill.execute must NOT be invoked for schema-invalid args; "
            f"observed {recorder.execute_calls - before} call(s) for {args!r}"
        )

    _check()
