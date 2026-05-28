"""Property 12 — Mistral function-definition conformance.

From ``design.md §Correctness Properties``:

    *For any* registered Skill ``S``, the dictionary returned by
    ``mistral_tool_definitions()[S.name]`` SHALL satisfy the Mistral
    function-calling schema validator, SHALL have
    ``parameters.type == "object"``, SHALL not contain JSON-Schema
    keywords outside the Mistral-supported subset, and SHALL
    round-trip through ``json.dumps`` / ``json.loads`` without
    information loss.

This file implements that universal quantification with Hypothesis.

Strategy
--------

We quantify over Skill manifests rather than over already-mapped tool
payloads, because the property is about the behaviour of
:meth:`MistralSchemaValidator.to_mistral_tool`. The strategy is a
two-step composition:

1. Draw a Mistral tool payload from
   :func:`tests.strategies.mistral_tool_payloads`. That strategy
   already emits the exact ``{"type": "function", "function": {...}}``
   shape :meth:`MistralSchemaValidator.to_mistral_tool` is supposed to
   produce, restricted to Mistral-supported scalar property types.
2. Project the payload back into a *manifest dict* of the form
   ``{"name", "description", "json_schema"}`` that ``to_mistral_tool``
   accepts as input. The ``json_schema`` of the manifest is the
   ``parameters`` block of the original payload — i.e., the most
   realistic input shape a Skill author would produce.

Going manifest-first is what the task description calls out and what
the production seam (``SkillRegistry.mistral_tool_definitions``)
actually exercises: a ``SkillManifest`` per registered Skill, mapped
through the validator. Using the *output* of ``mistral_tool_payloads``
as the strategy would re-test the strategy rather than the production
code.

Property assertions
-------------------

For every generated manifest, the test calls
``MistralSchemaValidator().to_mistral_tool(manifest)`` and asserts:

1. ``result["type"] == "function"`` — the Mistral tool envelope.
2. ``result["function"]["parameters"]["type"] == "object"`` — the
   hard rule from CP15 that Mistral's function-calling endpoint
   refuses to accept anything else.
3. The schema in ``parameters`` contains no Mistral-unsupported
   keywords:

   * no ``$ref`` that points outside the current document (remote
     refs); local ``#/...`` refs are allowed.
   * no ``format`` value outside :data:`MistralSchemaValidator.ALLOWED_FORMATS`
     (currently the singleton ``"date-time"``).
   * no ``oneOf`` array that mixes scalar (``string``/``number``/
     ``integer``/``boolean``/``null``) and non-scalar (``object``/
     ``array``) branches.

   We assert this directly by re-running the validator on the result's
   ``parameters`` block (which is the design's contract: anything the
   validator accepts is in-subset; anything it rejects is out).
4. ``json.loads(json.dumps(result)) == result`` — the round-trip
   guarantee. Because the strategy generates only Mistral-supported
   scalar types and the validator normalises tuples / sets via
   ``json.dumps`` already, the recovered dict is deeply equal to the
   original.

Validates: Requirements 14.3, 19.4 (CP15)
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, strategies as st
from tests.strategies import mistral_tool_payloads

from jarvis.llm.mistral_schema import MistralSchemaError, MistralSchemaValidator

# ---------------------------------------------------------------------------
# Manifest strategy — derived from ``mistral_tool_payloads``
# ---------------------------------------------------------------------------


def _payload_to_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a Mistral tool payload back into a manifest-shaped dict.

    The Mistral tool payload shape is::

        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": ...}}

    A manifest, as accepted by
    :meth:`MistralSchemaValidator.to_mistral_tool`, takes the form::

        {"name": ..., "description": ..., "json_schema": ...}

    where ``json_schema`` is exactly what the payload calls
    ``parameters``. The mapping is therefore a pure rename and is the
    natural inverse of ``to_mistral_tool`` for the *strategy's* shape
    of generated payloads (a flat object schema with scalar
    properties).
    """

    function = payload["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "json_schema": function["parameters"],
    }


def _manifests() -> st.SearchStrategy[dict[str, Any]]:
    """Generate manifest dicts that ``to_mistral_tool`` should accept.

    We compose with :func:`mistral_tool_payloads` rather than building
    a parallel JSON-Schema strategy so the two property tests share a
    single source of generation truth: anything the strategy emits is,
    by construction, inside the Mistral subset.
    """

    return mistral_tool_payloads().map(_payload_to_manifest)


# ---------------------------------------------------------------------------
# Property
# ---------------------------------------------------------------------------


# Disallowed ``format`` values that ``MistralSchemaValidator`` rejects.
# A small, hand-picked set covering the most common JSON-Schema
# formats Skill authors might reach for. The property's ``format``
# rejection invariant is otherwise asserted by re-running
# ``validator.validate`` on the produced ``parameters`` block; this
# explicit list keeps the test self-contained and gives the failure
# message a meaningful hint.
_DISALLOWED_FORMATS: tuple[str, ...] = (
    "email",
    "uri",
    "uuid",
    "ipv4",
    "ipv6",
    "hostname",
    "regex",
)


@given(manifest=_manifests())
def test_to_mistral_tool_produces_conformant_function_definition(
    manifest: dict[str, Any],
) -> None:
    """``to_mistral_tool(manifest)`` produces a Mistral-conformant tool dict.

    Validates: Requirements 14.3, 19.4 (CP15)
    """

    validator = MistralSchemaValidator()
    result = validator.to_mistral_tool(manifest)

    # 1. Outer envelope: every Mistral tool definition is a
    #    ``{"type": "function", ...}`` dict. The shape is the literal
    #    contract Mistral's HTTP API enforces server-side, and
    #    ``to_mistral_tool`` is the only place in the codebase where
    #    that shape is constructed.
    assert isinstance(result, dict)
    assert result["type"] == "function"
    assert isinstance(result["function"], dict)

    function = result["function"]
    # ``name`` and ``description`` survive the mapping verbatim. Type
    # checks here guard against an accidental ``str`` -> ``bytes`` /
    # ``None`` regression in the manifest extractor.
    assert isinstance(function["name"], str) and function["name"]
    assert isinstance(function["description"], str)

    # 2. ``parameters.type == "object"`` is the hard CP15 rule. We
    #    re-state it here independently of the validator so a future
    #    refactor that loosens the validator does not silently
    #    invalidate the property.
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["type"] == "object", (
        f"Mistral parameters block must be type=object, got "
        f"{parameters.get('type')!r}"
    )

    # 3a. The output passes the Mistral subset validator. Any
    #     unsupported keyword (remote ``$ref``, mixed ``oneOf``,
    #     non-``date-time`` ``format``) would have already raised
    #     during ``to_mistral_tool``; re-running ``validate`` on the
    #     output gives the property an explicit, independent witness
    #     and means a regression that loosens ``to_mistral_tool``
    #     without correspondingly loosening ``validate`` would still
    #     surface here.
    validator.validate(parameters)

    # 3b. Spot-check the explicit "no unsupported keyword" invariants
    #     by walking the parameters tree. These are the exact
    #     keywords/values ``MistralSchemaValidator`` rejects, listed
    #     so the test fails with a precise message if a future
    #     strategy regression starts emitting them.
    for path, sub in _walk_subschemas(parameters):
        if not isinstance(sub, dict):
            continue
        ref = sub.get("$ref")
        if isinstance(ref, str):
            assert ref.startswith("#"), (
                f"{path}: remote $ref leaked through to the Mistral tool "
                f"definition: {ref!r}"
            )
        fmt = sub.get("format")
        if fmt is not None:
            assert fmt not in _DISALLOWED_FORMATS, (
                f"{path}.format={fmt!r} is in the explicit-disallowed list"
            )
            assert fmt in MistralSchemaValidator.ALLOWED_FORMATS, (
                f"{path}.format={fmt!r} is outside the Mistral allow-list "
                f"({sorted(MistralSchemaValidator.ALLOWED_FORMATS)})"
            )

    # 4. JSON round-trip: ``json.loads(json.dumps(result)) == result``.
    #    This is the textual contract Property 12 spells out — every
    #    value emitted is JSON-serialisable (no Python-only types like
    #    sets / tuples / datetimes survive the validator), and the
    #    recovered structure is deeply equal to the original.
    encoded = json.dumps(result)
    recovered = json.loads(encoded)
    assert recovered == result


# ---------------------------------------------------------------------------
# Sub-schema walker
# ---------------------------------------------------------------------------


def _walk_subschemas(schema: Any, path: str = "$") -> list[tuple[str, Any]]:
    """Yield every (path, sub-schema) pair reachable inside ``schema``.

    Mirrors the recursion in
    :meth:`MistralSchemaValidator._recurse_subschemas` but is
    intentionally permissive: any non-dict node terminates that
    branch silently because the property test only inspects dict
    sub-schemas for the ``$ref`` / ``format`` checks. Keeping a
    second, simpler walker here makes the property's invariant
    statement self-contained — readers do not have to re-derive what
    "every keyword that holds a sub-schema" means from production
    code.
    """

    out: list[tuple[str, Any]] = [(path, schema)]
    if not isinstance(schema, dict):
        return out

    # Logical combinators / single-schema keywords.
    for key in ("not", "if", "then", "else", "propertyNames", "contains",
                "additionalItems"):
        if key in schema:
            out.extend(_walk_subschemas(schema[key], f"{path}.{key}"))

    # Schema-list keywords.
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            for index, sub in enumerate(schema[key]):
                out.extend(_walk_subschemas(sub, f"{path}.{key}[{index}]"))

    # Schema-map keywords.
    for key in ("properties", "patternProperties", "definitions", "$defs"):
        if key in schema and isinstance(schema[key], dict):
            for name, sub in schema[key].items():
                out.extend(_walk_subschemas(sub, f"{path}.{key}.{name}"))

    # ``items`` is polymorphic (single schema OR list).
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            for index, sub in enumerate(items):
                out.extend(_walk_subschemas(sub, f"{path}.items[{index}]"))
        else:
            out.extend(_walk_subschemas(items, f"{path}.items"))

    # ``additionalProperties`` may be a bool *or* a sub-schema.
    if "additionalProperties" in schema and isinstance(
        schema["additionalProperties"], dict
    ):
        out.extend(
            _walk_subschemas(
                schema["additionalProperties"], f"{path}.additionalProperties"
            )
        )

    return out


# ---------------------------------------------------------------------------
# Concrete edge-case examples
# ---------------------------------------------------------------------------


def test_minimal_object_manifest_round_trips() -> None:
    """A bare ``type=object`` manifest produces a valid tool definition.

    The strategy may not always shrink to the empty-properties corner
    on its own, so we pin it as an explicit example. This guards
    against a regression where ``to_mistral_tool`` starts requiring a
    non-empty ``properties`` block (which would silently break MCP
    Skills that advertise zero-arg tools).

    Validates: Requirements 14.3, 19.4 (CP15)
    """

    manifest = {
        "name": "ZeroArgSkill",
        "description": "Skill that takes no arguments.",
        "json_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }

    result = MistralSchemaValidator().to_mistral_tool(manifest)

    assert result["type"] == "function"
    assert result["function"]["parameters"]["type"] == "object"
    assert json.loads(json.dumps(result)) == result


def test_manifest_with_date_time_format_is_accepted() -> None:
    """``format=date-time`` is the one allowed format and must survive.

    Validates: Requirements 14.3, 19.4 (CP15)
    """

    manifest = {
        "name": "ReminderSkill",
        "description": "Schedule a reminder.",
        "json_schema": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "format": "date-time"},
                "label": {"type": "string"},
            },
            "required": ["when"],
            "additionalProperties": False,
        },
    }

    result = MistralSchemaValidator().to_mistral_tool(manifest)

    when_schema = result["function"]["parameters"]["properties"]["when"]
    assert when_schema["format"] == "date-time"
    assert json.loads(json.dumps(result)) == result


def test_manifest_with_remote_ref_is_rejected() -> None:
    """Remote ``$ref`` must be rejected — out-of-subset.

    Asserting the *negative* path here keeps Property 12's contract
    bidirectional: in-subset manifests round-trip; out-of-subset
    manifests raise :class:`MistralSchemaError`. The property test
    above only checks the positive direction (the strategy never
    emits remote refs), so a unit-style negative example pins the
    other half.

    Validates: Requirements 14.3, 19.4 (CP15)
    """

    manifest = {
        "name": "BadSkill",
        "description": "References an external schema document.",
        "json_schema": {
            "type": "object",
            "properties": {
                "target": {"$ref": "https://example.com/schemas/x.json#/Foo"}
            },
        },
    }

    try:
        MistralSchemaValidator().to_mistral_tool(manifest)
    except MistralSchemaError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError(
            "to_mistral_tool must reject manifests with remote $ref"
        )
