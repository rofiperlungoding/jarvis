"""Unit tests for ``jarvis.llm.mistral_schema``.

Covers the three subset rules (``$ref`` locality, ``oneOf`` mixing,
``format`` allow-list), the ``to_mistral_tool`` mapping shape, and the
``json.dumps``/``json.loads`` round-trip guarantee documented in Property 12
of the design.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pytest

from jarvis.llm.mistral_schema import (
    MistralSchemaError,
    MistralSchemaValidator,
    to_mistral_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeManifest:
    """Duck-typed stand-in for the not-yet-defined ``SkillManifest``."""

    name: str
    description: str
    json_schema: dict[str, Any]


def _object_schema(**properties: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# validate(): happy paths
# ---------------------------------------------------------------------------


def test_validate_accepts_minimal_object_schema() -> None:
    MistralSchemaValidator().validate({"type": "object", "properties": {}})


def test_validate_accepts_typical_skill_schema() -> None:
    schema = _object_schema(
        application={"type": "string", "minLength": 1},
        max_results={"type": "integer", "minimum": 1, "maximum": 10},
        flags={
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
        },
    )
    schema["required"] = ["application"]
    MistralSchemaValidator().validate(schema)


def test_validate_accepts_date_time_format() -> None:
    schema = _object_schema(
        when={"type": "string", "format": "date-time"},
    )
    MistralSchemaValidator().validate(schema)


def test_validate_accepts_local_ref() -> None:
    schema = {
        "type": "object",
        "properties": {"target": {"$ref": "#/definitions/Target"}},
        "definitions": {
            "Target": {"type": "string", "enum": ["alpha", "beta"]},
        },
    }
    MistralSchemaValidator().validate(schema)


def test_validate_accepts_homogeneous_one_of_scalars() -> None:
    schema = _object_schema(
        value={
            "oneOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ]
        }
    )
    MistralSchemaValidator().validate(schema)


def test_validate_accepts_homogeneous_one_of_objects() -> None:
    schema = _object_schema(
        payload={
            "oneOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "array", "items": {"type": "integer"}},
            ]
        }
    )
    MistralSchemaValidator().validate(schema)


def test_validate_accepts_boolean_subschemas() -> None:
    # JSON Schema allows ``true`` / ``false`` as full sub-schemas.
    schema = {
        "type": "object",
        "properties": {"anything": True, "nothing": False},
    }
    MistralSchemaValidator().validate(schema)


def test_validate_recurses_through_items_list_form() -> None:
    # Tuple-validation form of ``items``.
    schema = _object_schema(
        pair={
            "type": "array",
            "items": [
                {"type": "string"},
                {"type": "integer"},
            ],
        }
    )
    MistralSchemaValidator().validate(schema)


# ---------------------------------------------------------------------------
# validate(): rejection paths
# ---------------------------------------------------------------------------


def test_validate_rejects_non_dict_top_level() -> None:
    with pytest.raises(MistralSchemaError, match="must be a dict"):
        MistralSchemaValidator().validate("not a schema")  # type: ignore[arg-type]


def test_validate_rejects_remote_ref_http() -> None:
    schema = _object_schema(
        target={"$ref": "https://example.com/schema.json#/Target"}
    )
    with pytest.raises(MistralSchemaError, match="remote \\$ref"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_remote_ref_relative_path() -> None:
    schema = _object_schema(target={"$ref": "other.json#/Target"})
    with pytest.raises(MistralSchemaError, match="remote \\$ref"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_non_string_ref() -> None:
    schema = _object_schema(target={"$ref": 42})
    with pytest.raises(MistralSchemaError, match=r"\$ref must be a string"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_unsupported_format() -> None:
    schema = _object_schema(
        email={"type": "string", "format": "email"},
    )
    with pytest.raises(MistralSchemaError, match="format = 'email'"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_unsupported_format_in_items() -> None:
    schema = _object_schema(
        emails={
            "type": "array",
            "items": {"type": "string", "format": "uri"},
        }
    )
    with pytest.raises(MistralSchemaError, match="format = 'uri'"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_one_of_mixing_scalar_and_object() -> None:
    schema = _object_schema(
        value={
            "oneOf": [
                {"type": "string"},
                {"type": "object", "properties": {"a": {"type": "string"}}},
            ]
        }
    )
    with pytest.raises(MistralSchemaError, match="oneOf mixes scalar"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_one_of_mixing_scalar_and_array() -> None:
    schema = _object_schema(
        value={
            "oneOf": [
                {"type": "integer"},
                {"type": "array", "items": {"type": "integer"}},
            ]
        }
    )
    with pytest.raises(MistralSchemaError, match="oneOf mixes scalar"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_one_of_branch_with_mixed_type_list() -> None:
    schema = _object_schema(
        value={"oneOf": [{"type": ["string", "object"]}]},
    )
    with pytest.raises(
        MistralSchemaError, match="mixes scalar and object/array"
    ):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_one_of_not_a_list() -> None:
    schema = _object_schema(value={"oneOf": {"type": "string"}})
    with pytest.raises(MistralSchemaError, match="oneOf must be a list"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_violation_inside_nested_definitions() -> None:
    schema = {
        "type": "object",
        "definitions": {
            "Bad": {"type": "string", "format": "uuid"},
        },
        "properties": {"x": {"$ref": "#/definitions/Bad"}},
    }
    with pytest.raises(MistralSchemaError, match=r"definitions\.Bad\.format"):
        MistralSchemaValidator().validate(schema)


def test_validate_rejects_additional_properties_wrong_type() -> None:
    schema = {"type": "object", "additionalProperties": "yes"}
    with pytest.raises(
        MistralSchemaError, match="additionalProperties must be a bool"
    ):
        MistralSchemaValidator().validate(schema)


def test_validate_accepts_additional_properties_schema() -> None:
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    MistralSchemaValidator().validate(schema)


# ---------------------------------------------------------------------------
# to_mistral_tool: shape
# ---------------------------------------------------------------------------


def test_to_mistral_tool_dict_input_returns_expected_shape() -> None:
    manifest = {
        "name": "LaunchAppSkill",
        "description": "Launch a known application by spoken name.",
        "json_schema": _object_schema(
            application={"type": "string", "minLength": 1},
        ),
    }
    tool = to_mistral_tool(manifest)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "LaunchAppSkill"
    assert tool["function"]["description"].startswith("Launch")
    assert tool["function"]["parameters"]["type"] == "object"
    # Top-level keys are exactly the documented set.
    assert set(tool["function"].keys()) == {"name", "description", "parameters"}
    assert set(tool.keys()) == {"type", "function"}


def test_to_mistral_tool_object_input_uses_attributes() -> None:
    manifest = _FakeManifest(
        name="WebSearchSkill",
        description="Search the web.",
        json_schema=_object_schema(
            query={"type": "string"},
            max_results={"type": "integer", "default": 5, "maximum": 10},
        ),
    )
    tool = MistralSchemaValidator().to_mistral_tool(manifest)
    assert tool["function"]["name"] == "WebSearchSkill"
    assert tool["function"]["parameters"]["properties"]["max_results"][
        "maximum"
    ] == 10


def test_to_mistral_tool_round_trips_through_json() -> None:
    manifest = _FakeManifest(
        name="ReminderSkill",
        description="Set a reminder.",
        json_schema=_object_schema(
            label={"type": "string"},
            trigger_at={"type": "string", "format": "date-time"},
        ),
    )
    tool = to_mistral_tool(manifest)
    encoded = json.dumps(tool)
    restored = json.loads(encoded)
    assert restored == tool


def test_to_mistral_tool_normalises_tuple_to_list() -> None:
    # Skill authors may write ``"required": ("a", "b")`` by accident; the
    # JSON round-trip should produce a list without losing information.
    manifest = {
        "name": "S",
        "description": "d",
        "json_schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ("a", "b"),
        },
    }
    tool = to_mistral_tool(manifest)
    assert tool["function"]["parameters"]["required"] == ["a", "b"]


# ---------------------------------------------------------------------------
# to_mistral_tool: rejection paths
# ---------------------------------------------------------------------------


def test_to_mistral_tool_rejects_non_object_parameters() -> None:
    manifest = {
        "name": "BadSkill",
        "description": "",
        "json_schema": {"type": "string"},
    }
    with pytest.raises(MistralSchemaError, match="type must be 'object'"):
        to_mistral_tool(manifest)


def test_to_mistral_tool_rejects_missing_dict_field() -> None:
    manifest = {"name": "X", "description": "d"}
    with pytest.raises(MistralSchemaError, match="missing required key"):
        to_mistral_tool(manifest)


def test_to_mistral_tool_rejects_object_missing_attribute() -> None:
    class Partial:
        name = "X"
        description = "d"

    with pytest.raises(MistralSchemaError, match="missing required attribute"):
        to_mistral_tool(Partial())


def test_to_mistral_tool_rejects_empty_name() -> None:
    manifest = {"name": "", "description": "d", "json_schema": {"type": "object"}}
    with pytest.raises(MistralSchemaError, match="name must be a non-empty"):
        to_mistral_tool(manifest)


def test_to_mistral_tool_rejects_non_serialisable_value() -> None:
    # ``set`` is not JSON-serialisable. We still walk the schema to validate
    # subset rules, then fail at the json.dumps step with a clear message.
    manifest = {
        "name": "X",
        "description": "d",
        "json_schema": {
            "type": "object",
            "properties": {"x": {"type": "string", "default": {1, 2, 3}}},
        },
    }
    with pytest.raises(MistralSchemaError, match="not JSON-serialisable"):
        to_mistral_tool(manifest)


def test_to_mistral_tool_rejects_subset_violation_with_path() -> None:
    manifest = {
        "name": "X",
        "description": "d",
        "json_schema": _object_schema(
            email={"type": "string", "format": "email"},
        ),
    }
    with pytest.raises(MistralSchemaError, match=r"properties\.email\.format"):
        to_mistral_tool(manifest)
