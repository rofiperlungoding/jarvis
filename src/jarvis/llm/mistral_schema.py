"""Mistral function-calling JSON Schema validator and tool mapper.

This module implements :class:`MistralSchemaValidator`, the gatekeeper that
ensures every Skill manifest's ``json_schema`` conforms to the JSON-Schema
draft-07 subset that Mistral's function-calling endpoint accepts before the
Skill is exposed to the LLM_Backend.

Design references:

* ``design.md`` — Dialog_Manager / Skill_Registry section: ``MistralSchemaValidator``
  rejects ``$ref`` to remote, ``oneOf`` mixing scalar and object/array types,
  and ``format`` keywords other than ``date-time``.
* ``design.md`` — Property 12 (Mistral function-definition conformance) /
  Requirements 14.3 and 19.4 / CP15: every generated function definition must
  have ``parameters.type == "object"``, must contain only Mistral-supported
  JSON-Schema keywords, and must round-trip through ``json.dumps`` /
  ``json.loads`` without information loss.

The validator is intentionally schema-format-agnostic about *which* extra
keywords it accepts — Mistral's published spec only fixes hard rules around
``$ref``, ``oneOf`` mixing, and ``format``. Other keywords (``properties``,
``required``, ``items``, ``enum``, ``description``, etc.) are passed through
unchanged. The walker simply recurses into every place a sub-schema can
appear so the three hard rules are enforced everywhere, not just at the top
level.

The top-level :class:`Skill` interface is being stood up in parallel
(task 10.1, ``src/jarvis/skills/base.py``). To avoid a circular import and
to keep this module independently testable, ``to_mistral_tool`` accepts
either:

* a :class:`SkillManifest`-shaped dict with ``name``, ``description``, and
  ``json_schema`` keys, or
* any duck-typed object exposing ``.name``, ``.description``, and
  ``.json_schema`` attributes.
"""

from __future__ import annotations

import json
from typing import Any, Final

__all__ = [
    "MistralSchemaError",
    "MistralSchemaValidator",
    "to_mistral_tool",
]


class MistralSchemaError(ValueError):
    """Raised when a JSON Schema fails the Mistral function-call subset rules.

    Inherits from :class:`ValueError` so callers that already catch malformed
    config / schema input as ``ValueError`` continue to work transparently.
    """


# ---------------------------------------------------------------------------
# Constants that capture the Mistral subset rules
# ---------------------------------------------------------------------------

# JSON Schema scalar (i.e., non-container) primitive types as defined in
# draft-07. ``null`` is included because ``oneOf`` may legitimately combine
# nullable scalars.
_SCALAR_TYPES: Final[frozenset[str]] = frozenset(
    {"string", "number", "integer", "boolean", "null"}
)

# Container types — these define the "object/array" half of the mixing rule
# in the design ("``oneOf`` mixing scalar and object types").
_NON_SCALAR_TYPES: Final[frozenset[str]] = frozenset({"object", "array"})

# Allowed ``format`` values for the Mistral subset. The design explicitly
# allows ``date-time``; everything else is rejected by the validator.
_ALLOWED_FORMATS: Final[frozenset[str]] = frozenset({"date-time"})

# Sub-schema container keywords — places we must recurse into so the three
# hard rules are enforced at any nesting level. Maps keyword name to its
# expected container shape:
#   "schema"         -> the value itself is a sub-schema
#   "schema_list"    -> the value is a list of sub-schemas
#   "schema_map"     -> the value is a mapping of name -> sub-schema
#   "items"          -> JSON Schema's polymorphic ``items``: schema OR list
_SCHEMA_KEYWORD: Final[str] = "schema"
_SCHEMA_LIST_KEYWORD: Final[str] = "schema_list"
_SCHEMA_MAP_KEYWORD: Final[str] = "schema_map"
_ITEMS_KEYWORD: Final[str] = "items"

_SUBSCHEMA_KEYWORDS: Final[dict[str, str]] = {
    # Logical combinators
    "allOf": _SCHEMA_LIST_KEYWORD,
    "anyOf": _SCHEMA_LIST_KEYWORD,
    "oneOf": _SCHEMA_LIST_KEYWORD,
    "not": _SCHEMA_KEYWORD,
    # Object-shape sub-schemas
    "properties": _SCHEMA_MAP_KEYWORD,
    "patternProperties": _SCHEMA_MAP_KEYWORD,
    "propertyNames": _SCHEMA_KEYWORD,
    # Array-shape sub-schemas
    "items": _ITEMS_KEYWORD,
    "additionalItems": _SCHEMA_KEYWORD,
    "contains": _SCHEMA_KEYWORD,
    # Conditional combinators (draft-07)
    "if": _SCHEMA_KEYWORD,
    "then": _SCHEMA_KEYWORD,
    "else": _SCHEMA_KEYWORD,
    # Reusable definitions
    "definitions": _SCHEMA_MAP_KEYWORD,
    "$defs": _SCHEMA_MAP_KEYWORD,
}

# ``additionalProperties`` may legally be a bool *or* a sub-schema. We treat
# the bool case as a no-op and recurse only when it's a dict, so it's
# special-cased outside ``_SUBSCHEMA_KEYWORDS``.


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class MistralSchemaValidator:
    """Validate JSON Schemas against the Mistral function-calling subset.

    The validator enforces three hard rules drawn from the design:

    1. ``$ref`` values must reference a *local* definition (string starting
       with ``"#"``). Remote refs (``http://...``, ``file://...``,
       ``schema.json#/foo``) are rejected so the LLM_Backend never has to
       fetch external documents.
    2. ``oneOf`` arrays must not mix *scalar* (string / number / integer /
       boolean / null) sub-schemas with *non-scalar* (object / array)
       sub-schemas. Mistral's tool parser treats these very differently and
       mixed branches confuse the model into emitting malformed arguments.
    3. The ``format`` keyword must be ``"date-time"`` if present at all;
       any other ``format`` value is rejected. The design notes that other
       formats may eventually be "downgraded" rather than rejected, but for
       now we surface them so Skill authors can fix them at registration
       time rather than seeing silent argument coercion at runtime.

    Other JSON Schema keywords (``type``, ``properties``, ``required``,
    ``enum``, ``description``, ``minimum``, ``maximum``, ``minLength``,
    ``maxLength``, ``pattern``, ``default``, ``additionalProperties``, ...)
    pass through unchanged.
    """

    # Class-level constants exposed for callers/tests that want to introspect
    # the subset rules without poking at module-private names.
    ALLOWED_FORMATS: Final[frozenset[str]] = _ALLOWED_FORMATS
    SCALAR_TYPES: Final[frozenset[str]] = _SCALAR_TYPES
    NON_SCALAR_TYPES: Final[frozenset[str]] = _NON_SCALAR_TYPES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, schema: Any) -> None:
        """Validate ``schema`` against the Mistral function-call subset.

        Walks the entire schema tree and raises :class:`MistralSchemaError`
        the moment a violation is detected. Designed to be called once per
        Skill at registration time; it is *not* a hot-path validator for
        per-call argument validation (use ``jsonschema.Draft7Validator`` for
        that, as documented in the design).

        Args:
            schema: A JSON Schema dictionary.

        Raises:
            MistralSchemaError: If ``schema`` violates any of the subset
                rules or is not a dictionary at all.
        """
        if not isinstance(schema, dict):
            raise MistralSchemaError(
                f"$: schema must be a dict, got {type(schema).__name__}"
            )
        self._walk(schema, path="$")

    def to_mistral_tool(self, manifest: Any) -> dict[str, Any]:
        """Map a Skill manifest to a Mistral function-definition dict.

        Args:
            manifest: Either a dict with ``name``, ``description``, and
                ``json_schema`` keys, or any object exposing those three
                attributes (e.g., a ``SkillManifest`` from
                ``src/jarvis/skills/base.py``).

        Returns:
            A dict shaped exactly like Mistral's tool definition payload::

                {
                    "type": "function",
                    "function": {
                        "name": <str>,
                        "description": <str>,
                        "parameters": <dict>,  # JSON Schema, draft-07 subset
                    },
                }

            The returned dict contains only JSON-serialisable values, so it
            survives ``json.loads(json.dumps(result))`` without loss
            (tuples are normalised to lists by the JSON round-trip).

        Raises:
            MistralSchemaError: If the manifest is missing required fields,
                ``parameters.type != "object"``, or the JSON Schema fails the
                Mistral subset rules.
        """
        name, description, parameters = self._extract_manifest_fields(manifest)
        if parameters.get("type") != "object":
            raise MistralSchemaError(
                "$.json_schema.type must be 'object' for Mistral tool "
                f"parameters, got {parameters.get('type')!r}"
            )
        # Run the subset checks before anything else so the error message
        # points at the *schema* rather than the JSON encoder.
        self.validate(parameters)
        # Round-trip the parameters through json.dumps/json.loads. This both
        # guarantees that every value is JSON-serialisable (Property 12 /
        # CP15: "round-trip through json.dumps/json.loads without
        # information loss") AND normalises Python-only types like tuples
        # and frozensets-of-strings into JSON arrays.
        try:
            clean_parameters = json.loads(json.dumps(parameters))
        except (TypeError, ValueError) as exc:
            raise MistralSchemaError(
                f"$.json_schema is not JSON-serialisable: {exc}"
            ) from exc
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": clean_parameters,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_manifest_fields(manifest: Any) -> tuple[str, str, dict[str, Any]]:
        """Pull ``(name, description, json_schema)`` from dict or object input.

        Centralised so the duck-typed contract is enforced in exactly one
        place. Raises :class:`MistralSchemaError` on missing or wrong-typed
        fields.
        """
        if isinstance(manifest, dict):
            try:
                name = manifest["name"]
                description = manifest["description"]
                parameters = manifest["json_schema"]
            except KeyError as exc:
                raise MistralSchemaError(
                    f"manifest dict is missing required key: {exc.args[0]!r}"
                ) from exc
        else:
            try:
                name = manifest.name
                description = manifest.description
                parameters = manifest.json_schema
            except AttributeError as exc:
                raise MistralSchemaError(
                    f"manifest object is missing required attribute: {exc}"
                ) from exc

        if not isinstance(name, str) or not name:
            raise MistralSchemaError(
                "manifest.name must be a non-empty string"
            )
        if not isinstance(description, str):
            raise MistralSchemaError(
                "manifest.description must be a string"
            )
        if not isinstance(parameters, dict):
            raise MistralSchemaError(
                "manifest.json_schema must be a dict"
            )
        return name, description, parameters

    def _walk(self, schema: Any, path: str) -> None:
        """Recursively validate ``schema`` at ``path``.

        ``path`` is a JSON-Path-ish string used for error messages so Skill
        authors can pinpoint the offending keyword.
        """
        if not isinstance(schema, dict):
            # Sub-schemas in JSON Schema are always objects (draft-07 also
            # allows boolean schemas: ``true`` / ``false``). Boolean schemas
            # are valid and have no keywords to validate.
            if isinstance(schema, bool):
                return
            raise MistralSchemaError(
                f"{path}: sub-schema must be a dict (or bool), "
                f"got {type(schema).__name__}"
            )

        self._check_ref(schema, path)
        self._check_format(schema, path)
        self._check_one_of_mixing(schema, path)
        self._recurse_subschemas(schema, path)

    # -- individual rule checks ----------------------------------------

    @staticmethod
    def _check_ref(schema: dict[str, Any], path: str) -> None:
        """Reject ``$ref`` values that point outside the current document."""
        if "$ref" not in schema:
            return
        ref = schema["$ref"]
        if not isinstance(ref, str):
            raise MistralSchemaError(
                f"{path}.$ref must be a string, got {type(ref).__name__}"
            )
        # Local refs always start with '#'. Bare '#' (the document root) and
        # '#/path/to/def' are accepted; anything else is treated as remote.
        if not ref.startswith("#"):
            raise MistralSchemaError(
                f"{path}.$ref must reference a local definition starting "
                f"with '#'; remote $ref is not supported: {ref!r}"
            )

    @classmethod
    def _check_format(cls, schema: dict[str, Any], path: str) -> None:
        """Reject ``format`` values outside the Mistral allow-list."""
        if "format" not in schema:
            return
        fmt = schema["format"]
        if fmt not in _ALLOWED_FORMATS:
            allowed = ", ".join(sorted(_ALLOWED_FORMATS))
            raise MistralSchemaError(
                f"{path}.format = {fmt!r} is not in the Mistral-supported "
                f"set ({{{allowed}}})"
            )

    def _check_one_of_mixing(self, schema: dict[str, Any], path: str) -> None:
        """Reject ``oneOf`` arrays that mix scalar and non-scalar branches."""
        if "oneOf" not in schema:
            return
        branches = schema["oneOf"]
        if not isinstance(branches, list):
            raise MistralSchemaError(
                f"{path}.oneOf must be a list, got {type(branches).__name__}"
            )
        categories: set[str] = set()
        for index, branch in enumerate(branches):
            category = self._categorize_branch(branch)
            if category == "mixed":
                # A single branch declared its own ``type`` as a mix of
                # scalar and non-scalar — surface it with the same error so
                # the failure mode is consistent.
                raise MistralSchemaError(
                    f"{path}.oneOf[{index}].type mixes scalar and "
                    "object/array types"
                )
            if category is not None:
                categories.add(category)
        if "scalar" in categories and "non_scalar" in categories:
            raise MistralSchemaError(
                f"{path}.oneOf mixes scalar (string/number/integer/boolean/"
                "null) and non-scalar (object/array) branches; Mistral "
                "requires homogeneous oneOf branches"
            )

    @staticmethod
    def _categorize_branch(branch: Any) -> str | None:  # noqa: PLR0911 - explicit branch enumeration is the clearest form
        """Bucket a ``oneOf`` branch by ``type`` into scalar / non-scalar.

        Returns ``"scalar"``, ``"non_scalar"``, ``"mixed"`` (when a single
        branch's ``type`` is itself a list mixing both kinds), or ``None``
        when no determination is possible (e.g., the branch uses ``$ref``,
        ``allOf``, or omits ``type`` altogether). ``None`` is treated as
        unconstrained — we only flag a violation when at least one scalar
        and one non-scalar branch are *positively* identified.
        """
        if not isinstance(branch, dict):
            return None
        branch_type = branch.get("type")
        if isinstance(branch_type, str):
            if branch_type in _SCALAR_TYPES:
                return "scalar"
            if branch_type in _NON_SCALAR_TYPES:
                return "non_scalar"
            return None
        if isinstance(branch_type, list):
            has_scalar = any(t in _SCALAR_TYPES for t in branch_type)
            has_non_scalar = any(t in _NON_SCALAR_TYPES for t in branch_type)
            if has_scalar and has_non_scalar:
                return "mixed"
            if has_scalar:
                return "scalar"
            if has_non_scalar:
                return "non_scalar"
        return None

    # -- recursion -----------------------------------------------------

    def _recurse_subschemas(self, schema: dict[str, Any], path: str) -> None:  # noqa: PLR0912 - dispatches over each JSON Schema keyword type
        """Walk into every keyword that can hold a sub-schema."""
        for keyword, kind in _SUBSCHEMA_KEYWORDS.items():
            if keyword not in schema:
                continue
            value = schema[keyword]
            sub_path = f"{path}.{keyword}"
            if kind == _SCHEMA_KEYWORD:
                self._walk(value, sub_path)
            elif kind == _SCHEMA_LIST_KEYWORD:
                if not isinstance(value, list):
                    raise MistralSchemaError(
                        f"{sub_path} must be a list, got {type(value).__name__}"
                    )
                for index, sub in enumerate(value):
                    self._walk(sub, f"{sub_path}[{index}]")
            elif kind == _SCHEMA_MAP_KEYWORD:
                if not isinstance(value, dict):
                    raise MistralSchemaError(
                        f"{sub_path} must be a dict, got {type(value).__name__}"
                    )
                for name, sub in value.items():
                    self._walk(sub, f"{sub_path}.{name}")
            elif kind == _ITEMS_KEYWORD:
                # ``items`` can be either a single sub-schema or a list of
                # sub-schemas (tuple validation, draft-07 §6.4).
                if isinstance(value, list):
                    for index, sub in enumerate(value):
                        self._walk(sub, f"{sub_path}[{index}]")
                else:
                    self._walk(value, sub_path)

        # ``additionalProperties`` is special-cased: bool means "allow or
        # forbid" and has no sub-schema to validate; only dict values
        # represent a constraint sub-schema.
        if "additionalProperties" in schema:
            ap = schema["additionalProperties"]
            if isinstance(ap, dict):
                self._walk(ap, f"{path}.additionalProperties")
            elif not isinstance(ap, bool):
                raise MistralSchemaError(
                    f"{path}.additionalProperties must be a bool or schema, "
                    f"got {type(ap).__name__}"
                )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def to_mistral_tool(manifest: Any) -> dict[str, Any]:
    """Module-level shortcut for :meth:`MistralSchemaValidator.to_mistral_tool`.

    The Skill_Registry usually owns a single :class:`MistralSchemaValidator`
    instance, but tests and one-off callers often want the simplest possible
    entry point. This wrapper instantiates a default validator and delegates.
    """
    return MistralSchemaValidator().to_mistral_tool(manifest)
