"""Built-in :class:`VolumeSkill` — system master volume control.

Wraps :meth:`jarvis.automation.platform.PlatformAdapter.set_volume`,
:meth:`~jarvis.automation.platform.PlatformAdapter.adjust_volume`, and
:meth:`~jarvis.automation.platform.PlatformAdapter.hotkey` behind a Skill
that the LLM can invoke through Mistral function calling. The acceptance
criteria fix the operation vocabulary (Requirement 4.3) at five values —
``set``, ``increase``, ``decrease``, ``mute``, ``unmute`` — together with
an optional integer ``level`` argument in the inclusive range ``[0, 100]``
that doubles as both the absolute target for ``set`` (Requirement 4.4)
and the delta for ``increase`` / ``decrease`` (Requirement 4.5). When
``level`` is omitted from an ``increase`` / ``decrease`` call the Skill
applies the documented default delta of 10 percent.

Operation → adapter call mapping
--------------------------------

* ``set``   → :meth:`PlatformAdapter.set_volume(level)`. ``level`` is
  required for this operation (the schema enforces the constraint via
  ``if``/``then``, so the registry surfaces missing levels as
  ``schema_violation`` and triggers the LLM retry loop documented in
  Requirement 14.5).
* ``increase`` → :meth:`PlatformAdapter.adjust_volume(+delta)` where
  ``delta`` is ``level`` if supplied, else :data:`DEFAULT_DELTA_PCT`.
* ``decrease`` → :meth:`PlatformAdapter.adjust_volume(-delta)` with the
  same delta resolution.
* ``mute`` / ``unmute`` → :meth:`PlatformAdapter.hotkey("volumemute")`.
  Windows exposes the system mute as a single ``VK_VOLUME_MUTE`` toggle
  rather than as separate set-mute / clear-mute primitives, which is why
  both Skill operations dispatch the same hotkey. The Skill records
  which user-facing operation was requested in the success payload so
  the Dialog_Manager can phrase its acknowledgement naturally.

Failure mapping
---------------

* :class:`PlatformNotSupportedError` from the adapter is converted to
  ``SkillResult.error("platform_not_supported", ...)``. This mirrors the
  contract in :mod:`jarvis.skills.builtin.media_control` and matches the
  error-taxonomy entry in :mod:`jarvis.skills.base`.
* ``ctx.platform_adapter`` missing or not satisfying the
  :class:`PlatformAdapter` Protocol → ``internal_error`` so the
  Dialog_Manager can surface the misconfiguration without crashing.
* Any other adapter exception bubbles up to the registry, which converts
  it to ``internal_error`` with a traceback id (Property 7 / CP10).

The Skill is non-destructive: changing the volume has no permanent
side effect, so :attr:`SkillManifest.destructive` is ``False`` and the
Authorization_Policy will not request user confirmation before dispatch
(Requirement 16.1).

Validates: Requirements 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    PlatformAdapter,
    PlatformNotSupportedError,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DELTA_PCT",
    "MUTE_HOTKEY",
    "SKILL",
    "VOLUME_OPERATIONS",
    "VolumeSkill",
]


# ---------------------------------------------------------------------------
# Operation vocabulary and defaults
# ---------------------------------------------------------------------------

#: Closed set of user-facing operations accepted by the Skill. Mirrors
#: Requirement 4.3 exactly. Exposed at module scope so tests and the
#: registry property tests can introspect the set without parsing the
#: schema dict.
VOLUME_OPERATIONS: Final[tuple[str, ...]] = (
    "set",
    "increase",
    "decrease",
    "mute",
    "unmute",
)

#: Default percentage change applied to ``increase`` / ``decrease`` when
#: the LLM omits the ``level`` argument (Requirement 4.5). The value is
#: pulled out as a constant so tests can assert it without re-deriving
#: the magic number from the requirement text.
DEFAULT_DELTA_PCT: Final[int] = 10

#: pyautogui virtual-key name forwarded to
#: :meth:`PlatformAdapter.hotkey` for ``mute`` / ``unmute``. Windows
#: implements ``VK_VOLUME_MUTE`` as a toggle, so the same key handles
#: both operations.
MUTE_HOTKEY: Final[str] = "volumemute"


# JSON Schema accepted by the LLM. Kept as a plain dict so it round-trips
# unchanged through ``json.dumps``/``json.loads`` (Property 12 / CP15)
# and so :class:`MistralSchemaValidator` can inspect it without
# surprises.
#
# The ``allOf``/``if``/``then`` clause encodes the operation-specific
# "level is required for set" invariant from Requirement 4.4 inside the
# schema itself. Doing so lets the :class:`SkillRegistry` surface the
# constraint as a ``schema_violation`` (Property 2 / CP2) and trigger
# the standard LLM retry loop (Requirement 14.5) when the model
# generates a ``set`` Tool_Call without a ``level``. ``additionalProperties:
# false`` is critical for the same reason: it gates extraneous arguments
# the Skill does not understand.
_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "description": (
                "Volume operation to perform. 'set' targets an absolute "
                "level (the 'level' field is required). 'increase' and "
                "'decrease' shift the master output volume by the "
                "supplied 'level' percentage points, defaulting to 10 "
                "when omitted. 'mute' and 'unmute' toggle the system "
                "mute via the OS volume-mute key."
            ),
            "enum": list(VOLUME_OPERATIONS),
        },
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Volume percentage in [0, 100]. Required for 'set'; "
                "treated as the delta for 'increase' and 'decrease' "
                "(default 10); ignored for 'mute' and 'unmute'."
            ),
        },
    },
    "required": ["operation"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {
                "properties": {"operation": {"const": "set"}},
                "required": ["operation"],
            },
            "then": {"required": ["level"]},
        }
    ],
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="volume",
    description=(
        "Control the system master output volume. Use 'set' with a "
        "'level' percentage to target an absolute volume, 'increase' "
        "or 'decrease' (with an optional 'level' delta defaulting to "
        "10) to shift the volume relatively, or 'mute'/'unmute' to "
        "toggle the system mute."
    ),
    json_schema=_JSON_SCHEMA,
    destructive=False,
    timeout_seconds=5.0,
    platforms=("windows",),
    source="builtin",
)


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class VolumeSkill:
    """Skill exposing master output volume controls."""

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Translate ``args`` into the appropriate adapter call.

        The ``args`` dict has already been validated against
        :data:`_JSON_SCHEMA` by the :class:`SkillRegistry` (Property 2 /
        CP2), so we can assume:

        * ``"operation"`` is present and is one of
          :data:`VOLUME_OPERATIONS`;
        * if ``"operation" == "set"``, ``"level"`` is present;
        * any ``"level"`` value is an integer in ``[0, 100]``.

        The executor still defends against context-level
        misconfiguration (no platform adapter, smuggled-in non-Protocol
        objects) and against drift between the schema and the
        :data:`VOLUME_OPERATIONS` literal by returning ``internal_error``
        with a clear message.
        """
        adapter = ctx.platform_adapter
        if adapter is None:
            return SkillResult.error(
                "internal_error",
                "VolumeSkill requires ctx.platform_adapter",
            )

        # ``isinstance`` against the runtime-checkable Protocol is a
        # cheap belt-and-braces guard; the SkillContext field is typed
        # ``Any`` to avoid an import cycle, so misconfigured contexts
        # can smuggle in unrelated objects.
        if not isinstance(adapter, PlatformAdapter):
            return SkillResult.error(
                "internal_error",
                "ctx.platform_adapter does not satisfy the PlatformAdapter "
                f"protocol (got {type(adapter).__name__})",
            )

        operation = args["operation"]
        level = args.get("level")

        try:
            return await self._dispatch(adapter, operation, level)
        except PlatformNotSupportedError as exc:
            # Mirror the platform error code into the SkillResult error
            # taxonomy. The message preserves the capability and any
            # adapter-supplied detail so the Dialog_Manager can speak a
            # useful explanation back to the user.
            logger.info(
                "volume: operation %r unsupported on platform %r: %s",
                operation,
                exc.platform,
                exc.detail,
            )
            return SkillResult.error(
                PLATFORM_NOT_SUPPORTED,
                str(exc),
            )

    @staticmethod
    async def _dispatch(
        adapter: PlatformAdapter, operation: str, level: int | None
    ) -> SkillResult:
        """Map ``operation`` to the corresponding adapter call.

        Pulled out as a helper so the public :meth:`execute` keeps a
        single ``try``/``except`` for :class:`PlatformNotSupportedError`
        and stays under the per-function return-statement budget.
        """
        if operation == "set":
            # Schema guarantees ``level`` is present and in range for
            # ``set``. The defensive ``None`` check is purely for
            # paranoia against a misconfigured registry.
            if level is None:  # pragma: no cover - schema enforces
                return SkillResult.error(
                    "internal_error",
                    "VolumeSkill 'set' requires a 'level' argument",
                )
            await adapter.set_volume(level)
            return SkillResult.success(
                value={"operation": operation, "level": level},
            )

        if operation in ("increase", "decrease"):
            magnitude = level if level is not None else DEFAULT_DELTA_PCT
            delta = magnitude if operation == "increase" else -magnitude
            await adapter.adjust_volume(delta)
            return SkillResult.success(
                value={
                    "operation": operation,
                    "delta": delta,
                    "magnitude": magnitude,
                },
            )

        if operation in ("mute", "unmute"):
            # Windows exposes ``VK_VOLUME_MUTE`` as a single toggle key.
            # Both Skill operations therefore press the same hotkey; the
            # Dialog_Manager can still phrase the acknowledgement using
            # ``operation`` from the result value because the
            # user-facing intent is preserved.
            await adapter.hotkey(MUTE_HOTKEY)
            return SkillResult.success(
                value={"operation": operation, "hotkey": MUTE_HOTKEY},
            )

        # Defence in depth: a drift between :data:`VOLUME_OPERATIONS`
        # and the schema enum would surface as an unknown operation
        # here. The schema enum prevents this in practice.
        return SkillResult.error(  # pragma: no cover - schema enforces
            "internal_error",
            f"unknown volume operation {operation!r}; expected one "
            f"of {list(VOLUME_OPERATIONS)!r}",
        )


# Module-level export consumed by :meth:`SkillRegistry.discover`. Built-in
# Skills are registered explicitly during application bootstrap, but
# exposing ``SKILL`` here keeps the discovery contract uniform between
# built-in and user-supplied modules and lets tests load the module via
# the same code path.
SKILL: Skill = VolumeSkill()
