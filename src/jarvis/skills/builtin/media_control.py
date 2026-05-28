"""Built-in :class:`MediaControlSkill` — transport controls via media keys.

Wraps :meth:`jarvis.automation.platform.PlatformAdapter.media_key` behind a
Skill that the LLM can invoke through Mistral function calling. The
acceptance criteria fix the user-facing action vocabulary (Requirement
4.1) at five values — ``play``, ``pause``, ``next``, ``previous``,
``stop`` — and require the corresponding Windows media-key event be sent
to the operating system (Requirement 4.2).

The Skill does the small amount of glue that lives between the LLM's
``action`` argument and the lower-level ``MediaKey`` literal accepted by
the adapter. In particular:

* ``play`` and ``pause`` both map to the platform's ``"play_pause"``
  key. Windows exposes a single ``VK_MEDIA_PLAY_PAUSE`` virtual key —
  there are no separate "play" or "pause" scancodes — so collapsing the
  two skill actions onto the same media key matches the OS behaviour.
  The Skill still records *which* user-facing action was requested in
  the success payload so the Dialog_Manager can phrase its
  acknowledgement naturally ("playing", "paused", …).
* ``previous`` is renamed to the adapter's ``"prev"`` literal. We keep
  the longer ``previous`` form on the user-facing side because that is
  what Requirement 4.1 specifies and because the LLM's argument
  generator favours unambiguous English.

Failure mapping
---------------

* :class:`PlatformNotSupportedError` from the adapter is converted to
  ``SkillResult.error("platform_not_supported", ...)`` — this is the
  contract the Skills layer has with adapters that cannot service a
  capability on the current OS (see ``design.md §Automation_Service``
  and the error-taxonomy entry in :mod:`jarvis.skills.base`).
* Any other adapter exception bubbles up to the registry, which
  converts it to ``SkillResult.error("internal_error", ...)`` — the
  Skill itself does not pre-empt that handling because doing so would
  swallow tracebacks that are useful for forensics.

The Skill is non-destructive: pressing a media key has no permanent
side effect, so :attr:`SkillManifest.destructive` is ``False`` and the
Authorization_Policy will not request user confirmation before
dispatch (Requirement 16.1).

Validates: Requirements 4.1, 4.2
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    MediaKey,
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
    "ACTION_TO_MEDIA_KEY",
    "MEDIA_CONTROL_ACTIONS",
    "SKILL",
    "MediaControlSkill",
]


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

#: Closed set of user-facing actions accepted by the Skill. Mirrors
#: Requirement 4.1 exactly. Exposed at module scope so tests and the
#: registry property tests can introspect the set without importing the
#: schema dict.
MEDIA_CONTROL_ACTIONS: Final[tuple[str, ...]] = (
    "play",
    "pause",
    "next",
    "previous",
    "stop",
)

#: Mapping from the user-facing action vocabulary to the
#: :data:`jarvis.automation.platform.MediaKey` literal accepted by the
#: adapter. ``play`` / ``pause`` both target the OS toggle key, and
#: ``previous`` is renamed to ``prev`` to match the adapter contract.
#: The mapping is total — every value in :data:`MEDIA_CONTROL_ACTIONS`
#: has an entry — and the Skill relies on that totality so the
#: ``KeyError`` branch below is purely defensive.
ACTION_TO_MEDIA_KEY: Final[dict[str, MediaKey]] = {
    "play": "play_pause",
    "pause": "play_pause",
    "next": "next",
    "previous": "prev",
    "stop": "stop",
}


# JSON Schema accepted by the LLM. Kept as a plain dict so it round-trips
# unchanged through ``json.dumps``/``json.loads`` (Property 12 / CP15) and
# so :class:`MistralSchemaValidator` can inspect it without surprises.
# ``additionalProperties: false`` is critical: it is the gate that lets
# the registry return ``schema_violation`` for a Tool_Call carrying
# arguments the Skill does not understand.
_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": (
                "Transport control to apply. 'play' and 'pause' both send "
                "the play/pause toggle to the operating system because "
                "Windows exposes a single key for the two states."
            ),
            "enum": list(MEDIA_CONTROL_ACTIONS),
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="media_control",
    description=(
        "Press a system media key to control the active media player. "
        "Use 'play' or 'pause' to toggle playback, 'next'/'previous' to "
        "skip tracks, and 'stop' to halt playback."
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


class MediaControlSkill:
    """Skill exposing transport controls (play/pause/next/previous/stop)."""

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Translate ``args["action"]`` into a media-key event.

        The ``args`` dict has already been validated against
        :data:`_JSON_SCHEMA` by the :class:`SkillRegistry` (Property 2 /
        CP2), so we can assume ``"action"`` is present and is one of the
        five accepted values. We still defend against drift between the
        schema and :data:`ACTION_TO_MEDIA_KEY` by returning
        ``internal_error`` if the lookup misses — that situation
        indicates a programmer error in this module, not an
        ill-formed Tool_Call.
        """
        adapter = ctx.platform_adapter
        if adapter is None:
            # The registry will not normally invoke us without a
            # platform adapter, but the dataclass default allows it for
            # test-injection convenience. We surface it as
            # ``internal_error`` because the Skill cannot fulfill its
            # contract without an OS-level side effect.
            return SkillResult.error(
                "internal_error",
                "MediaControlSkill requires ctx.platform_adapter",
            )

        # ``isinstance`` against the runtime-checkable Protocol is a
        # cheap belt-and-braces guard; the SkillContext field is typed
        # ``Any`` to avoid an import cycle, so misconfigured contexts can
        # smuggle in unrelated objects.
        if not isinstance(adapter, PlatformAdapter):
            return SkillResult.error(
                "internal_error",
                "ctx.platform_adapter does not satisfy the PlatformAdapter "
                f"protocol (got {type(adapter).__name__})",
            )

        action = args["action"]
        try:
            media_key = ACTION_TO_MEDIA_KEY[action]
        except KeyError:  # pragma: no cover - schema enum prevents this
            return SkillResult.error(
                "internal_error",
                f"unknown media-control action {action!r}; expected one "
                f"of {list(MEDIA_CONTROL_ACTIONS)!r}",
            )

        try:
            await adapter.media_key(media_key)
        except PlatformNotSupportedError as exc:
            # Mirror the platform error code into the SkillResult error
            # taxonomy. The message preserves the capability and any
            # adapter-supplied detail so the Dialog_Manager can speak a
            # useful explanation back to the user.
            logger.info(
                "media_control: media_key %r unsupported on platform %r: %s",
                media_key,
                exc.platform,
                exc.detail,
            )
            return SkillResult.error(
                PLATFORM_NOT_SUPPORTED,
                str(exc),
            )

        return SkillResult.success(
            value={"action": action, "media_key": media_key},
        )


# Module-level export consumed by :meth:`SkillRegistry.discover`. Built-in
# Skills are registered explicitly during application bootstrap, but
# exposing ``SKILL`` here keeps the discovery contract uniform between
# built-in and user-supplied modules and lets tests load the module via
# the same code path.
SKILL: Skill = MediaControlSkill()
