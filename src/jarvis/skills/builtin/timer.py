"""Built-in :class:`TimerSkill` — start a countdown timer.

The Skill_Registry exposes ``TimerSkill`` to the LLM so the user can ask
JARVIS to start a countdown ("set a five minute timer"). When the model
emits the corresponding Tool_Call, this Skill validates the duration,
delegates persistence to :meth:`jarvis.reminders.service.ReminderService.add_timer`,
and returns a structured payload describing the freshly-scheduled
reminder so the Dialog_Manager can read the confirmation back to the
user.

Argument schema
---------------

``duration_seconds`` is the only required field, restricted to a strictly
positive integer (Requirement 6.3). ``label`` is optional; when omitted
the persistence layer stores the empty string so the metadata table's
``NOT NULL`` constraint stays satisfied without a sentinel
(:meth:`ReminderService.add_timer` handles the normalisation).

Context contract
----------------

The Skill expects ``ctx.extras["reminder_service"]`` to hold a started
:class:`ReminderService`. The application bootstrap
(``src/jarvis/app.py``, task 19.1) injects the service there during
startup. If the entry is missing — for example, in unit tests or when the
service crashed at boot — the executor returns ``internal_error`` rather
than ``not_supported``: a missing dependency is a wiring bug, not a
platform limitation.

Error mapping
-------------

* ``schema_violation`` — caught by the Skill_Registry before ``execute``
  runs; this module never returns it directly.
* ``internal_error`` — ``ctx.extras["reminder_service"]`` is missing or
  :meth:`add_timer` raised an unexpected exception. The Skill_Registry
  back-stops generic exceptions, but we trap them here to attach a
  Skill-specific message and avoid the registry's traceback overhead.

Validates: Requirements 6.3, 6.4
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = ["SKILL", "TimerSkill"]


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


# The schema mirrors Requirement 6.3: ``duration_seconds`` is an integer
# field strictly greater than zero; ``label`` is an optional string. We
# declare ``additionalProperties: false`` so a hallucinated extra field
# is caught by ``Draft7Validator`` before reaching the executor — the
# Mistral subset accepts the keyword unchanged.
_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "TimerSkillArguments",
    "description": (
        "Arguments for starting a countdown timer. ``duration_seconds`` is "
        "the countdown length in seconds (must be greater than zero). "
        "``label`` is an optional human-readable description (e.g. "
        '"pasta", "tea steeping").'
    ),
    "properties": {
        "duration_seconds": {
            "type": "integer",
            "description": (
                "Countdown length in seconds. Must be strictly greater "
                "than zero (Requirement 6.3)."
            ),
            "minimum": 1,
            "exclusiveMinimum": 0,
        },
        "label": {
            "type": "string",
            "description": (
                "Optional human-readable label spoken / shown when the "
                "timer fires. Omit if the timer is unnamed."
            ),
            "minLength": 1,
        },
    },
    "required": ["duration_seconds"],
    "additionalProperties": False,
}


_REMINDER_SERVICE_KEY: Final[str] = "reminder_service"


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class TimerSkill:
    """Start a countdown timer via :class:`ReminderService`.

    The Skill is a thin adapter: argument validation is owned by the
    Skill_Registry through the JSON Schema (Property 2 / CP2), so the
    executor only needs to (a) resolve the :class:`ReminderService` from
    the :class:`SkillContext`, (b) forward the call to
    :meth:`ReminderService.add_timer`, and (c) translate the resulting
    :class:`Reminder` into a JSON-serialisable success payload.
    """

    manifest: SkillManifest = SkillManifest(
        name="TimerSkill",
        description=(
            "Start a countdown timer for a given number of seconds. "
            "When the countdown reaches zero, JARVIS surfaces a "
            "Windows toast notification and (when the user is engaged) "
            "speaks the optional label aloud."
        ),
        json_schema=_SCHEMA,
        # Setting a timer is non-destructive: it does not exfiltrate
        # data, mutate user files, or run code. The Authorization_Policy
        # therefore lets it through without confirmation (Requirement
        # 16.1 lists the destructive Skills explicitly; TimerSkill is
        # intentionally absent).
        destructive=False,
        # ReminderService persists to a local SQLite file on every
        # supported platform; the timer Skill itself has no Windows-
        # specific dependencies, so we expose it everywhere the Skill
        # can plausibly run. The platform_not_supported gate fires only
        # if the host platform tag is missing from this tuple.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Persist a new countdown timer and report back its identity.

        Pre-conditions enforced by the Skill_Registry's JSON-Schema gate:

        * ``args["duration_seconds"]`` is an ``int`` > 0;
        * ``args["label"]``, if present, is a non-empty string.

        The executor still defends against semantic anomalies (a missing
        :class:`ReminderService` in ``ctx.extras``, an unexpected
        exception inside ``add_timer``) so the Dialog_Manager always
        sees a structured :class:`SkillResult` rather than an exception.
        """

        duration_seconds = args["duration_seconds"]
        label = args.get("label")

        reminder_service = ctx.extras.get(_REMINDER_SERVICE_KEY)
        if reminder_service is None:
            # ``not_supported`` would suggest the host platform cannot
            # run timers; the more accurate framing here is that the
            # application bootstrap did not wire the service into the
            # Skill context. Surface as ``internal_error`` so the
            # Dialog_Manager apologises rather than steering the user
            # toward an irrelevant troubleshooting path.
            logger.error(
                "TimerSkill invoked without a ReminderService on "
                "ctx.extras[%r]; check application bootstrap",
                _REMINDER_SERVICE_KEY,
            )
            return SkillResult.error(
                "internal_error",
                "reminder service is unavailable",
            )

        try:
            reminder = await reminder_service.add_timer(
                duration_seconds=duration_seconds,
                label=label,
            )
        except (TypeError, ValueError) as exc:
            # The schema gate already rejects the obvious offenders, but
            # ReminderService.add_timer enforces its own invariants
            # (positive int, str-or-None label) and raises ``ValueError``
            # / ``TypeError`` if a future schema change loosens the
            # contract. Map them onto ``schema_violation`` so the LLM
            # gets a chance to retry rather than treating the failure
            # as a runtime fault.
            logger.warning(
                "ReminderService.add_timer rejected duration=%r label=%r: %s",
                duration_seconds,
                label,
                exc,
            )
            return SkillResult.error(
                "schema_violation",
                f"invalid timer arguments: {exc}",
            )

        # Compose a JSON-serialisable success payload mirroring the
        # ReminderService.add_timer return shape. ``trigger_at`` is
        # already a timezone-aware UTC datetime; we serialise via
        # ``isoformat`` so the value survives ``json.dumps`` /
        # ``json.loads`` round-trips when the Dialog_Manager forwards
        # the value back to the model.
        return SkillResult.success(
            value={
                "reminder_id": reminder.reminder_id,
                "kind": reminder.kind,
                "label": reminder.label,
                "duration_seconds": reminder.duration_seconds,
                "trigger_at": reminder.trigger_at.isoformat(),
                "seq": reminder.seq,
            }
        )


#: The plugin discovery hook expected by :class:`SkillRegistry.discover`.
#: Built-in Skills are loaded as plain modules on the registry's plugin
#: path; the registry imports the module and binds whatever object is
#: assigned to ``SKILL`` on the module's top level.
SKILL: Skill = TimerSkill()
