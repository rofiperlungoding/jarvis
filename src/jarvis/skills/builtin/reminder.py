"""Built-in reminder skills.

This module ships two Skills that expose the persistent
:class:`~jarvis.reminders.service.ReminderService` to the LLM_Backend
through Mistral function-calling tool definitions:

* :class:`ReminderSkill` (Requirements 6.1, 6.2) — schedules a single
  named reminder for an ISO-8601 ``trigger_at`` instant. The Skill
  forwards the request to :meth:`ReminderService.add`, which is the
  same code path used by :class:`TimerSkill` (Task 18.9), so reminders
  and timers share the same SQLite persistence layer and APScheduler
  job store. That shared persistence is what underwrites the
  cross-restart guarantee in Requirement 6.6.

* :class:`ListReminderSkill` (Requirement 6.7) — reports every pending
  reminder, alarm, and timer in ``(trigger_at, seq)`` order. The
  ordering is the public contract of
  :meth:`ReminderService.list_pending`, which is in turn what
  Property 10 / CP13 verifies; tests in this package exercise both the
  underlying service ordering and the Skill's faithful pass-through.

Why ``ctx.extras`` instead of a typed :class:`SkillContext` field?
------------------------------------------------------------------
The :class:`~jarvis.skills.base.SkillContext` data model in
``src/jarvis/skills/base.py`` does not (yet) declare a typed
``reminder_service`` field; it offers ``providers`` (HTTP-shaped
external services) and an open-ended ``extras`` mapping for "MCP /
test-injected fakes". A reminder service is neither HTTP-shaped nor
MCP-sourced, so injecting it via ``extras`` under the well-known
:data:`REMINDER_SERVICE_EXTRAS_KEY` is the cleanest forward-compatible
choice. Task 19.x (application wiring) installs the service under that
key when assembling the context for every Tool_Call. If a future
refactor promotes ``reminder_service`` to a typed field, only this
module needs to switch lookup paths.

Error handling
--------------
Both Skills surface errors via the closed
:data:`~jarvis.skills.base.SkillErrorCode` taxonomy:

* ``internal_error`` — the reminder service is missing from the
  context (i.e., the dispatcher misconfigured the run) or
  :meth:`ReminderService.add` / :meth:`ReminderService.list_pending`
  raised an unexpected error. The dispatcher already surfaces
  unhandled exceptions as ``internal_error`` (Property 7 / CP10), but
  catching them here lets us add a Skill-specific message that helps
  the LLM recover (e.g., "I have not been wired up to reminder
  storage").
* ``schema_violation`` — the JSON Schema check on registration accepts
  ``trigger_at`` as a string with ``format: "date-time"``. The
  registry's :class:`Draft7Validator` only enforces type, *not*
  format (the registry does not register a format checker); we
  therefore parse ``trigger_at`` ourselves and return
  ``schema_violation`` when the LLM hands us a string that is not a
  parseable ISO-8601 timestamp or is missing a timezone. The
  Dialog_Manager's two-retry loop (Requirement 14.5) then asks the
  LLM to fix the argument.

Validates: Requirements 6.1, 6.2, 6.7
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Final

from jarvis.reminders.service import Reminder, ReminderService
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Key under which application wiring (Task 19.x) installs the
#: :class:`ReminderService` instance into :attr:`SkillContext.extras`.
#: Exposed as a module-level constant so tests and the eventual
#: ``app.py`` bootstrap share a single source of truth.
REMINDER_SERVICE_EXTRAS_KEY: Final[str] = "reminder_service"


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------


# JSON Schema for :class:`ReminderSkill`.
#
# * ``label`` is a non-empty string (Requirement 6.1 — every reminder
#   carries a human-readable label).
# * ``trigger_at`` is a string with ``format: "date-time"``. Mistral's
#   subset validator allows ``date-time`` (see
#   :data:`jarvis.llm.mistral_schema.MistralSchemaValidator.ALLOWED_FORMATS`),
#   and the LLM hint helps the model emit a parseable ISO-8601 instant.
# * ``additionalProperties: false`` keeps the LLM honest: any extra key
#   the model invents is rejected up front by the registry's argument
#   validator with ``schema_violation`` rather than silently ignored.
_REMINDER_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable description of the reminder.",
        },
        "trigger_at": {
            "type": "string",
            "format": "date-time",
            "description": (
                "ISO-8601 timestamp (with timezone) at which the reminder "
                "should fire."
            ),
        },
    },
    "required": ["label", "trigger_at"],
    "additionalProperties": False,
}

_LIST_REMINDER_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_service(ctx: SkillContext) -> ReminderService | None:
    """Return the :class:`ReminderService` injected via ``ctx.extras``.

    Returns ``None`` when the key is missing or maps to a value that
    does not look like a :class:`ReminderService` (e.g., a stale
    placeholder). The duck-typed shape check — presence of ``add`` and
    ``list_pending`` coroutines — keeps the Skill testable with
    lightweight fakes that mimic only the surface area we touch.
    """

    candidate = ctx.extras.get(REMINDER_SERVICE_EXTRAS_KEY)
    if candidate is None:
        return None
    # Real production callers always pass a :class:`ReminderService`.
    # Tests pass minimal fakes that expose ``add`` / ``list_pending``;
    # we accept both via a structural probe so the Skill stays
    # easy to drive in isolation.
    if not (hasattr(candidate, "add") and hasattr(candidate, "list_pending")):
        return None
    return candidate  # type: ignore[no-any-return]


def _parse_trigger_at(raw: Any) -> datetime | None:
    """Parse an ISO-8601 ``trigger_at`` value to an aware UTC datetime.

    Returns ``None`` when ``raw`` is not a string, is not a parseable
    ISO-8601 timestamp, or is missing a timezone offset. The "must be
    timezone-aware" guard mirrors :meth:`ReminderService.add`'s own
    validation; catching it here lets us surface a single
    ``schema_violation`` to the LLM regardless of which layer detected
    the problem.

    Notes on parsing
    ----------------
    Python 3.11's :meth:`datetime.fromisoformat` accepts the full
    ISO-8601 grammar including the trailing ``"Z"`` shorthand for UTC,
    so we do not need an external dependency. We deliberately do not
    coerce naive datetimes to UTC: silently re-interpreting a naive
    "5 pm" as "5 pm UTC" would mis-fire reminders for any user not
    living on the prime meridian. Returning ``None`` and asking the
    LLM to retry is the correct behaviour.
    """

    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed


def _serialize_reminder(reminder: Reminder) -> dict[str, Any]:
    """Render a :class:`Reminder` as a JSON-friendly dict.

    The Skill returns the reminder list inside :attr:`SkillResult.value`,
    which the dispatcher subsequently embeds in a tool-result message
    for the LLM. JSON serialisation must therefore be lossless and
    timezone-explicit — :meth:`datetime.isoformat` delivers both.
    """

    return {
        "reminder_id": reminder.reminder_id,
        "kind": reminder.kind,
        "label": reminder.label,
        "trigger_at": reminder.trigger_at.isoformat(),
        "duration_seconds": reminder.duration_seconds,
        "seq": reminder.seq,
        "created_at": reminder.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# ReminderSkill
# ---------------------------------------------------------------------------


class ReminderSkill:
    """Schedule a single reminder via :meth:`ReminderService.add`.

    See module docstring for the design rationale; the manifest below
    is the single Mistral-facing contract.
    """

    manifest: SkillManifest = SkillManifest(
        name="ReminderSkill",
        description=(
            "Schedule a reminder that fires once at the specified "
            "ISO-8601 timestamp. Persists across application restarts."
        ),
        json_schema=_REMINDER_JSON_SCHEMA,
        destructive=False,
        # 30s is plenty for an SQLite insert + APScheduler add_job.
        timeout_seconds=30.0,
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Persist the reminder and return its identifier.

        ``args`` has already been validated against
        :attr:`manifest.json_schema` by the
        :class:`~jarvis.skills.registry.SkillRegistry`, so structural
        checks here only need to defend against the format-keyword gap
        described in the module docstring.
        """

        service = _resolve_service(ctx)
        if service is None:
            return SkillResult.error(
                "internal_error",
                (
                    "ReminderSkill requires a ReminderService under "
                    f"ctx.extras[{REMINDER_SERVICE_EXTRAS_KEY!r}]; none was "
                    "supplied. The dispatcher's run-context is "
                    "misconfigured."
                ),
            )

        label = args.get("label")
        if not isinstance(label, str) or not label:
            # Defence-in-depth: the JSON Schema enforces non-empty
            # strings, but the registry's validator runs *before*
            # us. Returning ``schema_violation`` here keeps the
            # invariant that empty labels never reach the service.
            return SkillResult.error(
                "schema_violation",
                "label must be a non-empty string",
            )

        trigger_at = _parse_trigger_at(args.get("trigger_at"))
        if trigger_at is None:
            return SkillResult.error(
                "schema_violation",
                (
                    "trigger_at must be a timezone-aware ISO-8601 "
                    "timestamp (e.g., '2025-01-01T12:00:00+00:00')."
                ),
            )

        try:
            reminder = await service.add(label, trigger_at)
        except (ValueError, TypeError) as exc:
            # ReminderService raises ValueError for naive datetimes
            # and empty labels — re-surface as schema_violation so the
            # LLM can retry with a corrected argument.
            return SkillResult.error("schema_violation", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            # The dispatcher would convert this to ``internal_error``
            # anyway, but we make the failure mode explicit here so
            # the audit log carries a clearer message.
            logger.exception("ReminderService.add raised unexpectedly")
            return SkillResult.error(
                "internal_error",
                f"failed to persist reminder: {type(exc).__name__}: {exc}",
            )

        return SkillResult.success(_serialize_reminder(reminder))


# ---------------------------------------------------------------------------
# ListReminderSkill
# ---------------------------------------------------------------------------


class ListReminderSkill:
    """Return all pending reminders, alarms, and timers.

    Mirrors :meth:`ReminderService.list_pending` directly; the only
    transformation is converting :class:`Reminder` instances into JSON-
    friendly dicts. Ordering — ``(trigger_at, seq)`` ascending — is
    preserved because the underlying SQL query already produces it
    (Property 10 / CP13).
    """

    manifest: SkillManifest = SkillManifest(
        name="ListReminderSkill",
        description=(
            "List all pending reminders, alarms, and timers in "
            "ascending fire-time order."
        ),
        json_schema=_LIST_REMINDER_JSON_SCHEMA,
        destructive=False,
        timeout_seconds=30.0,
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        service = _resolve_service(ctx)
        if service is None:
            return SkillResult.error(
                "internal_error",
                (
                    "ListReminderSkill requires a ReminderService under "
                    f"ctx.extras[{REMINDER_SERVICE_EXTRAS_KEY!r}]; none was "
                    "supplied. The dispatcher's run-context is "
                    "misconfigured."
                ),
            )

        try:
            reminders = await service.list_pending()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ReminderService.list_pending raised unexpectedly")
            return SkillResult.error(
                "internal_error",
                f"failed to list reminders: {type(exc).__name__}: {exc}",
            )

        return SkillResult.success(
            {"reminders": [_serialize_reminder(r) for r in reminders]}
        )


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------


#: Built-in skills exported by this module. The application bootstrap
#: (Task 19.x) iterates over :data:`SKILLS` and calls
#: :meth:`SkillRegistry.register` for each entry. The plugin
#: discovery loop in :class:`SkillRegistry` looks for a top-level
#: ``SKILL`` attribute (singular) and is *not* used for built-ins; we
#: therefore use the more idiomatic plural ``SKILLS`` list here so a
#: single module file can ship related Skills together (Requirements
#: 6.1, 6.7 are naturally co-located on the reminder service).
SKILLS: Final[list[Skill]] = [ReminderSkill(), ListReminderSkill()]


__all__ = [
    "REMINDER_SERVICE_EXTRAS_KEY",
    "SKILLS",
    "ListReminderSkill",
    "ReminderSkill",
]
