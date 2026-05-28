"""Built-in ``CalendarSkill``.

Implements the ``CalendarSkill`` referenced from ``design.md §Built-in
Skills`` and Requirements 7.5, 7.6, and 7.7. The Skill is a thin
adapter between the LLM_Backend's tool-calling interface and the
:class:`~jarvis.automation.providers.calendar.CalendarClient` provider
that the application bootstrap wires through
:attr:`SkillContext.providers` under the ``"calendar"`` key.

Three operations
----------------

The Skill multiplexes three distinct calendar operations behind a single
``operation`` discriminator so the LLM only has to discover one tool:

* ``list_today`` (Requirement 7.5) — returns events overlapping the
  user's current local day. No additional arguments.
* ``list_range`` (Requirement 7.5) — returns events overlapping the
  inclusive ``[start, end]`` window. Both bounds are required ISO-8601
  timestamps.
* ``create_event`` (Requirement 7.6) — creates a new event with a
  ``title`` and ``start``/``end`` bounds. This operation is registered
  as a destructive operation in
  :class:`~jarvis.config.schema.AuthorizationConfig.destructive_operations`,
  so the :class:`AuthorizationPolicy` requires user confirmation before
  the registry dispatches it (Requirement 16.1, 16.2).

The manifest itself declares ``destructive=False`` because two of the
three operations are read-only. Operation-level destructive
classification is the right tool here: the policy reads the
``operation`` discriminator out of the Tool_Call arguments and gates
``create_event`` (and only ``create_event``) on user confirmation.

JSON Schema shape
-----------------

The argument schema uses ``allOf`` + ``if/then`` blocks (draft-07) to
require the right per-operation fields:

* When ``operation == "list_range"`` we require ``start`` and ``end``.
* When ``operation == "create_event"`` we require ``title``, ``start``,
  and ``end``.
* When ``operation == "list_today"`` no additional fields are required.

This keeps the Skill discoverable as a single tool while still letting
the registry's :class:`Draft7Validator` reject obviously malformed
calls (e.g., ``create_event`` without a title) at dispatch time —
satisfying Property 2 / CP2's "schema_violation iff is_valid is false"
contract.

Error translation
-----------------

The Skill maps the structured outcomes raised by
:class:`CalendarClient` into the closed
:data:`~jarvis.skills.base.SkillErrorCode` taxonomy:

* :class:`ProviderError` ``"missing_credentials"`` →
  :meth:`SkillResult.error` ``"missing_credentials"`` (Requirement 5.6
  mirrored for the calendar provider).
* :class:`ProviderError` ``"provider_unavailable"`` →
  :meth:`SkillResult.error` ``"provider_unavailable"`` (Requirement
  7.7).
* :class:`NetworkPolicyViolation` → :meth:`SkillResult.error`
  ``"access_denied"``. The provider has already recorded the
  ``policy_violation`` audit row before raising (Requirement 13.6) so we
  do not double-write through the registry's :class:`PolicyViolation`
  adapter.
* :class:`ValueError` from the client (naive datetime, invalid window) →
  ``"schema_violation"`` so the Dialog_Manager's two-retry loop
  (Requirement 14.5) gives the LLM a chance to correct the call.

Plugin discovery
----------------

The :class:`SkillRegistry` looks for a top-level ``SKILL`` attribute
when loading a plugin file (see ``registry._load_plugin_file``). The
module exposes the singleton instance under that exact name so the
same module can be discovered both as a built-in (registered
programmatically by the bootstrap) and, in tests, as a generic plugin.

Validates: Requirements 7.5, 7.6, 7.7, 16.1, 16.2
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = ["CALENDAR_PROVIDER_KEY", "SCHEMA", "SKILL", "CalendarSkill"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Key under which the application bootstrap installs the
#: :class:`CalendarClient` instance into :attr:`SkillContext.providers`.
#: Documented in ``src/jarvis/skills/base.py`` alongside the other
#: provider keys (``"weather"``, ``"news"``, ``"email"``,
#: ``"web_search"``).
CALENDAR_PROVIDER_KEY: Final[str] = "calendar"

#: Skill name surfaced to the LLM. Pinned as a constant because the
#: ``[authorization].destructive_operations`` config refers to the Skill
#: by this exact name; renaming would silently disable the
#: ``create_event`` confirmation gate.
_SKILL_NAME: Final[str] = "CalendarSkill"

#: Closed set of operations the Skill exposes. Kept as a tuple so the
#: ``enum`` keyword in the JSON Schema and the runtime branching stay
#: in sync.
_OPERATIONS: Final[tuple[str, ...]] = ("list_today", "list_range", "create_event")


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def _build_schema() -> dict[str, Any]:
    """Build the Mistral-compatible argument schema for the Skill.

    The schema uses ``allOf`` + ``if/then`` blocks rather than ``oneOf``
    over the operation discriminator so the LLM sees a single,
    well-typed ``operation`` field with a closed enum. ``oneOf`` over
    object branches would mix scalar / object branches the moment a
    branch's required-set differed, which the
    :class:`MistralSchemaValidator` would (rightly) reject. The
    conditional approach satisfies the validator's subset rules and
    composes cleanly with :class:`Draft7Validator`'s ``is_valid``
    semantics.

    The base ``properties`` block declares every possible field as
    optional; the ``allOf`` blocks then promote specific fields to
    required for the matching operations.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": (
                    "Calendar operation to perform: 'list_today' returns "
                    "events on the current local day, 'list_range' returns "
                    "events between two timestamps, and 'create_event' "
                    "creates a new event (requires user confirmation)."
                ),
            },
            "start": {
                "type": "string",
                "format": "date-time",
                "description": (
                    "ISO-8601 timestamp (with timezone) for the start of "
                    "the range or the new event."
                ),
            },
            "end": {
                "type": "string",
                "format": "date-time",
                "description": (
                    "ISO-8601 timestamp (with timezone) for the end of "
                    "the range or the new event."
                ),
            },
            "title": {
                "type": "string",
                "minLength": 1,
                "description": "Title / summary of the new event.",
            },
        },
        "required": ["operation"],
        "allOf": [
            {
                "if": {
                    "properties": {"operation": {"const": "list_range"}},
                    "required": ["operation"],
                },
                "then": {
                    "required": ["start", "end"],
                },
            },
            {
                "if": {
                    "properties": {"operation": {"const": "create_event"}},
                    "required": ["operation"],
                },
                "then": {
                    "required": ["title", "start", "end"],
                },
            },
        ],
    }


SCHEMA: Final[dict[str, Any]] = _build_schema()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso_aware(raw: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into a timezone-aware :class:`datetime`.

    Returns ``None`` when ``raw`` is not a string, fails to parse, or
    is missing a timezone offset. The "must be timezone-aware" guard
    mirrors the underlying :class:`CalendarClient` validation; catching
    it here lets us surface a single ``schema_violation`` regardless
    of which layer detected the issue.
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


def _translate_provider_exception(exc: BaseException) -> SkillResult:
    """Map a provider-raised exception onto the documented error code.

    Centralised translation so the per-operation branches do not
    duplicate the per-exception fan-out. The mapping mirrors the rules
    documented in the module docstring.
    """

    if isinstance(exc, ProviderError):
        # ProviderError carries one of {"missing_credentials",
        # "provider_unavailable"}. Both map 1:1 onto the SkillResult
        # error taxonomy.
        return SkillResult.error(exc.error_code, str(exc))
    if isinstance(exc, NetworkPolicyViolation):
        return SkillResult.error(
            "access_denied",
            f"calendar host blocked by network allowlist: {exc}",
        )
    if isinstance(exc, httpx.TimeoutException):
        message = (
            f"calendar request timed out: {exc}"
            if str(exc)
            else "calendar request timed out"
        )
        return SkillResult.error("timeout", message)
    # ValueError / TypeError fall through here. The provider client
    # uses :class:`ValueError` for shape problems (naive datetimes,
    # ``end < start``) the JSON Schema cannot fully express.
    return SkillResult.error(
        "schema_violation",
        f"calendar provider rejected arguments: {exc}",
    )


# ---------------------------------------------------------------------------
# CalendarSkill
# ---------------------------------------------------------------------------


class CalendarSkill:
    """Skill that proxies to the configured :class:`CalendarClient`.

    Stateless: a single instance is reused across invocations, with
    each ``execute`` call receiving the per-call :class:`SkillContext`
    produced by the :class:`SkillRegistry`. The provider lookup is
    deferred to :meth:`execute` rather than resolved at construction
    so the same instance can be registered before the providers are
    fully wired (the discovery path in :meth:`SkillRegistry.discover`
    runs at startup, before the run-loop in :func:`jarvis.app.main`
    populates every :class:`SkillContext`).
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name=_SKILL_NAME,
        description=(
            "Read or modify the user's calendar. Supports listing today's "
            "events, listing events in a range, and creating new events. "
            "Creating events requires explicit user confirmation."
        ),
        json_schema=SCHEMA,
        # Manifest-level destructive=False because two of three operations
        # are read-only. The :class:`AuthorizationPolicy` consults
        # ``[authorization].destructive_operations`` for the per-operation
        # gate (CalendarSkill.operation == "create_event" → destructive).
        destructive=False,
        timeout_seconds=30.0,
        # Calendar API access is OS-agnostic; declare every supported
        # platform so Requirement 15.4's gating does not block the Skill
        # on macOS / Linux builds.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Run the requested calendar operation.

        ``args`` has already been validated against
        :attr:`manifest.json_schema` by the
        :class:`~jarvis.skills.registry.SkillRegistry`, so structural
        checks here only need to defend against semantic errors the
        schema cannot express (e.g., naive ISO timestamps, ``end <=
        start``) and against a misconfigured run-context that did not
        wire a calendar provider.
        """

        operation = args.get("operation")
        if operation not in _OPERATIONS:
            # Defence-in-depth: the JSON Schema's ``enum`` enforces
            # this, but the registry's validator runs *before* us. A
            # smuggled-in op should never reach the dispatch branches.
            return SkillResult.error(
                "schema_violation",
                f"operation must be one of {list(_OPERATIONS)!r}",
            )

        # Provider resolution. The Skill cannot invent a client, so a
        # missing provider mapping surfaces as
        # ``provider_unavailable`` — the Dialog_Manager will tell the
        # user "the calendar provider isn't configured" rather than
        # asking them to repeat the request.
        client = ctx.providers.get(CALENDAR_PROVIDER_KEY) if ctx.providers else None
        if client is None:
            logger.warning(
                "CalendarSkill invoked without a %r provider in context",
                CALENDAR_PROVIDER_KEY,
            )
            return SkillResult.error(
                "provider_unavailable",
                "no calendar provider is configured",
            )

        if operation == "list_today":
            return await self._list_today(client)
        if operation == "list_range":
            return await self._list_range(client, args)
        # ``create_event`` is the only remaining branch; the
        # :class:`AuthorizationPolicy` has already obtained the user's
        # confirmation before the registry dispatched us here
        # (Requirement 16.2). The Skill does not re-prompt.
        return await self._create_event(client, args)

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    @staticmethod
    async def _list_today(client: Any) -> SkillResult:
        try:
            events = await client.list_today()
        except (
            ProviderError,
            NetworkPolicyViolation,
            httpx.TimeoutException,
            ValueError,
            TypeError,
        ) as exc:
            return _translate_provider_exception(exc)
        return SkillResult.success(
            {
                "operation": "list_today",
                "events": list(events) if events is not None else [],
            }
        )

    @staticmethod
    async def _list_range(client: Any, args: dict[str, Any]) -> SkillResult:
        start = _parse_iso_aware(args.get("start"))
        end = _parse_iso_aware(args.get("end"))
        if start is None:
            return SkillResult.error(
                "schema_violation",
                (
                    "start must be a timezone-aware ISO-8601 timestamp "
                    "(e.g., '2025-01-01T09:00:00+00:00')."
                ),
            )
        if end is None:
            return SkillResult.error(
                "schema_violation",
                (
                    "end must be a timezone-aware ISO-8601 timestamp "
                    "(e.g., '2025-01-01T17:00:00+00:00')."
                ),
            )
        if end < start:
            return SkillResult.error(
                "schema_violation",
                "end must be greater than or equal to start",
            )

        try:
            events = await client.list_range(start, end)
        except (
            ProviderError,
            NetworkPolicyViolation,
            httpx.TimeoutException,
            ValueError,
            TypeError,
        ) as exc:
            return _translate_provider_exception(exc)
        return SkillResult.success(
            {
                "operation": "list_range",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "events": list(events) if events is not None else [],
            }
        )

    @staticmethod
    async def _create_event(client: Any, args: dict[str, Any]) -> SkillResult:
        raw_title = args.get("title")
        if not isinstance(raw_title, str) or not raw_title.strip():
            return SkillResult.error(
                "schema_violation",
                "title must be a non-empty string",
            )
        title = raw_title.strip()

        start = _parse_iso_aware(args.get("start"))
        end = _parse_iso_aware(args.get("end"))
        if start is None:
            return SkillResult.error(
                "schema_violation",
                (
                    "start must be a timezone-aware ISO-8601 timestamp "
                    "(e.g., '2025-01-01T09:00:00+00:00')."
                ),
            )
        if end is None:
            return SkillResult.error(
                "schema_violation",
                (
                    "end must be a timezone-aware ISO-8601 timestamp "
                    "(e.g., '2025-01-01T17:00:00+00:00')."
                ),
            )
        if end <= start:
            return SkillResult.error(
                "schema_violation",
                "end must be strictly greater than start for create_event",
            )

        try:
            event = await client.create_event(title, start, end)
        except (
            ProviderError,
            NetworkPolicyViolation,
            httpx.TimeoutException,
            ValueError,
            TypeError,
        ) as exc:
            return _translate_provider_exception(exc)
        return SkillResult.success(
            {
                "operation": "create_event",
                "event": dict(event) if isinstance(event, dict) else event,
            }
        )


# ---------------------------------------------------------------------------
# Plugin handle
# ---------------------------------------------------------------------------


#: Module-level singleton consumed by :meth:`SkillRegistry.discover`.
#: Typed as :class:`CalendarSkill` rather than the :class:`Skill`
#: Protocol because the latter declares ``manifest`` as a writable
#: variable while we expose it as a :data:`Final` class attribute; the
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: CalendarSkill = CalendarSkill()
