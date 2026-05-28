"""Weather built-in skill.

Implements :class:`WeatherSkill`, the Skill the Dialog_Manager invokes
when the LLM emits a ``WeatherSkill`` Tool_Call (Requirement 7.1). The
skill is the thin Skill-layer translation between the LLM's tool-call
arguments and the
:class:`~jarvis.automation.providers.weather.WeatherClient` configured
at application bootstrap.

Responsibilities
----------------

* Declare a Mistral-compatible JSON Schema with a single optional
  ``location`` string field. Per Requirement 7.1 the field defaults to
  the user's configured home location, which lives on the underlying
  :class:`WeatherClient`'s ``provider_config.default_location`` —
  passing ``None`` (or omitting the field entirely) lets the client
  apply that default. Including the field as optional in the schema
  matches the design's "argument schema accepts an optional 'location'
  string field" wording verbatim and lets the LLM emit either form.
* Look up the ``"weather"`` provider in :class:`SkillContext.providers`
  and call :meth:`WeatherClient.fetch`. Successful responses carry the
  current conditions and a 24-hour forecast (Requirement 7.2 — the
  client returns 8 entries from OpenWeather's 3-hour-granularity
  forecast endpoint, covering 24 hours).
* Translate the structured failure modes the client raises into the
  documented Error Taxonomy codes:

  * :class:`ProviderError` with code ``"missing_credentials"`` →
    :data:`SkillResult.error_code` ``"missing_credentials"``. Pulled
    from ``provider_config.api_key_credential`` either being unset or
    missing from the :class:`CredentialStore`. The Dialog_Manager
    follows up with the documented credential-setup flow.
  * :class:`ProviderError` with code ``"provider_unavailable"`` →
    ``"provider_unavailable"`` per Requirement 7.7 (OpenWeather
    returned a non-2xx, the body could not be parsed as JSON, or
    coordinates were absent).
  * :class:`NetworkPolicyViolation` → ``"access_denied"`` per
    Requirement 13.6 (the configured ``network_destination_allowlist``
    refused ``api.openweathermap.org``). The provider client has
    already recorded a ``policy_violation`` audit row before raising,
    so we deliberately do not record another.
  * :class:`httpx.TimeoutException` → ``"timeout"``. Handled here even
    though the :class:`WeatherClient` already maps timeouts onto
    :class:`ProviderError("provider_unavailable")` — defending the
    Skill layer against a future client refactor that lets the raw
    :mod:`httpx` exception leak through keeps the user-facing error
    code stable.
* Surface a missing provider entry in :class:`SkillContext.providers`
  as ``"provider_unavailable"`` rather than ``"internal_error"`` so the
  Dialog_Manager renders "the weather provider isn't configured"
  rather than asking the user to repeat the request.

The success payload mirrors the shape :class:`WeatherClient.fetch`
returns — ``{"location": str, "current": dict, "forecast": list}`` —
and is forwarded to the LLM verbatim so it can summarise the conditions
and forecast in its spoken response.

Validates: Requirements 7.1, 7.2, 7.7
"""

from __future__ import annotations

import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = ["SCHEMA", "SKILL", "WeatherSkill"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider key used to look the weather client up out of
#: :class:`SkillContext.providers`. Matches the documented mapping in
#: ``src/jarvis/skills/base.py`` ("``"weather"``") and the bootstrap
#: wiring in ``src/jarvis/app.py``.
_PROVIDER_KEY: Final[str] = "weather"

#: Skill name surfaced to the LLM. Pinned as a constant because
#: Requirement 7.1 anchors the wording ("WeatherSkill") and changing
#: it would silently break configured trusted-action allowlists.
_SKILL_NAME: Final[str] = "WeatherSkill"

#: Description handed to the LLM as ``function.description``. Concise,
#: action-oriented, and references the optional ``location`` field so
#: the model knows it can omit it for "what's the weather?".
_SKILL_DESCRIPTION: Final[str] = (
    "Fetch current weather conditions and a 24-hour forecast for a "
    "location. If 'location' is omitted, the user's configured home "
    "location is used."
)

#: Maximum length of a free-form ``location`` string. OpenWeather's
#: ``q`` parameter does not advertise an explicit limit; 200 characters
#: is generous enough for "City, Region, CountryCode"-style queries
#: while keeping the per-turn payload bounded against a hostile LLM.
_MAX_LOCATION_LENGTH: Final[int] = 200


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


# JSON Schema for the LLM-facing tool arguments. ``location`` is
# optional (``required`` is empty) and constrained to a non-empty
# string when supplied. ``additionalProperties: false`` keeps the LLM
# from smuggling arbitrary fields through this Skill — future
# extensions (e.g., ``units``) will live in their own schema additions.
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "Weather",
    "description": (
        "Retrieve current weather conditions and a 24-hour forecast "
        "for an optional 'location'. When omitted, the configured "
        "home location is used."
    ),
    "properties": {
        "location": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_LOCATION_LENGTH,
            "description": (
                "Free-form location (e.g. 'London,GB' or "
                "'San Francisco'). Defaults to the configured home "
                "location when omitted."
            ),
        },
    },
    "required": [],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# WeatherSkill
# ---------------------------------------------------------------------------


class WeatherSkill:
    """Skill that proxies to a configured :class:`WeatherClient`.

    Stateless: the same instance can be safely registered once and used
    for every Tool_Call. The provider client is resolved from the
    :class:`SkillContext` on every call so a credential rotation or
    config reload (re-wiring ``ctx.providers["weather"]``) takes effect
    on the next request without re-registering the Skill.
    """

    manifest: SkillManifest = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=SCHEMA,
        destructive=False,
        # Weather lookup is read-only and platform-agnostic; declare
        # every supported platform so Requirement 15.4's gating does
        # not block the Skill on a future macOS / Linux build.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Run the configured weather provider against ``args["location"]``.

        Argument validation here is intentionally a thin layer on top of
        the registry's JSON Schema check: the registry has already
        validated types and required-ness, so we only enforce the
        *semantic* rules (a supplied ``location`` must be non-empty
        after stripping; omitting the field entirely defers to the
        provider's configured ``default_location``).
        """
        # 1. Argument validation. Pulled into a helper so the early-exit
        #    branches do not pile up directly inside ``execute``.
        validated = self._validate_args(args)
        if isinstance(validated, SkillResult):
            return validated
        location = validated  # may be ``None`` to use the configured default

        # 2. Provider resolution. The Skill cannot invent a client, so
        #    a missing provider mapping surfaces as
        #    ``provider_unavailable`` — the Dialog_Manager will tell
        #    the user "the weather provider isn't configured" rather
        #    than asking them to repeat the request.
        client = ctx.providers.get(_PROVIDER_KEY) if ctx.providers else None
        if client is None:
            logger.warning(
                "WeatherSkill invoked without a 'weather' provider in context"
            )
            return SkillResult.error(
                "provider_unavailable",
                "no weather provider is configured",
            )

        # 3. Provider invocation + error translation.
        try:
            payload = await client.fetch(location)
        except (ProviderError, NetworkPolicyViolation, httpx.TimeoutException) as exc:
            return self._translate_provider_exception(exc)

        # 4. Success payload. The :class:`WeatherClient` already shapes
        #    the response into ``{"location", "current", "forecast"}``;
        #    we forward it verbatim so the LLM can compose its summary.
        #    A non-dict (defensive) is surfaced as
        #    ``provider_unavailable`` so the Dialog_Manager treats it
        #    consistently with other upstream response anomalies.
        if not isinstance(payload, dict):
            return SkillResult.error(
                "provider_unavailable",
                "weather provider returned an unexpected response shape",
            )
        return SkillResult.success(value=dict(payload))

    @classmethod
    def _translate_provider_exception(cls, exc: BaseException) -> SkillResult:
        """Map a provider-raised exception onto the documented error code.

        Centralised translation so :meth:`execute` does not duplicate
        per-exception branching. The mapping follows the design's Error
        Taxonomy:

        * :class:`ProviderError` → its own ``error_code`` verbatim
          (one of ``missing_credentials`` / ``provider_unavailable``).
        * :class:`NetworkPolicyViolation` → ``access_denied`` (the
          allowlist already recorded a ``policy_violation`` audit row).
        * :class:`httpx.TimeoutException` → ``timeout`` (defence in
          depth against a future client refactor that propagates the
          raw transport exception).
        """
        if isinstance(exc, ProviderError):
            return SkillResult.error(exc.error_code, str(exc))
        if isinstance(exc, NetworkPolicyViolation):
            return SkillResult.error(
                "access_denied",
                f"weather lookup blocked by network policy: {exc}",
            )
        # ``httpx.TimeoutException`` falls through here.
        message = (
            f"weather lookup timed out: {exc}"
            if str(exc)
            else "weather lookup timed out"
        )
        return SkillResult.error("timeout", message)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _validate_args(cls, args: dict[str, Any]) -> SkillResult | str | None:
        """Validate ``args`` and return the resolved location.

        Returns one of:

        * a :class:`SkillResult` when validation fails (caller surfaces
          it directly);
        * a ``str`` — the cleaned, non-empty location to forward;
        * ``None`` — the field was omitted entirely, in which case the
          underlying provider falls back to its configured
          ``default_location`` per Requirement 7.1.

        Splitting validation out of :meth:`execute` keeps each method's
        branching shallow and matches the pattern in
        :class:`WebSearchSkill`.
        """
        if "location" not in args or args["location"] is None:
            # Omitted entirely (or explicitly null) — defer to the
            # provider's configured default.
            return None

        raw = args["location"]
        if not isinstance(raw, str):
            return SkillResult.error(
                "schema_violation",
                "WeatherSkill 'location' must be a string",
            )
        location = raw.strip()
        if not location:
            return SkillResult.error(
                "schema_violation",
                "WeatherSkill 'location' must be non-empty after stripping",
            )
        if len(location) > _MAX_LOCATION_LENGTH:
            return SkillResult.error(
                "schema_violation",
                (
                    f"WeatherSkill 'location' must be at most "
                    f"{_MAX_LOCATION_LENGTH} characters"
                ),
            )
        return location


# ---------------------------------------------------------------------------
# Module-level Skill registration handle
# ---------------------------------------------------------------------------


# The :class:`SkillRegistry` discovers built-in skills via the
# convention of a top-level ``SKILL`` attribute. Exposing the singleton
# here keeps every built-in skill addressable through a uniform import.
SKILL: Skill = WeatherSkill()
