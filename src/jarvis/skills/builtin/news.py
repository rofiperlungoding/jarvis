"""News headlines built-in skill.

Implements :class:`NewsSkill`, the Skill the Dialog_Manager invokes when
the LLM emits a ``NewsSkill`` Tool_Call (Requirement 7.3). The skill is
a thin Skill-layer translation between the LLM's tool-call arguments
and the :class:`~jarvis.automation.providers.news.NewsClient` configured
at application bootstrap.

Responsibilities
----------------

* Declare the JSON Schema for the LLM-facing arguments per Requirement
  7.3: ``topic`` is optional, ``max_items`` defaults to 5 and is capped
  at 10. The schema mirrors :class:`WebSearchSkill`'s ``max_results``
  shape so the registry's :class:`~jsonschema.Draft7Validator` rejects
  out-of-range integers before dispatch (Property 2 / CP2).
* Validate the caller-supplied arguments beyond the JSON Schema rule
  set the registry already enforces. ``topic`` is normalised by
  stripping whitespace; an empty post-strip ``topic`` is treated as
  "use the configured default" (which the provider expands). Empty
  defaults — i.e. neither argument nor configuration carries a topic —
  are surfaced as ``schema_violation`` so the LLM gets a chance to
  retry with a topic.
* Resolve the ``"news"`` provider from :class:`SkillContext.providers`
  and call :meth:`NewsClient.fetch`. Translate the structured failure
  modes the client raises into the documented Error Taxonomy codes:

  * :class:`ProviderError` is mapped 1:1 onto the matching
    :class:`SkillResult` error code (``missing_credentials``,
    ``provider_unavailable``).
  * :class:`NetworkPolicyViolation` → ``access_denied`` per Requirement
    13.6 (the configured allowlist refused the destination; the
    provider already wrote the ``policy_violation`` audit row).
  * :class:`httpx.TimeoutException` → ``timeout``. The base client's
    retry budget is already spent by the time this propagates.
* Shape the success payload so the LLM has both summarisable text and
  citable URLs. ``value["articles"]`` carries the canonical list of
  ``{"title", "source", "url", "published_at", "description"}`` rows;
  ``value["cited_urls"]`` is the deduplicated tuple-as-list of source
  URLs in result order so the Dialog_Manager can render them straight
  into :attr:`AssistantResponse.cited_urls` (mirrors
  :class:`WebSearchSkill`'s ``cited_urls`` for transcript-log
  consistency); ``value["headlines"]`` is a pre-formatted bullet list
  (`"- {title} — {source} ({url})"`) ready for the LLM to read aloud.

Validates: Requirements 7.3, 7.4, 7.7
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

__all__ = ["DEFAULT_MAX_ITEMS", "MAX_ITEMS_HARD_CAP", "SKILL", "NewsSkill"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider key used to look the news client up out of
#: :class:`SkillContext.providers`. Matches the documented mapping in
#: ``src/jarvis/skills/base.py`` (``"news"``).
_PROVIDER_KEY: Final[str] = "news"

#: Skill name surfaced to the LLM. Pinned as a constant because
#: Requirement 7.3 anchors the wording ("NewsSkill") and changing it
#: would silently break configured trusted-action allowlists.
_SKILL_NAME: Final[str] = "NewsSkill"

#: Description handed to the LLM as ``function.description``. Concise,
#: action-oriented, and avoids implementation details so the model picks
#: the tool for its capability rather than its internals.
_SKILL_DESCRIPTION: Final[str] = (
    "Fetch the top news headlines for a topic. Returns titles, source "
    "names, URLs, and short descriptions. Use when the user asks for "
    "current events or news on a particular subject."
)

#: Default ``max_items`` when the caller omits the field. Matches
#: Requirement 7.3's documented default of 5 and the value carried by
#: :data:`jarvis.automation.providers.news._DEFAULT_MAX_ITEMS`.
DEFAULT_MAX_ITEMS: Final[int] = 5

#: Hard ceiling on ``max_items``. Matches Requirement 7.3's "capped at
#: 10" rule and the value enforced inside :class:`NewsClient`.
MAX_ITEMS_HARD_CAP: Final[int] = 10

#: Maximum length, in characters, of a single rendered headline line.
#: The LLM is free to rewrite the summary, but capping it here keeps the
#: per-turn payload bounded for very chatty providers.
_MAX_HEADLINE_LINE_LENGTH: Final[int] = 280


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def _build_json_schema() -> dict[str, Any]:
    """Build the JSON Schema describing the Skill's arguments.

    Pulled into a helper so the constants :data:`DEFAULT_MAX_ITEMS` and
    :data:`MAX_ITEMS_HARD_CAP` flow into the schema *and* into the
    runtime clamp without drift. The schema is draft-07 compliant and
    stays inside the Mistral function-calling subset (no ``$ref``, no
    mixed ``oneOf``, no exotic ``format``).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Topic or keyword to fetch headlines for. When "
                    "omitted, the configured default topic is used."
                ),
            },
            "max_items": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_ITEMS_HARD_CAP,
                "default": DEFAULT_MAX_ITEMS,
                "description": (
                    f"Number of headlines to return. Defaults to 5, capped at {MAX_ITEMS_HARD_CAP}."
                ),
            },
        },
        "required": [],
    }


# ---------------------------------------------------------------------------
# NewsSkill
# ---------------------------------------------------------------------------


class NewsSkill:
    """Skill that proxies to a configured :class:`NewsClient`.

    Stateless: the same instance can be safely registered once and used
    for every Tool_Call. The provider client is resolved from the
    :class:`SkillContext` on every call so a future provider rotation
    (e.g., NewsAPI → another vendor) does not require re-registering
    the Skill.
    """

    manifest: SkillManifest = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=_build_json_schema(),
        destructive=False,
        # News fetch is read-only and platform-agnostic; declare every
        # supported platform so Requirement 15.4's gating does not block
        # the Skill on a future macOS / Linux build.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Run the configured news provider against the supplied topic.

        Argument validation here is intentionally a thin layer on top of
        the registry's JSON Schema check: the registry has already
        validated types and required-ness, so we only enforce the
        *semantic* rules (string topic, integer max_items, runtime
        clamp).
        """
        # 1. Argument validation. Pulled into a helper so the early-exit
        #    branches do not pile up directly inside ``execute``.
        validated = self._validate_args(args)
        if isinstance(validated, SkillResult):
            return validated
        topic, max_items = validated

        # 2. Provider resolution. The Skill cannot invent a client, so a
        #    missing provider mapping surfaces as
        #    ``provider_unavailable`` — the Dialog_Manager will tell the
        #    user "the news provider isn't configured" rather than
        #    asking them to repeat the request.
        client = ctx.providers.get(_PROVIDER_KEY) if ctx.providers else None
        if client is None:
            logger.warning("NewsSkill invoked without a 'news' provider in context")
            return SkillResult.error(
                "provider_unavailable",
                "no news provider is configured",
            )

        # 3. Provider invocation + error translation.
        try:
            articles = await client.fetch(topic, max_items=max_items)
        except (
            ProviderError,
            NetworkPolicyViolation,
            httpx.TimeoutException,
            ValueError,
            TypeError,
        ) as exc:
            return self._translate_provider_exception(exc)

        # 4. Success payload assembly. ``articles == []`` is still a
        #    successful result — the Dialog_Manager will tell the user
        #    "no headlines found, would you like to try a different
        #    topic?" rather than treating the empty list as an error.
        return SkillResult.success(
            self._build_success_value(
                topic=topic,
                max_items=max_items,
                articles=list(articles),
            )
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_args(cls, args: dict[str, Any]) -> SkillResult | tuple[str | None, int]:
        """Validate ``args`` and return ``(topic, max_items)`` on success.

        Returns a :class:`SkillResult` when validation fails so the
        caller can surface the error directly. Splitting validation out
        of :meth:`execute` keeps each method's branching shallow enough
        to satisfy the project's pylint return-count budget while
        preserving the documented per-rule error messages.

        ``topic`` is normalised by stripping whitespace; the empty
        post-strip case is converted to ``None`` so the provider's
        default-topic fallback path is exercised.
        """
        raw_topic = args.get("topic")
        if raw_topic is None:
            topic: str | None = None
        elif not isinstance(raw_topic, str):
            return SkillResult.error(
                "schema_violation",
                "NewsSkill 'topic' must be a string",
            )
        else:
            stripped = raw_topic.strip()
            topic = stripped if stripped else None

        max_items = cls._resolve_max_items(args.get("max_items"))
        if isinstance(max_items, SkillResult):
            return max_items
        return topic, max_items

    @staticmethod
    def _resolve_max_items(raw: Any) -> int | SkillResult:
        """Clamp ``raw`` to ``[1, MAX_ITEMS_HARD_CAP]`` or fail gracefully.

        The JSON Schema enforces ``integer, minimum 1, maximum 10`` so
        the registry's validator rejects out-of-range values before we
        get here; the secondary clamp protects against a future caller
        bypassing the schema (e.g., a test or a hand-built dispatch).
        Booleans are rejected even though ``bool`` is a subclass of
        ``int``: ``max_items=True`` quietly meaning ``1`` would be a
        nasty footgun.
        """
        if raw is None:
            return DEFAULT_MAX_ITEMS
        if isinstance(raw, bool):
            return SkillResult.error(
                "schema_violation",
                "NewsSkill 'max_items' must be an integer, not bool",
            )
        if not isinstance(raw, int):
            return SkillResult.error(
                "schema_violation",
                "NewsSkill 'max_items' must be an integer",
            )
        if raw < 1:
            return 1
        if raw > MAX_ITEMS_HARD_CAP:
            return MAX_ITEMS_HARD_CAP
        return raw

    # ------------------------------------------------------------------
    # Error translation
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_provider_exception(exc: BaseException) -> SkillResult:
        """Map a provider-raised exception onto the documented error code.

        Centralised translation so :meth:`execute` does not duplicate the
        per-exception branching. The mapping follows the design's Error
        Taxonomy:

        * :class:`ProviderError` → matching error code on the
          :class:`SkillResult` (``missing_credentials`` /
          ``provider_unavailable``). Per Requirement 7.7, the
          Dialog_Manager renders ``provider_unavailable`` as a
          user-visible "the news provider is unavailable" message.
        * :class:`NetworkPolicyViolation` → ``access_denied`` per
          Requirement 13.6 (the allowlist already recorded a
          ``policy_violation`` audit row).
        * :class:`httpx.TimeoutException` → ``timeout`` (retry budget
          already exhausted by :class:`ProviderClient`). NewsAPI's
          provider currently re-wraps timeouts as
          :class:`ProviderError`; we keep this branch defensively in
          case a future provider plumbs the timeout straight through.
        * :class:`ValueError` / :class:`TypeError` → ``schema_violation``
          so the LLM gets a Requirement-14.5 retry chance.
        """
        if isinstance(exc, ProviderError):
            # ``ProviderError.error_code`` is a closed
            # ``ProviderErrorCode`` literal type whose values are a
            # strict subset of :data:`SkillErrorCode`. Forwarding the
            # code preserves the design's 1:1 mapping at the Skill
            # boundary.
            return SkillResult.error(
                exc.error_code,
                str(exc),
            )
        if isinstance(exc, NetworkPolicyViolation):
            return SkillResult.error(
                "access_denied",
                f"news fetch blocked by network policy: {exc}",
            )
        if isinstance(exc, httpx.TimeoutException):
            message = f"news fetch timed out: {exc}" if str(exc) else "news fetch timed out"
            return SkillResult.error("timeout", message)
        # ValueError / TypeError fall through here.
        return SkillResult.error(
            "schema_violation",
            f"news fetch rejected arguments: {exc}",
        )

    # ------------------------------------------------------------------
    # Success payload
    # ------------------------------------------------------------------

    @classmethod
    def _build_success_value(
        cls,
        *,
        topic: str | None,
        max_items: int,
        articles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the structured payload returned to the Dialog_Manager.

        The payload mirrors :class:`WebSearchSkill`'s success shape so
        the Dialog_Manager can route both Skills' results through the
        same transcript-rendering path:

        * ``topic`` — echoed back so the audit trail (and the LLM's
          tool-result message) carry the canonical topic the client
          actually issued. ``None`` is preserved when the provider's
          configured default was used implicitly.
        * ``max_items`` — echoed back so the LLM knows how many rows
          it asked for, useful when the actual list is shorter.
        * ``articles`` — the per-row list in the canonical
          ``{"title", "source", "url", "published_at", "description"}``
          shape produced by :class:`NewsClient`.
        * ``cited_urls`` — deduplicated, order-preserving list of
          source URLs. Mirrors the per-row ``url`` field but is
          convenient for callers that only need citations.
        * ``headlines`` — bullet list ready for the LLM to read.
        """
        cited_urls: list[str] = []
        seen_urls: set[str] = set()
        for row in articles:
            url = row.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                cited_urls.append(url)

        headlines = cls._format_headline_lines(topic=topic, articles=articles)

        return {
            "topic": topic,
            "max_items": max_items,
            "articles": list(articles),
            "cited_urls": cited_urls,
            "headlines": headlines,
        }

    @staticmethod
    def _format_headline_lines(*, topic: str | None, articles: list[dict[str, Any]]) -> list[str]:
        """Render the per-article headline lines.

        Returns a single-element list explaining the absence of
        results when ``articles`` is empty. Otherwise produces one line
        per row in the form ``"- {title} — {source} ({url})"`` with the
        per-line length bounded by :data:`_MAX_HEADLINE_LINE_LENGTH`.
        """
        if not articles:
            descriptor = repr(topic) if topic else "the configured default topic"
            return [f"No headlines were found for {descriptor}."]

        lines: list[str] = []
        for row in articles:
            title = (row.get("title") or "").strip()
            source = (row.get("source") or "").strip()
            url = (row.get("url") or "").strip()

            # Build the line piecewise so missing fields do not produce
            # awkward double separators ("—  ()" etc.).
            head = title if title else "(untitled)"
            body = f" — {source}" if source else ""
            tail = f" ({url})" if url else ""
            line = f"- {head}{body}{tail}"

            if len(line) > _MAX_HEADLINE_LINE_LENGTH:
                # Preserve the URL suffix when truncating so citations
                # remain intact even when the title is verbose.
                budget = _MAX_HEADLINE_LINE_LENGTH - len(tail) - 1
                budget = max(budget, 0)
                truncated_head = (head + body)[:budget].rstrip()
                line = f"- {truncated_head}…{tail}"
            lines.append(line)
        return lines


# ---------------------------------------------------------------------------
# Module-level Skill registration handle
# ---------------------------------------------------------------------------


# The :class:`SkillRegistry` discovers built-in skills via the convention
# of a top-level ``SKILL`` attribute. Exposing the singleton here keeps
# every built-in skill addressable through a uniform import.
SKILL: Skill = NewsSkill()
