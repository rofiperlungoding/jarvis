"""Web search built-in skill.

Implements :class:`WebSearchSkill`, the Skill the Dialog_Manager invokes
when the LLM emits a ``WebSearchSkill`` Tool_Call (Requirement 3.1). The
skill is the thin Skill-layer translation between the LLM's tool-call
arguments and the :class:`~jarvis.automation.providers.search.WebSearchClient`
configured at application bootstrap.

Responsibilities
----------------

* Validate the caller-supplied ``query`` and ``max_results`` arguments
  beyond the JSON-Schema rule set the registry already enforces. ``query``
  must be non-empty after stripping; ``max_results`` is silently clamped
  to ``[1, 10]`` so the LLM cannot ask for more results than the design
  promises (Requirement 3.1).
* Look up the ``"web_search"`` provider in :class:`SkillContext.providers`
  and call :meth:`WebSearchClient.search`. Translate the structured
  failure modes the client raises into the documented Error Taxonomy
  codes:

  * :class:`WebSearchError` with HTTP 401/403 →
    :data:`SkillResult.error_code` ``"missing_credentials"``. This is
    the documented "the user has no API key configured" path
    (Requirement 5.6 mirrored for the search provider) and triggers the
    Dialog_Manager's credential setup flow.
  * Any other :class:`WebSearchError` → ``"provider_unavailable"`` per
    Requirement 7.7 (the upstream returned a non-2xx and retries were
    exhausted by :class:`ProviderClient`).
  * :class:`NetworkPolicyViolation` → ``"access_denied"`` per
    Requirement 13.6 (the configured allowlist refused the destination).
  * :class:`httpx.TimeoutException` → ``"timeout"``. The base client's
    retry budget is already spent by the time this propagates.
* Shape the success payload so the LLM has both summarisable text and
  citable URLs:

  * ``value["results"]`` carries the list of ``{"title", "url", "snippet"}``
    rows in the canonical order returned by the provider.
  * ``value["cited_urls"]`` is the deduplicated tuple-as-list of source
    URLs in result order. The Dialog_Manager copies this into the
    :attr:`AssistantResponse.cited_urls` slot when it logs the turn,
    satisfying Requirement 3.3 ("cite source URLs in the textual
    transcript shown in the UI log").
  * ``value["summary_lines"]`` is a pre-formatted bullet list ready for
    the LLM to use directly when composing the spoken summary
    (Requirement 3.3). Each line is shaped ``"- {title} — {snippet}
    ({url})"`` and is the single place where the Skill stitches title,
    snippet, and URL together.
* Honour Requirement 3.4 by returning a *successful* result even when
  the provider has zero matches: ``value["results"] == []`` and a
  one-line ``value["summary_lines"]`` explaining the absence so the
  Dialog_Manager can speak "no results, would you like to refine the
  query?" without re-engineering the negative path.

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.automation.providers.search import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_HARD_CAP,
    WebSearchError,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = ["SKILL", "WebSearchSkill"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider key used to look the search client up out of
#: :class:`SkillContext.providers`. Matches the documented mapping in
#: ``src/jarvis/skills/base.py`` ("``"web_search"``").
_PROVIDER_KEY: Final[str] = "web_search"

#: Skill name surfaced to the LLM. Pinned as a constant because
#: Requirement 3.1 anchors the wording ("WebSearchSkill") and changing
#: it would silently break configured trusted-action allowlists.
_SKILL_NAME: Final[str] = "WebSearchSkill"

#: Description handed to the LLM as ``function.description``. Concise,
#: action-oriented, and avoids implementation details so the model picks
#: the tool for its capability rather than its internals.
_SKILL_DESCRIPTION: Final[str] = (
    "Search the web for current information and return the top results "
    "with titles, URLs, and snippets. Use when the user asks for "
    "information likely outside the model's training data."
)

#: HTTP status codes that indicate a credential problem rather than a
#: generic upstream outage. 407 (proxy authentication) is included for
#: completeness even though none of the configured providers use a proxy
#: today.
_MISSING_CREDENTIALS_STATUS: Final[frozenset[int]] = frozenset({401, 403, 407})

#: Maximum length, in characters, of a single rendered summary line. The
#: LLM is free to rewrite the summary, but capping it here keeps the
#: per-turn payload bounded for very chatty providers.
_MAX_SUMMARY_LINE_LENGTH: Final[int] = 280


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def _build_json_schema() -> dict[str, Any]:
    """Build the JSON Schema describing the Skill's arguments.

    Pulled into a helper so the constants ``DEFAULT_MAX_RESULTS`` and
    ``MAX_RESULTS_HARD_CAP`` flow into the schema *and* into the
    runtime clamp without drift. The schema is draft-07 compliant and
    stays inside the Mistral function-calling subset (no ``$ref``, no
    mixed ``oneOf``, no exotic ``format``).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Free-text search query. Should be a fully-formed "
                    "question or topic; the skill does not modify it."
                ),
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RESULTS_HARD_CAP,
                "default": DEFAULT_MAX_RESULTS,
                "description": (
                    "Number of results to return. Defaults to 5, capped "
                    f"at {MAX_RESULTS_HARD_CAP}."
                ),
            },
        },
        "required": ["query"],
    }


# ---------------------------------------------------------------------------
# WebSearchSkill
# ---------------------------------------------------------------------------


class WebSearchSkill:
    """Skill that proxies to a configured :class:`WebSearchClient`.

    Stateless: the same instance can be safely registered once and used
    for every Tool_Call. The provider client is resolved from the
    :class:`SkillContext` on every call so a future provider rotation
    (e.g., Tavily → Bing) does not require re-registering the Skill.
    """

    manifest: SkillManifest = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=_build_json_schema(),
        destructive=False,
        # Web search is read-only and platform-agnostic; declare every
        # supported platform so Requirement 15.4's gating does not block
        # the Skill on a future macOS / Linux build.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Run the configured search provider against ``args["query"]``.

        Argument validation here is intentionally a thin layer on top of
        the registry's JSON Schema check: the registry has already
        validated types and required-ness, so we only enforce the
        *semantic* rules (non-empty query after stripping) and the
        runtime clamp on ``max_results``.
        """
        # 1. Argument validation. Pulled into a helper so the early-exit
        #    branches do not pile up directly inside ``execute``.
        validated = self._validate_args(args)
        if isinstance(validated, SkillResult):
            return validated
        query, max_results = validated

        # 2. Provider resolution. The Skill cannot invent a client, so a
        #    missing provider mapping surfaces as
        #    ``provider_unavailable`` — the Dialog_Manager will tell the
        #    user "the search provider isn't configured" rather than
        #    asking them to repeat the request.
        client = ctx.providers.get(_PROVIDER_KEY) if ctx.providers else None
        if client is None:
            logger.warning(
                "WebSearchSkill invoked without a 'web_search' provider in context"
            )
            return SkillResult.error(
                "provider_unavailable",
                "no web search provider is configured",
            )

        # 3. Provider invocation + error translation.
        try:
            results = await client.search(query, max_results=max_results)
        except (
            WebSearchError,
            NetworkPolicyViolation,
            httpx.TimeoutException,
            ValueError,
            TypeError,
        ) as exc:
            return self._translate_provider_exception(exc)

        # 4. Success payload assembly. We always return ``ok=True`` even
        #    when ``results == []`` — Requirement 3.4 explicitly puts
        #    "zero results" on the success side of the boundary so the
        #    Dialog_Manager can offer to refine the query without
        #    treating the empty list as an error.
        return SkillResult.success(
            self._build_success_value(query=query, results=results)
        )

    @classmethod
    def _validate_args(cls, args: dict[str, Any]) -> SkillResult | tuple[str, int]:
        """Validate ``args`` and return ``(query, max_results)`` on success.

        Returns a :class:`SkillResult` when validation fails so the
        caller can surface the error directly. Splitting validation out
        of :meth:`execute` keeps each method's branching shallow enough
        to satisfy the project's pylint return-count budget while
        preserving the documented per-rule error messages.
        """
        # ``args`` has already been validated against the JSON Schema by
        # the registry, so missing keys mean the dispatcher misbehaved —
        # surface as ``internal_error`` rather than ``schema_violation``.
        try:
            raw_query = args["query"]
        except KeyError:
            return SkillResult.error(
                "internal_error",
                "WebSearchSkill received args without 'query'",
            )

        if not isinstance(raw_query, str):
            return SkillResult.error(
                "schema_violation",
                "WebSearchSkill 'query' must be a string",
            )
        query = raw_query.strip()
        if not query:
            return SkillResult.error(
                "schema_violation",
                "WebSearchSkill 'query' must be non-empty after stripping",
            )

        max_results = cls._resolve_max_results(args.get("max_results"))
        if isinstance(max_results, SkillResult):
            return max_results
        return query, max_results

    @classmethod
    def _translate_provider_exception(cls, exc: BaseException) -> SkillResult:
        """Map a provider-raised exception onto the documented error code.

        Centralised translation so :meth:`execute` does not duplicate the
        per-exception branching. The mapping follows the design's Error
        Taxonomy:

        * :class:`WebSearchError` 401/403/407 → ``missing_credentials``;
          other status codes → ``provider_unavailable``.
        * :class:`NetworkPolicyViolation` → ``access_denied`` (the
          allowlist already recorded a ``policy_violation`` audit row).
        * :class:`httpx.TimeoutException` → ``timeout`` (retry budget
          already exhausted by :class:`ProviderClient`).
        * :class:`ValueError` / :class:`TypeError` → ``schema_violation``
          so the LLM gets a Requirement-14.5 retry chance.
        """
        if isinstance(exc, WebSearchError):
            return cls._translate_websearch_error(exc)
        if isinstance(exc, NetworkPolicyViolation):
            return SkillResult.error(
                "access_denied",
                f"web search blocked by network policy: {exc}",
            )
        if isinstance(exc, httpx.TimeoutException):
            message = (
                f"web search timed out: {exc}" if str(exc) else "web search timed out"
            )
            return SkillResult.error("timeout", message)
        # ValueError / TypeError fall through here.
        return SkillResult.error(
            "schema_violation",
            f"web search rejected arguments: {exc}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_max_results(raw: Any) -> int | SkillResult:
        """Clamp ``raw`` to ``[1, MAX_RESULTS_HARD_CAP]`` or fail gracefully.

        The JSON Schema enforces ``integer, minimum 1, maximum 10`` so
        the registry's validator rejects out-of-range values before we
        get here; the secondary clamp protects against a future caller
        bypassing the schema (e.g., a test or a hand-built dispatch).
        Booleans are rejected even though ``bool`` is a subclass of
        ``int``: ``max_results=True`` quietly meaning ``1`` would be a
        nasty footgun.
        """
        if raw is None:
            return DEFAULT_MAX_RESULTS
        if isinstance(raw, bool):
            return SkillResult.error(
                "schema_violation",
                "WebSearchSkill 'max_results' must be an integer, not bool",
            )
        if not isinstance(raw, int):
            return SkillResult.error(
                "schema_violation",
                "WebSearchSkill 'max_results' must be an integer",
            )
        if raw < 1:
            return 1
        if raw > MAX_RESULTS_HARD_CAP:
            return MAX_RESULTS_HARD_CAP
        return raw

    @staticmethod
    def _translate_websearch_error(exc: WebSearchError) -> SkillResult:
        """Map a :class:`WebSearchError` to the documented error code."""
        if exc.status_code in _MISSING_CREDENTIALS_STATUS:
            return SkillResult.error(
                "missing_credentials",
                (
                    f"web search provider {exc.provider!r} returned HTTP "
                    f"{exc.status_code}; configure the API key for the "
                    "provider"
                ),
            )
        return SkillResult.error(
            "provider_unavailable",
            (
                f"web search provider {exc.provider!r} returned HTTP "
                f"{exc.status_code}"
            ),
        )

    @classmethod
    def _build_success_value(
        cls,
        *,
        query: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the structured payload returned to the Dialog_Manager.

        Three fields populate the payload:

        * ``query`` — echoed back so the audit trail (and the LLM's
          tool-result message) carry the canonical query string the
          client actually issued.
        * ``results`` — the per-row list in the uniform
          ``{"title", "url", "snippet"}`` shape.
        * ``cited_urls`` — deduplicated, order-preserving list of source
          URLs. Mirrors the per-row ``url`` field but is convenient for
          callers that only need citations.
        * ``summary_lines`` — bullet list ready for the LLM to read.
        """
        cited_urls: list[str] = []
        seen_urls: set[str] = set()
        for row in results:
            url = row.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                cited_urls.append(url)

        summary_lines = cls._format_summary_lines(query=query, results=results)

        return {
            "query": query,
            "results": list(results),
            "cited_urls": cited_urls,
            "summary_lines": summary_lines,
        }

    @staticmethod
    def _format_summary_lines(
        *, query: str, results: list[dict[str, Any]]
    ) -> list[str]:
        """Render the per-result summary lines.

        Returns a single-element list explaining the absence of
        results when ``results`` is empty (Requirement 3.4). Otherwise
        produces one line per row in the form
        ``"- {title} — {snippet} ({url})"`` with the per-line length
        bounded by :data:`_MAX_SUMMARY_LINE_LENGTH`.
        """
        if not results:
            return [f"No results were found for {query!r}."]

        lines: list[str] = []
        for row in results:
            title = (row.get("title") or "").strip()
            snippet = (row.get("snippet") or "").strip()
            url = (row.get("url") or "").strip()

            # Build the line piecewise so missing fields do not produce
            # awkward double separators ("—  ()" etc.).
            head = title if title else "(untitled)"
            body = f" — {snippet}" if snippet else ""
            tail = f" ({url})" if url else ""
            line = f"- {head}{body}{tail}"

            if len(line) > _MAX_SUMMARY_LINE_LENGTH:
                # Preserve the URL suffix when truncating so citations
                # remain intact even when the snippet is verbose.
                budget = _MAX_SUMMARY_LINE_LENGTH - len(tail) - 1
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
SKILL: Skill = WebSearchSkill()
