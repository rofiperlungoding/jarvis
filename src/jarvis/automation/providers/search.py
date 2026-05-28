"""Web search provider client (Tavily / Bing / DuckDuckGo).

This module implements :class:`WebSearchClient`, the per-provider transport
that backs :class:`WebSearchSkill` (``design.md §Automation_Service`` and
Requirement 3 in ``requirements.md``). Three concrete providers are
supported, selected at construction time via ``provider_config.provider``:

* ``"tavily"`` — POST ``https://api.tavily.com/search`` with the query and
  ``max_results`` carried in the JSON body. Auth is the Tavily API key
  pulled from the configured :class:`CredentialStore` slot.
* ``"bing"`` — GET ``https://api.bing.microsoft.com/v7.0/search`` with the
  query and ``count`` in the query string. Auth is the
  ``Ocp-Apim-Subscription-Key`` header, again pulled from the
  :class:`CredentialStore`.
* ``"duckduckgo"`` — GET ``https://api.duckduckgo.com/?q=...&format=json``.
  DuckDuckGo's instant-answer API is keyless, so no credential lookup
  occurs and the client tolerates a missing credential slot.

All three providers share a single uniform output shape: a list of
``{"title": str, "url": str, "snippet": str}`` dictionaries. Per
Requirement 3.1 the caller-requested ``max_results`` is clamped to the
client's hard ceiling of :data:`MAX_RESULTS_HARD_CAP` (10) and the
provider's own response is also truncated to the clamped value, so a
chatty upstream cannot push more rows back than the design promises.
The default ``max_results`` is :data:`DEFAULT_MAX_RESULTS` (5),
also matching Requirement 3.1.

The class extends :class:`ProviderClient` so it inherits the cross-cutting
concerns the base layer documents:

* 5 s read timeout (Requirement 7.7);
* exponential-backoff retries on transient 5xx / timeout failures;
* one ``network_egress`` audit row per logical call against the
  configured ``network_destination_allowlist`` (Requirements 13.4, 13.6).

A blocked destination — for example a misconfiguration that asks for
``"bing"`` while the allowlist only includes ``api.tavily.com`` — raises
:class:`NetworkPolicyViolation` from the base class so the calling Skill
can translate it into the documented ``provider_unavailable`` /
``access_denied`` error code at the Skill boundary.

Validates: Requirements 3.1, 3.2, 3.4, 7.7, 13.4, 13.6
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final, Literal, TypedDict
from urllib.parse import urlencode

import httpx

from jarvis.automation.providers.http import (
    DEFAULT_PROVIDER_MAX_ATTEMPTS,
    DEFAULT_PROVIDER_TIMEOUT_S,
    ProviderClient,
)
from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialBackend

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "MAX_RESULTS_HARD_CAP",
    "SearchProvider",
    "SearchResult",
    "WebSearchClient",
    "WebSearchError",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard ceiling on the number of results a single ``search`` call may
#: return, applied independently of any caller- or provider-supplied
#: value. Mirrors Requirement 3.1's ``max_results <= 10`` rule.
MAX_RESULTS_HARD_CAP: Final[int] = 10

#: Default ``max_results`` when the caller does not supply one. Matches
#: Requirement 3.1's documented default of 5.
DEFAULT_MAX_RESULTS: Final[int] = 5

#: Set of providers the client knows how to drive. Kept as a frozenset so
#: ``provider in _SUPPORTED_PROVIDERS`` is O(1) and so the constructor can
#: mention the legal values in its error message without recomputing.
_SUPPORTED_PROVIDERS: Final[frozenset[str]] = frozenset({"tavily", "bing", "duckduckgo"})

#: Per-provider justification strings persisted on every ``network_egress``
#: row. The Privacy_Dashboard surfaces these verbatim, so the wording is
#: deliberately user-facing rather than internal.
_DEFAULT_JUSTIFICATION: Final[str] = "web search"

#: Per-provider canonical endpoints. Centralised here so subclasses (or a
#: future test seam) can override one URL without re-implementing the
#: whole provider branch.
_TAVILY_URL: Final[str] = "https://api.tavily.com/search"
_BING_URL: Final[str] = "https://api.bing.microsoft.com/v7.0/search"
_DUCKDUCKGO_URL: Final[str] = "https://api.duckduckgo.com/"


SearchProvider = Literal["tavily", "bing", "duckduckgo"]


class SearchResult(TypedDict):
    """One result row in the uniform output shape used by every provider.

    The three keys mirror Requirement 3.2 (``title``, ``url``, ``snippet``)
    and are always present in every row returned by :meth:`WebSearchClient.search`,
    even when an upstream provider omits one — missing values are coerced
    to the empty string so the downstream :class:`WebSearchSkill` does not
    need to defend against partial dictionaries.
    """

    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class WebSearchError(RuntimeError):
    """Raised when the upstream provider returns a non-2xx response.

    Carries the HTTP ``status_code`` so the calling Skill can translate it
    into the appropriate :class:`SkillResult` error code: 401/403 → the
    ``CredentialUpdateFlow`` (rotation), 4xx → ``provider_unavailable``
    surfaced to the user, exhausted-retry 5xx → ``provider_unavailable``.
    """

    def __init__(self, *, provider: SearchProvider, status_code: int, body: str) -> None:
        super().__init__(
            f"web search provider {provider!r} returned HTTP {status_code}"
        )
        self.provider: SearchProvider = provider
        self.status_code: int = status_code
        self.body: str = body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    """Return ``value`` as a string, coercing ``None`` / non-strings safely.

    Provider responses occasionally carry ``None`` for an optional snippet
    (DuckDuckGo's ``RelatedTopics`` entries do this for the section
    headings) or a non-string scalar that Pydantic-style strict mode would
    reject. We coerce here so the uniform :class:`SearchResult` shape is
    always satisfied, but we deliberately stop short of HTML-stripping —
    that is the Skill's job, since the Skill renders the snippet to the
    user and may want to keep markup in a future structured output mode.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _provider_config_value(provider_config: Any, key: str, default: Any = None) -> Any:
    """Return ``provider_config[key]`` whether it is a model or a mapping.

    The constructor is documented to accept both a
    :class:`~jarvis.config.schema.ProvidersSearchConfig` pydantic instance
    and a plain ``dict`` so callers in tests do not have to import the
    schema module just to spin up a client. Pydantic v2 models support
    attribute access, so this helper falls back through both forms.
    """
    if hasattr(provider_config, key):
        return getattr(provider_config, key)
    if isinstance(provider_config, dict):
        return provider_config.get(key, default)
    return default


# ---------------------------------------------------------------------------
# WebSearchClient
# ---------------------------------------------------------------------------


class WebSearchClient(ProviderClient):
    """HTTP client that fans out to Tavily, Bing, or DuckDuckGo.

    Parameters
    ----------
    audit_log:
        :class:`AuditLog` instance forwarded to :class:`ProviderClient` for
        ``network_egress`` and ``policy_violation`` rows.
    network_allowlist:
        Iterable of allowed hostnames forwarded to :class:`ProviderClient`.
        At least the configured provider's host (``api.tavily.com``,
        ``api.bing.microsoft.com``, or ``api.duckduckgo.com``) must be
        listed or every search call will be blocked with a
        ``policy_violation`` audit row.
    credential_store:
        :class:`CredentialBackend` used to look up the per-provider API
        key by name (``provider_config.api_key_credential``). DuckDuckGo
        does not require credentials, so a missing slot is tolerated for
        that provider; for Tavily and Bing a missing credential causes
        :meth:`search` to raise :class:`WebSearchError` with HTTP 401 so
        the Skill layer can surface ``missing_credentials``.
    provider_config:
        Either a :class:`~jarvis.config.schema.ProvidersSearchConfig`
        instance or a plain mapping with the same shape. The relevant
        fields are ``provider`` (one of ``tavily``, ``bing``, or
        ``duckduckgo``) and ``api_key_credential`` (the credential slot
        to read). The ``max_results_default`` and ``max_results_cap``
        fields are accepted for forward compatibility but the hard cap of
        10 (Requirement 3.1) is always enforced regardless.
    client:
        Optional pre-configured :class:`httpx.AsyncClient`. Forwarded to
        :class:`ProviderClient`; primarily used in tests via
        :class:`httpx.MockTransport`.
    timeout_seconds:
        Read timeout in seconds (default 5 s, Requirement 7.7).
    max_attempts:
        Total number of attempts on retryable failures (default 3).
    """

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        network_allowlist: Iterable[str],
        credential_store: CredentialBackend | None,
        provider_config: Any,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_S,
        max_attempts: int = DEFAULT_PROVIDER_MAX_ATTEMPTS,
    ) -> None:
        provider = _provider_config_value(provider_config, "provider")
        if not isinstance(provider, str) or provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                "provider_config.provider must be one of "
                f"{sorted(_SUPPORTED_PROVIDERS)!r}, got {provider!r}"
            )

        api_key_credential = _provider_config_value(
            provider_config, "api_key_credential", default=None
        )
        if api_key_credential is not None and not isinstance(api_key_credential, str):
            raise TypeError(
                "provider_config.api_key_credential must be a string or None"
            )

        super().__init__(
            audit_log=audit_log,
            network_allowlist=network_allowlist,
            justification=_DEFAULT_JUSTIFICATION,
            skill_name="WebSearchClient",
            client=client,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )

        self._provider: SearchProvider = provider  # type: ignore[assignment]
        self._credential_store: CredentialBackend | None = credential_store
        self._api_key_credential: str | None = api_key_credential

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> SearchProvider:
        """The configured provider identifier."""
        return self._provider

    @property
    def api_key_credential(self) -> str | None:
        """The credential-store slot consulted for the provider API key."""
        return self._api_key_credential

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[SearchResult]:
        """Run ``query`` against the configured provider.

        Parameters
        ----------
        query:
            User query string. Must be non-empty after stripping; an empty
            query raises :class:`ValueError` rather than wasting a
            provider call.
        max_results:
            Caller-requested result count. Clamped to ``[1, 10]`` per
            Requirement 3.1 — values below 1 are raised to 1, values
            above 10 are lowered to 10. Non-integer values raise
            :class:`TypeError`.

        Returns
        -------
        list[SearchResult]
            Up to ``max_results`` rows in the uniform
            ``{"title", "url", "snippet"}`` shape (Requirement 3.2). May
            be empty when the provider has no matches (Requirement 3.4 —
            the calling Skill is responsible for the user-facing "no
            results" message).

        Raises
        ------
        ValueError
            If ``query`` is empty after stripping.
        TypeError
            If ``max_results`` is not an integer.
        WebSearchError
            If the provider returns a non-2xx response (after retries on
            5xx have been exhausted by :class:`ProviderClient`).
        NetworkPolicyViolation
            If the provider's host is not on the configured allowlist.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must be non-empty")
        if not isinstance(max_results, bool) and not isinstance(max_results, int):
            # ``bool`` is a subclass of ``int`` in Python; reject it so
            # ``search(q, max_results=True)`` does not silently mean 1.
            raise TypeError("max_results must be an int")
        if isinstance(max_results, bool):
            raise TypeError("max_results must be an int, not bool")

        clamped = self._clamp_max_results(max_results)

        if self._provider == "tavily":
            results = await self._search_tavily(cleaned, clamped)
        elif self._provider == "bing":
            results = await self._search_bing(cleaned, clamped)
        elif self._provider == "duckduckgo":
            results = await self._search_duckduckgo(cleaned, clamped)
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError(
                f"unsupported provider routed past constructor: {self._provider!r}"
            )

        # Truncate after the provider call as well so an upstream that
        # ignored our request and returned more rows than asked cannot
        # exceed the design's documented cap.
        return results[:clamped]

    # ------------------------------------------------------------------
    # Internals — clamping and credential lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_max_results(requested: int) -> int:
        """Return ``requested`` clamped to ``[1, MAX_RESULTS_HARD_CAP]``.

        Requirement 3.1 says ``max_results`` defaults to 5 and is capped
        at 10. Callers that pass values outside the legal range are
        silently corrected rather than rejected so a generous
        :class:`WebSearchSkill` schema does not have to duplicate the
        clamp logic for itself.
        """
        if requested < 1:
            return 1
        if requested > MAX_RESULTS_HARD_CAP:
            return MAX_RESULTS_HARD_CAP
        return int(requested)

    def _resolve_api_key(self) -> str | None:
        """Read the configured API key from the :class:`CredentialBackend`.

        Returns ``None`` when no credential slot is configured *or* when
        the slot is configured but the backend has no value for it. The
        per-provider helpers translate these two cases differently:
        DuckDuckGo tolerates the missing key (its instant-answer API is
        keyless), while Tavily and Bing surface a synthetic 401 so the
        Skill layer can route the user through the credential setup
        flow (Requirement 5.6, mirrored here for non-email providers).
        """
        if self._api_key_credential is None or self._credential_store is None:
            return None
        try:
            value = self._credential_store.get(self._api_key_credential)
        except Exception:
            # A missing or corrupted DPAPI blob raises rather than
            # returning ``None``. Treat that as "no key available" so the
            # provider branch below can produce the same user-visible
            # error path as a totally unconfigured slot, instead of
            # bubbling a stack trace up through the Dialog_Manager.
            return None
        if value is None or value == "":
            return None
        return value

    # ------------------------------------------------------------------
    # Internals — per-provider drivers
    # ------------------------------------------------------------------

    async def _search_tavily(
        self, query: str, max_results: int
    ) -> list[SearchResult]:
        """Drive the Tavily search API.

        Tavily expects a JSON POST body with ``api_key``, ``query``, and
        ``max_results``. The successful response shape is
        ``{"results": [{"title", "url", "content"}, ...]}``. We map
        ``content`` to ``snippet`` so the uniform output shape is
        preserved.
        """
        api_key = self._resolve_api_key()
        if not api_key:
            # Synthesize a 401 so the Skill maps this to
            # ``missing_credentials``. Tavily itself returns 401 in the
            # same scenario, so the calling Skill's downstream handling
            # is identical whether the failure is local or remote.
            raise WebSearchError(
                provider="tavily",
                status_code=401,
                body="api key not configured",
            )

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
        }
        response = await self.post(_TAVILY_URL, json=payload)
        self._raise_for_status(response, provider="tavily")
        body = self._safe_json(response)
        raw_results = body.get("results", []) if isinstance(body, dict) else []
        return [
            SearchResult(
                title=_coerce_str(item.get("title")),
                url=_coerce_str(item.get("url")),
                snippet=_coerce_str(item.get("content") or item.get("snippet")),
            )
            for item in raw_results
            if isinstance(item, dict)
        ]

    async def _search_bing(
        self, query: str, max_results: int
    ) -> list[SearchResult]:
        """Drive the Bing Web Search v7 API.

        Bing accepts ``q`` and ``count`` as query-string parameters and
        requires the ``Ocp-Apim-Subscription-Key`` header. Successful
        responses carry a ``webPages.value`` array whose entries have
        ``name``, ``url``, and ``snippet``.
        """
        api_key = self._resolve_api_key()
        if not api_key:
            raise WebSearchError(
                provider="bing",
                status_code=401,
                body="api key not configured",
            )

        params = {"q": query, "count": str(max_results)}
        url = f"{_BING_URL}?{urlencode(params)}"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        response = await self.get(url, headers=headers)
        self._raise_for_status(response, provider="bing")
        body = self._safe_json(response)
        web_pages = (
            body.get("webPages", {}) if isinstance(body, dict) else {}
        )
        raw_results = (
            web_pages.get("value", []) if isinstance(web_pages, dict) else []
        )
        return [
            SearchResult(
                title=_coerce_str(item.get("name")),
                url=_coerce_str(item.get("url")),
                snippet=_coerce_str(item.get("snippet")),
            )
            for item in raw_results
            if isinstance(item, dict)
        ]

    async def _search_duckduckgo(
        self, query: str, max_results: int
    ) -> list[SearchResult]:
        """Drive the DuckDuckGo instant-answer API.

        DuckDuckGo exposes a keyless JSON endpoint at
        ``https://api.duckduckgo.com/?q=...&format=json``. Results live
        under ``RelatedTopics``; entries that are sub-categories
        themselves carry a ``Topics`` list rather than a ``FirstURL`` and
        are flattened so the caller sees a single uniform list. The
        ``Text`` field doubles as both title and snippet — we duplicate
        it into both columns since DuckDuckGo does not separate them.
        """
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        url = f"{_DUCKDUCKGO_URL}?{urlencode(params)}"
        response = await self.get(url)
        self._raise_for_status(response, provider="duckduckgo")
        body = self._safe_json(response)
        topics = (
            body.get("RelatedTopics", []) if isinstance(body, dict) else []
        )

        flattened: list[dict[str, Any]] = []
        for entry in topics:
            if not isinstance(entry, dict):
                continue
            if "Topics" in entry and isinstance(entry["Topics"], list):
                # Sub-category: descend one level.
                for sub in entry["Topics"]:
                    if isinstance(sub, dict):
                        flattened.append(sub)
            else:
                flattened.append(entry)
            if len(flattened) >= max_results:
                # Stop traversing once we have enough rows; DuckDuckGo
                # responses can be quite large and the design's cap is
                # the only ground truth that matters.
                break

        results: list[SearchResult] = []
        for item in flattened[:max_results]:
            text = _coerce_str(item.get("Text"))
            results.append(
                SearchResult(
                    title=text,
                    url=_coerce_str(item.get("FirstURL")),
                    snippet=text,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internals — response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_status(
        response: httpx.Response, *, provider: SearchProvider
    ) -> None:
        """Translate non-2xx responses into :class:`WebSearchError`.

        :class:`ProviderClient.request` already retries 5xx responses up
        to ``max_attempts`` times before handing the final response back,
        so by the time we land here a 5xx genuinely means the upstream is
        unavailable from the user's point of view (Requirement 7.7 →
        ``provider_unavailable``). 4xx responses come straight through
        without retry; the Skill caller is expected to discriminate on
        ``status_code`` (401/403 vs the rest).
        """
        if 200 <= response.status_code < 300:
            return
        # ``response.text`` decodes the body using the negotiated charset;
        # we cap the captured size so a hostile upstream cannot inflate
        # the in-memory exception message.
        body = response.text[:1024]
        raise WebSearchError(
            provider=provider,
            status_code=response.status_code,
            body=body,
        )

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        """Return ``response.json()`` or ``{}`` on a malformed body.

        A malformed JSON body from an upstream we trusted enough to have
        on the allowlist still happens occasionally (Bing has been known
        to return HTML error pages on rate-limit). Falling back to an
        empty dict means the per-provider extractor returns an empty
        result list, which the calling Skill renders as the documented
        "no results" message (Requirement 3.4) instead of crashing the
        Dialog_Manager.
        """
        try:
            return response.json()
        except (ValueError, httpx.DecodingError):
            return {}
