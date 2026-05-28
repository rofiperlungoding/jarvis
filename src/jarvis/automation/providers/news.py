"""NewsAPI provider client.

Implements the ``NewsClient`` referenced by ``NewsSkill`` (Requirement
7.3, 7.4, 7.7). The client fetches the top headlines for a configurable
``topic``, capping the number of items at ten per Requirement 7.3.

Endpoint: ``GET https://newsapi.org/v2/top-headlines?q={topic}&pageSize=...``
with the API key passed via the ``X-Api-Key`` request header. The header
form is preferred over the ``apiKey`` query string parameter so the key
never lands in HTTP-server access logs.

Validates: Requirements 5.6, 7.3, 7.4, 7.7
"""

from __future__ import annotations

import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import ProviderClient
from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialBackend

logger = logging.getLogger(__name__)

__all__ = ["NewsClient"]


_NEWSAPI_HOST: Final[str] = "newsapi.org"
_NEWSAPI_BASE_URL: Final[str] = f"https://{_NEWSAPI_HOST}"

#: Per Requirement 7.3, ``max_items`` defaults to 5 and is capped at 10.
_DEFAULT_MAX_ITEMS: Final[int] = 5
_MAX_ITEMS_CAP: Final[int] = 10


class NewsClient(ProviderClient):
    """NewsAPI-backed news headlines client.

    Parameters mirror :class:`WeatherClient`. ``provider_config`` MUST
    expose ``api_key_credential`` (string), ``default_topic`` (string)
    and ``timeout_seconds`` (float) attributes.
    """

    PROVIDER_NAME: Final[str] = "newsapi"

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        network_allowlist: list[str] | tuple[str, ...] | frozenset[str],
        credential_store: CredentialBackend,
        provider_config: Any,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            audit_log=audit_log,
            network_allowlist=network_allowlist,
            justification="news lookup",
            skill_name="NewsClient",
            client=client,
            timeout_seconds=float(getattr(provider_config, "timeout_seconds", 5.0)),
        )
        self._credentials: CredentialBackend = credential_store
        self._config: Any = provider_config

    @property
    def default_topic(self) -> str:
        """Return the configured default topic."""
        return str(getattr(self._config, "default_topic", ""))

    async def fetch(
        self,
        topic: str | None = None,
        max_items: int = _DEFAULT_MAX_ITEMS,
    ) -> list[dict[str, Any]]:
        """Return the top ``max_items`` headlines for ``topic``.

        ``topic`` is forwarded as the ``q`` query parameter. ``None`` (or
        an empty string) falls back to the configured ``default_topic``.

        ``max_items`` is clamped to the inclusive range ``[1, 10]``: zero
        or negative values are treated as the default (5), and any value
        above ten is capped at ten per Requirement 7.3.

        Returns:
            A list of ``{title, source, url, published_at, description}``
            dicts ordered as returned by NewsAPI. Empty list if NewsAPI
            reports zero matching articles.

        Raises:
            ProviderError(missing_credentials): the configured credential
                is absent from the credential store.
            ProviderError(provider_unavailable): NewsAPI returned a non-2xx
                response, the request timed out, or the body could not be
                parsed.
        """
        resolved_topic = (topic or self.default_topic).strip()
        if not resolved_topic:
            raise ProviderError(
                "provider_unavailable",
                "no topic provided and no default_topic configured",
                provider=self.PROVIDER_NAME,
            )

        clamped = self._clamp_max_items(max_items)
        api_key = self._read_api_key()

        params = {
            "q": resolved_topic,
            "pageSize": clamped,
        }
        headers = {"X-Api-Key": api_key}

        try:
            response = await self.get(
                f"{_NEWSAPI_BASE_URL}/v2/top-headlines",
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "provider_unavailable",
                f"NewsAPI request timed out: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"NewsAPI request failed: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                f"NewsAPI returned HTTP {response.status_code}",
                provider=self.PROVIDER_NAME,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"NewsAPI returned non-JSON body: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        # NewsAPI uses ``status`` to communicate logical errors even on a
        # 200 response (e.g., ``"error"``, ``"apiKeyInvalid"``).
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise ProviderError(
                "provider_unavailable",
                f"NewsAPI returned non-ok payload: {payload!r}",
                provider=self.PROVIDER_NAME,
            )

        articles_raw = payload.get("articles", [])
        if not isinstance(articles_raw, list):
            return []
        articles_raw = articles_raw[:clamped]
        return [self._normalise_article(item) for item in articles_raw if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_max_items(value: int) -> int:
        """Clamp ``max_items`` to ``[1, 10]`` per Requirement 7.3."""
        if not isinstance(value, int) or isinstance(value, bool):
            return _DEFAULT_MAX_ITEMS
        if value <= 0:
            return _DEFAULT_MAX_ITEMS
        if value > _MAX_ITEMS_CAP:
            return _MAX_ITEMS_CAP
        return value

    @staticmethod
    def _normalise_article(article: dict[str, Any]) -> dict[str, Any]:
        """Reduce a NewsAPI article dict to the fields downstream needs.

        Keeping the shape narrow makes the LLM-visible payload predictable
        and avoids leaking auxiliary metadata that NewsAPI returns
        (``urlToImage``, ``content`` excerpts) into the audit log.
        """
        source = article.get("source")
        source_name: str | None = None
        if isinstance(source, dict):
            name = source.get("name")
            source_name = str(name) if isinstance(name, str) else None

        return {
            "title": article.get("title"),
            "source": source_name,
            "url": article.get("url"),
            "published_at": article.get("publishedAt"),
            "description": article.get("description"),
        }

    def _read_api_key(self) -> str:
        credential_name = str(getattr(self._config, "api_key_credential", ""))
        if not credential_name:
            raise ProviderError(
                "missing_credentials",
                "providers.news.api_key_credential is not configured",
                provider=self.PROVIDER_NAME,
            )
        try:
            value = self._credentials.get(credential_name)
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(
                "missing_credentials",
                f"unable to read credential {credential_name!r}: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        if not value:
            raise ProviderError(
                "missing_credentials",
                f"credential {credential_name!r} is not set in the credential store",
                provider=self.PROVIDER_NAME,
            )
        return value
