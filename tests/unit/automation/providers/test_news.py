"""Unit tests for ``jarvis.automation.providers.news.NewsClient``.

Validates: Requirements 5.6, 7.3, 7.4, 7.7
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.news import NewsClient
from jarvis.security.audit_log import AuditLog
from jarvis.utils.time_source import FakeTimeSource


@dataclass
class _StubConfig:
    api_key_credential: str = "news/api_key"
    default_topic: str = "technology"
    timeout_seconds: float = 5.0


class _DictBackend:
    def __init__(self, items: dict[str, str] | None = None) -> None:
        self._items: dict[str, str] = dict(items or {})

    def set(self, name: str, value: str) -> None:
        self._items[name] = value

    def get(self, name: str) -> str | None:
        return self._items.get(name)

    def delete(self, name: str) -> None:
        self._items.pop(name, None)

    def list_names(self) -> list[str]:
        return sorted(self._items)

    def wipe(self) -> None:
        self._items.clear()


@pytest.fixture()
def audit_log(tmp_path: Path) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=FakeTimeSource(now=datetime(2024, 1, 1, tzinfo=UTC)),
        run_id="news-test",
    )
    yield log
    log.close()


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _build_httpx(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _articles_payload(count: int) -> dict[str, Any]:
    return {
        "status": "ok",
        "totalResults": count,
        "articles": [
            {
                "source": {"id": None, "name": f"src-{i}"},
                "author": "a",
                "title": f"title-{i}",
                "description": f"desc-{i}",
                "url": f"https://example.invalid/{i}",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z",
                "content": "...",
            }
            for i in range(count)
        ],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fetch_returns_normalised_articles(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_articles_payload(5))

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "k"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        articles = _run(client.fetch("ai"))
    finally:
        _run(client.aclose())

    assert len(articles) == 5
    article = articles[0]
    assert set(article) == {"title", "source", "url", "published_at", "description"}
    assert article["source"] == "src-0"
    assert article["title"] == "title-0"


def test_max_items_capped_at_ten(audit_log: AuditLog) -> None:
    """Requirement 7.3 — ``max_items`` is capped at 10."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_articles_payload(15))

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "k"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        articles = _run(client.fetch("ai", max_items=42))
    finally:
        _run(client.aclose())

    assert len(articles) == 10
    # The pageSize parameter on the wire is also clamped.
    assert b"pageSize=10" in bytes(captured[0].url.query)


def test_default_topic_used_when_argument_is_none(audit_log: AuditLog) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_articles_payload(1))

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "k"}),
        provider_config=_StubConfig(default_topic="science"),
        client=_build_httpx(handler),
    )

    try:
        _run(client.fetch())
    finally:
        _run(client.aclose())

    assert b"q=science" in bytes(captured[0].url.query)


def test_api_key_passed_via_header_not_query(audit_log: AuditLog) -> None:
    """The API key must travel in ``X-Api-Key``, never in the query string."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_articles_payload(1))

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "supersecret"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        _run(client.fetch("ai"))
    finally:
        _run(client.aclose())

    assert captured[0].headers.get("X-Api-Key") == "supersecret"
    assert b"supersecret" not in bytes(captured[0].url.query)


# ---------------------------------------------------------------------------
# Failure mappings
# ---------------------------------------------------------------------------


def test_missing_credential_raises_missing_credentials(
    audit_log: AuditLog,
) -> None:
    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend(),
        provider_config=_StubConfig(),
    )
    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("ai"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "missing_credentials"


def test_logical_error_in_payload_maps_to_provider_unavailable(
    audit_log: AuditLog,
) -> None:
    """NewsAPI uses ``status`` even on a 200 response to flag errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "error", "code": "apiKeyInvalid", "message": "no"},
        )

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "k"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("ai"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"


def test_4xx_response_maps_to_provider_unavailable(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "rate limit"})

    client = NewsClient(
        audit_log=audit_log,
        network_allowlist=["newsapi.org"],
        credential_store=_DictBackend({"news/api_key": "k"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("ai"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"
