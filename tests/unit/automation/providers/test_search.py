"""Unit tests for ``jarvis.automation.providers.search.WebSearchClient``.

Covers the four design-level promises Task 17.3 makes:

* Provider switching via ``provider_config.provider`` selects between
  Tavily, Bing, and DuckDuckGo with the documented per-provider request
  shapes (Requirements 3.1, 3.2).
* ``max_results`` is clamped to ``[1, 10]`` regardless of caller input or
  upstream over-delivery (Requirement 3.1).
* Each provider's response is normalised to the uniform
  ``{"title", "url", "snippet"}`` shape (Requirement 3.2).
* Zero results / non-2xx responses surface in the way the Skill layer
  expects (Requirement 3.4): empty list on no matches, ``WebSearchError``
  with the upstream status code on 4xx.

The :class:`httpx.MockTransport` pattern from ``test_http.py`` is reused
so every test is hermetic and fast.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from datetime import UTC, datetime
import json as _json
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.automation.providers.search import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_HARD_CAP,
    WebSearchClient,
    WebSearchError,
)
from jarvis.config.schema import ProvidersSearchConfig
from jarvis.security.audit_log import AuditLog
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_source() -> FakeTimeSource:
    return FakeTimeSource(now=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))


@pytest.fixture()
def audit_log(tmp_path: Path, time_source: FakeTimeSource) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=time_source,
        run_id="search-test",
    )
    yield log
    log.close()


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _StubCredentialStore:
    """Minimal :class:`CredentialBackend` returning a fixed value.

    Tests that exercise an unconfigured / failing credential lookup pass
    ``None`` for the credential store directly instead of stubbing this
    helper, so the implementation is also exercised against the documented
    ``credential_store=None`` shape.
    """

    def __init__(self, value: str | None = "TEST-API-KEY") -> None:
        self._value = value
        self.calls: list[str] = []

    def set(self, name: str, value: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def get(self, name: str) -> str | None:
        self.calls.append(name)
        return self._value

    def delete(self, name: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def list_names(self) -> list[str]:  # pragma: no cover - unused
        return []

    def wipe(self) -> None:  # pragma: no cover - unused
        return


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_unknown_provider(audit_log: AuditLog) -> None:
    with pytest.raises(ValueError, match=r"provider_config\.provider"):
        WebSearchClient(
            audit_log=audit_log,
            network_allowlist=["api.tavily.com"],
            credential_store=_StubCredentialStore(),
            provider_config={"provider": "yahoo", "api_key_credential": "search/key"},
        )


def test_constructor_accepts_dict_provider_config(audit_log: AuditLog) -> None:
    pc = WebSearchClient(
        audit_log=audit_log,
        network_allowlist=["api.tavily.com"],
        credential_store=_StubCredentialStore(),
        provider_config={
            "provider": "tavily",
            "api_key_credential": "search/api_key",
        },
    )
    try:
        assert pc.provider == "tavily"
        assert pc.api_key_credential == "search/api_key"
        assert pc.skill_name == "WebSearchClient"
    finally:
        _run(pc.aclose())


def test_constructor_accepts_pydantic_provider_config(audit_log: AuditLog) -> None:
    cfg = ProvidersSearchConfig(provider="bing", api_key_credential="search/bing_key")
    pc = WebSearchClient(
        audit_log=audit_log,
        network_allowlist=["api.bing.microsoft.com"],
        credential_store=_StubCredentialStore(),
        provider_config=cfg,
    )
    try:
        assert pc.provider == "bing"
        assert pc.api_key_credential == "search/bing_key"
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _build_search_client(
    audit_log: AuditLog,
    handler: Any,
    *,
    provider: str = "tavily",
    allowlist: list[str] | None = None,
    credential_store: Any = None,
) -> WebSearchClient:
    if allowlist is None:
        allowlist = {
            "tavily": ["api.tavily.com"],
            "bing": ["api.bing.microsoft.com"],
            "duckduckgo": ["api.duckduckgo.com"],
        }[provider]
    if credential_store is None and provider != "duckduckgo":
        credential_store = _StubCredentialStore()
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return WebSearchClient(
        audit_log=audit_log,
        network_allowlist=allowlist,
        credential_store=credential_store,
        provider_config={
            "provider": provider,
            "api_key_credential": "search/api_key",
        },
        client=client,
    )


def test_search_rejects_empty_query(audit_log: AuditLog) -> None:
    pc = _build_search_client(audit_log, handler=lambda r: httpx.Response(200, json={}))
    try:
        with pytest.raises(ValueError, match="non-empty"):
            _run(pc.search("   "))
    finally:
        _run(pc.aclose())


def test_search_rejects_non_string_query(audit_log: AuditLog) -> None:
    pc = _build_search_client(audit_log, handler=lambda r: httpx.Response(200, json={}))
    try:
        with pytest.raises(TypeError):
            _run(pc.search(123))  # type: ignore[arg-type]
    finally:
        _run(pc.aclose())


def test_search_rejects_bool_max_results(audit_log: AuditLog) -> None:
    pc = _build_search_client(audit_log, handler=lambda r: httpx.Response(200, json={}))
    try:
        with pytest.raises(TypeError):
            _run(pc.search("python", max_results=True))  # type: ignore[arg-type]
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# Tavily provider
# ---------------------------------------------------------------------------


def test_tavily_search_posts_json_body_with_api_key(audit_log: AuditLog) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Python.org",
                        "url": "https://python.org",
                        "content": "The official Python homepage.",
                    },
                    {
                        "title": "PEP 8",
                        "url": "https://peps.python.org/pep-0008/",
                        "content": "Style guide.",
                    },
                ]
            },
        )

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        results = _run(pc.search("python", max_results=2))
    finally:
        _run(pc.aclose())

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    assert req.url.host == "api.tavily.com"
    body = req.read()
    payload = _json.loads(body)
    assert payload["api_key"] == "TEST-API-KEY"
    assert payload["query"] == "python"
    assert payload["max_results"] == 2

    assert results == [
        {
            "title": "Python.org",
            "url": "https://python.org",
            "snippet": "The official Python homepage.",
        },
        {
            "title": "PEP 8",
            "url": "https://peps.python.org/pep-0008/",
            "snippet": "Style guide.",
        },
    ]


def test_tavily_search_missing_api_key_raises_401(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("provider should not be hit when key is missing")

    pc = _build_search_client(
        audit_log,
        handler,
        provider="tavily",
        credential_store=_StubCredentialStore(value=None),
    )
    try:
        with pytest.raises(WebSearchError) as excinfo:
            _run(pc.search("anything"))
        assert excinfo.value.status_code == 401
        assert excinfo.value.provider == "tavily"
    finally:
        _run(pc.aclose())


def test_tavily_search_returns_empty_on_no_results(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        results = _run(pc.search("query"))
        assert results == []
    finally:
        _run(pc.aclose())


def test_tavily_search_normalizes_missing_fields(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": None, "url": "https://x.example", "content": None},
                    {"url": "https://y.example"},
                ]
            },
        )

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        results = _run(pc.search("query"))
    finally:
        _run(pc.aclose())

    assert results == [
        {"title": "", "url": "https://x.example", "snippet": ""},
        {"title": "", "url": "https://y.example", "snippet": ""},
    ]


# ---------------------------------------------------------------------------
# Bing provider
# ---------------------------------------------------------------------------


def test_bing_search_uses_subscription_key_header(audit_log: AuditLog) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "webPages": {
                    "value": [
                        {
                            "name": "Wiki",
                            "url": "https://en.wikipedia.org/wiki/Python",
                            "snippet": "Python is a programming language.",
                        }
                    ]
                }
            },
        )

    pc = _build_search_client(audit_log, handler, provider="bing")
    try:
        results = _run(pc.search("python", max_results=3))
    finally:
        _run(pc.aclose())

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "GET"
    assert req.url.host == "api.bing.microsoft.com"
    assert req.url.params["q"] == "python"
    assert req.url.params["count"] == "3"
    assert req.headers["Ocp-Apim-Subscription-Key"] == "TEST-API-KEY"

    assert results == [
        {
            "title": "Wiki",
            "url": "https://en.wikipedia.org/wiki/Python",
            "snippet": "Python is a programming language.",
        }
    ]


def test_bing_search_returns_empty_on_missing_webpages(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    pc = _build_search_client(audit_log, handler, provider="bing")
    try:
        results = _run(pc.search("python"))
        assert results == []
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# DuckDuckGo provider
# ---------------------------------------------------------------------------


def test_duckduckgo_search_works_without_credentials(audit_log: AuditLog) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "RelatedTopics": [
                    {
                        "Text": "Python (programming language)",
                        "FirstURL": "https://duckduckgo.com/Python",
                    },
                    {
                        "Topics": [
                            {
                                "Text": "Python software foundation",
                                "FirstURL": "https://duckduckgo.com/PSF",
                            }
                        ]
                    },
                ]
            },
        )

    pc = _build_search_client(
        audit_log, handler, provider="duckduckgo", credential_store=None
    )
    try:
        results = _run(pc.search("python", max_results=5))
    finally:
        _run(pc.aclose())

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "GET"
    assert req.url.host == "api.duckduckgo.com"
    assert req.url.params["format"] == "json"
    # No auth header is required for DuckDuckGo.
    assert "Ocp-Apim-Subscription-Key" not in req.headers
    assert "Authorization" not in req.headers

    assert results == [
        {
            "title": "Python (programming language)",
            "url": "https://duckduckgo.com/Python",
            "snippet": "Python (programming language)",
        },
        {
            "title": "Python software foundation",
            "url": "https://duckduckgo.com/PSF",
            "snippet": "Python software foundation",
        },
    ]


# ---------------------------------------------------------------------------
# max_results clamping (Requirement 3.1)
# ---------------------------------------------------------------------------


def test_max_results_default_is_five(audit_log: AuditLog) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_json.loads(request.read()))
        return httpx.Response(200, json={"results": []})

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        _run(pc.search("query"))
    finally:
        _run(pc.aclose())
    assert seen[0]["max_results"] == DEFAULT_MAX_RESULTS == 5


def test_max_results_capped_at_ten(audit_log: AuditLog) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_json.loads(request.read()))
        return httpx.Response(200, json={"results": []})

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        _run(pc.search("query", max_results=99))
    finally:
        _run(pc.aclose())
    assert seen[0]["max_results"] == MAX_RESULTS_HARD_CAP == 10


def test_max_results_floor_one(audit_log: AuditLog) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_json.loads(request.read()))
        return httpx.Response(200, json={"results": []})

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        _run(pc.search("query", max_results=0))
    finally:
        _run(pc.aclose())
    assert seen[0]["max_results"] == 1


def test_response_truncated_to_clamp(audit_log: AuditLog) -> None:
    """Even if the upstream over-delivers, we never return more than the cap."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"r{i}", "url": f"https://x/{i}", "content": "s"}
                    for i in range(50)
                ]
            },
        )

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        results = _run(pc.search("query", max_results=3))
    finally:
        _run(pc.aclose())
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Error / status handling (Requirement 3.4, 7.7)
# ---------------------------------------------------------------------------


def test_non_2xx_raises_websearcherror_with_status_code(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        with pytest.raises(WebSearchError) as excinfo:
            _run(pc.search("query"))
        assert excinfo.value.status_code == 403
        assert excinfo.value.provider == "tavily"
    finally:
        _run(pc.aclose())


def test_malformed_json_treated_as_no_results(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    pc = _build_search_client(audit_log, handler, provider="bing")
    try:
        results = _run(pc.search("query"))
        assert results == []
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# Audit log integration (Requirements 13.4, 13.6)
# ---------------------------------------------------------------------------


def test_search_records_one_network_egress_per_call(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    pc = _build_search_client(audit_log, handler, provider="tavily")
    try:
        _run(pc.search("query"))
    finally:
        _run(pc.aclose())

    entries = audit_log.entries()
    egress = [e for e in entries if e.kind == "network_egress"]
    assert len(egress) == 1
    assert egress[0].skill == "WebSearchClient"
    assert egress[0].justification == "web search"
    assert egress[0].destination == "https://api.tavily.com"


def test_search_blocked_destination_records_policy_violation(
    audit_log: AuditLog,
) -> None:
    """An allowlist that omits the provider host blocks the call."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("blocked request must not reach the wire")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    pc = WebSearchClient(
        audit_log=audit_log,
        network_allowlist=["api.allowed.invalid"],  # tavily host NOT on list
        credential_store=_StubCredentialStore(),
        provider_config={
            "provider": "tavily",
            "api_key_credential": "search/api_key",
        },
        client=client,
    )
    try:
        with pytest.raises(NetworkPolicyViolation):
            _run(pc.search("query"))
    finally:
        _run(pc.aclose())

    entries = audit_log.entries()
    violations = [e for e in entries if e.kind == "policy_violation"]
    assert len(violations) == 1
    assert violations[0].skill == "WebSearchClient"
