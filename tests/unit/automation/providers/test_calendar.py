"""Unit tests for ``jarvis.automation.providers.calendar.CalendarClient``.

Validates: Requirements 5.6, 7.5, 7.6, 7.7
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.calendar import CalendarClient
from jarvis.automation.providers.errors import ProviderError
from jarvis.security.audit_log import AuditLog
from jarvis.utils.time_source import FakeTimeSource


@dataclass
class _StubConfig:
    oauth_credential: str = "calendar/oauth_token"
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
        run_id="calendar-test",
    )
    yield log
    log.close()


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _build_httpx(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# list_range
# ---------------------------------------------------------------------------


def test_list_range_returns_normalised_events(audit_log: AuditLog) -> None:
    response_payload = {
        "items": [
            {
                "id": "abc",
                "summary": "Standup",
                "start": {"dateTime": "2024-01-01T09:00:00Z"},
                "end": {"dateTime": "2024-01-01T09:30:00Z"},
                "htmlLink": "https://cal/x",
                "status": "confirmed",
                "extra": "ignored",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_payload)

    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    try:
        events = _run(client.list_range(start, end))
    finally:
        _run(client.aclose())

    assert len(events) == 1
    event = events[0]
    assert event["id"] == "abc"
    assert event["title"] == "Standup"
    assert event["start"] == "2024-01-01T09:00:00Z"
    assert event["end"] == "2024-01-01T09:30:00Z"
    assert event["html_link"] == "https://cal/x"
    assert event["status"] == "confirmed"
    assert "extra" not in event


def test_list_range_passes_bearer_token(audit_log: AuditLog) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"items": []})

    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok-x"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        _run(
            client.list_range(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            )
        )
    finally:
        _run(client.aclose())

    assert captured[0].headers.get("Authorization") == "Bearer tok-x"


def test_list_range_naive_datetime_rejected(audit_log: AuditLog) -> None:
    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
    )
    try:
        with pytest.raises(ValueError, match="timezone-aware"):
            _run(
                client.list_range(
                    datetime(2024, 1, 1),  # naive
                    datetime(2024, 1, 2, tzinfo=UTC),
                )
            )
    finally:
        _run(client.aclose())


def test_list_range_end_before_start_rejected(audit_log: AuditLog) -> None:
    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
    )
    try:
        with pytest.raises(ValueError, match=r"end must be"):
            _run(
                client.list_range(
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                )
            )
    finally:
        _run(client.aclose())


# ---------------------------------------------------------------------------
# create_event
# ---------------------------------------------------------------------------


def test_create_event_posts_summary_and_bounds(audit_log: AuditLog) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "new-id",
                "summary": "New thing",
                "start": {"dateTime": "2024-01-01T10:00:00Z"},
                "end": {"dateTime": "2024-01-01T11:00:00Z"},
                "htmlLink": "https://cal/new",
                "status": "confirmed",
            },
        )

    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    start = datetime(2024, 1, 1, 10, tzinfo=UTC)
    end = start + timedelta(hours=1)
    try:
        event = _run(client.create_event("New thing", start, end))
    finally:
        _run(client.aclose())

    assert event["id"] == "new-id"
    assert event["title"] == "New thing"
    assert captured[0].method == "POST"
    body = captured[0].read()
    assert b"New thing" in body
    assert b"2024-01-01T10:00:00" in body


def test_create_event_rejects_empty_title(audit_log: AuditLog) -> None:
    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
    )
    try:
        with pytest.raises(ValueError, match="title"):
            _run(
                client.create_event(
                    "   ",
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 1, tzinfo=UTC),
                )
            )
    finally:
        _run(client.aclose())


def test_create_event_end_must_be_after_start(audit_log: AuditLog) -> None:
    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
    )
    try:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="end must be"):
            _run(client.create_event("Title", ts, ts))
    finally:
        _run(client.aclose())


# ---------------------------------------------------------------------------
# Credentials / failure mappings
# ---------------------------------------------------------------------------


def test_missing_oauth_token_raises_missing_credentials(
    audit_log: AuditLog,
) -> None:
    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend(),
        provider_config=_StubConfig(),
    )
    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(
                client.list_range(
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                )
            )
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "missing_credentials"


def test_4xx_response_maps_to_provider_unavailable(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "denied"}})

    client = CalendarClient(
        audit_log=audit_log,
        network_allowlist=["www.googleapis.com"],
        credential_store=_DictBackend({"calendar/oauth_token": "tok"}),
        provider_config=_StubConfig(),
        client=_build_httpx(handler),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(
                client.list_range(
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                )
            )
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"
