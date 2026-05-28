"""Unit tests for ``jarvis.automation.providers.weather.WeatherClient``.

Validates: Requirements 5.6, 7.1, 7.2, 7.7
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
from jarvis.automation.providers.weather import WeatherClient
from jarvis.security.audit_log import AuditLog
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubConfig:
    api_key_credential: str = "weather/api_key"
    default_location: str = "Bandung,ID"
    timeout_seconds: float = 5.0


class _DictBackend:
    """In-memory ``CredentialBackend`` for tests."""

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
def time_source() -> FakeTimeSource:
    return FakeTimeSource(now=datetime(2024, 6, 1, tzinfo=UTC))


@pytest.fixture()
def audit_log(tmp_path: Path, time_source: FakeTimeSource) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=time_source,
        run_id="weather-test",
    )
    yield log
    log.close()


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _build_httpx_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _success_handler(
    *,
    current: dict[str, Any],
    forecast: dict[str, Any],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/weather"):
            return httpx.Response(200, json=current)
        if request.url.path.endswith("/forecast"):
            return httpx.Response(200, json=forecast)
        return httpx.Response(404)

    return handler


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fetch_returns_current_and_24h_forecast(audit_log: AuditLog) -> None:
    """24-hour forecast is exactly 8 entries (3 h x 8 = 24 h)."""
    forecast_entries = [{"dt": i, "main": {"temp": 20 + i}} for i in range(40)]
    handler = _success_handler(
        current={"coord": {"lat": -6.9, "lon": 107.6}, "main": {"temp": 25}},
        forecast={"list": forecast_entries},
    )
    backend = _DictBackend({"weather/api_key": "key123"})
    config = _StubConfig()

    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=backend,
        provider_config=config,
        client=_build_httpx_client(handler),
    )

    try:
        result = _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    assert result["location"] == "Jakarta,ID"
    assert result["current"]["main"]["temp"] == 25
    assert len(result["forecast"]) == 8
    assert result["forecast"][0]["dt"] == 0
    assert result["forecast"][7]["dt"] == 7


def test_fetch_uses_default_location_when_none_provided(
    audit_log: AuditLog,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path.endswith("/weather"):
            return httpx.Response(
                200, json={"coord": {"lat": 1.0, "lon": 2.0}}
            )
        return httpx.Response(200, json={"list": []})

    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(default_location="Bandung,ID"),
        client=_build_httpx_client(handler),
    )

    try:
        _ = _run(client.fetch())
    finally:
        _run(client.aclose())

    # The first call (current) must have used the configured default.
    assert any(
        b"q=Bandung" in bytes(req.url.query) for req in captured
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_missing_credential_raises_provider_error(audit_log: AuditLog) -> None:
    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend(),  # empty
        provider_config=_StubConfig(),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "missing_credentials"


def test_unconfigured_credential_name_raises_missing_credentials(
    audit_log: AuditLog,
) -> None:
    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(api_key_credential=""),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "missing_credentials"


# ---------------------------------------------------------------------------
# Provider unavailable mappings
# ---------------------------------------------------------------------------


def test_upstream_4xx_maps_to_provider_unavailable(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad key"})

    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(),
        client=_build_httpx_client(handler),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"


def test_missing_coord_in_current_response_raises(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"main": {"temp": 25}})  # no coord

    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(),
        client=_build_httpx_client(handler),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"


def test_empty_location_and_no_default_raises(audit_log: AuditLog) -> None:
    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(default_location=""),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.fetch(""))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"


# ---------------------------------------------------------------------------
# Audit / allowlist integration
# ---------------------------------------------------------------------------


def test_allowed_destination_records_two_egress_rows(
    audit_log: AuditLog,
) -> None:
    """Each of the two HTTP calls (current + forecast) is audited once."""
    handler = _success_handler(
        current={"coord": {"lat": 1.0, "lon": 2.0}},
        forecast={"list": [{"dt": 1}]},
    )
    client = WeatherClient(
        audit_log=audit_log,
        network_allowlist=["api.openweathermap.org"],
        credential_store=_DictBackend({"weather/api_key": "key"}),
        provider_config=_StubConfig(),
        client=_build_httpx_client(handler),
    )

    try:
        _run(client.fetch("Jakarta,ID"))
    finally:
        _run(client.aclose())

    rows = audit_log.entries()
    egress = [r for r in rows if r.kind == "network_egress"]
    assert len(egress) == 2
    assert all(r.skill == "WeatherClient" for r in egress)
    assert all(r.justification == "weather lookup" for r in egress)
