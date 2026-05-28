"""Unit tests for ``jarvis.automation.providers.http.ProviderClient``.

Covers the three responsibilities the base layer is documented to own:

* **Read-timeout discipline (Requirement 7.7).** The default 5 s timeout is
  applied when no client is injected, and the constructor rejects
  non-positive values so a misconfiguration cannot silently disable the
  budget.
* **Exponential-backoff retries on transient failure.** Successive 5xx
  responses and :class:`httpx.TimeoutException` failures are retried up to
  ``max_attempts`` times; the eventual outcome (final 5xx response or
  reraised timeout) is surfaced to the caller. Successful responses do
  not trigger retries, and 4xx responses bypass retry entirely.
* **Network egress audit + allowlist enforcement (Requirements 13.4,
  13.6).** Allowed destinations produce exactly one ``network_egress``
  audit row per logical call (regardless of retry count) carrying the
  configured justification; blocked destinations produce a
  ``policy_violation`` row and raise :class:`NetworkPolicyViolation`
  *before* the request reaches the wire.

The tests use :class:`httpx.MockTransport` to drive the underlying
:class:`httpx.AsyncClient` rather than reaching for the network, mirroring
the pattern already established for ``OllamaBackend``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.automation.providers.http import (
    DEFAULT_PROVIDER_TIMEOUT_S,
    NetworkPolicyViolation,
    ProviderClient,
)
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
        run_id="provider-test",
    )
    yield log
    log.close()


def _run(coro: Awaitable[Any]) -> Any:
    """Synchronously execute ``coro`` on a fresh event loop.

    Mirrors the helper used in ``test_audit_log.py``; avoids the
    pytest-asyncio per-test loop machinery so the file remains a plain
    sync pytest module.
    """
    return asyncio.run(coro)  # type: ignore[arg-type]


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
) -> httpx.AsyncClient:
    """Wrap ``handler`` in an :class:`httpx.AsyncClient` via MockTransport."""
    # MockTransport accepts both sync and async handlers at runtime;
    # its overload set is just narrower than the union we want to use.
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_constructor_defaults_timeout_to_five_seconds(audit_log: AuditLog) -> None:
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
    )
    try:
        assert pc.timeout_seconds == DEFAULT_PROVIDER_TIMEOUT_S == 5.0
        assert pc.justification == "weather lookup"
        # Default skill_name falls back to the runtime class name so that
        # subclasses don't need to repeat the identifier.
        assert pc.skill_name == "ProviderClient"
    finally:
        _run(pc.aclose())


def test_constructor_rejects_empty_justification(audit_log: AuditLog) -> None:
    with pytest.raises(ValueError, match="justification"):
        ProviderClient(
            audit_log=audit_log,
            network_allowlist=["api.example.invalid"],
            justification="",
        )


def test_constructor_rejects_non_positive_timeout(audit_log: AuditLog) -> None:
    with pytest.raises(ValueError, match="timeout"):
        ProviderClient(
            audit_log=audit_log,
            network_allowlist=["api.example.invalid"],
            justification="weather lookup",
            timeout_seconds=0,
        )


def test_constructor_rejects_zero_max_attempts(audit_log: AuditLog) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ProviderClient(
            audit_log=audit_log,
            network_allowlist=["api.example.invalid"],
            justification="weather lookup",
            max_attempts=0,
        )


def test_allowlist_is_lowercased_and_deduplicated(audit_log: AuditLog) -> None:
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["API.Example.Invalid", "api.example.invalid", ""],
        justification="weather lookup",
    )
    try:
        assert pc.allowlist == frozenset({"api.example.invalid"})
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# Allowlist enforcement (Requirement 13.6)
# ---------------------------------------------------------------------------


def test_blocked_destination_records_policy_violation_and_raises(
    audit_log: AuditLog,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.allowed.invalid"],
        justification="weather lookup",
        client=client,
    )

    try:
        async def driver() -> None:
            await pc.get("https://api.attacker.invalid/v1/data?token=secret")

        with pytest.raises(NetworkPolicyViolation) as exc_info:
            _run(driver())

        # The wire was never touched.
        assert calls == []

        # The exception carries the parsed host and a redacted destination.
        assert exc_info.value.host == "api.attacker.invalid"
        assert exc_info.value.destination == "https://api.attacker.invalid"

        # Exactly one audit row, of kind ``policy_violation``, with no
        # query string leaked into the destination column.
        entries = audit_log.entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.kind == "policy_violation"
        assert entry.skill == "ProviderClient"
        assert entry.destination == "https://api.attacker.invalid"
        assert entry.outcome == "blocked"
        assert "api.attacker.invalid" in (entry.justification or "")
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_relative_url_without_host_is_blocked(audit_log: AuditLog) -> None:
    """A URL we cannot identify cannot be meaningfully audited; block it."""
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.allowed.invalid"],
        justification="weather lookup",
    )
    try:
        async def driver() -> None:
            await pc.get("/relative/path")

        with pytest.raises(NetworkPolicyViolation):
            _run(driver())

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].kind == "policy_violation"
    finally:
        _run(pc.aclose())


def test_empty_allowlist_blocks_every_destination(audit_log: AuditLog) -> None:
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=[],
        justification="weather lookup",
    )
    try:
        async def driver() -> None:
            await pc.get("https://api.example.invalid/v1/anything")

        with pytest.raises(NetworkPolicyViolation):
            _run(driver())

        assert audit_log.count() == 1
        assert audit_log.entries()[0].kind == "policy_violation"
    finally:
        _run(pc.aclose())


# ---------------------------------------------------------------------------
# Network egress audit (Requirement 13.4)
# ---------------------------------------------------------------------------


def test_allowed_destination_records_single_network_egress_row(
    audit_log: AuditLog,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        skill_name="WeatherClient",
        client=client,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get(
                "https://api.example.invalid/v1/forecast?lat=1&lon=2"
            )

        response = _run(driver())
        assert response.status_code == 200

        # The wire saw exactly the request we issued.
        assert len(captured) == 1
        assert str(captured[0].url) == (
            "https://api.example.invalid/v1/forecast?lat=1&lon=2"
        )

        # And the audit log carries a single ``network_egress`` row with
        # the configured justification, scoped to the subclass skill name.
        entries = audit_log.entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.kind == "network_egress"
        assert entry.skill == "WeatherClient"
        # Path / query are stripped to avoid leaking secrets.
        assert entry.destination == "https://api.example.invalid"
        assert entry.justification == "weather lookup"
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_retries_do_not_multiply_audit_rows(audit_log: AuditLog) -> None:
    """One logical call ⇒ one ``network_egress`` row, even with retries."""
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 3:
            return httpx.Response(503, text="upstream busy")
        return httpx.Response(200, json={"ok": True})

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=3,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get("https://api.example.invalid/v1/forecast")

        response = _run(driver())
        assert response.status_code == 200
        assert counter["n"] == 3  # two 5xx + one OK

        entries = audit_log.entries()
        assert len(entries) == 1
        assert entries[0].kind == "network_egress"
    finally:
        _run(pc.aclose())
        _run(client.aclose())


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_5xx_responses_are_retried_up_to_max_attempts(
    audit_log: AuditLog,
) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(503, text="always busy")

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=3,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get("https://api.example.invalid/v1/forecast")

        # When the budget is exhausted we get the *final* 5xx response
        # back so the caller can map it to ``provider_unavailable``.
        response = _run(driver())
        assert response.status_code == 503
        assert counter["n"] == 3
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_4xx_responses_bypass_retry(audit_log: AuditLog) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(404, text="not found")

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=3,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get("https://api.example.invalid/v1/forecast")

        response = _run(driver())
        assert response.status_code == 404
        assert counter["n"] == 1  # 4xx is not retryable
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_timeouts_are_retried_then_reraised(audit_log: AuditLog) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        raise httpx.ReadTimeout("upstream stalled", request=request)

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=3,
    )

    try:
        async def driver() -> None:
            await pc.get("https://api.example.invalid/v1/forecast")

        with pytest.raises(httpx.ReadTimeout):
            _run(driver())

        assert counter["n"] == 3  # tried, retried, retried
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_successful_first_attempt_does_not_retry(audit_log: AuditLog) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, json={"ok": True})

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=5,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get("https://api.example.invalid/v1/forecast")

        response = _run(driver())
        assert response.status_code == 200
        assert counter["n"] == 1
    finally:
        _run(pc.aclose())
        _run(client.aclose())


def test_max_attempts_one_disables_retries(audit_log: AuditLog) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(503, text="busy")

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
        max_attempts=1,
    )

    try:
        async def driver() -> httpx.Response:
            return await pc.get("https://api.example.invalid/v1/forecast")

        response = _run(driver())
        assert response.status_code == 503
        assert counter["n"] == 1
    finally:
        _run(pc.aclose())
        _run(client.aclose())


# ---------------------------------------------------------------------------
# HTTP method surface
# ---------------------------------------------------------------------------


def test_method_helpers_dispatch_with_correct_verb(audit_log: AuditLog) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200)

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
    )

    try:
        async def driver() -> None:
            await pc.get("https://api.example.invalid/")
            await pc.post("https://api.example.invalid/", json={})
            await pc.put("https://api.example.invalid/", json={})
            await pc.patch("https://api.example.invalid/", json={})
            await pc.delete("https://api.example.invalid/")
            await pc.head("https://api.example.invalid/")

        _run(driver())
        assert seen == ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
    finally:
        _run(pc.aclose())
        _run(client.aclose())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_aclose_does_not_close_injected_client(audit_log: AuditLog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    client = _build_client(handler)
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
        client=client,
    )

    _run(pc.aclose())
    # The injected client was *not* closed — caller still owns it.
    assert client.is_closed is False
    _run(client.aclose())


def test_request_after_close_raises(audit_log: AuditLog) -> None:
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
    )
    _run(pc.aclose())

    async def driver() -> None:
        await pc.get("https://api.example.invalid/")

    with pytest.raises(RuntimeError, match="closed"):
        _run(driver())


def test_async_context_manager_closes_owned_client(audit_log: AuditLog) -> None:
    async def driver() -> ProviderClient:
        async with ProviderClient(
            audit_log=audit_log,
            network_allowlist=["api.example.invalid"],
            justification="weather lookup",
        ) as pc:
            return pc

    pc = _run(driver())

    async def post_close() -> None:
        await pc.get("https://api.example.invalid/")

    with pytest.raises(RuntimeError, match="closed"):
        _run(post_close())


def test_aclose_is_idempotent(audit_log: AuditLog) -> None:
    pc = ProviderClient(
        audit_log=audit_log,
        network_allowlist=["api.example.invalid"],
        justification="weather lookup",
    )
    _run(pc.aclose())
    _run(pc.aclose())  # must not raise
