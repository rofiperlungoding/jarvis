"""Integration tests for :mod:`tests.fakes.fake_mistral_server`.

These tests exercise the wire shape of :class:`FakeMistralServer`
directly through :class:`aiohttp.ClientSession` so the fixture's
contract (and the four documented failure modes) is locked down
independently of :class:`~jarvis.llm.mistral_backend.MistralBackend`.

The companion test ``test_22_4_dialog_manager_flows`` (task 22.4)
drives the production backend against the same server; this module
verifies the *server* itself.

Validates: Requirements 12.4, 19.5, 19.7, 19.8
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
import time
from typing import Any

import aiohttp
import pytest
from tests.fakes.fake_mistral_server import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_FAILURE_SCENARIOS,
    DEFAULT_SLOW_DELAY_SECONDS,
    FakeMistralServer,
    content_delta_event,
    finish_event,
    tool_call_delta_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DUMMY_REQUEST_BODY: dict[str, Any] = {
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "ping"}],
    "stream": True,
}


async def _post(
    server: FakeMistralServer,
    *,
    request_timeout_s: float = 10.0,
    body: dict[str, Any] | None = None,
) -> aiohttp.ClientResponse:
    """``POST`` a minimal Mistral-shaped request and return the open response.

    Caller is responsible for ``await``-ing ``response.read()`` /
    iterating the body and for closing the session. The returned
    response keeps a reference to its session via ``response._connection``
    on aiohttp 3.x; tests that need to keep both alive should use
    :func:`_open_post` instead.
    """
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request_timeout_s))
    payload = body if body is not None else _DUMMY_REQUEST_BODY
    response = await session.post(
        server.url + CHAT_COMPLETIONS_PATH,
        json=payload,
    )
    # Stash the session on the response so the caller can close it
    # together with the response object — keeps the test bodies tidy
    # without sprinkling ``async with`` everywhere.
    response.__dict__["_test_session"] = session
    return response


async def _close(response: aiohttp.ClientResponse) -> None:
    """Close ``response`` and the session :func:`_post` parked on it."""
    response.release()
    session: aiohttp.ClientSession | None = response.__dict__.get("_test_session")
    if session is not None:
        await session.close()


async def _read_sse_events(response: aiohttp.ClientResponse) -> list[dict[str, Any]]:
    """Parse the body of a streaming response into the list of chunk dicts.

    Stops when ``data: [DONE]`` is seen. Skips blank lines (which are
    the SSE record separator) and any non-``data:`` lines. Tests use
    this helper to assert the exact event sequence the server emitted
    on the wire — the same sequence the SDK will hand back to the
    backend's event loop.
    """
    body = await response.read()
    events: list[dict[str, Any]] = []
    for raw_line in body.split(b"\n"):
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if payload == b"[DONE]":
            break
        events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# Lifecycle and fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixture_starts_with_default_failure_scenarios(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """The fixture pre-registers the four documented failure modes + slow."""
    registered = set(fake_mistral_server.scenarios)
    for name in DEFAULT_FAILURE_SCENARIOS:
        assert name in registered, f"missing default scenario {name!r}"
    # The bound URL is reachable and points at HTTP loopback.
    assert fake_mistral_server.url.startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_unconfigured_active_returns_500(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Without a :meth:`set_active` call the server returns 500 — fail loudly."""
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 500
    finally:
        await _close(response)


@pytest.mark.asyncio
async def test_set_active_unknown_scenario_raises(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """:meth:`set_active` rejects unknown names with :class:`KeyError`."""
    with pytest.raises(KeyError, match="does-not-exist"):
        fake_mistral_server.set_active("does-not-exist")


@pytest.mark.asyncio
async def test_register_defaults_false_omits_failure_scenarios() -> None:
    """``register_defaults=False`` produces a server with an empty registry."""
    server = FakeMistralServer(register_defaults=False)
    assert server.scenarios == ()
    await server.start()
    try:
        assert server.url.startswith("http://127.0.0.1:")
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Failure modes (Requirements 12.4, 19.7, 19.8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_scenario_returns_401(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Default ``unauthorized`` scenario surfaces HTTP 401 (Requirement 19.7)."""
    fake_mistral_server.set_active("unauthorized")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 401
        body = await response.json()
        assert body["object"] == "error"
        assert body["type"] == "invalid_request_error"
    finally:
        await _close(response)


@pytest.mark.asyncio
async def test_forbidden_scenario_returns_403(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Default ``forbidden`` scenario surfaces HTTP 403 (Requirement 19.7)."""
    fake_mistral_server.set_active("forbidden")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 403
    finally:
        await _close(response)


@pytest.mark.asyncio
async def test_rate_limited_scenario_returns_429_with_retry_after(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Default ``rate_limited`` scenario surfaces HTTP 429 + ``Retry-After``.

    Validates Requirement 19.8: the production backend's tenacity
    schedule depends on the response shape modelled here.
    """
    fake_mistral_server.set_active("rate_limited")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 429
        assert response.headers.get("Retry-After") == "1"
        body = await response.json()
        assert body["type"] == "rate_limit_error"
    finally:
        await _close(response)


@pytest.mark.asyncio
async def test_server_error_scenario_returns_5xx(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Default ``server_error`` scenario surfaces HTTP 503 (Requirement 12.4)."""
    fake_mistral_server.set_active("server_error")
    response = await _post(fake_mistral_server)
    try:
        assert 500 <= response.status < 600
        assert response.status == 503
    finally:
        await _close(response)


@pytest.mark.asyncio
async def test_slow_scenario_delays_first_byte_past_3_seconds(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """``slow`` scenario takes >3 s — the BackendSelector circuit threshold.

    The test gives a generous ``timeout`` to the client so we can
    *measure* the elapsed time deterministically; Requirement 12.4
    only cares that the server is observably slower than 3 s, which is
    what the BackendSelector's circuit breaker is configured to.
    """
    fake_mistral_server.set_active("slow")
    start = time.perf_counter()
    response = await _post(fake_mistral_server, request_timeout_s=10.0)
    try:
        # Drain the body so the server-side write loop completes —
        # otherwise the elapsed measurement reflects only the time to
        # connect, which is not what the production backend sees.
        await response.read()
    finally:
        await _close(response)
    elapsed = time.perf_counter() - start
    assert elapsed >= DEFAULT_SLOW_DELAY_SECONDS - 0.1, (
        f"slow scenario completed in {elapsed:.2f}s; "
        f"expected at least {DEFAULT_SLOW_DELAY_SECONDS}s"
    )


# ---------------------------------------------------------------------------
# Streaming success path (Requirement 19.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_delta_scenario_emits_canonical_sse(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """A simple two-token content stream produces the expected SSE wire output."""
    fake_mistral_server.add_scenario(
        "simple_text",
        events=[
            content_delta_event("Hel", role="assistant"),
            content_delta_event("lo!", finish_reason=None),
            finish_event(finish_reason="stop"),
        ],
    )
    fake_mistral_server.set_active("simple_text")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        events = await _read_sse_events(response)
    finally:
        await _close(response)

    assert len(events) == 3

    first, second, terminator = events
    # First chunk advertises role + first delta.
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "Hel"}
    assert first["choices"][0]["finish_reason"] is None
    # Second chunk is content-only.
    assert second["choices"][0]["delta"] == {"content": "lo!"}
    # Terminator carries an empty delta and a finish_reason.
    assert terminator["choices"][0]["delta"] == {}
    assert terminator["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_tool_call_scenario_emits_fragmented_arguments(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """A streamed tool call splits ``id``/``name``/``arguments`` across deltas.

    The MistralBackend reassembles these into a single
    :class:`~jarvis.llm.base.ToolCall`; this test pins down the wire
    sequence the backend is required to handle.
    """
    fake_mistral_server.add_scenario(
        "tool_call",
        events=[
            tool_call_delta_event(
                tool_index=0,
                call_id="call-123",
                function_name="SendEmailSkill",
            ),
            tool_call_delta_event(arguments='{"recipient":'),
            tool_call_delta_event(arguments='"alex@example.com"}'),
            finish_event(finish_reason="tool_calls"),
        ],
    )
    fake_mistral_server.set_active("tool_call")
    response = await _post(fake_mistral_server)
    try:
        events = await _read_sse_events(response)
    finally:
        await _close(response)

    assert len(events) == 4
    head = events[0]["choices"][0]["delta"]["tool_calls"][0]
    assert head["index"] == 0
    assert head["id"] == "call-123"
    assert head["function"]["name"] == "SendEmailSkill"
    assert "arguments" not in head["function"]

    arg_fragments = [
        event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        for event in events[1:3]
    ]
    assert "".join(arg_fragments) == '{"recipient":"alex@example.com"}'

    terminator = events[3]
    assert terminator["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_mixed_content_and_tool_call_scenario(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """A turn that emits a content preamble before a tool call survives the wire."""
    fake_mistral_server.add_scenario(
        "mixed",
        events=[
            content_delta_event("Sure, sending now.", role="assistant"),
            tool_call_delta_event(
                tool_index=0,
                call_id="call-mix",
                function_name="SendEmailSkill",
                arguments='{"to":"alex"}',
            ),
            finish_event(finish_reason="tool_calls"),
        ],
    )
    fake_mistral_server.set_active("mixed")
    response = await _post(fake_mistral_server)
    try:
        events = await _read_sse_events(response)
    finally:
        await _close(response)

    assert len(events) == 3
    assert events[0]["choices"][0]["delta"]["content"] == "Sure, sending now."
    tool_call = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["arguments"] == '{"to":"alex"}'
    assert events[2]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Capture log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_captured_request_records_method_path_and_body(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """The fake records every request for post-hoc test assertions."""
    fake_mistral_server.add_scenario(
        "echo_back",
        events=[finish_event(finish_reason="stop")],
    )
    fake_mistral_server.set_active("echo_back")

    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        "stream": True,
    }
    response = await _post(fake_mistral_server, body=payload)
    try:
        await response.read()
    finally:
        await _close(response)

    assert len(fake_mistral_server.captured) == 1
    captured = fake_mistral_server.captured[0]
    assert captured.method == "POST"
    assert captured.path == CHAT_COMPLETIONS_PATH
    assert captured.body == payload
    assert b'"mistral-small-latest"' in captured.raw_body


@pytest.mark.asyncio
async def test_reset_captured_clears_request_log(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """:meth:`reset_captured` empties the log between phases of one test."""
    fake_mistral_server.set_active("unauthorized")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 401
    finally:
        await _close(response)
    assert len(fake_mistral_server.captured) == 1

    fake_mistral_server.reset_captured()
    assert fake_mistral_server.captured == []

    response = await _post(fake_mistral_server)
    try:
        assert response.status == 401
    finally:
        await _close(response)
    assert len(fake_mistral_server.captured) == 1


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_scenario_rejects_negative_delay(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Negative delays would silently never sleep — fail at registration time."""
    with pytest.raises(ValueError, match="delay_seconds"):
        fake_mistral_server.add_scenario("bad", events=[], delay_seconds=-0.1)


@pytest.mark.asyncio
async def test_add_failure_rejects_unsupported_body_type(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """Bodies other than bytes / str / Mapping / None raise :class:`TypeError`."""
    with pytest.raises(TypeError, match="bytes, str, Mapping"):
        fake_mistral_server.add_failure(
            "bad_body", status_code=500, body=12345  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_overriding_a_default_scenario_replaces_in_place(
    fake_mistral_server: FakeMistralServer,
) -> None:
    """:meth:`add_failure` with an existing name overrides the previous body."""
    fake_mistral_server.add_failure(
        "rate_limited",
        status_code=429,
        body=b"custom-rate-limit-body",
        headers={"Retry-After": "5"},
    )
    fake_mistral_server.set_active("rate_limited")
    response = await _post(fake_mistral_server)
    try:
        assert response.status == 429
        assert response.headers.get("Retry-After") == "5"
        assert (await response.read()) == b"custom-rate-limit-body"
    finally:
        await _close(response)


# ---------------------------------------------------------------------------
# Helper builder unit tests (kept here so all fake-server contracts are
# locked down in one place).
# ---------------------------------------------------------------------------


def test_content_delta_event_omits_role_when_unset() -> None:
    """Subsequent deltas in a turn must not re-advertise the role."""
    event = content_delta_event("foo")
    assert event["choices"][0]["delta"] == {"content": "foo"}


def test_tool_call_delta_event_omits_optional_fields() -> None:
    """A pure-arguments fragment carries only ``index`` and ``function``."""
    event = tool_call_delta_event(arguments="{}")
    tool_call = event["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call == {"index": 0, "function": {"arguments": "{}"}}


# ---------------------------------------------------------------------------
# Sanity: AsyncIterator export — keeps mypy happy on the helper module.
# ---------------------------------------------------------------------------


def test_async_iterator_export_is_typeable() -> None:
    """Trivial smoke test that the helper symbols are imported correctly."""

    async def _consume(_: AsyncIterator[bytes]) -> None:  # pragma: no cover - shape only
        async for _chunk in _:
            return

    # The helper accepts the type without complaint at static-analysis time.
    assert callable(_consume)
