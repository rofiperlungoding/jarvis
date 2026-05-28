"""In-process fake Mistral la Plateforme HTTP server for integration tests.

This module hosts :class:`FakeMistralServer`, an :mod:`aiohttp` web app
fixture that replays canned ``chat.stream`` Server-Sent-Event sequences
over the wire so :class:`~jarvis.llm.mistral_backend.MistralBackend` can
be exercised end-to-end without touching the real Mistral API.

Wire shape mirrored
-------------------

The real Mistral cloud accepts a ``POST`` to
``/v1/chat/completions`` whose JSON body sets ``stream: true``. The
response carries ``Content-Type: text/event-stream`` and is a sequence
of Server-Sent-Event records of the form::

    data: {"id":"...","object":"chat.completion.chunk", ...}

    data: {"id":"...","object":"chat.completion.chunk", ...}

    data: [DONE]

(Each record is terminated by a blank line — that is, ``\\n\\n``.) The
SSE chunks themselves match the OpenAI-streaming shape that the
``mistralai`` SDK consumes: ``choices[i].delta.content`` for token
deltas, ``choices[i].delta.tool_calls`` for streamed tool-call
fragments (carrying ``index``, optionally ``id`` and
``function.name`` once, plus a sequence of ``function.arguments``
JSON-string fragments that the SDK concatenates), and
``choices[i].finish_reason`` for terminal markers (``"stop"``,
``"tool_calls"``).

The fake reproduces that shape verbatim so the production backend can
be wired to it through nothing more than its ``endpoint`` constructor
kwarg.

What lives here
---------------

* :class:`FakeMistralServer` — manages a registry of named
  :class:`Scenario` definitions, an active-scenario selector, and the
  bound :mod:`aiohttp` runner.
* :class:`Scenario` and :class:`CapturedRequest` — value objects used
  for scenario registration and request log assertions.
* Helper builders :func:`content_delta_event`,
  :func:`tool_call_delta_event` and :func:`finish_event` — produce the
  SSE chunk payloads in their canonical Mistral shape so tests do not
  have to hand-roll JSON.
* :func:`fake_mistral_server` — a ``pytest_asyncio`` fixture that
  yields a started server with the four documented failure scenarios
  pre-registered (and tears down on test exit).

Default failure scenarios
-------------------------

Every fresh :class:`FakeMistralServer` has the following named
scenarios pre-registered so tests do not have to assemble each one
from scratch:

* ``"unauthorized"`` — HTTP 401, JSON error body (Requirement 19.7).
* ``"forbidden"`` — HTTP 403, JSON error body (Requirement 19.7).
* ``"rate_limited"`` — HTTP 429 with ``Retry-After: 1`` header
  (Requirement 19.8).
* ``"server_error"`` — HTTP 503 (Requirement 12.4).
* ``"slow"`` — 200 OK SSE response that sleeps **>3 s** before
  emitting the first chunk so the
  :class:`~jarvis.llm.selector.BackendSelector` 3 s circuit-breaker
  timeout has a deterministic trigger.

Tests can replace any of these by calling :meth:`FakeMistralServer.add_scenario`
or :meth:`FakeMistralServer.add_failure` with the same name; the new
definition overrides the default in-place.

Validates: Requirements 12.4, 19.5, 19.7, 19.8
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, Final

from aiohttp import web
import pytest_asyncio

__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_FAILURE_SCENARIOS",
    "DEFAULT_MODEL",
    "DEFAULT_SLOW_DELAY_SECONDS",
    "CapturedRequest",
    "FakeMistralServer",
    "Scenario",
    "content_delta_event",
    "fake_mistral_server",
    "finish_event",
    "tool_call_delta_event",
]


# ---------------------------------------------------------------------------
# Wire constants
# ---------------------------------------------------------------------------


#: Mistral's chat-completions endpoint path. The :mod:`mistralai` SDK
#: appends this to the configured ``server_url`` / ``endpoint``.
CHAT_COMPLETIONS_PATH: Final[str] = "/v1/chat/completions"

#: Default model id stamped into chunk payloads when the scenario
#: builder helpers are called without an override.
DEFAULT_MODEL: Final[str] = "mistral-large-latest"

#: Slow-scenario delay. The BackendSelector's circuit breaker opens at
#: 3 s (Requirement 12.4); 3.5 s gives the breaker an unambiguous
#: signal without making integration tests gratuitously slow.
DEFAULT_SLOW_DELAY_SECONDS: Final[float] = 3.5


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapturedRequest:
    """A request the fake observed, for post-hoc test assertions.

    The body is parsed as JSON when possible. ``body`` is ``None`` when
    the request had no body or could not be parsed; tests that need the
    raw bytes can read ``raw_body`` instead.
    """

    method: str
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, Any] | None
    raw_body: bytes


@dataclass
class Scenario:
    """Declarative description of one canned response.

    The ``events`` sequence is the ordered list of SSE chunk payloads
    (Python dicts ready to be ``json.dumps``-ed). Scenarios with
    ``is_failure=True`` ignore ``events`` and instead respond with
    ``status``, ``headers`` and ``body`` verbatim — used to model HTTP
    401 / 403 / 429 / 5xx.

    The class is intentionally mutable so tests can poke a registered
    scenario between turns (for example, swap its ``events`` for a
    different content sequence) without having to delete and re-add it.
    """

    name: str
    events: tuple[Mapping[str, Any], ...] = ()
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    delay_seconds: float = 0.0
    body: bytes = b""
    is_failure: bool = False


# ---------------------------------------------------------------------------
# Default failure-scenario names (exposed as a tuple so tests can iterate)
# ---------------------------------------------------------------------------


#: Canonical names for the documented failure modes pre-registered on
#: every :class:`FakeMistralServer`. Useful for parametrising tests.
DEFAULT_FAILURE_SCENARIOS: Final[tuple[str, ...]] = (
    "unauthorized",
    "forbidden",
    "rate_limited",
    "server_error",
    "slow",
)


# ---------------------------------------------------------------------------
# SSE chunk builder helpers
# ---------------------------------------------------------------------------


def content_delta_event(
    text: str,
    *,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-fake-0",
    model: str = DEFAULT_MODEL,
    created: int = 0,
    role: str | None = None,
) -> dict[str, Any]:
    """Return one SSE chunk payload carrying a single content delta.

    Mistral's first chunk on a turn typically also carries
    ``role: "assistant"``; pass ``role="assistant"`` on the first
    invocation to model that. Subsequent deltas should leave ``role``
    at its default of ``None`` so the field is omitted.
    """
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    delta["content"] = text
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def tool_call_delta_event(
    *,
    tool_index: int = 0,
    call_id: str | None = None,
    function_name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-fake-0",
    model: str = DEFAULT_MODEL,
    created: int = 0,
) -> dict[str, Any]:
    """Return one SSE chunk payload carrying a tool-call fragment.

    The Mistral / OpenAI streaming contract spreads each tool call
    across several deltas: the first usually carries ``id`` and
    ``function.name``; subsequent deltas append to
    ``function.arguments`` until the JSON is complete. Every fragment
    references the same ``index`` so the SDK can reassemble them.

    Pass any subset of ``call_id`` / ``function_name`` / ``arguments``;
    fields left at ``None`` are omitted from the resulting payload, so
    a single helper covers every fragment in the sequence.
    """
    function: dict[str, Any] = {}
    if function_name is not None:
        function["name"] = function_name
    if arguments is not None:
        function["arguments"] = arguments
    tool_call: dict[str, Any] = {"index": tool_index}
    if call_id is not None:
        tool_call["id"] = call_id
    if function:
        tool_call["function"] = function
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [tool_call]},
                "finish_reason": finish_reason,
            }
        ],
    }


def finish_event(
    *,
    finish_reason: str = "stop",
    chunk_id: str = "chatcmpl-fake-0",
    model: str = DEFAULT_MODEL,
    created: int = 0,
) -> dict[str, Any]:
    """Return the terminal SSE chunk carrying only ``finish_reason``.

    Mistral closes a turn with an empty-delta chunk whose
    ``finish_reason`` is one of ``"stop"`` (text-only), ``"tool_calls"``
    (function-calling turn) or ``"length"`` (max-tokens cut-off). The
    helper defaults to ``"stop"`` because that's the most common case
    in tests.
    """
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }


# ---------------------------------------------------------------------------
# FakeMistralServer
# ---------------------------------------------------------------------------


class FakeMistralServer:
    """An :mod:`aiohttp` web server replaying canned Mistral responses.

    Instantiate, optionally register extra scenarios, and call
    :meth:`start` to bind to an ephemeral local port. The returned URL
    is the value to inject into :class:`MistralBackend.endpoint` for
    the duration of the test.

    Parameters
    ----------
    host:
        Bind address. Defaults to ``"127.0.0.1"`` so the listener is
        not visible to other machines on the network — every test runs
        against ``localhost``.
    register_defaults:
        When ``True`` (the default) the four documented failure-mode
        scenarios plus the ``"slow"`` scenario are pre-registered
        before :meth:`start` returns. Pass ``False`` for tests that
        want absolute control over the scenario registry.

    Attributes
    ----------
    captured:
        Append-only list of every request the server has received, in
        receipt order. Tests assert against this list to verify the
        backend sent the right shape on the wire.

    Notes
    -----
    The server is single-active-scenario at any moment — call
    :meth:`set_active` between turns to switch. There is no per-request
    routing beyond that, intentionally: the unit and PBT layers cover
    request-shape variability already; the integration layer's job is
    to verify that the backend behaves correctly for one specific
    response shape per turn.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        register_defaults: bool = True,
    ) -> None:
        self._host = host
        self._scenarios: dict[str, Scenario] = {}
        self._active: str | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._url: str | None = None
        self.captured: list[CapturedRequest] = []
        if register_defaults:
            self._register_default_scenarios()

    # ------------------------------------------------------------------
    # Scenario registration
    # ------------------------------------------------------------------

    def add_scenario(
        self,
        name: str,
        *,
        events: Sequence[Mapping[str, Any]],
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Register a successful streaming scenario.

        Parameters
        ----------
        name:
            Identifier passed to :meth:`set_active`. Re-using an
            existing name overrides the previous definition in-place,
            which is how tests customise a default scenario without
            building a new server.
        events:
            Ordered SSE chunk payloads. Each dict is encoded as
            ``data: {json}\\n\\n``; the server appends ``data: [DONE]``
            after the last one. Use the helper builders
            (:func:`content_delta_event`, :func:`tool_call_delta_event`,
            :func:`finish_event`) to assemble these in the canonical
            Mistral shape.
        status:
            HTTP status code. Defaults to ``200``; tests rarely need
            to override this on a streaming scenario.
        headers:
            Extra response headers. ``Content-Type`` is forced to
            ``text/event-stream`` for streaming scenarios; pass other
            headers (e.g., custom ``Mistral-*`` markers) here.
        delay_seconds:
            Delay between request receipt and the first response byte.
            Use this for the ``"slow"`` scenario (>3 s) so the
            BackendSelector's circuit-breaker timeout fires.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("scenario name must be a non-empty string")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        self._scenarios[name] = Scenario(
            name=name,
            events=tuple(dict(event) for event in events),
            status=status,
            headers=dict(headers or {}),
            delay_seconds=float(delay_seconds),
            is_failure=False,
        )

    def add_failure(
        self,
        name: str,
        *,
        status_code: int,
        body: bytes | str | Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Register a failure scenario (401 / 403 / 429 / 5xx / slow).

        Parameters
        ----------
        name:
            Identifier passed to :meth:`set_active`.
        status_code:
            HTTP status code returned to the client.
        body:
            Response body. ``bytes`` is sent verbatim; ``str`` is
            UTF-8 encoded; a ``Mapping`` is JSON-encoded with
            ``Content-Type: application/json`` (unless the caller
            supplied ``Content-Type`` in ``headers``). ``None`` sends
            an empty body.
        headers:
            Extra response headers. For HTTP 429 callers SHOULD pass
            ``{"Retry-After": "1"}`` so the production backend's retry
            schedule sees a realistic value (Requirement 19.8).
        delay_seconds:
            Delay before responding. Use this together with a 200
            status to model a slow successful response, or with a 5xx
            to model "down + slow".
        """
        if not isinstance(name, str) or not name:
            raise ValueError("scenario name must be a non-empty string")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")

        merged_headers: dict[str, str] = dict(headers or {})
        encoded: bytes
        if body is None:
            encoded = b""
        elif isinstance(body, bytes):
            encoded = body
        elif isinstance(body, str):
            encoded = body.encode("utf-8")
        elif isinstance(body, Mapping):
            encoded = json.dumps(dict(body)).encode("utf-8")
            merged_headers.setdefault("Content-Type", "application/json")
        else:
            raise TypeError(
                "body must be bytes, str, Mapping, or None; "
                f"got {type(body).__name__}"
            )

        self._scenarios[name] = Scenario(
            name=name,
            events=(),
            status=int(status_code),
            headers=merged_headers,
            delay_seconds=float(delay_seconds),
            body=encoded,
            is_failure=True,
        )

    def set_active(self, name: str) -> None:
        """Select the scenario served on the next ``POST``.

        Raises
        ------
        KeyError:
            If ``name`` has not been registered.
        """
        if name not in self._scenarios:
            raise KeyError(f"unknown scenario: {name!r}")
        self._active = name

    @property
    def active(self) -> str | None:
        """Name of the currently selected scenario, or ``None``."""
        return self._active

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Tuple of registered scenario names in insertion order."""
        return tuple(self._scenarios)

    def reset_captured(self) -> None:
        """Drop the captured request log."""
        self.captured.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> str:
        """Bind to an ephemeral port and return the base URL.

        The URL is suitable to inject into
        :class:`~jarvis.llm.mistral_backend.MistralBackend.endpoint`
        directly — the SDK appends :data:`CHAT_COMPLETIONS_PATH` for
        every request.
        """
        if self._runner is not None:
            raise RuntimeError("FakeMistralServer is already running")

        app = web.Application()
        app.router.add_post(CHAT_COMPLETIONS_PATH, self._handle_chat_completions)

        runner = web.AppRunner(app)
        await runner.setup()
        # ``port=0`` asks the OS for an ephemeral port — by far the
        # safest choice in test environments where a fixed port would
        # collide with another test run or with developer services.
        site = web.TCPSite(runner, host=self._host, port=0)
        await site.start()

        self._runner = runner
        self._site = site
        # ``site.name`` would be the canonical "scheme://host:port"
        # string, but aiohttp does not update ``site._port`` after a
        # ``port=0`` ephemeral bind — ``site.name`` would resolve to
        # ``http://127.0.0.1:0`` and would not be reachable. Pull the
        # actual port off the underlying server socket instead.
        sockets = getattr(runner, "addresses", None)
        bound_port: int | None = None
        if sockets:
            # ``addresses`` is a list of ``(host, port, ...)`` tuples;
            # for a TCPSite with a single socket the second element is
            # the port.
            first = sockets[0]
            if isinstance(first, tuple) and len(first) >= 2 and isinstance(first[1], int):
                bound_port = first[1]
        if bound_port is None:
            # Fallback: walk the runner's server's underlying asyncio
            # server. ``getsockname`` always returns the kernel-assigned
            # port even when the original bind requested 0.
            asyncio_server = getattr(site, "_server", None)
            if asyncio_server is not None and asyncio_server.sockets:
                bound_port = asyncio_server.sockets[0].getsockname()[1]
        if bound_port is None:
            raise RuntimeError(
                "FakeMistralServer.start() could not determine the bound port"
            )
        self._url = f"http://{self._host}:{bound_port}"
        return self._url

    async def stop(self) -> None:
        """Stop the server and release the port. Safe to call twice."""
        runner = self._runner
        self._runner = None
        self._site = None
        self._url = None
        if runner is not None:
            await runner.cleanup()

    @property
    def url(self) -> str:
        """Bound base URL. Raises :class:`RuntimeError` before :meth:`start`."""
        if self._url is None:
            raise RuntimeError("FakeMistralServer is not running; call start() first")
        return self._url

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register_default_scenarios(self) -> None:
        """Pre-register the four documented failure modes + ``"slow"``."""
        self.add_failure(
            "unauthorized",
            status_code=401,
            body={"object": "error", "message": "Invalid API key", "type": "invalid_request_error"},
        )
        self.add_failure(
            "forbidden",
            status_code=403,
            body={"object": "error", "message": "Forbidden", "type": "forbidden"},
        )
        self.add_failure(
            "rate_limited",
            status_code=429,
            body={"object": "error", "message": "Too Many Requests", "type": "rate_limit_error"},
            headers={"Retry-After": "1"},
        )
        self.add_failure(
            "server_error",
            status_code=503,
            body={"object": "error", "message": "Service Unavailable", "type": "server_error"},
        )
        # Slow scenario: 200 OK, a single short content delta, but
        # >3 s before the first byte so the BackendSelector circuit
        # breaker has a deterministic trigger (Requirement 12.4).
        self.add_scenario(
            "slow",
            events=[
                content_delta_event("OK", role="assistant"),
                finish_event(finish_reason="stop"),
            ],
            delay_seconds=DEFAULT_SLOW_DELAY_SECONDS,
        )

    async def _handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse:
        """Serve the active scenario for a ``POST /v1/chat/completions``.

        The request body is captured before the response is composed so
        even a scenario that crashes the connection (e.g., a slow
        scenario that's interrupted by the client's timeout) leaves a
        record of what was sent.
        """
        raw_body = await request.read()
        parsed: Mapping[str, Any] | None
        try:
            decoded = json.loads(raw_body) if raw_body else None
            parsed = decoded if isinstance(decoded, dict) else None
        except json.JSONDecodeError:
            parsed = None
        self.captured.append(
            CapturedRequest(
                method=request.method,
                path=request.path,
                headers=dict(request.headers),
                body=parsed,
                raw_body=bytes(raw_body),
            )
        )

        if self._active is None:
            return web.json_response(
                {
                    "object": "error",
                    "message": "no active scenario configured",
                    "type": "fake_misconfiguration",
                },
                status=500,
            )
        scenario = self._scenarios[self._active]

        if scenario.delay_seconds > 0:
            await asyncio.sleep(scenario.delay_seconds)

        if scenario.is_failure:
            return web.Response(
                status=scenario.status,
                body=scenario.body,
                headers=dict(scenario.headers),
            )

        # Streaming success path. ``Cache-Control: no-cache`` and
        # ``Connection: keep-alive`` mirror the headers the real
        # Mistral API sends; some SSE clients refuse to start parsing
        # without them.
        sse_headers: dict[str, str] = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        sse_headers.update(scenario.headers)
        response = web.StreamResponse(status=scenario.status, headers=sse_headers)
        await response.prepare(request)
        for event in scenario.events:
            chunk = f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
            await response.write(chunk)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(name="fake_mistral_server")
async def fake_mistral_server() -> AsyncIterator[FakeMistralServer]:
    """Yield a started :class:`FakeMistralServer` with default scenarios.

    The fixture binds the server to an ephemeral local port on
    ``__aenter__`` and calls :meth:`FakeMistralServer.stop` on
    ``__aexit__``, so the listener never leaks across tests. Tests
    typically call :meth:`FakeMistralServer.set_active` to pick a
    scenario before issuing a request:

    .. code-block:: python

        @pytest.mark.asyncio
        async def test_unauthorized(
            fake_mistral_server: FakeMistralServer,
        ) -> None:
            fake_mistral_server.set_active("unauthorized")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    fake_mistral_server.url + CHAT_COMPLETIONS_PATH,
                    json={...},
                ) as response:
                    assert response.status == 401
    """
    server = FakeMistralServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
