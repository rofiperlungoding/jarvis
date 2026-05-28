"""Unit tests for ``jarvis.llm.selector.BackendSelector``.

Covers the circuit-breaker state machine described in design.md
"Mistral → Local Fallback Flow" and Requirement 12.4. The test
double :class:`FakeBackend` records every call and is configurable
to:

* enter quickly (the happy path),
* enter after a configurable :func:`asyncio.sleep` (drives the
  ``asyncio.wait_for`` timeout branch — the ">3 s" trigger),
* raise on entry (drives the ``httpx.TimeoutException`` and
  ``httpx.HTTPStatusError`` branches).

A :class:`~jarvis.utils.time_source.FakeTimeSource` is injected so
tests step the cool-down deterministically without sleeping for
real seconds.

Coverage map:

* Closed state routes to the primary; on success the breaker stays
  closed.
* Primary timeout (``asyncio.wait_for`` exceeded) trips the
  breaker; ``on_flip`` fires once.
* Primary 5xx HTTP error trips the breaker.
* While open, calls go straight to the fallback; ``on_flip`` is
  not fired again.
* After ``cool_down`` expires, the next call probes the primary
  (half-open).
* Successful probe collapses the breaker back to closed.
* Failing probe re-trips the breaker; ``on_flip`` fires again.
* Equivalence: the ``messages``, ``tools`` and ``**kwargs``
  forwarded to whichever backend is actually selected match the
  caller's input verbatim — Property 14 basis.
* Non-5xx HTTP errors (401 / 403 / 404) propagate without tripping
  and without falling back.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
from typing import Any

import httpx
import pytest

from jarvis.llm.base import ContentDeltaEvent, LLMEvent
from jarvis.llm.selector import BackendSelector
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _CallConfig:
    """Frozen snapshot of a :class:`FakeBackend`'s knobs at call time.

    We snapshot rather than read-through so a test can mutate the
    backend's configuration *between* calls — e.g., make the first
    call fail and the second succeed — without retroactively
    rewriting the in-flight context manager.
    """

    enter_exc: BaseException | None
    enter_sleep: float
    events: list[LLMEvent]
    iter_exc: BaseException | None


@dataclass
class _RecordedCall:
    """One observed invocation of :meth:`FakeBackend.stream`.

    Stores the *exact* arguments handed to the backend so equivalence
    assertions (Property 14) compare against caller intent rather
    than against any later mutation.
    """

    messages: Any
    tools: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeBackend:
    """Configurable :class:`~jarvis.llm.base.LLMBackend` test double.

    Attributes act as the *next-call* configuration; they may be
    mutated between calls. Each call snapshots them into a
    :class:`_CallConfig` so subsequent mutations do not alter an
    already-running stream.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[_RecordedCall] = []
        # Per-call knobs, all optional with sensible defaults.
        self.enter_exc: BaseException | None = None
        self.enter_sleep: float = 0.0
        self.events: list[LLMEvent] | None = None
        self.iter_exc: BaseException | None = None

    # -- LLMBackend protocol ------------------------------------------------

    def stream(
        self,
        messages: Any,
        *,
        tools: Any,
        **kwargs: Any,
    ) -> Any:
        # Record the call BEFORE entering the context manager so
        # tests can verify "primary was attempted but fallback was
        # used" in the trip path.
        self.calls.append(
            _RecordedCall(messages=messages, tools=tools, kwargs=dict(kwargs))
        )
        snapshot = _CallConfig(
            enter_exc=self.enter_exc,
            enter_sleep=self.enter_sleep,
            events=list(self.events) if self.events is not None
            else [ContentDeltaEvent(text=f"from-{self.name}")],
            iter_exc=self.iter_exc,
        )
        return _fake_stream_cm(snapshot)


@asynccontextmanager
async def _fake_stream_cm(cfg: _CallConfig) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    """Async context manager mirroring the real backends' shape."""
    if cfg.enter_sleep > 0:
        await asyncio.sleep(cfg.enter_sleep)
    if cfg.enter_exc is not None:
        raise cfg.enter_exc

    async def _events() -> AsyncIterator[LLMEvent]:
        for ev in cfg.events:
            yield ev
        if cfg.iter_exc is not None:
            raise cfg.iter_exc

    yield _events()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    """Construct an :class:`httpx.HTTPStatusError` carrying ``status``."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )


async def _drain(selector: BackendSelector, **call_kwargs: Any) -> list[LLMEvent]:
    """Open ``selector.stream`` with sample payload and collect all events."""
    messages = call_kwargs.pop(
        "messages",
        [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Hello."},
        ],
    )
    tools = call_kwargs.pop("tools", [])
    out: list[LLMEvent] = []
    async with selector.stream(messages, tools=tools, **call_kwargs) as events:
        async for event in events:
            out.append(event)
    return out


def _make_selector(
    primary: FakeBackend,
    fallback: FakeBackend,
    *,
    timeout_seconds: float = 0.05,
    cool_down_seconds: float = 1.0,
    on_flip: Any = None,
    time_source: FakeTimeSource | None = None,
) -> tuple[BackendSelector, FakeTimeSource, list[int]]:
    """Build a selector with a fake clock and a counting on_flip.

    Returns ``(selector, fake_time, flip_counts)``. ``flip_counts[0]``
    holds the number of times the on_flip callback has fired so far.
    Tests that need a custom ``on_flip`` can pass one explicitly.
    """
    fake_time = time_source if time_source is not None else FakeTimeSource()
    counts = [0]

    def _default_on_flip() -> None:
        counts[0] += 1

    selector = BackendSelector(
        primary,
        fallback,
        timeout_seconds=timeout_seconds,
        cool_down_seconds=cool_down_seconds,
        time_source=fake_time,
        on_flip=on_flip if on_flip is not None else _default_on_flip,
    )
    return selector, fake_time, counts


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_zero_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            BackendSelector(
                FakeBackend("p"),
                FakeBackend("f"),
                timeout_seconds=0.0,
            )

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            BackendSelector(
                FakeBackend("p"),
                FakeBackend("f"),
                timeout_seconds=-1.0,
            )

    def test_rejects_negative_cool_down(self) -> None:
        with pytest.raises(ValueError, match="cool_down_seconds"):
            BackendSelector(
                FakeBackend("p"),
                FakeBackend("f"),
                cool_down_seconds=-0.5,
            )

    def test_initial_state_is_closed(self) -> None:
        primary, fallback = FakeBackend("p"), FakeBackend("f")
        selector, _, _ = _make_selector(primary, fallback)
        assert not selector.is_open


# ---------------------------------------------------------------------------
# Closed-state behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClosedState:
    async def test_routes_to_primary_when_closed(self) -> None:
        primary, fallback = FakeBackend("p"), FakeBackend("f")
        selector, _, flip_counts = _make_selector(primary, fallback)

        events = await _drain(selector)

        assert len(primary.calls) == 1
        assert fallback.calls == []
        assert events == [ContentDeltaEvent(text="from-p")]
        assert flip_counts[0] == 0
        assert not selector.is_open

    async def test_successful_call_keeps_circuit_closed(self) -> None:
        primary, fallback = FakeBackend("p"), FakeBackend("f")
        selector, _, flip_counts = _make_selector(primary, fallback)

        for _ in range(3):
            await _drain(selector)

        assert len(primary.calls) == 3
        assert fallback.calls == []
        assert flip_counts[0] == 0
        assert not selector.is_open


# ---------------------------------------------------------------------------
# Trip triggers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTripsCircuit:
    async def test_primary_timeout_opens_circuit_and_falls_back(self) -> None:
        # primary's __aenter__ sleeps far longer than the configured
        # 50 ms budget — exercises the asyncio.wait_for/TimeoutError
        # branch (the ">3 s" trigger from the design).
        primary = FakeBackend("p")
        primary.enter_sleep = 0.5
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(
            primary, fallback, timeout_seconds=0.05
        )

        events = await _drain(selector)

        assert len(primary.calls) == 1
        assert len(fallback.calls) == 1
        assert events == [ContentDeltaEvent(text="from-f")]
        assert flip_counts[0] == 1
        assert selector.is_open

    async def test_httpx_timeout_exception_opens_circuit(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ReadTimeout("upstream slow")
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(primary, fallback)

        events = await _drain(selector)

        assert len(primary.calls) == 1
        assert len(fallback.calls) == 1
        assert events == [ContentDeltaEvent(text="from-f")]
        assert flip_counts[0] == 1
        assert selector.is_open

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    async def test_5xx_status_error_opens_circuit(self, status: int) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(status)
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(primary, fallback)

        events = await _drain(selector)

        assert len(primary.calls) == 1
        assert len(fallback.calls) == 1
        assert events == [ContentDeltaEvent(text="from-f")]
        assert flip_counts[0] == 1
        assert selector.is_open

    async def test_on_flip_failure_does_not_break_dialog(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A misbehaving notification callback is logged and swallowed."""

        def _broken_callback() -> None:
            raise RuntimeError("notification engine offline")

        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        selector, _, _ = _make_selector(
            primary, fallback, on_flip=_broken_callback
        )

        with caplog.at_level(logging.ERROR, logger="jarvis.llm.selector"):
            events = await _drain(selector)

        assert events == [ContentDeltaEvent(text="from-f")]
        assert any("on_flip" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Open-state behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenState:
    async def test_open_routes_to_fallback_without_touching_primary(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(500)
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(
            primary, fallback, cool_down_seconds=10.0
        )

        # First call trips the breaker.
        await _drain(selector)
        assert selector.is_open
        assert flip_counts[0] == 1

        # Re-arm the primary so it *would* succeed if it were
        # called — the open-state contract says it must not be.
        primary.enter_exc = None
        primary.calls.clear()
        fallback.calls.clear()

        events = await _drain(selector)

        assert primary.calls == []
        assert len(fallback.calls) == 1
        assert events == [ContentDeltaEvent(text="from-f")]
        # No new flip — the breaker was already open.
        assert flip_counts[0] == 1
        assert selector.is_open

    async def test_multiple_open_calls_do_not_re_fire_on_flip(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ConnectTimeout("dead")
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(
            primary, fallback, cool_down_seconds=10.0
        )

        # Trip + drive several open-state turns.
        for _ in range(4):
            await _drain(selector)

        assert flip_counts[0] == 1
        # primary is only attempted on the trip turn; subsequent
        # turns short-circuit straight to fallback.
        assert len(primary.calls) == 1
        assert len(fallback.calls) == 4


# ---------------------------------------------------------------------------
# Half-open / cool-down expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHalfOpen:
    async def test_cool_down_expiry_probes_primary(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        selector, fake_time, flip_counts = _make_selector(
            primary, fallback, cool_down_seconds=1.0
        )

        # Trip the breaker.
        await _drain(selector)
        assert bool(selector.is_open) is True
        assert len(primary.calls) == 1

        # Just before cool-down expires the breaker is still open;
        # primary MUST NOT be probed.
        fake_time.advance(0.5)
        assert bool(selector.is_open) is True
        await _drain(selector)
        assert len(primary.calls) == 1  # unchanged

        # Push the clock past the cool-down. Re-arm primary so the
        # probe can succeed and we can verify it was actually called.
        fake_time.advance(0.6)  # total elapsed = 1.1 > 1.0
        primary.enter_exc = None
        assert bool(selector.is_open) is False

        events = await _drain(selector)

        # Primary was probed (and succeeded).
        assert len(primary.calls) == 2
        assert events == [ContentDeltaEvent(text="from-p")]
        assert flip_counts[0] == 1  # still just the original trip

    async def test_successful_probe_resets_to_closed(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ReadTimeout("slow")
        fallback = FakeBackend("f")
        selector, fake_time, flip_counts = _make_selector(
            primary, fallback, cool_down_seconds=1.0
        )

        # Trip.
        await _drain(selector)
        assert bool(selector.is_open) is True

        # Cool-down expires; primary recovers.
        fake_time.advance(1.5)
        primary.enter_exc = None
        await _drain(selector)
        assert bool(selector.is_open) is False

        # Subsequent calls go straight to primary without touching
        # the fallback.
        primary.calls.clear()
        fallback.calls.clear()
        for _ in range(3):
            await _drain(selector)

        assert len(primary.calls) == 3
        assert fallback.calls == []
        assert flip_counts[0] == 1
        assert not selector.is_open

    async def test_failed_probe_retrips_and_fires_on_flip_again(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(502)
        fallback = FakeBackend("f")
        selector, fake_time, flip_counts = _make_selector(
            primary, fallback, cool_down_seconds=1.0
        )

        # Trip #1.
        await _drain(selector)
        assert flip_counts[0] == 1
        assert selector.is_open

        # Cool-down expires; primary is still unhealthy.
        fake_time.advance(1.5)
        events = await _drain(selector)

        # Probe failed → re-trip → on_flip fires a second time and
        # the call is routed to the fallback.
        assert flip_counts[0] == 2
        assert selector.is_open
        assert events == [ContentDeltaEvent(text="from-f")]
        assert len(primary.calls) == 2  # original + failed probe
        assert len(fallback.calls) == 2

        # Inside the fresh cool-down window — no extra flips.
        fake_time.advance(0.1)
        await _drain(selector)
        assert flip_counts[0] == 2
        assert len(primary.calls) == 2  # no probe yet
        assert len(fallback.calls) == 3


# ---------------------------------------------------------------------------
# Non-5xx HTTP errors propagate without tripping (Requirement 19.7 path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNon5xxPropagates:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_status_propagates_and_does_not_trip(self, status: int) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(status)
        fallback = FakeBackend("f")
        selector, _, flip_counts = _make_selector(primary, fallback)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await _drain(selector)

        assert excinfo.value.response.status_code == status
        # Primary attempted; fallback NEVER reached for non-5xx.
        assert len(primary.calls) == 1
        assert fallback.calls == []
        assert flip_counts[0] == 0
        assert not selector.is_open


# ---------------------------------------------------------------------------
# Equivalence: messages / tools / kwargs forwarded unchanged (Property 14)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRequestEquivalence:
    """The selector forwards request payloads unchanged regardless of
    which backend it picks.

    This is the unit-test basis for Property 14: any message list and
    tool-definition list passed to ``selector.stream`` arrives at the
    chosen backend with equal content. The breaker decides *which*
    backend serves the turn, not *what* it sees.
    """

    async def test_payload_forwarded_unchanged_to_primary(self) -> None:
        primary, fallback = FakeBackend("p"), FakeBackend("f")
        selector, _, _ = _make_selector(primary, fallback)

        messages = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "What's the weather?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "WeatherSkill",
                    "description": "Look up weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        await _drain(selector, messages=messages, tools=tools, model="m1")

        assert len(primary.calls) == 1
        call = primary.calls[0]
        assert call.messages == messages
        assert call.tools == tools
        assert call.kwargs == {"model": "m1"}
        assert fallback.calls == []

    async def test_payload_forwarded_unchanged_to_fallback_when_open(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        selector, _, _ = _make_selector(
            primary, fallback, cool_down_seconds=10.0
        )

        # Trip the breaker on a throw-away call so the next call
        # routes straight through to the fallback in the open state.
        await _drain(selector)
        assert selector.is_open
        primary.calls.clear()
        fallback.calls.clear()

        messages = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Hi from open."},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "EchoSkill",
                    "description": "Echo a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]
        await _drain(
            selector, messages=messages, tools=tools, model="x", temperature=0.7
        )

        assert primary.calls == []
        assert len(fallback.calls) == 1
        call = fallback.calls[0]
        assert call.messages == messages
        assert call.tools == tools
        assert call.kwargs == {"model": "x", "temperature": 0.7}

    async def test_primary_and_fallback_observe_equal_payloads(self) -> None:
        """Property-14 spirit: the same caller payload reaches whichever
        backend the breaker selects, with equal ``messages``, ``tools``
        and ``**kwargs``.
        """
        primary = FakeBackend("p")
        fallback = FakeBackend("f")
        selector, _, _ = _make_selector(
            primary, fallback, cool_down_seconds=10.0
        )

        messages = [
            {"role": "system", "content": "You are JARVIS."},
            {"role": "user", "content": "Same payload, both backends."},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "PingSkill",
                    "description": "Reply pong.",
                    "parameters": {"type": "object"},
                },
            }
        ]
        kwargs: dict[str, Any] = {"model": "m", "temperature": 0.2}

        # Closed: primary serves the turn.
        await _drain(selector, messages=messages, tools=tools, **kwargs)
        assert len(primary.calls) == 1
        assert fallback.calls == []

        # Trip the breaker so the next call goes to the fallback.
        primary.enter_exc = httpx.ReadTimeout("trip")
        await _drain(selector, messages=messages, tools=tools, **kwargs)
        primary.enter_exc = None  # not relevant — circuit now open

        # Open: the next call hits the fallback directly.
        await _drain(selector, messages=messages, tools=tools, **kwargs)

        # Pick the closed-path primary call and the open-path fallback
        # call and assert they observed identical payloads.
        primary_observed = primary.calls[0]
        fallback_observed = fallback.calls[-1]
        assert primary_observed.messages == fallback_observed.messages == messages
        assert primary_observed.tools == fallback_observed.tools == tools
        assert primary_observed.kwargs == fallback_observed.kwargs == kwargs

    async def test_caller_mutation_after_call_does_not_affect_recorded_payload(
        self,
    ) -> None:
        """The selector snapshots ``messages``/``tools`` (defensive
        ``list(...)``), so mutating the caller's lists after the
        call does not retroactively alter what reached the backend.
        """
        primary, fallback = FakeBackend("p"), FakeBackend("f")
        selector, _, _ = _make_selector(primary, fallback)

        messages = [{"role": "system", "content": "S"}]
        tools: list[Any] = []

        await _drain(selector, messages=messages, tools=tools)

        # Mutate caller-owned lists.
        messages.append({"role": "user", "content": "later"})
        tools.append({"type": "function", "function": {}})

        # The recorded backend payload still reflects the call-time
        # snapshot (a single system message, no tools).
        recorded = primary.calls[0]
        assert recorded.messages == [{"role": "system", "content": "S"}]
        assert recorded.tools == []
