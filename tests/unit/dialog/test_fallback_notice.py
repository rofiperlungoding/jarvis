"""Unit tests for ``jarvis.dialog.fallback_notice``.

Validates the wiring contract of task 13.5 — the spoken notice that
fires when :class:`jarvis.llm.selector.BackendSelector` opens its
circuit and switches to the local Ollama fallback (Requirement 12.4):

* Circuit open emits the notice via TTS exactly once.
* Subsequent requests during the open window do not re-emit.
* Notification text contains the persona honorific.
* A TTS exception raised while speaking the notice does not crash the
  dialog flow.

The tests compose a real :class:`BackendSelector` (the actual one-shot
machinery is the contract under test) with the fake LLM backends from
``test_backend_selector`` and a tiny in-memory ``TTSEngine`` double.

Validates: Requirement 12.4
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

from jarvis.dialog.fallback_notice import (
    DEFAULT_FALLBACK_NOTICE_TEMPLATE,
    build_backend_fallback_notice,
    format_fallback_notice,
)
from jarvis.dialog.persona import default_jarvis_persona
from jarvis.llm.base import ContentDeltaEvent, LLMEvent
from jarvis.llm.selector import BackendSelector
from jarvis.utils.time_source import FakeTimeSource

# ---------------------------------------------------------------------------
# Inline test doubles
# ---------------------------------------------------------------------------
#
# We deliberately duplicate the small ``FakeBackend`` / ``_drain`` /
# ``_http_status_error`` helpers used in
# ``tests/unit/llm/test_backend_selector.py`` rather than importing them
# across test modules. Cross-test imports would require a top-level
# ``tests/__init__.py`` — which deliberately does not exist so each
# ``tests/unit/<area>/`` package is independent. The duplication is
# tiny and the failure mode (a divergence here vs. there) is a benign
# documentation issue, not a runtime bug.


@dataclass
class _CallConfig:
    enter_exc: BaseException | None
    enter_sleep: float
    events: list[LLMEvent]


@dataclass
class _RecordedCall:
    messages: Any
    tools: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeBackend:
    """Minimal :class:`LLMBackend` test double for circuit-breaker scenarios."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[_RecordedCall] = []
        self.enter_exc: BaseException | None = None
        self.enter_sleep: float = 0.0
        self.events: list[LLMEvent] | None = None

    def stream(
        self,
        messages: Any,
        *,
        tools: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            _RecordedCall(messages=messages, tools=tools, kwargs=dict(kwargs))
        )
        snapshot = _CallConfig(
            enter_exc=self.enter_exc,
            enter_sleep=self.enter_sleep,
            events=(
                list(self.events)
                if self.events is not None
                else [ContentDeltaEvent(text=f"from-{self.name}")]
            ),
        )
        return _fake_stream_cm(snapshot)


@asynccontextmanager
async def _fake_stream_cm(
    cfg: _CallConfig,
) -> AsyncIterator[AsyncIterator[LLMEvent]]:
    if cfg.enter_sleep > 0:
        await asyncio.sleep(cfg.enter_sleep)
    if cfg.enter_exc is not None:
        raise cfg.enter_exc

    async def _events() -> AsyncIterator[LLMEvent]:
        for ev in cfg.events:
            yield ev

    yield _events()


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )


async def _drain(
    selector: BackendSelector, **call_kwargs: Any
) -> list[LLMEvent]:
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

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingTTS:
    """Records every ``speak`` call. Conforms structurally to ``TTSEngine``."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._playing: bool = False

    async def speak(self, text: str) -> None:
        # Store the text synchronously *before* awaiting anything so
        # tests that drain the loop see the notice even if a later
        # await would have cancelled the task.
        self.spoken.append(text)

    async def stop(self) -> None:
        self._playing = False

    def is_playing(self) -> bool:
        return self._playing

    async def aclose(self) -> None:  # pragma: no cover - not exercised
        pass


class _ExplodingTTS(_RecordingTTS):
    """Like :class:`_RecordingTTS` but ``speak`` always raises."""

    def __init__(self, exc: BaseException | None = None) -> None:
        super().__init__()
        self._exc = exc or RuntimeError("audio device offline")

    async def speak(self, text: str) -> None:
        # Still record the attempt so tests can verify the bridge
        # *tried* to speak before the failure.
        self.spoken.append(text)
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _settle() -> None:
    """Yield to the loop a couple of times so background tasks finish.

    The bridge schedules the speak via ``loop.create_task``; the
    task body is a tiny ``await tts.speak(...)``. A single ``sleep(0)``
    is enough on CPython, but two yields make the suite robust to
    future TTS implementations that briefly suspend.
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _build_selector(
    primary: FakeBackend,
    fallback: FakeBackend,
    tts: _RecordingTTS,
    *,
    honorific: str = "sir",
    template: str | None = None,
    cool_down_seconds: float = 1.0,
    timeout_seconds: float = 0.05,
) -> tuple[BackendSelector, FakeTimeSource]:
    """Wire a real selector + the production fallback notice callback."""
    fake_time = FakeTimeSource()
    on_flip = build_backend_fallback_notice(
        tts, honorific=honorific, template=template
    )
    selector = BackendSelector(
        primary,
        fallback,
        timeout_seconds=timeout_seconds,
        cool_down_seconds=cool_down_seconds,
        time_source=fake_time,
        on_flip=on_flip,
    )
    return selector, fake_time


# ---------------------------------------------------------------------------
# format_fallback_notice
# ---------------------------------------------------------------------------


class TestFormatFallbackNotice:
    def test_default_template_includes_honorific(self) -> None:
        text = format_fallback_notice("sir")

        assert "sir" in text
        # The cinematic phrasing from the design document.
        assert text == "The cloud is being slow, sir. Switching to local."

    def test_custom_honorific_renders_into_default_template(self) -> None:
        text = format_fallback_notice("madam")

        assert "madam" in text
        assert "sir" not in text

    def test_custom_template_is_honoured(self) -> None:
        text = format_fallback_notice(
            "boss",
            template="Cloud's down, {honorific}.",
        )

        assert text == "Cloud's down, boss."

    def test_template_without_placeholder_is_returned_verbatim(self) -> None:
        # ``str.format`` happily ignores unused kwargs, so a template
        # without ``{honorific}`` is a no-op rather than an error.
        text = format_fallback_notice("sir", template="Switching to local.")

        assert text == "Switching to local."

    def test_default_constant_matches_design_phrasing(self) -> None:
        # Pin the constant so accidental edits are surfaced in code
        # review rather than only at runtime.
        assert (
            DEFAULT_FALLBACK_NOTICE_TEMPLATE
            == "The cloud is being slow, {honorific}. Switching to local."
        )


# ---------------------------------------------------------------------------
# Bridge behaviour wired up to a real BackendSelector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeaksOnceOnTrip:
    async def test_circuit_open_emits_notice_via_tts_exactly_once(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, _ = _build_selector(primary, fallback, tts)

        events = await _drain(selector)
        await _settle()

        # Fallback served the turn, so the user got a real answer.
        assert events == [ContentDeltaEvent(text="from-f")]
        # And exactly one notice landed on the TTS queue.
        assert tts.spoken == [
            "The cloud is being slow, sir. Switching to local."
        ]
        assert selector.is_open

    async def test_notice_text_uses_persona_honorific(self) -> None:
        # Exercise the same wiring app.py will perform: persona's
        # honorific feeds into the bridge.
        persona = default_jarvis_persona()
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ReadTimeout("upstream slow")
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, _ = _build_selector(
            primary, fallback, tts, honorific=persona.honorific
        )

        await _drain(selector)
        await _settle()

        assert len(tts.spoken) == 1
        assert persona.honorific in tts.spoken[0]
        assert tts.spoken[0].endswith("Switching to local.")

    async def test_custom_honorific_propagates_to_speak(self) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(502)
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, _ = _build_selector(
            primary, fallback, tts, honorific="madam"
        )

        await _drain(selector)
        await _settle()

        assert tts.spoken == [
            "The cloud is being slow, madam. Switching to local."
        ]


@pytest.mark.asyncio
class TestNoReEmissionWhileOpen:
    async def test_subsequent_requests_during_open_window_do_not_re_emit(
        self,
    ) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ConnectTimeout("dead")
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, _ = _build_selector(
            primary, fallback, tts, cool_down_seconds=10.0
        )

        # Trip + several open-state turns.
        for _ in range(5):
            await _drain(selector)
        await _settle()

        # The notice fires exactly once — at the trip — even though
        # the fallback served five turns.
        assert tts.spoken == [
            "The cloud is being slow, sir. Switching to local."
        ]
        assert len(fallback.calls) == 5

    async def test_failed_recovery_probe_re_emits_notice(self) -> None:
        # The selector's contract is "per-flip, not per-lifetime": a
        # second flip after a successful (or, here, failed) recovery
        # IS news for the user, so the bridge speaks again. This is
        # the same one-shot semantics — one notice per transition.
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, fake_time = _build_selector(
            primary, fallback, tts, cool_down_seconds=1.0
        )

        # Trip #1.
        await _drain(selector)
        await _settle()
        assert len(tts.spoken) == 1

        # Cool-down expires; primary still unhealthy → re-trip.
        fake_time.advance(1.5)
        await _drain(selector)
        await _settle()

        # Two notices total — one per flip.
        assert len(tts.spoken) == 2
        assert tts.spoken[0] == tts.spoken[1]

    async def test_recovery_then_re_trip_re_emits_notice(self) -> None:
        # Trip → recover → fail again → notice fires again.
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(503)
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, fake_time = _build_selector(
            primary, fallback, tts, cool_down_seconds=1.0
        )

        # Trip #1.
        await _drain(selector)
        await _settle()
        assert len(tts.spoken) == 1

        # Cool-down expires; primary recovers.
        fake_time.advance(1.5)
        primary.enter_exc = None
        await _drain(selector)
        await _settle()
        assert not selector.is_open
        # Recovery does not speak — only flips into the open state do.
        assert len(tts.spoken) == 1

        # Primary fails again.
        primary.enter_exc = httpx.ReadTimeout("slow again")
        await _drain(selector)
        await _settle()

        assert len(tts.spoken) == 2

    async def test_calls_to_fallback_alone_never_emit(self) -> None:
        # A selector that has never tripped should be silent on the
        # TTS side regardless of how many turns it serves.
        primary = FakeBackend("p")
        fallback = FakeBackend("f")
        tts = _RecordingTTS()
        selector, _ = _build_selector(primary, fallback, tts)

        for _ in range(4):
            await _drain(selector)
        await _settle()

        assert tts.spoken == []
        assert not selector.is_open


@pytest.mark.asyncio
class TestExceptionIsolation:
    async def test_tts_exception_does_not_crash_dialog_flow(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The bridge must swallow TTS errors so a broken audio device
        # cannot abort the dialog turn — Requirement 17.1 spirit.
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(500)
        fallback = FakeBackend("f")
        tts = _ExplodingTTS()
        selector, _ = _build_selector(primary, fallback, tts)

        with caplog.at_level(logging.ERROR, logger="jarvis.dialog.fallback_notice"):
            events = await _drain(selector)
            await _settle()

        # The user still got a normal response from the fallback.
        assert events == [ContentDeltaEvent(text="from-f")]
        # The bridge attempted to speak.
        assert tts.spoken == [
            "The cloud is being slow, sir. Switching to local."
        ]
        # And the failure was logged with traceback.
        assert any(
            "fallback notice" in rec.getMessage()
            for rec in caplog.records
        )

    async def test_tts_exception_does_not_propagate_through_selector(
        self,
    ) -> None:
        # The selector's stream() must not surface a TTS error to the
        # Dialog_Manager — even if the exception happened on the
        # in-flight turn that triggered the trip.
        primary = FakeBackend("p")
        primary.enter_exc = httpx.ReadTimeout("slow")
        fallback = FakeBackend("f")
        tts = _ExplodingTTS(RuntimeError("device gone"))
        selector, _ = _build_selector(primary, fallback, tts)

        # Should NOT raise.
        events: list[LLMEvent] = []
        async with selector.stream(
            [{"role": "user", "content": "hi"}], tools=[]
        ) as stream:
            async for event in stream:
                events.append(event)

        assert events == [ContentDeltaEvent(text="from-f")]
        # Drain the scheduled task; the exception is swallowed inside
        # the bridge's ``_speak_notice`` wrapper, not raised here.
        await _settle()

    async def test_repeated_trips_with_failing_tts_do_not_break_dialog(
        self,
    ) -> None:
        primary = FakeBackend("p")
        primary.enter_exc = _http_status_error(500)
        fallback = FakeBackend("f")
        tts = _ExplodingTTS()
        selector, fake_time = _build_selector(
            primary, fallback, tts, cool_down_seconds=1.0
        )

        # Trip #1, then probe the half-open and re-trip.
        await _drain(selector)
        await _settle()
        fake_time.advance(1.5)
        await _drain(selector)
        await _settle()

        # Both flips logged a speak attempt and both turns served
        # successfully via the fallback.
        assert len(tts.spoken) == 2
        assert len(fallback.calls) == 2


# ---------------------------------------------------------------------------
# Bridge behaviour outside an event loop
# ---------------------------------------------------------------------------


class TestNoRunningLoop:
    def test_callback_is_safe_to_invoke_without_running_loop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ``BackendSelector`` only ever calls ``on_flip`` from inside
        # an async stream, but we defend against misuse anyway:
        # invoking the callback from sync code MUST NOT crash.
        tts = _RecordingTTS()
        on_flip = build_backend_fallback_notice(tts)

        with caplog.at_level(logging.WARNING, logger="jarvis.dialog.fallback_notice"):
            on_flip()  # synchronous invocation — no loop running

        # Nothing was spoken (no loop to schedule on) and the misuse
        # was logged for the operator.
        assert tts.spoken == []
        assert any(
            "no running event loop" in rec.getMessage()
            for rec in caplog.records
        )
