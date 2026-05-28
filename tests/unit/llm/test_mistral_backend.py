"""Unit tests for ``jarvis.llm.mistral_backend.MistralBackend``.

These tests exercise the backend with a hand-rolled fake async client
so the production ``mistralai`` SDK is not required at test time. The
fakes mirror the surface area documented in the design:

* ``client.chat.stream_async(**payload)`` is an async function returning
  an async iterator of SSE-shaped events.
* Events expose a ``data`` attribute whose ``choices`` list each carry
  a ``delta`` and an optional ``finish_reason``.

Coverage map:

* Streaming content / tool_call event translation (Requirement 19.4, 19.5).
* Tool-call fragment reassembly across multiple deltas
  (:class:`ToolCallEvent` re-assembly contract).
* HTTP 401 / 403 → :class:`MistralAuthError` (Requirement 19.7).
* HTTP 429 retry-and-recover and retry-exhaustion
  (Requirement 19.8).
* HTTP 5xx → :class:`MistralStreamError` without retry
  (Requirement 12.4 routing trigger).
* API key sourced from the Credential_Store and registered with the
  log redaction filter (Requirement 19.3, CP11).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json
import logging
from typing import Any

import pytest

from jarvis.llm.base import (
    ContentDeltaEvent,
    ToolCallEvent,
)
from jarvis.llm.mistral_backend import (
    DEFAULT_API_KEY_CREDENTIAL,
    DEFAULT_MISTRAL_ENDPOINT,
    DEFAULT_MISTRAL_MODEL,
    MistralAuthError,
    MistralBackend,
    MistralCredentialError,
    MistralCredentialMissingError,
    MistralRateLimitError,
    MistralStreamError,
)
from jarvis.security.log_redaction import LogRedactionFilter

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeFunctionDelta:
    """Stand-in for ``mistralai`` ``FunctionCall`` delta payload."""

    name: str | None = None
    arguments: Any = None  # str fragment OR final dict


@dataclass
class _FakeToolCallDelta:
    """Stand-in for one element of ``delta.tool_calls``."""

    id: str | None = None
    function: _FakeFunctionDelta | None = None
    index: int = 0


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[_FakeToolCallDelta] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta = field(default_factory=_FakeDelta)
    finish_reason: str | None = None


@dataclass
class _FakeChunk:
    """The payload object normally accessed via ``event.data``."""

    choices: list[_FakeChoice] = field(default_factory=list)


@dataclass
class _FakeEvent:
    """The SSE-style envelope emitted by the SDK."""

    data: _FakeChunk


def _content_event(text: str, finish_reason: str | None = None) -> _FakeEvent:
    return _FakeEvent(
        data=_FakeChunk(
            choices=[
                _FakeChoice(
                    delta=_FakeDelta(content=text),
                    finish_reason=finish_reason,
                )
            ]
        )
    )


def _tool_event(
    *deltas: _FakeToolCallDelta,
    finish_reason: str | None = None,
) -> _FakeEvent:
    return _FakeEvent(
        data=_FakeChunk(
            choices=[
                _FakeChoice(
                    delta=_FakeDelta(tool_calls=list(deltas)),
                    finish_reason=finish_reason,
                )
            ]
        )
    )


class _FakeStream:
    """Async iterator over a fixed list of fake events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class _RaisingStream:
    """Async iterator that raises on creation; used to exercise the open path."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        raise self._exc


class _SDKError(Exception):
    """Stand-in for ``mistralai.SDKError`` carrying a ``status_code``."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


@dataclass
class _FakeChat:
    """Programmable ``client.chat`` namespace.

    ``script`` is a list of either:
      * a list of fake events (success), or
      * a callable that raises an exception (failure).

    Each call to :meth:`stream_async` consumes one entry. The list is
    cycled so a test can compose retry sequences (e.g., two 429s
    followed by success).
    """

    script: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def stream_async(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("FakeChat.stream_async called with empty script")
        next_step = self.script.pop(0)
        if callable(next_step):
            # Failure path: invoke to raise.
            next_step()
        if isinstance(next_step, BaseException):
            raise next_step
        return _FakeStream(next_step)


class _FakeClient:
    """The ``mistralai.Mistral`` client surface our backend touches."""

    def __init__(self, chat: _FakeChat) -> None:
        self.chat = chat


class _FakeCredentialStore:
    """In-memory :class:`CredentialBackend` for the ``from_credential_store`` tests."""

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(store or {})

    def set(self, name: str, value: str) -> None:
        self._store[name] = value

    def get(self, name: str) -> str | None:
        return self._store.get(name)

    def delete(self, name: str) -> None:
        self._store.pop(name, None)

    def list_names(self) -> list[str]:
        return sorted(self._store)

    def wipe(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_messages() -> list[Any]:
    return [
        {"role": "system", "content": "You are JARVIS."},
        {"role": "user", "content": "Hello."},
    ]


def _sample_tools() -> list[Any]:
    return [
        {
            "type": "function",
            "function": {
                "name": "WeatherSkill",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


def _make_backend(
    fake_chat: _FakeChat,
    *,
    api_key: str = "sk-test-secret-12345",
    max_retries: int = 3,
    retry_backoff_initial_ms: int = 1,
    log_filter: LogRedactionFilter | None = None,
) -> MistralBackend:
    return MistralBackend(
        api_key=api_key,
        client=_FakeClient(fake_chat),
        max_retries=max_retries,
        retry_backoff_initial_ms=retry_backoff_initial_ms,
        log_redaction_filter=log_filter,
    )


# ---------------------------------------------------------------------------
# Construction / configuration tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults_match_design(self) -> None:
        backend = _make_backend(_FakeChat(script=[]))
        assert backend.endpoint == DEFAULT_MISTRAL_ENDPOINT
        assert backend.model == DEFAULT_MISTRAL_MODEL
        assert backend.max_retries == 3

    def test_rejects_empty_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            MistralBackend(api_key="", client=_FakeClient(_FakeChat()))

    def test_rejects_negative_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            MistralBackend(
                api_key="k",
                client=_FakeClient(_FakeChat()),
                max_retries=-1,
            )

    def test_repr_does_not_leak_api_key(self) -> None:
        secret = "sk-redacted-12345-very-secret"
        backend = _make_backend(_FakeChat(), api_key=secret)
        rendered = repr(backend)
        assert secret not in rendered
        assert "endpoint=" in rendered

    def test_str_does_not_leak_api_key(self) -> None:
        # ``str()`` falls back to ``__repr__`` by default; defence in depth.
        secret = "sk-mistral-no-leak-test"
        backend = _make_backend(_FakeChat(), api_key=secret)
        assert secret not in str(backend)

    def test_api_key_registered_with_log_filter(self) -> None:
        secret = "sk-redactor-target-9999"
        log_filter = LogRedactionFilter()
        _make_backend(_FakeChat(), api_key=secret, log_filter=log_filter)

        # Build a real LogRecord and run it through the filter; the
        # secret must be absent from the formatted message.
        record = logging.LogRecord(
            name="jarvis.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=f"sending request with key {secret}",
            args=None,
            exc_info=None,
        )
        log_filter.filter(record)
        assert secret not in record.getMessage()

    def test_api_key_is_not_in_public_attributes(self) -> None:
        # Property 8 / CP11: no public attribute should expose the key.
        secret = "sk-deep-introspection-zzzz"
        backend = _make_backend(_FakeChat(), api_key=secret)

        public_names = [n for n in dir(backend) if not n.startswith("_")]
        for name in public_names:
            value = getattr(backend, name, None)
            if isinstance(value, str):
                assert secret not in value, f"leaked via public attr {name}"


# ---------------------------------------------------------------------------
# from_credential_store
# ---------------------------------------------------------------------------


class TestFromCredentialStore:
    def test_pulls_key_from_store(self) -> None:
        store = _FakeCredentialStore({DEFAULT_API_KEY_CREDENTIAL: "sk-from-store"})
        chat = _FakeChat()
        backend = MistralBackend.from_credential_store(
            store,
            client=_FakeClient(chat),
        )
        assert backend.endpoint == DEFAULT_MISTRAL_ENDPOINT

    def test_missing_key_raises_credential_missing(self) -> None:
        store = _FakeCredentialStore()  # empty
        with pytest.raises(MistralCredentialMissingError):
            MistralBackend.from_credential_store(
                store,
                client=_FakeClient(_FakeChat()),
            )

    def test_empty_string_key_raises_credential_missing(self) -> None:
        store = _FakeCredentialStore({DEFAULT_API_KEY_CREDENTIAL: ""})
        with pytest.raises(MistralCredentialMissingError):
            MistralBackend.from_credential_store(
                store,
                client=_FakeClient(_FakeChat()),
            )

    def test_custom_credential_name(self) -> None:
        store = _FakeCredentialStore({"corp/mistral": "sk-corp"})
        backend = MistralBackend.from_credential_store(
            store,
            api_key_credential_name="corp/mistral",
            client=_FakeClient(_FakeChat()),
        )
        assert backend.model == DEFAULT_MISTRAL_MODEL


# ---------------------------------------------------------------------------
# Streaming: content deltas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestContentStreaming:
    async def test_yields_content_delta_events(self) -> None:
        events = [
            _content_event("Hello"),
            _content_event(", "),
            _content_event("sir.", finish_reason="stop"),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[Any] = []
        async with backend.stream(_sample_messages(), tools=[]) as stream:
            async for event in stream:
                out.append(event)

        assert all(isinstance(e, ContentDeltaEvent) for e in out)
        assert "".join(e.text for e in out) == "Hello, sir."

    async def test_skips_empty_content_heartbeats(self) -> None:
        events = [
            _content_event(""),
            _content_event("Hi"),
            _content_event("", finish_reason="stop"),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[Any] = []
        async with backend.stream(_sample_messages(), tools=[]) as stream:
            async for event in stream:
                out.append(event)

        assert len(out) == 1
        assert out[0].text == "Hi"

    async def test_payload_is_forwarded_to_client(self) -> None:
        chat = _FakeChat(script=[[_content_event("ok", finish_reason="stop")]])
        backend = _make_backend(chat)

        async with backend.stream(_sample_messages(), tools=_sample_tools()):
            pass

        assert len(chat.calls) == 1
        call = chat.calls[0]
        assert call["model"] == DEFAULT_MISTRAL_MODEL
        assert call["messages"] == _sample_messages()
        assert call["tools"] == _sample_tools()

    async def test_empty_tools_omitted_from_payload(self) -> None:
        chat = _FakeChat(script=[[_content_event("ok", finish_reason="stop")]])
        backend = _make_backend(chat)

        async with backend.stream(_sample_messages(), tools=[]):
            pass

        call = chat.calls[0]
        assert "tools" not in call

    async def test_per_call_model_override(self) -> None:
        chat = _FakeChat(script=[[_content_event("ok", finish_reason="stop")]])
        backend = _make_backend(chat)

        async with backend.stream(
            _sample_messages(),
            tools=[],
            model="mistral-small-latest",
        ):
            pass

        assert chat.calls[0]["model"] == "mistral-small-latest"


# ---------------------------------------------------------------------------
# Streaming: tool-call reassembly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolCallReassembly:
    async def test_reassembles_streamed_fragments(self) -> None:
        # The model emits the call across four fragments: id, name,
        # then two argument string fragments. Final terminal reason
        # is ``tool_calls``.
        events = [
            _tool_event(_FakeToolCallDelta(id="call-abc", index=0)),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(name="WeatherSkill"),
                    index=0,
                )
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(arguments='{"location":'),
                    index=0,
                )
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(arguments=' "Bandung,ID"}'),
                    index=0,
                ),
                finish_reason="tool_calls",
            ),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[Any] = []
        async with backend.stream(_sample_messages(), tools=_sample_tools()) as stream:
            async for event in stream:
                out.append(event)

        tool_events = [e for e in out if isinstance(e, ToolCallEvent)]
        assert len(tool_events) == 1
        tc = tool_events[0].tool_call
        assert tc.id == "call-abc"
        assert tc.skill_name == "WeatherSkill"
        assert tc.arguments == {"location": "Bandung,ID"}
        # raw_arguments should preserve the original concatenated bytes.
        assert json.loads(tc.raw_arguments) == {"location": "Bandung,ID"}

    async def test_multiple_tool_calls_emitted_in_index_order(self) -> None:
        # The model emits two tool calls in a single response. They
        # arrive interleaved by index but must surface in index order.
        events = [
            _tool_event(
                _FakeToolCallDelta(id="b", index=1),
                _FakeToolCallDelta(id="a", index=0),
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(name="SkillB"),
                    index=1,
                ),
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(name="SkillA"),
                    index=0,
                ),
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(arguments="{}"),
                    index=1,
                ),
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(arguments="{}"),
                    index=0,
                ),
                finish_reason="tool_calls",
            ),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[ToolCallEvent] = []
        async with backend.stream(_sample_messages(), tools=_sample_tools()) as stream:
            async for event in stream:
                if isinstance(event, ToolCallEvent):
                    out.append(event)

        assert [e.tool_call.skill_name for e in out] == ["SkillA", "SkillB"]
        assert [e.tool_call.id for e in out] == ["a", "b"]

    async def test_dict_arguments_adopted_wholesale(self) -> None:
        # Some SDK paths emit ``arguments`` as a parsed dict on the
        # final fragment instead of a JSON string.
        events = [
            _tool_event(
                _FakeToolCallDelta(
                    id="x",
                    function=_FakeFunctionDelta(name="EchoSkill"),
                    index=0,
                )
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(
                        arguments={"text": "hello", "count": 3}
                    ),
                    index=0,
                ),
                finish_reason="tool_calls",
            ),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[ToolCallEvent] = []
        async with backend.stream(_sample_messages(), tools=_sample_tools()) as stream:
            async for event in stream:
                if isinstance(event, ToolCallEvent):
                    out.append(event)

        assert len(out) == 1
        tc = out[0].tool_call
        assert tc.arguments == {"text": "hello", "count": 3}
        # raw_arguments must be JSON-parseable back to the same value.
        assert json.loads(tc.raw_arguments) == {"text": "hello", "count": 3}

    async def test_synthetic_id_when_model_omits_one(self) -> None:
        events = [
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(name="NoIdSkill"),
                    index=0,
                )
            ),
            _tool_event(
                _FakeToolCallDelta(
                    function=_FakeFunctionDelta(arguments="{}"),
                    index=0,
                ),
                finish_reason="tool_calls",
            ),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[ToolCallEvent] = []
        async with backend.stream(_sample_messages(), tools=_sample_tools()) as stream:
            async for event in stream:
                if isinstance(event, ToolCallEvent):
                    out.append(event)

        assert len(out) == 1
        assert out[0].tool_call.id.startswith("mistral-synth-")

    async def test_flushes_pending_tool_calls_at_end_of_stream(self) -> None:
        # No terminal finish_reason; reassembly must still happen on
        # stream exhaustion.
        events = [
            _tool_event(
                _FakeToolCallDelta(
                    id="c",
                    function=_FakeFunctionDelta(name="S", arguments='{"k":1}'),
                    index=0,
                )
            ),
        ]
        chat = _FakeChat(script=[events])
        backend = _make_backend(chat)

        out: list[ToolCallEvent] = []
        async with backend.stream(_sample_messages(), tools=_sample_tools()) as stream:
            async for event in stream:
                if isinstance(event, ToolCallEvent):
                    out.append(event)

        assert len(out) == 1
        assert out[0].tool_call.skill_name == "S"
        assert out[0].tool_call.arguments == {"k": 1}


# ---------------------------------------------------------------------------
# HTTP error semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorMapping:
    @pytest.mark.parametrize("status", [401, 403])
    async def test_401_403_raises_credential_error(self, status: int) -> None:
        chat = _FakeChat(script=[_SDKError(status)])
        backend = _make_backend(chat)

        with pytest.raises(MistralAuthError) as excinfo:
            async with backend.stream(_sample_messages(), tools=[]) as stream:
                async for _ in stream:
                    pass

        assert excinfo.value.status_code == status
        # Backwards-compat alias points to the same class.
        assert MistralCredentialError is MistralAuthError

    async def test_429_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two 429s, then a success on the third attempt.
        success_events = [_content_event("ok", finish_reason="stop")]
        chat = _FakeChat(
            script=[_SDKError(429), _SDKError(429), success_events]
        )
        backend = _make_backend(chat, max_retries=3, retry_backoff_initial_ms=1)

        # Replace asyncio.sleep with a no-op so retry waits are instant.
        async def _instant_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        out: list[Any] = []
        async with backend.stream(_sample_messages(), tools=[]) as stream:
            async for event in stream:
                out.append(event)

        assert len(out) == 1
        assert isinstance(out[0], ContentDeltaEvent)
        assert out[0].text == "ok"
        assert len(chat.calls) == 3

    async def test_429_exhausted_raises_rate_limit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Four 429s — total_attempts = max_retries + 1 = 4.
        chat = _FakeChat(
            script=[_SDKError(429), _SDKError(429), _SDKError(429), _SDKError(429)]
        )
        backend = _make_backend(chat, max_retries=3, retry_backoff_initial_ms=1)

        async def _instant_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        with pytest.raises(MistralRateLimitError) as excinfo:
            async with backend.stream(_sample_messages(), tools=[]) as stream:
                async for _ in stream:
                    pass

        # Failure-mode taxonomy stable code.
        assert excinfo.value.code == "rate_limited"
        assert len(chat.calls) == 4

    async def test_max_retries_zero_disables_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat = _FakeChat(script=[_SDKError(429)])
        backend = _make_backend(chat, max_retries=0, retry_backoff_initial_ms=1)

        async def _instant_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        with pytest.raises(MistralRateLimitError):
            async with backend.stream(_sample_messages(), tools=[]) as stream:
                async for _ in stream:
                    pass

        # Exactly one attempt before exhaustion.
        assert len(chat.calls) == 1

    async def test_5xx_raises_stream_error_without_retry(self) -> None:
        chat = _FakeChat(script=[_SDKError(503)])
        backend = _make_backend(chat, max_retries=3, retry_backoff_initial_ms=1)

        with pytest.raises(MistralStreamError) as excinfo:
            async with backend.stream(_sample_messages(), tools=[]) as stream:
                async for _ in stream:
                    pass

        # Auth and rate-limit errors are subclasses; ensure we got the
        # *generic* one (no retry budget consumed).
        assert not isinstance(excinfo.value, MistralAuthError)
        assert not isinstance(excinfo.value, MistralRateLimitError)
        assert len(chat.calls) == 1

    async def test_non_http_exception_raises_stream_error(self) -> None:
        # An exception with no recognisable status code is treated as
        # a generic backend failure (Requirement 12.4 path).
        chat = _FakeChat(script=[RuntimeError("connection reset")])
        backend = _make_backend(chat)

        with pytest.raises(MistralStreamError):
            async with backend.stream(_sample_messages(), tools=[]) as stream:
                async for _ in stream:
                    pass


# ---------------------------------------------------------------------------
# Stream lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamLifecycle:
    async def test_stream_aclose_called_on_normal_exit(self) -> None:
        events = [_content_event("ok", finish_reason="stop")]
        # Capture the FakeStream so we can inspect ``closed`` after.
        closure_holder: dict[str, _FakeStream] = {}

        original = _FakeChat.stream_async

        async def patched(self: _FakeChat, **kwargs: Any) -> _FakeStream:
            stream = await original(self, **kwargs)
            closure_holder["stream"] = stream
            return stream

        chat = _FakeChat(script=[events])
        chat.stream_async = patched.__get__(chat, _FakeChat)  # type: ignore[method-assign]
        backend = _make_backend(chat)

        async with backend.stream(_sample_messages(), tools=[]) as stream:
            async for _ in stream:
                pass

        assert closure_holder["stream"].closed is True

    async def test_stream_aclose_called_on_early_exit(self) -> None:
        events = [
            _content_event("first"),
            _content_event("second"),
            _content_event("third", finish_reason="stop"),
        ]
        closure_holder: dict[str, _FakeStream] = {}

        original = _FakeChat.stream_async

        async def patched(self: _FakeChat, **kwargs: Any) -> _FakeStream:
            stream = await original(self, **kwargs)
            closure_holder["stream"] = stream
            return stream

        chat = _FakeChat(script=[events])
        chat.stream_async = patched.__get__(chat, _FakeChat)  # type: ignore[method-assign]
        backend = _make_backend(chat)

        async with backend.stream(_sample_messages(), tools=[]) as stream:
            async for event in stream:
                # Break after first delta to exercise early-exit teardown.
                assert isinstance(event, ContentDeltaEvent)
                break

        assert closure_holder["stream"].closed is True
