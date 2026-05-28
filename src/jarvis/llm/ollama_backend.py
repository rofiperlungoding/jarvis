"""Local Mistral fallback over the Ollama OpenAI-compatible chat endpoint.

This module implements :class:`OllamaBackend`, the local-LLM half of the
:class:`~jarvis.llm.base.LLMBackend` Protocol. It is selected by
:class:`~jarvis.llm.selector.BackendSelector` when the cloud Mistral
endpoint is unhealthy (Requirement 12.4) and serves the same
``messages`` / ``tools`` request shape as the Mistral backend so the
Dialog_Manager remains backend-agnostic (Property 14 / CP equivalence).

Why hand-rolled HTTP rather than the ``ollama`` Python SDK
---------------------------------------------------------

Two reasons:

1. The official ``ollama`` SDK at the time of writing does not surface
   the raw NDJSON line stream — it materialises whole messages and so
   forfeits the per-token latency win that lets us hit the 800 ms
   Latency_Budget at sentence boundaries (Requirement 12.1, 19.5).
2. Property 14 / CP (Backend fallback equivalence shape) requires that
   the JSON payload sent to Ollama contains the *same* ``messages`` and
   a ``tools`` array of equal length whose entries share ``name`` and
   ``parameters`` with the Mistral payload. Driving the wire format
   directly with :class:`httpx.AsyncClient` is the simplest way to
   guarantee that pass-through with no SDK transformations in between.

NDJSON event shape
------------------

Ollama's ``/api/chat`` endpoint streams newline-delimited JSON. Each
line carries an incremental assistant ``message`` plus a ``done`` flag::

    {"model": "mistral", "message": {"role": "assistant",
        "content": "Hello"}, "done": false}
    {"model": "mistral", "message": {"role": "assistant",
        "content": ", sir"}, "done": false}
    {"model": "mistral", "message": {"role": "assistant",
        "content": "", "tool_calls": [
            {"function": {"name": "WeatherSkill",
                          "arguments": {"location": "Bandung,ID"}}}
        ]}, "done": false}
    {"model": "mistral", "done": true, "done_reason": "stop", ...}

We translate each line into the backend-neutral
:class:`~jarvis.llm.base.LLMEvent` discriminated union exactly as the
Mistral backend does:

* Non-empty ``message.content`` deltas become :class:`ContentDeltaEvent`.
* Each entry of ``message.tool_calls`` becomes a single
  :class:`ToolCallEvent`. Ollama emits the call's ``arguments`` as a
  parsed dict on most models and as a JSON string on others — we
  normalise to *both* (``arguments`` parsed, ``raw_arguments`` original)
  so :class:`~jarvis.llm.base.ToolCall` can be round-tripped byte-equal
  for CP1.

Validates: Requirement 12.4
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import json
from typing import Any, Final
import uuid

import httpx

from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    Stream,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)

__all__ = [
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OLLAMA_MODEL",
    "OllamaBackend",
    "OllamaStreamError",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default Ollama daemon address. Matches ``llm.fallback.endpoint`` in
#: ``src/jarvis/config/default.toml`` and the URL referenced in the
#: design's fallback flow diagram.
DEFAULT_OLLAMA_ENDPOINT: Final[str] = "http://localhost:11434"

#: Default model id. Matches ``llm.fallback.model`` in ``default.toml``;
#: the operator can override per-call by passing ``model=...`` to
#: :meth:`OllamaBackend.stream`.
DEFAULT_OLLAMA_MODEL: Final[str] = "mistral"

#: Translation table from generic LLM kwargs (the names the
#: Dialog_Manager and BackendSelector pass through unchanged) to the
#: Ollama-specific keys nested under the ``options`` block. Anything not
#: listed here is forwarded verbatim to Ollama's top-level body when
#: it's a known top-level field, or dropped otherwise (we'd rather emit
#: a clean payload than silently misroute a parameter).
_KWARG_TO_OLLAMA_OPTION: Final[dict[str, str]] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "num_predict",
    "num_predict": "num_predict",
    "seed": "seed",
    "random_seed": "seed",  # Mistral spelling
    "stop": "stop",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}

#: Top-level Ollama request fields that the Dialog_Manager / config may
#: pass through as kwargs (rather than nesting under ``options``).
_OLLAMA_TOPLEVEL_PASSTHROUGH: Final[frozenset[str]] = frozenset(
    {"format", "keep_alive", "system", "template"}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OllamaStreamError(RuntimeError):
    """Raised when the Ollama daemon embeds an ``"error"`` field in the
    NDJSON stream after the HTTP status was OK.

    Distinguished from :class:`httpx.HTTPStatusError` (transport-level
    failure) so :class:`~jarvis.llm.selector.BackendSelector` can treat
    it as a hard backend failure that should *not* feed the circuit
    breaker recovery path — if the local daemon itself is reporting an
    error mid-stream, the Mistral cloud almost certainly is not the
    cause.
    """


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OllamaBackend:
    """Stream chat completions from a local Ollama daemon.

    Conforms structurally to :class:`~jarvis.llm.base.LLMBackend`. The
    request shape mirrors Mistral so :class:`BackendSelector` can route
    a turn to either backend without rewriting the payload.

    Parameters
    ----------
    endpoint:
        Base URL of the Ollama daemon. Defaults to
        :data:`DEFAULT_OLLAMA_ENDPOINT`. A trailing ``/`` is stripped so
        callers may pass either ``http://localhost:11434`` or
        ``http://localhost:11434/``.
    model:
        Default model id appended to every request when the caller does
        not override it via ``model=`` in :meth:`stream` kwargs. Defaults
        to :data:`DEFAULT_OLLAMA_MODEL`.
    client:
        Optional pre-configured :class:`httpx.AsyncClient`. When
        supplied, the backend uses it for every request and does *not*
        close it on context exit — ownership stays with the caller. This
        makes the backend trivially testable with ``respx`` /
        ``httpx.MockTransport`` and avoids spawning a fresh TCP pool for
        every dialog turn in production.
    connect_timeout_s:
        Connect / write / pool timeout in seconds. Read timeout is held
        at :data:`None` because streaming responses are arbitrarily long
        and the upstream Voice_Pipeline already enforces its own hard
        timeout (Requirement 17.3).
    """

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        model: str = DEFAULT_OLLAMA_MODEL,
        client: httpx.AsyncClient | None = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("OllamaBackend.endpoint must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("OllamaBackend.model must be a non-empty string")
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._client = client
        self._connect_timeout_s = connect_timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[Stream]:
        """Open a streaming ``/api/chat`` request and return the event stream.

        See :class:`~jarvis.llm.base.LLMBackend` for the contract
        documentation. The returned async context manager is a
        :class:`contextlib._AsyncGeneratorContextManager`; use it as::

            async with backend.stream(messages, tools=tools) as events:
                async for event in events:
                    ...

        The HTTP connection is held open for the duration of the
        ``async with`` block and is closed when the block exits, even
        if the consumer breaks out of the inner ``async for`` early.
        """
        return self._stream(list(messages), list(tools), dict(kwargs))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Stream]:
        """Implementation detail backing :meth:`stream`.

        Wrapped in :func:`contextlib.asynccontextmanager` so the
        body's ``yield`` produces a context manager whose ``__aexit__``
        guarantees the underlying ``httpx`` response and (when owned)
        client are closed. Splitting the public ``stream`` from this
        helper keeps the Protocol's plain-method signature without
        forcing the caller through an extra ``await``.
        """
        payload = self._build_payload(messages, tools, kwargs)
        url = f"{self._endpoint}/api/chat"

        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._connect_timeout_s,
                    read=None,
                    write=self._connect_timeout_s,
                    pool=self._connect_timeout_s,
                )
            )
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                yield self._iter_events(response)
        finally:
            if owns_client:
                await client.aclose()

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate ``(messages, tools, kwargs)`` into Ollama's wire body.

        Property 14 invariant: ``messages`` and ``tools`` are forwarded
        unchanged. Tool entries already carry the
        ``{"type": "function", "function": {"name", "description",
        "parameters"}}`` shape produced by
        :func:`jarvis.llm.mistral_schema.to_mistral_tool`, which is
        exactly what Ollama's OpenAI-compatible endpoint expects.
        """
        # Pop the model override before iterating remaining kwargs so we
        # don't accidentally route it through `_KWARG_TO_OLLAMA_OPTION`.
        model = kwargs.pop("model", self._model)
        if not isinstance(model, str) or not model:
            raise ValueError("model kwarg must be a non-empty string")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        # Per Property 14: an empty `tools` list still implies "no
        # function calling on this turn" — emit it only when non-empty so
        # we don't perturb Ollama's parser with the empty array on models
        # that warn on it.
        if tools:
            payload["tools"] = tools

        # Translate the generic kwargs into the nested `options` block.
        options: dict[str, Any] = {}
        for kwarg_key in list(kwargs):
            ollama_key = _KWARG_TO_OLLAMA_OPTION.get(kwarg_key)
            if ollama_key is not None:
                options[ollama_key] = kwargs.pop(kwarg_key)

        # Allow callers to pass an explicit `options` dict (e.g., loaded
        # straight from a config table). Caller-supplied keys win on
        # conflict so the config file remains the source of truth when
        # both paths are used.
        explicit_options = kwargs.pop("options", None)
        if explicit_options is not None:
            if not isinstance(explicit_options, dict):
                raise TypeError("`options` kwarg must be a dict if provided")
            options = {**options, **explicit_options}
        if options:
            payload["options"] = options

        # Top-level Ollama-native fields the caller may forward verbatim.
        for passthrough_key in _OLLAMA_TOPLEVEL_PASSTHROUGH:
            if passthrough_key in kwargs:
                payload[passthrough_key] = kwargs.pop(passthrough_key)

        # Anything left in `kwargs` is unknown — stay strict so the
        # Dialog_Manager's request shape is auditable.
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"OllamaBackend.stream() got unexpected keyword arguments: {unexpected}"
            )

        return payload

    @staticmethod
    async def _iter_events(response: httpx.Response) -> AsyncIterator[LLMEvent]:
        """Translate Ollama's NDJSON response into :class:`LLMEvent` values.

        The generator terminates either when Ollama emits ``done: true``
        or when the underlying line iterator is exhausted (which is
        how :mod:`httpx` signals end-of-stream).
        """
        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped:
                # Blank lines are allowed in NDJSON; skip without ending.
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # A malformed line is unusual but should not poison the
                # whole stream; the next valid line still carries the
                # `done` flag. Drop and keep reading.
                continue
            if not isinstance(obj, dict):
                continue

            # Ollama may surface daemon-level errors mid-stream when the
            # HTTP status was 200. Treat that as a hard failure so the
            # Dialog_Manager / BackendSelector can react rather than
            # silently producing partial output.
            error = obj.get("error")
            if isinstance(error, str) and error:
                raise OllamaStreamError(error)

            message = obj.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content != "":
                    yield ContentDeltaEvent(text=content)

                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw_tc in tool_calls:
                        event = OllamaBackend._tool_call_event(raw_tc)
                        if event is not None:
                            yield event

            # `done: true` marks the final line of the stream.
            if obj.get("done") is True:
                return

    @staticmethod
    def _tool_call_event(raw: Any) -> ToolCallEvent | None:
        """Normalise an Ollama ``tool_calls[i]`` element to :class:`ToolCallEvent`.

        Returns ``None`` when the element is malformed. Two argument
        encodings are accepted because models hosted by Ollama disagree:

        * Most models emit a parsed JSON ``object`` (``dict``). We keep
          it as ``arguments`` and re-serialise via :func:`json.dumps`
          for ``raw_arguments`` so CP1 round-trip is preserved.
        * Some models emit a JSON-encoded ``string``. We parse it for
          ``arguments`` and keep the original string for
          ``raw_arguments``.

        Ollama does not assign tool-call ids. We synthesise a UUID-based
        id so the Dialog_Manager can correlate the upcoming
        :class:`~jarvis.llm.base.ToolMessage` reply with this call,
        matching how :class:`MistralBackend` will treat its own ids.
        """
        if not isinstance(raw, dict):
            return None
        function = raw.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return None

        arguments = function.get("arguments", {})
        parsed: dict[str, Any]
        raw_arguments: str
        if isinstance(arguments, str):
            raw_arguments = arguments
            try:
                decoded = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                # An un-parseable argument string is a model bug; surface
                # an empty dict so Skill schema validation downstream
                # produces a clean `schema_violation` rather than a JSON
                # error in the dispatch loop.
                decoded = {}
            parsed = decoded if isinstance(decoded, dict) else {}
        elif isinstance(arguments, dict):
            parsed = arguments
            try:
                raw_arguments = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                # Fall back to a best-effort string form; the parsed
                # dict is still authoritative for Skill dispatch.
                raw_arguments = json.dumps(
                    {k: str(v) for k, v in arguments.items()},
                    ensure_ascii=False,
                )
        else:
            return None

        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"ollama-{uuid.uuid4().hex}"

        return ToolCallEvent(
            tool_call=ToolCall(
                id=call_id,
                skill_name=name,
                arguments=parsed,
                raw_arguments=raw_arguments,
            )
        )
