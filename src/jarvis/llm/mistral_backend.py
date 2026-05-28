"""Mistral la Plateforme cloud backend for the Dialog_Manager.

This module implements :class:`MistralBackend`, the cloud half of the
:class:`~jarvis.llm.base.LLMBackend` Protocol. It is the *default*
backend selected by :class:`~jarvis.llm.selector.BackendSelector` when
the Mistral API is reachable (Requirements 19.1, 19.5) and is wrapped
underneath the BackendSelector's circuit breaker so the local Ollama
fallback can take over on degradation (Requirement 12.4).

Why this module exists
----------------------

The Dialog_Manager needs an async streaming chat completion that:

* Streams content deltas to the SentenceAccumulator → TTS pipeline so
  speech synthesis can begin at the first sentence boundary
  (Requirements 12.2, 19.5).
* Maps each Skill registered in :class:`SkillRegistry` to a Mistral
  function-calling tool definition, and reassembles streamed tool-call
  fragments into a single :class:`~jarvis.llm.base.ToolCall` so the
  Dialog_Manager never has to reason about partial invocations
  (Requirement 19.4).
* Pulls the API key from
  :class:`~jarvis.security.credential_store.CredentialStore` at startup
  and never writes the key value to logs, telemetry, or any persisted
  file outside the credential store (Requirement 19.3, CP11).
* Handles HTTP error semantics specified by Requirement 19:

  - **HTTP 401 / 403** raises :class:`MistralCredentialError`. The
    Dialog_Manager catches it and routes the next turn through
    ``CredentialUpdateFlow``, which calls
    ``CredentialStore.set("mistral/api_key", ...)`` (Requirement 19.7).
  - **HTTP 429** retries with tenacity exponential backoff up to three
    additional attempts (four total). When the retry budget is
    exhausted, raises :class:`MistralRateLimitError`, whose ``code``
    is ``"rate_limited"`` per the design's failure-mode taxonomy
    (Requirement 19.8).
  - All other failures raise :class:`MistralStreamError` so the
    BackendSelector can open its circuit (Requirement 12.4).

Why lazy import of ``mistralai``
--------------------------------

``mistralai`` ships with heavyweight optional dependencies (``httpx``
TLS pools, pydantic models). We import it at first use rather than at
module-import time so:

1. Tests that exercise the event-mapping and retry logic with a fake
   async client never have to install or mock-import ``mistralai``.
2. Linux / macOS dev environments where the cloud backend is disabled
   in the user config can avoid the cost altogether.

The same pattern is used by :class:`OllamaBackend` for its httpx
client, and by :class:`KeyringBackend` for the optional ``keyring``
package.

Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import json
import logging
from typing import Any, Final, cast
import uuid

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jarvis.llm.base import (
    ContentDeltaEvent,
    LLMEvent,
    Message,
    Stream,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
)
from jarvis.security.credential_store import CredentialBackend
from jarvis.security.log_redaction import LogRedactionFilter

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_API_KEY_CREDENTIAL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MISTRAL_ENDPOINT",
    "DEFAULT_MISTRAL_MODEL",
    "DEFAULT_RETRY_BACKOFF_INITIAL_MS",
    "MistralAuthError",
    "MistralBackend",
    "MistralCredentialError",
    "MistralCredentialMissingError",
    "MistralRateLimitError",
    "MistralStreamError",
]


# ---------------------------------------------------------------------------
# Constants (mirror ``[llm.mistral]`` in ``src/jarvis/config/default.toml``)
# ---------------------------------------------------------------------------

#: Default cloud endpoint. Matches ``llm.mistral.endpoint`` in
#: ``default.toml`` and Requirement 19.1.
DEFAULT_MISTRAL_ENDPOINT: Final[str] = "https://api.mistral.ai"

#: Default model id. Matches ``llm.mistral.model`` in ``default.toml``
#: and Requirement 19.2. Operators can override per-call by passing
#: ``model=...`` to :meth:`MistralBackend.stream` or by changing the
#: backend's ``model`` constructor argument (Requirement 19.6).
DEFAULT_MISTRAL_MODEL: Final[str] = "mistral-large-latest"

#: Default credential name under which the API key is stored. Matches
#: ``llm.mistral.api_key_credential`` in ``default.toml`` and is the
#: name passed to ``CredentialStore.set`` by ``CredentialUpdateFlow``
#: (Requirement 19.3, 19.7).
DEFAULT_API_KEY_CREDENTIAL: Final[str] = "mistral/api_key"

#: Default number of *additional* attempts on HTTP 429 (Requirement 19.8).
#: The total number of attempts is therefore ``DEFAULT_MAX_RETRIES + 1``.
DEFAULT_MAX_RETRIES: Final[int] = 3

#: Default initial backoff in milliseconds; doubled on each retry up to
#: ``8 s``. The exact schedule is opaque to callers — only the maximum
#: number of retries is observable.
DEFAULT_RETRY_BACKOFF_INITIAL_MS: Final[int] = 200

#: Finish reasons that signal the model has emitted a complete batch of
#: tool calls (or completed normally). When we see one of these on a
#: streamed choice we can flush the corresponding accumulator slot.
_TERMINAL_FINISH_REASONS: Final[frozenset[str]] = frozenset(
    {"stop", "tool_calls", "length", "error", "model_length", "content_filter"}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MistralStreamError(RuntimeError):
    """Generic Mistral backend failure surfaced to the BackendSelector.

    Carries the original exception as ``__cause__`` so the selector and
    its audit log can attribute the cause without coupling to the SDK's
    error hierarchy.
    """


class MistralAuthError(MistralStreamError):
    """Raised when the Mistral API rejects the API key.

    Includes both HTTP 401 (invalid / revoked key) and HTTP 403
    (forbidden, e.g., API key valid but lacks the required entitlement).
    The Dialog_Manager catches this and delegates to
    ``CredentialUpdateFlow`` so the user can supply a new key
    (Requirement 19.7).

    Distinct from :class:`MistralRateLimitError` because the recovery
    path is different — re-trying with the same key is futile.
    """

    #: HTTP status code that triggered the error. Always 401 or 403.
    status_code: int

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# Backwards-compat alias kept short so call sites can ``except
# MistralCredentialError`` without importing two symbols. The "auth"
# spelling is preserved as the canonical class name because that is
# what the design's failure-mode table uses.
MistralCredentialError = MistralAuthError


class MistralCredentialMissingError(MistralStreamError):
    """Raised when no API key is available in the Credential_Store.

    Surfaces a different recovery path than 401/403 (the user has never
    set a key) so the Dialog_Manager can offer onboarding instead of
    re-prompting an existing-key replacement.
    """


class MistralRateLimitError(MistralStreamError):
    """Raised when HTTP 429 retries are exhausted (Requirement 19.8).

    The ``code`` attribute is the string ``"rate_limited"`` exactly as
    listed in the design's failure-mode taxonomy, so audit log entries
    and Dialog_Manager error responses can use the literal value
    without extra translation.
    """

    #: Failure-mode taxonomy code. Stable string, do not localise.
    code: Final[str] = "rate_limited"


# ---------------------------------------------------------------------------
# Internal retry signal
# ---------------------------------------------------------------------------


class _Retryable429(Exception):  # noqa: N818 - internal sentinel, not user-facing
    """Internal marker raised inside the tenacity retry loop on HTTP 429.

    Kept private to this module so callers cannot accidentally catch it
    and bypass the public :class:`MistralRateLimitError` contract. The
    original SDK exception is attached via ``__cause__`` so the final
    :class:`MistralRateLimitError` can re-attach it on exhaustion.
    """


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class MistralBackend:
    """Stream chat completions from Mistral la Plateforme.

    Conforms structurally to :class:`~jarvis.llm.base.LLMBackend`. The
    backend pulls its API key from the
    :class:`~jarvis.security.credential_store.CredentialStore` (via the
    :meth:`from_credential_store` classmethod) and registers the key
    value with an optional :class:`LogRedactionFilter` so that any
    accidental logging of the key — by JARVIS code, by ``mistralai``,
    or by ``httpx`` — is scrubbed before the record reaches a handler
    (Requirement 19.3, CP11).

    Parameters
    ----------
    api_key:
        The Mistral API key. Held only on the instance and never
        written to disk or logs by this class. Pass through
        :meth:`from_credential_store` in production code; the direct
        keyword is exposed primarily for tests.
    endpoint:
        Base URL of the Mistral API. Defaults to
        :data:`DEFAULT_MISTRAL_ENDPOINT`. Trailing ``/`` is stripped so
        either spelling is accepted.
    model:
        Default model id. Defaults to :data:`DEFAULT_MISTRAL_MODEL`.
        Per Requirement 19.6, the operator can override per-call by
        passing ``model=...`` to :meth:`stream`.
    max_retries:
        Maximum *additional* attempts on HTTP 429 before raising
        :class:`MistralRateLimitError`. Defaults to
        :data:`DEFAULT_MAX_RETRIES` (3), giving four total attempts.
    retry_backoff_initial_ms:
        Initial backoff in milliseconds. Subsequent waits double up to
        a cap of 8 s. Defaults to
        :data:`DEFAULT_RETRY_BACKOFF_INITIAL_MS` (200).
    request_timeout_ms:
        Optional per-request read timeout passed to the underlying
        ``mistralai`` client. ``None`` (the default) lets the SDK use
        its own default, which the BackendSelector's 3 s circuit
        breaker timeout supersedes anyway (Requirement 12.4).
    client:
        Optional pre-configured ``mistralai.Mistral`` (or compatible)
        async client. Supplying this short-circuits the lazy-import +
        construction logic; useful for tests with a fake. The backend
        does not call ``aclose`` on a caller-supplied client, so
        ownership stays with the caller.
    log_redaction_filter:
        Optional :class:`LogRedactionFilter` to register the API key
        with. Strongly recommended in production so accidental logging
        of the key (e.g., in an HTTP error message that echoes a
        ``Authorization`` header) is scrubbed.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = DEFAULT_MISTRAL_ENDPOINT,
        model: str = DEFAULT_MISTRAL_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_initial_ms: int = DEFAULT_RETRY_BACKOFF_INITIAL_MS,
        request_timeout_ms: int | None = None,
        client: Any = None,
        log_redaction_filter: LogRedactionFilter | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            # Empty / non-string keys are caught here so the eventual
            # 401 from the Mistral API doesn't surface as a confusing
            # auth error from the cloud.
            raise ValueError("MistralBackend.api_key must be a non-empty string")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("MistralBackend.endpoint must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("MistralBackend.model must be a non-empty string")
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("MistralBackend.max_retries must be a non-negative int")
        if (
            not isinstance(retry_backoff_initial_ms, int)
            or retry_backoff_initial_ms <= 0
        ):
            raise ValueError(
                "MistralBackend.retry_backoff_initial_ms must be a positive int"
            )

        # Store api_key in a name-mangled slot so `vars(self)` and
        # default `__repr__` cannot leak it. `__repr__` is also
        # overridden to provide a redacted representation.
        self.__api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._retry_backoff_initial_s = retry_backoff_initial_ms / 1000.0
        self._request_timeout_ms = request_timeout_ms
        self._client = client  # may be None; resolved lazily

        # Register the secret with the redaction filter as early as
        # possible — *before* any code path that might log the key.
        if log_redaction_filter is not None:
            log_redaction_filter.register_secret(api_key)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_credential_store(
        cls,
        credential_store: CredentialBackend,
        *,
        api_key_credential_name: str = DEFAULT_API_KEY_CREDENTIAL,
        log_redaction_filter: LogRedactionFilter | None = None,
        **kwargs: Any,
    ) -> MistralBackend:
        """Construct a backend by fetching the API key from a credential store.

        Implements the "pull API key from CredentialStore at startup"
        contract from Requirement 19.3. Raises
        :class:`MistralCredentialMissingError` when the credential is
        not present so ``app.py`` can surface an onboarding flow
        instead of crashing the dialog loop.

        Parameters
        ----------
        credential_store:
            Any object satisfying the
            :class:`~jarvis.security.credential_store.CredentialBackend`
            Protocol. In production this is the DPAPI-backed
            :class:`CredentialStore`; tests typically pass a fake.
        api_key_credential_name:
            The credential name under which the key is stored. Defaults
            to :data:`DEFAULT_API_KEY_CREDENTIAL`.
        log_redaction_filter:
            Forwarded to ``__init__`` so the freshly-fetched key is
            registered for redaction before this method returns.
        **kwargs:
            Forwarded to :meth:`__init__` (``endpoint``, ``model``,
            ``max_retries``, etc.).
        """
        api_key = credential_store.get(api_key_credential_name)
        if api_key is None or api_key == "":
            raise MistralCredentialMissingError(
                f"no Mistral API key found under credential name "
                f"{api_key_credential_name!r}; run CredentialUpdateFlow"
            )
        return cls(
            api_key=api_key,
            log_redaction_filter=log_redaction_filter,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Default model id used when ``stream`` is called without override."""
        return self._model

    @property
    def endpoint(self) -> str:
        """Base URL of the Mistral API used by this backend."""
        return self._endpoint

    @property
    def max_retries(self) -> int:
        """Maximum *additional* attempts on HTTP 429."""
        return self._max_retries

    def __repr__(self) -> str:
        # Defence in depth: never include the API key in any
        # representation. Logs that catch unhandled exceptions
        # involving the backend instance routinely run ``repr(self)``,
        # so this is the last line of defence before the redaction
        # filter on the logger (Requirement 19.3).
        return (
            f"MistralBackend(endpoint={self._endpoint!r}, "
            f"model={self._model!r}, max_retries={self._max_retries})"
        )

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[Stream]:
        """Open a streaming ``chat.stream_async`` request and return the event stream.

        See :class:`~jarvis.llm.base.LLMBackend` for the contract
        documentation. The returned async context manager opens the
        streaming HTTP connection on ``__aenter__``, retries on HTTP
        429 up to :attr:`max_retries` times, and tears down the
        connection on ``__aexit__`` even when the consumer breaks out
        of the inner ``async for`` early.

        Raises
        ------
        :class:`MistralCredentialError`
            On HTTP 401 / 403 (Requirement 19.7).
        :class:`MistralRateLimitError`
            On HTTP 429 after exhausting retries (Requirement 19.8).
        :class:`MistralStreamError`
            On any other API or transport failure.
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
        """Implementation backing :meth:`stream`."""
        client = self._resolve_client()
        payload = self._build_payload(messages, tools, kwargs)

        raw_stream = await self._open_stream_with_retry(client, payload)
        # ``raw_stream`` is an async iterator/generator. Some SDK
        # versions return an awaitable that yields an async iterator on
        # ``await``; the retry helper has already unwrapped that for
        # us, so we can iterate directly here.
        try:
            yield self._iter_events(raw_stream)
        finally:
            # If the SDK's stream object exposes an explicit close,
            # call it so connection pool slots are released even when
            # the consumer breaks out of the ``async for`` early.
            aclose = getattr(raw_stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    # Best effort — never let teardown errors mask the
                    # original consumer-side exception.
                    logger.debug(
                        "Mistral stream aclose raised during teardown",
                        exc_info=True,
                    )

    # ---- Client construction ----------------------------------------

    def _resolve_client(self) -> Any:
        """Return the Mistral async client, lazily importing the SDK.

        We delay the import so test environments without ``mistralai``
        installed can still exercise event mapping and retry logic
        with a caller-supplied fake.
        """
        if self._client is not None:
            return self._client
        try:
            from mistralai import (  # noqa: PLC0415 - optional dep loaded only when backend is used
                Mistral,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when missing
            raise MistralStreamError(
                "MistralBackend requires the `mistralai` package. "
                "Install it with `pip install mistralai`."
            ) from exc
        # ``server_url`` is the public knob used by the SDK to override
        # the base endpoint. ``api_key`` is forwarded directly; the SDK
        # holds it in memory and attaches it as the ``Authorization``
        # bearer header on each request.
        client_kwargs: dict[str, Any] = {
            "api_key": self.__api_key,
            "server_url": self._endpoint,
        }
        if self._request_timeout_ms is not None:
            client_kwargs["timeout_ms"] = self._request_timeout_ms
        self._client = Mistral(**client_kwargs)
        return self._client

    # ---- Payload construction ---------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate ``(messages, tools, kwargs)`` into ``chat.stream_async`` kwargs.

        Property 14 invariant: ``messages`` is forwarded unchanged and
        ``tools`` is forwarded unchanged when non-empty so the request
        shape is identical to the OllamaBackend's payload (modulo SDK
        wrapping).
        """
        model = kwargs.pop("model", self._model)
        if not isinstance(model, str) or not model:
            raise ValueError("model kwarg must be a non-empty string")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        # Forward any caller-supplied generation knobs verbatim. The
        # Mistral SDK is liberal about which kwargs it accepts; we
        # don't pre-filter so future SDK versions don't require an
        # update here just to expose a new sampling parameter.
        for key, value in kwargs.items():
            if key in payload:
                # Defensive: never let a stray kwarg override the
                # canonical model / messages / tools assignments above.
                raise TypeError(
                    f"MistralBackend.stream() refusing to override "
                    f"reserved key {key!r}"
                )
            payload[key] = value

        return payload

    # ---- Retry wrapper ----------------------------------------------

    async def _open_stream_with_retry(
        self,
        client: Any,
        payload: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Open the SDK stream with HTTP-429 backoff (Requirement 19.8).

        The retry budget covers the *initial* connection only — once
        the stream is producing events we no longer retry, because a
        mid-stream 429 is exceptional and the BackendSelector's circuit
        breaker is the right tool for that.

        Translation strategy:

        * The SDK's raw exception is caught here once and routed
          through :meth:`_classify_and_raise`. That helper raises
          :class:`_Retryable429` for HTTP 429 (which tenacity catches
          and uses to schedule the next attempt) and the public
          :class:`MistralAuthError` / :class:`MistralStreamError`
          for everything else (which tenacity's ``retry_if`` predicate
          rejects, so they propagate out of the loop unchanged).
        * Tenacity's ``RetryError`` wraps an exhausted-retry outcome.
          We unwrap and surface :class:`MistralRateLimitError`.
        """
        # ``stop_after_attempt(n)`` allows up to ``n`` attempts total.
        # Requirement 19.8: at most 3 retries → 4 attempts total.
        total_attempts = self._max_retries + 1
        logger.info(
            "Mistral request: model=%s msg_count=%d tool_count=%d",
            payload.get("model"),
            len(payload.get("messages", [])),
            len(payload.get("tools", []) or []),
        )
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_Retryable429),
                stop=stop_after_attempt(total_attempts),
                wait=wait_exponential(
                    multiplier=self._retry_backoff_initial_s,
                    min=self._retry_backoff_initial_s,
                    max=8.0,
                ),
                reraise=False,
            ):
                with attempt:
                    raw_stream = await self._invoke_stream_async(client, payload)
                    return raw_stream
        except RetryError as retry_exc:
            # Tenacity exhausted the retry budget on 429. Surface the
            # public ``rate_limited`` error per Requirement 19.8.
            last_attempt = retry_exc.last_attempt
            cause: BaseException | None = None
            if last_attempt is not None:
                try:
                    cause = last_attempt.exception()
                except Exception:
                    cause = None
            # The cause is a ``_Retryable429`` whose ``__cause__`` is
            # the original SDK exception. Walk one level down so the
            # public error's ``__cause__`` is the SDK exception, which
            # is what audit log code and tests assert against.
            sdk_cause = getattr(cause, "__cause__", None) or cause
            raise MistralRateLimitError(
                "Mistral API rate limit exceeded; "
                f"retried {total_attempts} time(s)"
            ) from sdk_cause
        # Defensive: if the loop completes without returning the
        # ``AsyncRetrying`` invariant has been violated. We never
        # expect to reach here in practice.
        raise MistralStreamError(
            "Mistral chat.stream_async returned without yielding a stream"
        )

    async def _invoke_stream_async(
        self,
        client: Any,
        payload: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Call the SDK's ``chat.stream_async`` and normalise the result.

        Translates raw SDK exceptions into the public hierarchy
        (or :class:`_Retryable429` for HTTP 429). Some SDK versions
        return an awaitable that resolves to the async iterator;
        others return the iterator directly. This helper accepts
        either shape so the retry loop never has to introspect.
        """
        try:
            result = client.chat.stream_async(**payload)
        except Exception as exc:
            # ``_classify_and_raise`` always raises; the trailing
            # ``raise`` is for type-checker satisfaction only.
            self._classify_and_raise(exc)
            raise  # pragma: no cover

        # ``stream_async`` is conventionally an async function: the
        # call returns a coroutine that yields the async iterator on
        # ``await``. But some test fakes (and old SDK versions) return
        # the iterator synchronously. Detect by checking for
        # ``__await__`` and act accordingly.
        if hasattr(result, "__await__"):
            try:
                result = await result
            except Exception as exc:
                self._classify_and_raise(exc)
                raise  # pragma: no cover
        return cast("AsyncIterator[Any]", result)

    # ---- Error translation ------------------------------------------

    def _classify_and_raise(self, exc: BaseException) -> None:
        """Translate an SDK / transport exception to the public hierarchy."""
        # Attempt to extract a status code; ``mistralai`` exposes one
        # on its ``SDKError`` subclasses, and httpx exceptions expose
        # one on their ``response`` attribute. We use ``getattr``
        # exclusively so we don't import the SDK's error classes here
        # (preserving the lazy-import contract).
        status = self._extract_status_code(exc)
        logger.warning(
            "Mistral stream error: status=%s exc_type=%s msg=%s",
            status,
            exc.__class__.__name__,
            exc,
        )
        if status in (401, 403):
            raise MistralAuthError(
                f"Mistral API rejected the API key (HTTP {status}); "
                "trigger CredentialUpdateFlow to refresh it.",
                status_code=status,
            ) from exc
        if status == 429:
            # Wrap in the internal retry signal so tenacity will
            # retry. The original exception is preserved as
            # ``__cause__``.
            retryable = _Retryable429("HTTP 429 Too Many Requests")
            retryable.__cause__ = exc
            raise retryable
        # Anything else (5xx, network, parse errors) is a generic
        # stream error. The BackendSelector's circuit breaker treats
        # this as a backend failure (Requirement 12.4).
        raise MistralStreamError(
            f"Mistral chat.stream_async failed: {exc!s}"
        ) from exc

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        """Best-effort extraction of an HTTP status code from ``exc``.

        Walks the conventional attribute names used by ``mistralai``
        (``status_code``), ``httpx`` (``response.status_code``), and
        the OpenAI-style SDKs (``http_status``). Returns ``None`` if
        none of them are present, in which case the exception is
        treated as a non-HTTP failure.
        """
        # Direct attribute, used by mistralai 1.x SDKError.
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        # OpenAI-style.
        status = getattr(exc, "http_status", None)
        if isinstance(status, int):
            return status
        # httpx.HTTPStatusError attaches the response.
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
        return None

    # ---- Event translation ------------------------------------------

    @staticmethod
    async def _iter_events(raw_stream: AsyncIterator[Any]) -> AsyncIterator[LLMEvent]:
        """Translate the SDK's async event stream into :class:`LLMEvent` values.

        Mistral streams content deltas alongside *fragments* of tool
        calls. Each fragment carries an ``index`` that identifies the
        slot it belongs to and may set the ``id``, the function
        ``name``, or append to the function ``arguments`` JSON string.
        We accumulate per-index fragments and emit a single
        :class:`ToolCallEvent` once the matching choice's
        ``finish_reason`` becomes one of the terminal reasons (or the
        stream ends).

        The contract documented on :class:`ToolCallEvent` —
        "Backends are responsible for re-assembling the streamed
        function call fragments" — is enforced here so the
        Dialog_Manager only ever sees fully-formed tool calls.
        """
        pending: dict[int, dict[str, Any]] = {}

        async for event in raw_stream:
            # The mistralai SDK wraps each Server-Sent-Event in a
            # ``CompletionEvent`` whose payload is exposed under
            # ``.data``. Some test fakes hand us the payload object
            # directly. Fall back to the event itself when ``.data``
            # is absent so both shapes work.
            payload = getattr(event, "data", event)
            choices = getattr(payload, "choices", None)
            if not choices:
                continue
            for choice in choices:
                delta = getattr(choice, "delta", None)
                finish_reason = getattr(choice, "finish_reason", None)

                # ----- Content delta -----
                if delta is not None:
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content != "":
                        yield ContentDeltaEvent(text=content)

                    # ----- Tool-call fragments -----
                    tool_calls = getattr(delta, "tool_calls", None)
                    if isinstance(tool_calls, list):
                        for tc_delta in tool_calls:
                            MistralBackend._absorb_tool_call_fragment(
                                pending, tc_delta
                            )

                # ----- Flush completed tool calls on terminal reasons -----
                if (
                    isinstance(finish_reason, str)
                    and finish_reason in _TERMINAL_FINISH_REASONS
                    and pending
                ):
                    for emitted in MistralBackend._drain_pending(pending):
                        yield emitted

        # End-of-stream: flush any tool calls the model failed to
        # terminate explicitly. Mistral's API has been observed to
        # close the stream without a ``tool_calls`` finish_reason on
        # short responses, so this fallback keeps the contract intact.
        if pending:
            for emitted in MistralBackend._drain_pending(pending):
                yield emitted

    @staticmethod
    def _absorb_tool_call_fragment(
        pending: dict[int, dict[str, Any]],
        fragment: Any,
    ) -> None:
        """Merge a streamed tool-call fragment into the per-index accumulator.

        Three categories of fragment are recognised:

        1. ``id`` set on the fragment — the model has chosen the
           tool-call id for this slot. Stored verbatim.
        2. ``function.name`` set — the skill name. Stored verbatim
           (typically arrives once at the start of the slot).
        3. ``function.arguments`` set — either a JSON string fragment
           that we concatenate, or a fully-parsed dict that we adopt
           wholesale (some SDK versions emit a dict on the final
           fragment).
        """
        if fragment is None:
            return
        # Mistral's streaming tool-call fragments may carry an explicit
        # ``index``; when absent (single-tool turns) we default to 0.
        index_attr = getattr(fragment, "index", None)
        index = index_attr if isinstance(index_attr, int) else 0
        slot = pending.setdefault(
            index,
            {"id": "", "name": "", "args_buffer": "", "args_obj": None},
        )

        tc_id = getattr(fragment, "id", None)
        if isinstance(tc_id, str) and tc_id:
            slot["id"] = tc_id

        function = getattr(fragment, "function", None)
        if function is not None:
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                slot["name"] = name

            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str):
                # JSON string fragment — concatenate.
                slot["args_buffer"] = slot["args_buffer"] + arguments
            elif isinstance(arguments, dict):
                # Fully-formed object — adopt wholesale, replacing
                # whatever fragments we may have buffered. This is the
                # behaviour the SDK exhibits when ``json_mode`` is on.
                slot["args_obj"] = arguments

    @staticmethod
    def _drain_pending(
        pending: dict[int, dict[str, Any]],
    ) -> list[ToolCallEvent]:
        """Materialise every accumulated tool-call slot into events.

        Returns an empty list when the slot map is empty. Slots are
        emitted in ``index`` order so multi-tool turns surface in a
        deterministic sequence — the Dialog_Manager dispatches them in
        receipt order, so a stable order makes audit logs easier to
        read (CP9).
        """
        ordered_indices = sorted(pending.keys())
        emitted: list[ToolCallEvent] = []
        for index in ordered_indices:
            slot = pending[index]
            event = MistralBackend._materialise_slot(slot)
            if event is not None:
                emitted.append(event)
        pending.clear()
        return emitted

    @staticmethod
    def _materialise_slot(slot: dict[str, Any]) -> ToolCallEvent | None:
        """Turn an accumulator slot into a :class:`ToolCallEvent` or ``None``.

        Returns ``None`` when the slot is missing critical fields
        (e.g., the function name) so a malformed stream does not crash
        the dialog loop. Per the design's failure-mode taxonomy the
        Dialog_Manager treats a tool-call-less response as a normal
        text turn.
        """
        skill_name = slot.get("name") or ""
        if not skill_name:
            return None

        # Resolve arguments: prefer the explicit dict slot when set,
        # otherwise parse the concatenated JSON buffer. Parse failures
        # produce an empty arguments dict so Skill schema validation
        # downstream can return a clean ``schema_violation`` result.
        args_obj = slot.get("args_obj")
        args_buffer = slot.get("args_buffer") or ""
        parsed: dict[str, Any]
        raw_arguments: str
        if isinstance(args_obj, dict):
            parsed = args_obj
            try:
                raw_arguments = json.dumps(args_obj, ensure_ascii=False)
            except (TypeError, ValueError):
                raw_arguments = json.dumps(
                    {k: str(v) for k, v in args_obj.items()},
                    ensure_ascii=False,
                )
        else:
            raw_arguments = args_buffer
            if args_buffer == "":
                # Empty arguments are valid — the function takes no
                # parameters. Represent as ``{}`` to satisfy the
                # downstream JSON Schema validators that expect an
                # object.
                parsed = {}
                raw_arguments = "{}"
            else:
                try:
                    decoded = json.loads(args_buffer)
                except json.JSONDecodeError:
                    decoded = {}
                parsed = decoded if isinstance(decoded, dict) else {}

        call_id = slot.get("id")
        if not isinstance(call_id, str) or not call_id:
            # Mistral occasionally elides the id on short streams.
            # Synthesise one so the Dialog_Manager can correlate the
            # upcoming ``tool`` reply with this call. Use a stable
            # prefix so audit log readers can spot synthetic ids.
            call_id = f"mistral-synth-{uuid.uuid4().hex}"

        return ToolCallEvent(
            tool_call=ToolCall(
                id=call_id,
                skill_name=skill_name,
                arguments=parsed,
                raw_arguments=raw_arguments,
            )
        )
