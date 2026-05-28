"""Cloud / local LLM backend selector with a circuit-breaker fallback.

This module implements :class:`BackendSelector`, the third of the three
:class:`~jarvis.llm.base.LLMBackend` Protocol implementations described
in ``design.md``. It composes a *primary* backend (cloud Mistral) and a
*fallback* backend (local Ollama-hosted Mistral) behind a tiny circuit
breaker so the Dialog_Manager sees a single :class:`LLMBackend`
regardless of which one is actually serving a turn.

Why a circuit breaker rather than per-call try/fallback
-------------------------------------------------------

Two reasons:

1. **Latency budget (Requirement 12.1).** Falling back on every
   timeout would mean every degraded turn pays the full 3 s primary
   timeout *plus* the local model latency. Once we know Mistral is
   unhealthy, we want subsequent turns to skip the primary entirely
   for a cool-down window so the user only experiences the slow turn
   once.
2. **One-shot user notification.** Requirement 12.4 says the user
   SHALL be informed of the fallback. The Dialog_Manager wires the
   ``on_flip`` callback to a brief TTS announcement (\"The cloud is
   being slow, sir. Switching to local.\"). Speaking that on every
   single turn would be unbearable; speaking it once at the
   transition is what the design's Mistral → Local Fallback Flow
   shows.

State machine
-------------

The selector behaves as the classic three-state circuit breaker, with
the *half-open* state collapsed into the next probe:

* **Closed** (initial). ``stream()`` routes to the primary. Any timeout
  > ``timeout_seconds`` or HTTP 5xx response trips the breaker into
  *Open*.
* **Open**. ``stream()`` routes directly to the fallback for
  ``cool_down_seconds`` seconds. The first ``stream()`` call after the
  cool-down expires acts as the half-open probe.
* **Half-open** (implicit). The next probe call opens the primary
  again. Success collapses back to *Closed*; failure (timeout / 5xx)
  re-trips and fires ``on_flip`` once more — the callback is per-flip,
  not per-lifetime, because a second flip after a successful recovery
  is itself news for the user.

Trigger conditions
------------------

The breaker trips when the primary backend produces any of:

* :class:`asyncio.TimeoutError` raised by the
  ``asyncio.wait_for(..., timeout=timeout_seconds)`` wrapper on the
  primary's ``__aenter__`` (this is the ">3 s" path from the design).
* :class:`httpx.TimeoutException` (any subclass — connect, read,
  write, pool). The Mistral SDK and the hand-rolled OllamaBackend
  both surface httpx timeouts when their underlying client times out
  with a different budget than ours.
* :class:`httpx.HTTPStatusError` whose response status is in the
  ``[500, 600)`` range — exactly the "5xx" condition in
  Requirement 12.4.

Mid-stream failures (after the primary has yielded its first event)
also feed the breaker, but the in-flight call is *not* re-routed: the
exception propagates to the caller and the next turn benefits from the
now-open circuit. Restarting an LLM stream mid-turn would emit a
duplicate prefix to the TTS engine.

Property 14 invariant
---------------------

The selector forwards ``messages`` and ``**kwargs`` to whichever
backend it picks, **without copying or transforming them**, beyond the
defensive ``list(...)`` / ``dict(...)`` snapshots the
:func:`contextlib.asynccontextmanager` body needs to detach from
caller mutation. The two concrete backends already accept the same
``(messages, tools)`` shape (see
:mod:`jarvis.llm.ollama_backend._build_payload` and the design's
Mistral tool mapping), so by passing the request through unchanged the
selector trivially satisfies CP-14: any payload the primary would have
sent has the same ``messages`` and a ``tools`` array of equal length
with matching ``name``/``parameters`` keys.

Validates: Requirement 12.4
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import logging
from typing import Any, Final

import httpx

from jarvis.llm.base import LLMBackend, LLMEvent, Message, Stream, ToolDefinition
from jarvis.utils.time_source import SystemTimeSource, TimeSource

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COOL_DOWN_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "BackendSelector",
]


# ---------------------------------------------------------------------------
# Defaults — match design.md §"Mistral → Local Fallback Flow"
# ---------------------------------------------------------------------------

#: Per-call timeout budget for the primary backend. The design pins
#: this at 3 s; the configurable kwarg lets tests dial it down without
#: hard-coding a real-time wait.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 3.0

#: Cool-down before the half-open probe is allowed. The design pins
#: this at 30 s and the default ``llm.fallback.circuit_open_seconds``
#: in ``default.toml`` mirrors the value.
DEFAULT_COOL_DOWN_SECONDS: Final[float] = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_server_error(exc: httpx.HTTPStatusError) -> bool:
    """Return ``True`` if ``exc`` carries a 5xx response status.

    Defensive: an HTTPStatusError might in pathological cases lack a
    ``response`` (e.g., when the user passes a hand-built error in
    tests). Treat the absence as "not a 5xx" rather than crashing.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return False
    status = getattr(response, "status_code", None)
    return isinstance(status, int) and 500 <= status < 600


def _is_trip_error(exc: BaseException) -> bool:
    """Return ``True`` for exceptions that should open the circuit."""
    if isinstance(exc, TimeoutError):
        # ``asyncio.TimeoutError`` is aliased to the built-in
        # ``TimeoutError`` on Python 3.11+, so this single check
        # covers both spellings.
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_server_error(exc)
    return False


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class BackendSelector:
    """Route a streaming chat call to a primary or fallback :class:`LLMBackend`.

    Conforms structurally to :class:`~jarvis.llm.base.LLMBackend`. The
    Dialog_Manager treats the selector as just another backend, so the
    fallback is invisible above the protocol boundary.

    Parameters
    ----------
    primary:
        The cloud backend (typically
        :class:`jarvis.llm.mistral_backend.MistralBackend`). Tried
        first whenever the circuit is closed or half-open.
    fallback:
        The local backend (typically
        :class:`jarvis.llm.ollama_backend.OllamaBackend`). Used while
        the circuit is open and as the recovery path on a tripping
        error.
    timeout_seconds:
        Maximum seconds to wait for the primary to *enter* its
        streaming context manager. The design pins this at 3 s
        (Requirement 12.4); tests typically pass a small fraction of
        a second.
    cool_down_seconds:
        Seconds the circuit stays open after a trip before the next
        probe is allowed. The design pins this at 30 s.
    time_source:
        Injectable :class:`~jarvis.utils.time_source.TimeSource`. The
        breaker uses ``monotonic()`` exclusively, never
        :py:meth:`TimeSource.now`, because cool-downs measure elapsed
        wall-time-independent intervals (Requirement 17.3 spirit:
        a corrected wall clock MUST NOT shorten or extend the
        cool-down). Defaults to
        :class:`~jarvis.utils.time_source.SystemTimeSource`.
    on_flip:
        Optional zero-argument callback invoked exactly once each time
        the circuit transitions into the *open* state. The
        Dialog_Manager wires this to a brief TTS notice
        ("The cloud is being slow, sir. Switching to local.")
        per the design's fallback flow diagram. Exceptions raised by
        the callback are logged and swallowed so a misbehaving
        notification never breaks the dialog loop.
    """

    def __init__(
        self,
        primary: LLMBackend,
        fallback: LLMBackend,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cool_down_seconds: float = DEFAULT_COOL_DOWN_SECONDS,
        time_source: TimeSource | None = None,
        on_flip: Callable[[], None] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cool_down_seconds < 0:
            raise ValueError("cool_down_seconds must be non-negative")
        self._primary = primary
        self._fallback = fallback
        self._timeout_seconds = float(timeout_seconds)
        self._cool_down_seconds = float(cool_down_seconds)
        self._time = time_source if time_source is not None else SystemTimeSource()
        self._on_flip = on_flip
        # Monotonic timestamp of the last trip. ``None`` means closed.
        self._opened_at: float | None = None

    # ------------------------------------------------------------------
    # Public state inspection (handy for tests and the Dialog_Manager)
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """``True`` while the circuit is open (cool-down still active)."""
        return self._is_open()

    # ------------------------------------------------------------------
    # LLMBackend Protocol
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[Stream]:
        """Open a streaming chat completion against primary or fallback.

        See :class:`~jarvis.llm.base.LLMBackend` for the contract
        documentation. The selector decides at call time which
        backend to use based on the circuit state.

        The returned async context manager is the standard
        :func:`contextlib.asynccontextmanager` shape; use as::

            async with selector.stream(msgs, tools=tools) as events:
                async for event in events:
                    ...
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
        """Implementation backing :meth:`stream`.

        The two-branch structure is intentional: the @asynccontextmanager
        protocol requires *exactly one* ``yield``, so we determine the
        backend up front and only then yield its event stream.
        """
        if not self._is_open():
            # Closed or half-open: probe the primary.
            primary_cm = self._primary.stream(messages, tools=tools, **kwargs)
            tripped = False
            primary_stream: Stream | None = None
            try:
                # ``asyncio.wait_for`` enforces the per-call budget on
                # the primary's ``__aenter__``. On timeout it cancels
                # the underlying task; the @asynccontextmanager-backed
                # backends run their finally blocks via the resulting
                # CancelledError, so we MUST NOT also call
                # ``primary_cm.__aexit__`` afterwards — doing so would
                # be a context-manager protocol violation.
                primary_stream = await asyncio.wait_for(
                    primary_cm.__aenter__(),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                tripped = True
                self._trip()
            except httpx.TimeoutException:
                tripped = True
                self._trip()
            except httpx.HTTPStatusError as exc:
                if _is_server_error(exc):
                    tripped = True
                    self._trip()
                else:
                    # Non-5xx HTTP errors (4xx etc.) are not a primary
                    # health signal — bubble up so the Dialog_Manager
                    # can route 401/403 through CredentialUpdateFlow.
                    raise

            if not tripped and primary_stream is not None:
                # Successful entry on a possibly-half-open breaker
                # closes the circuit again.
                self._reset()
                exc_info: tuple[Any, Any, Any] = (None, None, None)
                try:
                    yield self._iter_primary(primary_stream)
                except BaseException as exc:
                    exc_info = (type(exc), exc, exc.__traceback__)
                    raise
                finally:
                    # Mirror the standard contextlib pattern: forward
                    # the outgoing exception (if any) into the wrapped
                    # ``__aexit__`` so the underlying backend can run
                    # its cleanup with full context. We deliberately
                    # ignore the boolean return: a wrapped CM that
                    # claims to suppress an exception we already
                    # observed via ``except`` cannot retroactively
                    # un-raise it from this ``finally`` block, and
                    # silently changing propagation here would mask
                    # genuine LLM-side errors from the Dialog_Manager.
                    await primary_cm.__aexit__(*exc_info)
                return

        # Either the breaker was already open, or the primary just
        # tripped it on entry. Either way, route to the fallback.
        async with self._fallback.stream(messages, tools=tools, **kwargs) as events:
            yield events

    async def _iter_primary(self, stream: Stream) -> AsyncIterator[LLMEvent]:
        """Forward primary events while watching for circuit-trip errors.

        Pass-through is a single ``async for`` to preserve event order
        and timing — the Dialog_Manager's :class:`SentenceAccumulator`
        depends on tokens arriving exactly as the backend emits them.
        Mid-stream tripping errors update the breaker state for the
        *next* turn, but the current call's exception is re-raised so
        the caller can decide what to surface to the user.
        """
        try:
            async for event in stream:
                yield event
        except BaseException as exc:
            if _is_trip_error(exc):
                self._trip()
            raise

    # ------------------------------------------------------------------
    # Circuit state
    # ------------------------------------------------------------------

    def _is_open(self) -> bool:
        if self._opened_at is None:
            return False
        elapsed = self._time.monotonic() - self._opened_at
        # Cool-down expired: behave as half-open. Don't clear
        # ``_opened_at`` here; the next successful primary probe
        # will do so in ``_reset``. Leaving it set lets a probe
        # failure re-trip and fire ``on_flip`` again from a known
        # state.
        return elapsed < self._cool_down_seconds

    def _trip(self) -> None:
        """Open the breaker and notify the user once."""
        self._opened_at = self._time.monotonic()
        logger.warning(
            "BackendSelector circuit opened; routing subsequent calls to fallback "
            "for %.0fs cool-down",
            self._cool_down_seconds,
        )
        if self._on_flip is not None:
            try:
                self._on_flip()
            except Exception:
                # Never let a misbehaving notification callback break
                # the dialog loop. Log with traceback so the operator
                # can debug after the fact.
                logger.exception("BackendSelector.on_flip callback raised")

    def _reset(self) -> None:
        """Close the breaker after a successful primary probe."""
        if self._opened_at is not None:
            logger.info("BackendSelector circuit closed; primary backend recovered")
        self._opened_at = None
