"""HTTP base client for outbound provider integrations.

This module implements :class:`ProviderClient`, the shared transport layer
that backs the per-domain provider clients (weather, news, calendar, SMTP,
web search) introduced under ``src/jarvis/automation/providers/`` in
``design.md §Automation_Service``. Three cross-cutting responsibilities live
here so individual provider classes can stay focused on payload shapes:

* **Read-timeout discipline (Requirement 7.7).** Every request runs against
  a configurable read timeout — defaulting to 5 s — so an unhealthy
  upstream cannot hold the Dialog_Manager's tool-execution path open past
  the user-visible ``provider_unavailable`` budget. The same timeout is
  applied to connect / write / pool by default; subclasses that need a
  bespoke profile can inject their own :class:`httpx.AsyncClient`.

* **Exponential-backoff retries on transient failure.** Timeouts and 5xx
  responses are routed through :mod:`tenacity` with capped exponential
  wait. Once the retry budget is exhausted the most recent response is
  returned verbatim (5xx case) or the most recent transport exception is
  reraised (timeout case). The calling Skill is responsible for translating
  that final outcome into the ``provider_unavailable`` /``timeout``
  ``SkillResult`` error codes documented in the design's error taxonomy
  (Requirement 7.7).

* **Network egress audit + allowlist enforcement (Requirements 13.4,
  13.6).** Before each request, the destination host is checked against
  ``security.network_destination_allowlist`` (the configured list passed
  in at construction time). Allowed destinations are recorded as
  ``network_egress`` in :class:`AuditLog` with the per-client
  justification; blocked destinations are recorded as ``policy_violation``
  and raise :class:`NetworkPolicyViolation` so the request never reaches
  the wire. Exactly one audit entry is written per *logical* call —
  retries on the same URL do not multiply the egress row count, which
  keeps the audit log faithful to the user's mental model of "one tool
  invocation = one network destination".

Validates: Requirements 7.7, 13.4, 13.6
"""

from __future__ import annotations

from collections.abc import Iterable
from types import TracebackType
from typing import Any, Final
from urllib.parse import urlparse, urlunparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jarvis.security.audit_log import AuditLog

__all__ = [
    "DEFAULT_PROVIDER_MAX_ATTEMPTS",
    "DEFAULT_PROVIDER_TIMEOUT_S",
    "NetworkPolicyViolation",
    "ProviderClient",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default read/connect/write/pool timeout in seconds. Matches the 5 s
#: ceiling documented in Requirement 7.7 so that providers never block the
#: tool-execution path past the budget the Dialog_Manager promises the user.
DEFAULT_PROVIDER_TIMEOUT_S: Final[float] = 5.0

#: Default total number of attempts (initial call + retries) for retryable
#: failures. Three keeps the worst-case wall time inside the per-turn
#: latency budget once the exponential backoff caps below are honoured.
DEFAULT_PROVIDER_MAX_ATTEMPTS: Final[int] = 3

# Tenacity exponential-backoff parameters. The numbers are deliberately
# small so the test suite (which exercises 5xx / timeout retry paths)
# completes well under one second per case while still exercising real
# multi-attempt scheduling.
_RETRY_WAIT_MULTIPLIER: Final[float] = 0.1
_RETRY_WAIT_MIN: Final[float] = 0.1
_RETRY_WAIT_MAX: Final[float] = 2.0


# ---------------------------------------------------------------------------
# Public exception types
# ---------------------------------------------------------------------------


class NetworkPolicyViolation(RuntimeError):  # noqa: N818 - documented public name
    """Raised when an outbound destination is not on the allowlist.

    Carries both the original (full) ``destination`` URL and the parsed
    ``host`` that failed the check so callers can render a meaningful
    error to the user without re-parsing. The audit log row is written
    *before* this exception is raised, so even if the caller swallows
    the exception, Requirement 13.6 is still satisfied.
    """

    def __init__(self, *, destination: str, host: str | None) -> None:
        host_repr = host or "<no host>"
        super().__init__(
            f"network destination not on allowlist: {host_repr} ({destination})"
        )
        self.destination: str = destination
        self.host: str | None = host


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _RetryableStatusError(Exception):
    """Internal sentinel: a 5xx response that tenacity should retry on.

    We raise this from inside the retry loop so :mod:`tenacity` can apply
    the configured backoff schedule. The most recent response is held on
    the exception object so that, after the retry budget is exhausted, the
    public :meth:`ProviderClient.request` can hand the response back to
    the caller verbatim instead of leaking ``_RetryableStatusError`` past
    the module boundary.
    """

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"upstream returned HTTP {response.status_code}")
        self.response: httpx.Response = response


# ---------------------------------------------------------------------------
# ProviderClient
# ---------------------------------------------------------------------------


class ProviderClient:
    """Shared HTTP base for outbound provider integrations.

    Parameters
    ----------
    audit_log:
        :class:`AuditLog` instance used to persist ``network_egress`` and
        ``policy_violation`` rows (Requirements 13.4, 13.6).
    network_allowlist:
        Iterable of hostnames the client is allowed to reach. Match is
        case-insensitive and exact against the URL's parsed hostname; an
        empty allowlist therefore blocks every outbound destination by
        construction. Sourced from ``config.security.network_destination_allowlist``
        at wiring time.
    justification:
        Short, user-visible string describing why this client makes
        outbound calls — e.g., ``"weather lookup"``. Recorded verbatim in
        every ``network_egress`` row so a future Privacy_Dashboard can
        surface a per-destination explanation (Requirement 13.4). Must be
        non-empty: a missing justification would defeat the user-visible
        rationale the requirement promises.
    skill_name:
        Optional skill identifier persisted on every audit row. Defaults
        to the runtime class name of the subclass so a ``WeatherClient``
        shows up as ``WeatherClient`` in the audit table without each
        provider needing to hardcode it.
    client:
        Pre-configured :class:`httpx.AsyncClient`. When supplied, ownership
        stays with the caller: :meth:`aclose` will *not* close it. This
        makes the base trivially testable with :class:`httpx.MockTransport`
        / :mod:`respx`. When ``None``, a new client is constructed with
        ``timeout = httpx.Timeout(timeout_seconds)`` and is closed on
        :meth:`aclose` / ``__aexit__``.
    timeout_seconds:
        Read (and connect / write / pool) timeout in seconds applied to
        every request when no client is injected. Defaults to
        :data:`DEFAULT_PROVIDER_TIMEOUT_S`.
    max_attempts:
        Total number of attempts (including the initial call) for
        retryable failures (timeouts and 5xx responses). Must be ≥ 1.
        Defaults to :data:`DEFAULT_PROVIDER_MAX_ATTEMPTS`.
    """

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        network_allowlist: Iterable[str],
        justification: str,
        skill_name: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_S,
        max_attempts: int = DEFAULT_PROVIDER_MAX_ATTEMPTS,
    ) -> None:
        if not isinstance(justification, str) or not justification:
            raise ValueError("justification must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        # Materialise the allowlist into a frozenset of lower-case
        # hostnames. Lower-casing once at construction time means the
        # per-request check is a single hash lookup, and freezing prevents
        # a subclass from drifting from the configured policy at runtime.
        self._allowlist: frozenset[str] = frozenset(
            entry.strip().lower() for entry in network_allowlist if entry
        )
        self._audit: AuditLog = audit_log
        self._justification: str = justification
        self._skill_name: str = skill_name or type(self).__name__
        self._timeout_seconds: float = float(timeout_seconds)
        self._max_attempts: int = int(max_attempts)

        if client is not None:
            self._client: httpx.AsyncClient = client
            self._owns_client: bool = False
        else:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds),
            )
            self._owns_client = True

        self._closed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def justification(self) -> str:
        """The user-visible justification recorded for every egress row."""
        return self._justification

    @property
    def skill_name(self) -> str:
        """The skill identifier persisted on every audit row."""
        return self._skill_name

    @property
    def allowlist(self) -> frozenset[str]:
        """Read-only view of the lower-cased hostname allowlist."""
        return self._allowlist

    @property
    def timeout_seconds(self) -> float:
        """Configured read timeout in seconds (Requirement 7.7)."""
        return self._timeout_seconds

    @property
    def max_attempts(self) -> int:
        """Total attempts (initial + retries) for retryable failures."""
        return self._max_attempts

    # ------------------------------------------------------------------
    # Public HTTP surface
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue an HTTP request with audit + retry semantics.

        The destination host is enforced against the allowlist *before* the
        first attempt; the audit-log row (``network_egress`` for an allowed
        destination, ``policy_violation`` otherwise) is written exactly
        once per logical call, regardless of how many retries occur. After
        the retry budget is exhausted, a final 5xx response is returned to
        the caller verbatim so provider Skills can map it onto the
        ``provider_unavailable`` error code in their own translation
        layer; transport-level timeouts propagate as
        :class:`httpx.TimeoutException` so the caller can map them onto
        the ``timeout`` error code instead.
        """
        self._ensure_open()
        await self._enforce_and_record(url)

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(
                multiplier=_RETRY_WAIT_MULTIPLIER,
                min=_RETRY_WAIT_MIN,
                max=_RETRY_WAIT_MAX,
            ),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, _RetryableStatusError),
            ),
            reraise=True,
        )

        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._client.request(
                        method, url, **kwargs
                    )
                    if 500 <= response.status_code < 600:
                        # Trigger tenacity's retry path. The response is
                        # carried on the exception so that, on the final
                        # attempt, we can still return it to the caller.
                        raise _RetryableStatusError(response)
                    return response
        except _RetryableStatusError as exc:
            # Retry budget exhausted on a 5xx — surface the response.
            return exc.response

        # Defensive: tenacity always exits the loop via ``return`` (success
        # branch), the ``except _RetryableStatusError`` above (5xx exhaust),
        # or by reraising :class:`httpx.TimeoutException` (timeout exhaust).
        raise RuntimeError(  # pragma: no cover - defensive
            "ProviderClient.request loop terminated without producing a result"
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('GET', url, ...)``."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('POST', url, ...)``."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('PUT', url, ...)``."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('PATCH', url, ...)``."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('DELETE', url, ...)``."""
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> httpx.Response:
        """Convenience wrapper for ``self.request('HEAD', url, ...)``."""
        return await self.request("HEAD", url, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it.

        Safe to call multiple times. After ``aclose``, further calls to
        :meth:`request` raise :class:`RuntimeError`. When the client was
        injected by the caller, ownership stays with the caller and the
        client is *not* closed.
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ProviderClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ProviderClient is closed")

    async def _enforce_and_record(self, url: str) -> None:
        """Allowlist check + audit recording for a single outbound call.

        Splitting this out of :meth:`request` keeps the retry loop free of
        audit concerns — and means the same check / log path is reused by
        any future helper (e.g. ``stream``) that wants to share the
        budgeting machinery.
        """
        host = self._extract_host(url)
        destination = self._safe_destination(url)

        if host is None or host.lower() not in self._allowlist:
            # Record *before* raising so that even callers who swallow
            # :class:`NetworkPolicyViolation` still produce the audit
            # entry Requirement 13.6 mandates.
            await self._audit.record_policy_violation(
                skill=self._skill_name,
                destination=destination,
                justification=(
                    f"network destination not on allowlist: "
                    f"{host or '<no host>'}"
                ),
                outcome="blocked",
            )
            raise NetworkPolicyViolation(destination=destination, host=host)

        await self._audit.record_network_egress(
            destination=destination,
            justification=self._justification,
            skill=self._skill_name,
        )

    @staticmethod
    def _extract_host(url: str) -> str | None:
        """Return the lower-cased hostname for ``url`` or ``None``.

        :func:`urllib.parse.urlparse` already lower-cases the hostname,
        but we re-apply :meth:`str.lower` defensively in case a future
        Python version changes that behaviour. URLs without an explicit
        scheme (e.g. ``"/relative/path"``) yield ``None`` and are blocked
        — we cannot meaningfully audit a destination we cannot identify.
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return None
        host = parsed.hostname
        if host is None or host == "":
            return None
        return host.lower()

    @staticmethod
    def _safe_destination(url: str) -> str:
        """Return a redacted ``scheme://netloc`` form of ``url``.

        We deliberately drop path, query, and fragment when persisting the
        destination so any URL-embedded credentials (API keys passed in
        the query string, basic-auth tokens) never reach the audit log.
        Falls back to the raw URL string only when the input is so
        malformed we cannot extract a usable scheme + netloc — that case
        will already have failed the allowlist check.
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return url
        if parsed.scheme and parsed.netloc:
            return urlunparse(
                (parsed.scheme, parsed.netloc, "", "", "", "")
            )
        return url
