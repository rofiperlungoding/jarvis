"""Google Calendar provider client.

Implements the ``CalendarClient`` referenced by ``CalendarSkill``
(Requirement 7.5, 7.6, 7.7). The client supports the three operations
constrained by the Skill manifest:

* ``list_today`` — events occurring on the user's current local day.
* ``list_range(start, end)`` — events overlapping the inclusive
  ``[start, end]`` window. ``start`` and ``end`` are timezone-aware
  :class:`datetime` instances.
* ``create_event(title, start, end)`` — create a new event. Marked
  destructive in the authorization config; the confirmation flow is
  applied by ``Authorization_Policy`` upstream of this client.

OAuth credentials are pulled from :class:`CredentialStore` under the
``oauth_credential`` configured key (default ``"calendar/oauth_token"``).
The stored value is treated as an opaque bearer token and is sent in the
``Authorization: Bearer ...`` header.

The client targets Google's Calendar v3 API at ``www.googleapis.com``.
The host is included in the default ``security.network_destination_allowlist``.

Validates: Requirements 5.6, 7.5, 7.6, 7.7
"""

from __future__ import annotations

from datetime import UTC, datetime, time
import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import ProviderClient
from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialBackend

logger = logging.getLogger(__name__)

__all__ = ["CalendarClient"]


_GCAL_HOST: Final[str] = "www.googleapis.com"
_GCAL_BASE_URL: Final[str] = f"https://{_GCAL_HOST}"
_GCAL_PRIMARY_CALENDAR: Final[str] = "primary"


class CalendarClient(ProviderClient):
    """Google Calendar v3 client."""

    PROVIDER_NAME: Final[str] = "google"

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        network_allowlist: list[str] | tuple[str, ...] | frozenset[str],
        credential_store: CredentialBackend,
        provider_config: Any,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            audit_log=audit_log,
            network_allowlist=network_allowlist,
            justification="calendar lookup",
            skill_name="CalendarClient",
            client=client,
            timeout_seconds=float(getattr(provider_config, "timeout_seconds", 5.0)),
        )
        self._credentials: CredentialBackend = credential_store
        self._config: Any = provider_config

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def list_today(self) -> list[dict[str, Any]]:
        """Return events that overlap the current local day.

        "Local day" is computed against the system clock's local timezone
        (via :meth:`datetime.now().astimezone()`). Tests that require
        deterministic boundaries should call :meth:`list_range` directly
        with a fake clock.
        """
        now_local = datetime.now().astimezone()
        local_tz = now_local.tzinfo or UTC
        start_of_day = datetime.combine(now_local.date(), time.min, tzinfo=local_tz)
        end_of_day = datetime.combine(now_local.date(), time.max, tzinfo=local_tz)
        return await self.list_range(start_of_day, end_of_day)

    async def list_range(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Return events overlapping the inclusive ``[start, end]`` window.

        ``start`` and ``end`` MUST be timezone-aware. Naive datetimes are
        rejected to avoid silently shifting the window when the host's
        local timezone changes.
        """
        self._require_aware(start, "start")
        self._require_aware(end, "end")
        if end < start:
            raise ValueError("end must be >= start")

        token = self._read_oauth_token()
        params = {
            "timeMin": _to_rfc3339(start),
            "timeMax": _to_rfc3339(end),
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        headers = {"Authorization": f"Bearer {token}"}

        url = (
            f"{_GCAL_BASE_URL}/calendar/v3/calendars/"
            f"{_GCAL_PRIMARY_CALENDAR}/events"
        )

        payload = await self._get_json(url, params=params, headers=headers)
        items = payload.get("items", [])
        if not isinstance(items, list):
            return []
        return [self._normalise_event(item) for item in items if isinstance(item, dict)]

    async def create_event(
        self, title: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        """Create a calendar event with the given title and bounds.

        The ``Authorization_Policy`` MUST have already obtained the user's
        confirmation by the time this method is invoked (Requirement 7.6).
        Returns the normalised event dict for the freshly-created event.
        """
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        self._require_aware(start, "start")
        self._require_aware(end, "end")
        if end <= start:
            raise ValueError("end must be > start")

        token = self._read_oauth_token()
        body = {
            "summary": title,
            "start": {"dateTime": _to_rfc3339(start)},
            "end": {"dateTime": _to_rfc3339(end)},
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        url = (
            f"{_GCAL_BASE_URL}/calendar/v3/calendars/"
            f"{_GCAL_PRIMARY_CALENDAR}/events"
        )

        try:
            response = await self.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar create_event timed out: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar create_event failed: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                (
                    f"Google Calendar create_event returned HTTP "
                    f"{response.status_code}"
                ),
                provider=self.PROVIDER_NAME,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar create_event returned non-JSON: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                "provider_unavailable",
                "Google Calendar create_event response was not a JSON object",
                provider=self.PROVIDER_NAME,
            )
        return self._normalise_event(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _require_aware(value: datetime, label: str) -> None:
        if not isinstance(value, datetime):
            raise TypeError(f"{label} must be a datetime instance")
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(f"{label} must be timezone-aware")

    @staticmethod
    def _normalise_event(event: dict[str, Any]) -> dict[str, Any]:
        """Reduce a Google Calendar event to the fields downstream needs."""
        start = event.get("start")
        end = event.get("end")
        return {
            "id": event.get("id"),
            "title": event.get("summary"),
            "start": start.get("dateTime") if isinstance(start, dict) else None,
            "end": end.get("dateTime") if isinstance(end, dict) else None,
            "html_link": event.get("htmlLink"),
            "status": event.get("status"),
        }

    def _read_oauth_token(self) -> str:
        credential_name = str(getattr(self._config, "oauth_credential", ""))
        if not credential_name:
            raise ProviderError(
                "missing_credentials",
                "providers.calendar.oauth_credential is not configured",
                provider=self.PROVIDER_NAME,
            )
        try:
            value = self._credentials.get(credential_name)
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(
                "missing_credentials",
                f"unable to read credential {credential_name!r}: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        if not value:
            raise ProviderError(
                "missing_credentials",
                f"credential {credential_name!r} is not set in the credential store",
                provider=self.PROVIDER_NAME,
            )
        return value

    async def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar request timed out: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar request failed: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar returned HTTP {response.status_code}",
                provider=self.PROVIDER_NAME,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"Google Calendar returned non-JSON body: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                "provider_unavailable",
                "Google Calendar response was not a JSON object",
                provider=self.PROVIDER_NAME,
            )
        return data


def _to_rfc3339(value: datetime) -> str:
    """Format ``value`` as RFC 3339 with a numeric offset.

    Google Calendar accepts ISO 8601, which is a strict subset of RFC
    3339. We deliberately convert to UTC so daylight-saving boundaries
    cannot trip up the API.
    """
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
