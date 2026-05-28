"""OpenWeather provider client.

Implements the ``WeatherClient`` referenced by ``WeatherSkill`` (Requirement
7.1, 7.2, 7.7). The client fetches *current conditions* and a *24-hour
forecast* for a free-form ``location`` string, pulling its API key from
:class:`CredentialStore` at request time so a missing credential surfaces
``missing_credentials`` rather than a stack trace (Requirement 5.6).

OpenWeather is queried via two endpoints:

* ``GET /data/2.5/weather?q={location}&units=metric&appid=...`` — current
  conditions. Used both for the "current" portion of the response and to
  resolve ``location`` to a (lat, lon) pair for the forecast call.
* ``GET /data/2.5/forecast?lat=...&lon=...&units=metric&appid=...`` — the
  3-hour granularity 5-day forecast. We slice the first eight entries so
  callers receive a 24-hour window (8 x 3 h = 24 h), satisfying the
  "24-hour forecast" wording in Requirement 7.2.

This module deliberately stays thin: the heavy lifting (timeouts, retries,
allowlist enforcement, audit logging) lives in :class:`ProviderClient`.

Validates: Requirements 5.6, 7.1, 7.2, 7.7
"""

from __future__ import annotations

import logging
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import ProviderClient
from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialBackend

logger = logging.getLogger(__name__)

__all__ = ["WeatherClient"]


#: Hostname of the OpenWeather API. Centralised here so the
#: ``security.network_destination_allowlist`` check in :class:`ProviderClient`
#: cannot drift accidentally — the test ``test_default_endpoint_host``
#: pins this constant against the documented allowlist entry.
_OPENWEATHER_HOST: Final[str] = "api.openweathermap.org"
_OPENWEATHER_BASE_URL: Final[str] = f"https://{_OPENWEATHER_HOST}"

#: Number of forecast entries to return. OpenWeather reports forecasts at
#: 3-hour granularity, so ``8`` covers the next 24 hours. Anchored as a
#: constant so :class:`WeatherSkill` and the tests reference the same
#: value.
_FORECAST_HOURS: Final[int] = 24
_FORECAST_STEP_HOURS: Final[int] = 3
_FORECAST_ENTRIES: Final[int] = _FORECAST_HOURS // _FORECAST_STEP_HOURS


class WeatherClient(ProviderClient):
    """OpenWeather-backed weather client.

    Parameters
    ----------
    audit_log:
        Forwarded to :class:`ProviderClient` for ``network_egress`` /
        ``policy_violation`` rows.
    network_allowlist:
        Iterable of allowed hostnames; ``api.openweathermap.org`` MUST be
        present for the client to function. Mismatches raise
        :class:`NetworkPolicyViolation` from the base class.
    credential_store:
        Backend that resolves the configured ``api_key_credential`` (e.g.
        ``"weather/api_key"``) to the API key string. Pulled at every
        :meth:`fetch` call so an out-of-band rotation takes effect on the
        next request.
    provider_config:
        Section-shaped object exposing ``api_key_credential``,
        ``default_location``, and ``timeout_seconds`` attributes. Typed
        as ``Any`` to avoid an import cycle on
        :class:`ProvidersWeatherConfig`.
    client:
        Optional pre-configured :class:`httpx.AsyncClient` for tests.
    """

    PROVIDER_NAME: Final[str] = "openweather"

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
            justification="weather lookup",
            skill_name="WeatherClient",
            client=client,
            timeout_seconds=float(getattr(provider_config, "timeout_seconds", 5.0)),
        )
        self._credentials: CredentialBackend = credential_store
        self._config: Any = provider_config

    @property
    def default_location(self) -> str:
        """Return the configured default location for fallbacks."""
        return str(getattr(self._config, "default_location", ""))

    async def fetch(self, location: str | None = None) -> dict[str, Any]:
        """Return current conditions and a 24-hour forecast for ``location``.

        ``location`` is forwarded verbatim to OpenWeather's ``q`` parameter
        (free-form, e.g. ``"Bandung,ID"``). Passing ``None`` (or an empty
        string) falls back to ``provider_config.default_location`` per
        Requirement 7.1. Returns a dictionary with the shape::

            {
                "location": "Bandung,ID",
                "current": {<openweather current payload>},
                "forecast": [<8 entries from /forecast.list>],
            }

        Raises:
            ProviderError(missing_credentials): the configured credential
                key is absent from the credential store.
            ProviderError(provider_unavailable): OpenWeather returned a
                non-2xx response, the request timed out, or the response
                body could not be parsed as JSON.
        """
        resolved_location = (location or self.default_location).strip()
        if not resolved_location:
            raise ProviderError(
                "provider_unavailable",
                "no location provided and no default_location configured",
                provider=self.PROVIDER_NAME,
            )

        api_key = self._read_api_key()

        params_current = {
            "q": resolved_location,
            "units": "metric",
            "appid": api_key,
        }
        current = await self._get_json(
            f"{_OPENWEATHER_BASE_URL}/data/2.5/weather",
            params=params_current,
        )

        coord = current.get("coord") if isinstance(current, dict) else None
        if not isinstance(coord, dict) or "lat" not in coord or "lon" not in coord:
            raise ProviderError(
                "provider_unavailable",
                f"OpenWeather did not return coordinates for {resolved_location!r}",
                provider=self.PROVIDER_NAME,
            )

        params_forecast = {
            "lat": coord["lat"],
            "lon": coord["lon"],
            "units": "metric",
            "appid": api_key,
        }
        forecast_payload = await self._get_json(
            f"{_OPENWEATHER_BASE_URL}/data/2.5/forecast",
            params=params_forecast,
        )

        forecast_list_raw: Any = (
            forecast_payload.get("list") if isinstance(forecast_payload, dict) else None
        )
        forecast_list = (
            forecast_list_raw[:_FORECAST_ENTRIES]
            if isinstance(forecast_list_raw, list)
            else []
        )

        return {
            "location": resolved_location,
            "current": current,
            "forecast": forecast_list,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_api_key(self) -> str:
        """Read the OpenWeather API key from the credential store."""
        credential_name = str(getattr(self._config, "api_key_credential", ""))
        if not credential_name:
            raise ProviderError(
                "missing_credentials",
                "providers.weather.api_key_credential is not configured",
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
        self, url: str, *, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Issue a GET and parse JSON, mapping failures onto ProviderError."""
        try:
            response = await self.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "provider_unavailable",
                f"OpenWeather request timed out: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"OpenWeather request failed: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                "provider_unavailable",
                (
                    f"OpenWeather returned HTTP {response.status_code} "
                    f"for {url}"
                ),
                provider=self.PROVIDER_NAME,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                "provider_unavailable",
                f"OpenWeather returned non-JSON body: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                "provider_unavailable",
                "OpenWeather response was not a JSON object",
                provider=self.PROVIDER_NAME,
            )
        return data
