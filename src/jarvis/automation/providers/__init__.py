"""Outbound provider clients for the Automation_Service.

Each provider client subclasses :class:`ProviderClient` so it inherits the
shared 5-second read timeout, exponential-backoff retries, network-egress
audit log entries, and ``security.network_destination_allowlist``
enforcement (Requirements 7.7, 13.4, 13.6). Per-provider modules add the
domain-specific request shapes and translate transport-level outcomes into
the structured :class:`ProviderError` taxonomy used by the calling Skills.
"""

from jarvis.automation.providers.calendar import CalendarClient
from jarvis.automation.providers.email import EmailClient
from jarvis.automation.providers.errors import (
    PROVIDER_ERROR_CODES,
    ProviderError,
    ProviderErrorCode,
)
from jarvis.automation.providers.http import (
    DEFAULT_PROVIDER_MAX_ATTEMPTS,
    DEFAULT_PROVIDER_TIMEOUT_S,
    NetworkPolicyViolation,
    ProviderClient,
)
from jarvis.automation.providers.news import NewsClient
from jarvis.automation.providers.weather import WeatherClient

__all__ = [
    "DEFAULT_PROVIDER_MAX_ATTEMPTS",
    "DEFAULT_PROVIDER_TIMEOUT_S",
    "PROVIDER_ERROR_CODES",
    "CalendarClient",
    "EmailClient",
    "NetworkPolicyViolation",
    "NewsClient",
    "ProviderClient",
    "ProviderError",
    "ProviderErrorCode",
    "WeatherClient",
]
