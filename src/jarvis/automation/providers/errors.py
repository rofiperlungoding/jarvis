"""Structured errors raised by provider clients.

Provider clients (``WeatherClient``, ``NewsClient``, ``CalendarClient``,
``EmailClient``) translate transport-level outcomes into one of the
:class:`SkillResult` error codes documented in ``design.md §Error Taxonomy``
so the calling Skill can convert them to a ``SkillResult.error(...)``
without repeating boilerplate per-provider.

Two codes are surfaced from this layer:

* ``missing_credentials`` — the credential entry the client needs is absent
  from :class:`CredentialStore`. Per Requirement 5.6 the Dialog_Manager
  guides the user through the credential-setup flow when this code is
  returned to the user.
* ``provider_unavailable`` — the upstream HTTP / SMTP service refused the
  request after the retry budget was exhausted, returned an error
  response, or timed out within the read timeout. Per Requirement 7.7 the
  Dialog_Manager simply informs the user in this case.

Skills consuming these errors typically look like::

    try:
        data = await self._weather.fetch(location)
    except ProviderError as exc:
        return SkillResult.error(exc.error_code, str(exc))

Validates: Requirements 5.6, 7.7
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = [
    "PROVIDER_ERROR_CODES",
    "ProviderError",
    "ProviderErrorCode",
]


ProviderErrorCode = Literal["missing_credentials", "provider_unavailable"]

#: Closed runtime-checkable mirror of :data:`ProviderErrorCode`.
PROVIDER_ERROR_CODES: Final[tuple[str, ...]] = (
    "missing_credentials",
    "provider_unavailable",
)


class ProviderError(RuntimeError):
    """Raised by a provider client when an upstream call cannot be satisfied.

    The exception carries the error code that the calling Skill should map
    onto :class:`SkillResult.error_code`. Treating these as exceptions
    rather than return values keeps the per-provider call sites readable
    while still preserving the structured taxonomy at the Skill boundary.
    """

    def __init__(
        self,
        error_code: ProviderErrorCode,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        if error_code not in PROVIDER_ERROR_CODES:
            raise ValueError(
                f"unknown provider error code: {error_code!r}; "
                f"expected one of {PROVIDER_ERROR_CODES!r}"
            )
        if not isinstance(message, str) or not message:
            raise ValueError("ProviderError message must be a non-empty string")
        super().__init__(message)
        self.error_code: ProviderErrorCode = error_code
        self.provider: str | None = provider
