"""SMTP email provider client.

Implements the ``EmailClient`` referenced by ``SendEmailSkill``
(Requirement 5.1, 5.2, 5.3, 5.6). The client opens an SMTP connection to
the configured host/port, optionally upgrades to TLS, authenticates with
credentials sourced from :class:`CredentialStore`, and submits an
RFC 5322-compatible message constructed via :class:`email.message.EmailMessage`.

Because :mod:`smtplib` is synchronous, the actual SMTP transaction runs on
a worker thread via :func:`asyncio.to_thread` so the dialog event loop
keeps moving while we wait on the upstream server. Allowlist enforcement
and audit recording are inherited from :class:`ProviderClient`: we
construct a synthetic ``smtp://host:port`` URL so the same
``network_egress`` / ``policy_violation`` accounting applies regardless of
the underlying transport.

The Authorization_Policy is responsible for prompting the user for
confirmation upstream of this client (Requirement 5.2). By the time
:meth:`send` is invoked, that gate has already been cleared.

Validates: Requirements 5.1, 5.2, 5.3, 5.6, 7.7
"""

from __future__ import annotations

import asyncio
from email.message import EmailMessage
import logging
import smtplib
import ssl
from typing import Any, Final

import httpx

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import ProviderClient
from jarvis.security.audit_log import AuditLog
from jarvis.security.credential_store import CredentialBackend

logger = logging.getLogger(__name__)

__all__ = ["EmailClient"]


_DEFAULT_TIMEOUT_S: Final[float] = 5.0


class EmailClient(ProviderClient):
    """SMTP email submission client.

    Parameters mirror the other provider clients. ``provider_config``
    MUST expose ``host`` (string), ``port`` (int), ``username_credential``
    (string), ``password_credential`` (string), and ``timeout_seconds``
    (float).
    """

    PROVIDER_NAME: Final[str] = "smtp"

    def __init__(
        self,
        *,
        audit_log: AuditLog,
        network_allowlist: list[str] | tuple[str, ...] | frozenset[str],
        credential_store: CredentialBackend,
        provider_config: Any,
        client: httpx.AsyncClient | None = None,
        ssl_context: ssl.SSLContext | None = None,
        smtp_factory: Any = None,
    ) -> None:
        # The base ``ProviderClient`` is engineered around an HTTP client
        # but its allowlist + audit accounting is content-agnostic, so we
        # subclass it and simply forward the bookkeeping. The optional
        # ``client`` argument is accepted for symmetry with the other
        # providers and is unused here beyond the lifecycle that the base
        # class manages.
        super().__init__(
            audit_log=audit_log,
            network_allowlist=network_allowlist,
            justification="email send",
            skill_name="EmailClient",
            client=client,
            timeout_seconds=float(getattr(provider_config, "timeout_seconds", _DEFAULT_TIMEOUT_S)),
        )
        self._credentials: CredentialBackend = credential_store
        self._config: Any = provider_config
        self._ssl_context: ssl.SSLContext = ssl_context or ssl.create_default_context()
        # ``smtp_factory`` lets tests inject a ``smtplib.SMTP`` lookalike
        # without monkey-patching the module. Defaults to the real
        # synchronous client which we drive on a worker thread.
        self._smtp_factory: Any = smtp_factory or smtplib.SMTP

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def send(self, recipient: str, subject: str, body: str) -> dict[str, Any]:
        """Submit an email to ``recipient`` via the configured SMTP server.

        ``recipient`` MUST be a single RFC 5321 address; comma-separated
        lists are rejected so the Authorization_Policy's spoken read-back
        and the actual transmission cannot disagree on the audience
        (Requirement 5.2). The ``From:`` header is taken from the
        configured SMTP username credential.

        Returns:
            ``{"recipient", "subject", "host", "port", "from"}``.

        Raises:
            ProviderError(missing_credentials): SMTP credentials absent
                from the credential store (Requirement 5.6).
            ProviderError(provider_unavailable): SMTP transmission failed
                or timed out within ``timeout_seconds`` (Requirement 7.7).
            NetworkPolicyViolation: the configured SMTP host is not on
                the network allowlist (Requirement 13.6).
        """
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError("recipient must be a non-empty string")
        if "," in recipient:
            raise ValueError(
                "recipient must be a single address; comma-separated lists "
                "are not supported"
            )
        if not isinstance(subject, str):
            raise TypeError("subject must be a string")
        if not isinstance(body, str):
            raise TypeError("body must be a string")

        host = str(getattr(self._config, "host", "")).strip()
        port = int(getattr(self._config, "port", 0))
        if not host or port <= 0:
            raise ProviderError(
                "provider_unavailable",
                "providers.email.host / providers.email.port not configured",
                provider=self.PROVIDER_NAME,
            )

        # Allowlist + audit accounting. ``_enforce_and_record`` writes a
        # ``network_egress`` row on success and raises
        # :class:`NetworkPolicyViolation` (after recording a
        # ``policy_violation`` row) on a blocked destination — exactly the
        # behaviour the HTTP providers get for free.
        destination = f"smtp://{host}:{port}"
        self._ensure_open()
        await self._enforce_and_record(destination)

        username = self._read_credential(
            getattr(self._config, "username_credential", ""),
            field_name="providers.email.username_credential",
        )
        password = self._read_credential(
            getattr(self._config, "password_credential", ""),
            field_name="providers.email.password_credential",
        )

        message = EmailMessage()
        message["From"] = username
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        try:
            await asyncio.to_thread(
                self._submit,
                host=host,
                port=port,
                username=username,
                password=password,
                message=message,
            )
        except ProviderError:
            raise
        except (smtplib.SMTPException, OSError, TimeoutError) as exc:
            # Any transport-level failure (connection refused, auth
            # failure, server error, network timeout) collapses to
            # ``provider_unavailable`` per the design's error taxonomy.
            raise ProviderError(
                "provider_unavailable",
                f"SMTP transmission failed: {exc}",
                provider=self.PROVIDER_NAME,
            ) from exc

        return {
            "recipient": recipient,
            "subject": subject,
            "host": host,
            "port": port,
            "from": username,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _submit(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        message: EmailMessage,
    ) -> None:
        """Synchronous SMTP submission, run via :func:`asyncio.to_thread`.

        Uses STARTTLS on the standard submission port (587) and any other
        non-465 port. Port 465 is reserved for implicit TLS, which would
        require :class:`smtplib.SMTP_SSL` — we surface a clear error in
        that case rather than silently choosing the wrong transport.
        """
        if port == 465:
            raise ProviderError(
                "provider_unavailable",
                "implicit-TLS SMTP (port 465) is not supported; "
                "use STARTTLS on port 587",
                provider=self.PROVIDER_NAME,
            )

        timeout = float(getattr(self._config, "timeout_seconds", _DEFAULT_TIMEOUT_S))
        with self._smtp_factory(host, port, timeout=timeout) as smtp:
            smtp.ehlo()
            # STARTTLS upgrades the connection to TLS in-band. We always
            # attempt it on the standard submission port to keep the
            # Mistral key / SMTP password off the wire in clear-text.
            try:
                smtp.starttls(context=self._ssl_context)
                smtp.ehlo()
            except smtplib.SMTPNotSupportedError:
                # Some test/dev servers (e.g., MailHog) advertise no TLS.
                # We log and continue: the network_destination_allowlist
                # already constrains where we can talk to.
                logger.warning(
                    "SMTP server %s:%d does not support STARTTLS",
                    host,
                    port,
                )
            smtp.login(username, password)
            smtp.send_message(message)

    def _read_credential(self, name: str | object, *, field_name: str) -> str:
        credential_name = str(name or "")
        if not credential_name:
            raise ProviderError(
                "missing_credentials",
                f"{field_name} is not configured",
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
