"""Unit tests for ``jarvis.automation.providers.email.EmailClient``.

Validates: Requirements 5.1, 5.2, 5.3, 5.6, 7.7, 13.6
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
import smtplib
from typing import Any, ClassVar

import pytest

from jarvis.automation.providers.email import EmailClient
from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.security.audit_log import AuditLog
from jarvis.utils.time_source import FakeTimeSource


@dataclass
class _StubConfig:
    host: str = "smtp.example.com"
    port: int = 587
    username_credential: str = "email/smtp_user"
    password_credential: str = "email/smtp_password"
    timeout_seconds: float = 5.0


class _DictBackend:
    def __init__(self, items: dict[str, str] | None = None) -> None:
        self._items: dict[str, str] = dict(items or {})

    def set(self, name: str, value: str) -> None:
        self._items[name] = value

    def get(self, name: str) -> str | None:
        return self._items.get(name)

    def delete(self, name: str) -> None:
        self._items.pop(name, None)

    def list_names(self) -> list[str]:
        return sorted(self._items)

    def wipe(self) -> None:
        self._items.clear()


class _FakeSMTP:
    """Minimal :class:`smtplib.SMTP` lookalike usable as a context manager.

    Records every method call so tests can assert the intended sequence
    (``ehlo`` → ``starttls`` → ``ehlo`` → ``login`` → ``send_message``).
    """

    instances: ClassVar[list[_FakeSMTP]] = []

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        starttls_unsupported: bool = False,
        login_failure: Exception | None = None,
        send_failure: Exception | None = None,
    ) -> None:
        self.host: str = host
        self.port: int = port
        self.timeout: float = timeout
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.sent_message: EmailMessage | None = None
        self._starttls_unsupported: bool = starttls_unsupported
        self._login_failure: Exception | None = login_failure
        self._send_failure: Exception | None = send_failure
        _FakeSMTP.instances.append(self)

    def __enter__(self) -> _FakeSMTP:
        self.calls.append(("__enter__", ()))
        return self

    def __exit__(self, *args: Any) -> None:
        self.calls.append(("__exit__", ()))

    def ehlo(self) -> None:
        self.calls.append(("ehlo", ()))

    def starttls(self, *, context: Any = None) -> None:
        if self._starttls_unsupported:
            raise smtplib.SMTPNotSupportedError("not supported")
        self.calls.append(("starttls", ()))

    def login(self, user: str, password: str) -> None:
        if self._login_failure is not None:
            raise self._login_failure
        self.calls.append(("login", (user, password)))

    def send_message(self, message: EmailMessage) -> None:
        if self._send_failure is not None:
            raise self._send_failure
        self.sent_message = message
        self.calls.append(("send_message", ()))


def _factory(**overrides: Any):
    """Return a callable suitable for ``smtp_factory=`` that reuses overrides."""

    def _build(host: str, port: int, *, timeout: float) -> _FakeSMTP:
        return _FakeSMTP(host, port, timeout=timeout, **overrides)

    return _build


@pytest.fixture()
def audit_log(tmp_path: Path) -> Iterator[AuditLog]:
    log = AuditLog(
        tmp_path / "audit.sqlite",
        time_source=FakeTimeSource(now=datetime(2024, 1, 1, tzinfo=UTC)),
        run_id="email-test",
    )
    yield log
    log.close()


@pytest.fixture(autouse=True)
def _reset_smtp_instances() -> Iterator[None]:
    _FakeSMTP.instances.clear()
    yield
    _FakeSMTP.instances.clear()


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_send_authenticates_and_submits_message(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "alice@example.com", "email/smtp_password": "pw"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        result = _run(
            client.send(
                "bob@example.com",
                "Hi Bob",
                "Body of the message.",
            )
        )
    finally:
        _run(client.aclose())

    assert result == {
        "recipient": "bob@example.com",
        "subject": "Hi Bob",
        "host": "smtp.example.com",
        "port": 587,
        "from": "alice@example.com",
    }

    assert len(_FakeSMTP.instances) == 1
    smtp = _FakeSMTP.instances[0]
    sequence = [name for name, _ in smtp.calls]
    assert sequence == [
        "__enter__",
        "ehlo",
        "starttls",
        "ehlo",
        "login",
        "send_message",
        "__exit__",
    ]
    assert smtp.sent_message is not None
    assert smtp.sent_message["From"] == "alice@example.com"
    assert smtp.sent_message["To"] == "bob@example.com"
    assert smtp.sent_message["Subject"] == "Hi Bob"


def test_send_records_network_egress_for_smtp(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    rows = audit_log.entries()
    assert len(rows) == 1
    entry = rows[0]
    assert entry.kind == "network_egress"
    assert entry.skill == "EmailClient"
    assert entry.destination == "smtp://smtp.example.com:587"
    assert entry.justification == "email send"


def test_send_continues_when_starttls_unsupported(audit_log: AuditLog) -> None:
    """MailHog and similar dev servers advertise no TLS — log and proceed."""
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(starttls_unsupported=True),
    )

    try:
        result = _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    assert result["recipient"] == "bob@example.com"
    smtp = _FakeSMTP.instances[0]
    sequence = [name for name, _ in smtp.calls]
    # No second ``ehlo`` after the failed STARTTLS, but ``login`` /
    # ``send_message`` still run.
    assert "starttls" not in sequence  # raised, not recorded
    assert "login" in sequence
    assert "send_message" in sequence


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_send_rejects_comma_separated_recipients(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        with pytest.raises(ValueError, match="comma"):
            _run(client.send("a@x.com,b@x.com", "Hi", "Body"))
    finally:
        _run(client.aclose())


def test_send_rejects_empty_recipient(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        with pytest.raises(ValueError, match="recipient"):
            _run(client.send("   ", "Hi", "Body"))
    finally:
        _run(client.aclose())


# ---------------------------------------------------------------------------
# Credentials / failure mappings
# ---------------------------------------------------------------------------


def test_missing_username_credential_raises_missing_credentials(
    audit_log: AuditLog,
) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_password": "p"}  # username missing
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "missing_credentials"


def test_smtp_auth_error_maps_to_provider_unavailable(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(
            login_failure=smtplib.SMTPAuthenticationError(535, b"bad password")
        ),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"


def test_implicit_tls_port_465_rejected(audit_log: AuditLog) -> None:
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["smtp.example.com"],
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(port=465),
        smtp_factory=_factory(),
    )

    try:
        with pytest.raises(ProviderError) as exc_info:
            _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    assert exc_info.value.error_code == "provider_unavailable"
    assert "465" in str(exc_info.value)


def test_blocked_host_raises_network_policy_violation(audit_log: AuditLog) -> None:
    """Off-allowlist SMTP hosts raise before the SMTP transaction runs."""
    client = EmailClient(
        audit_log=audit_log,
        network_allowlist=["other.invalid"],  # smtp.example.com is NOT here
        credential_store=_DictBackend(
            {"email/smtp_user": "u", "email/smtp_password": "p"}
        ),
        provider_config=_StubConfig(),
        smtp_factory=_factory(),
    )

    try:
        with pytest.raises(NetworkPolicyViolation):
            _run(client.send("bob@example.com", "Hi", "Body"))
    finally:
        _run(client.aclose())

    # No SMTP attempt was made.
    assert _FakeSMTP.instances == []
    # Audit log carries a policy_violation row.
    rows = audit_log.entries()
    assert len(rows) == 1
    assert rows[0].kind == "policy_violation"
