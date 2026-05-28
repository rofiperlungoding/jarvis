"""Unit tests for ``jarvis.skills.builtin.send_email``.

The Skill itself is intentionally thin — it adapts an injected
:class:`~jarvis.automation.providers.email.EmailClient`-shaped dependency
to the :class:`SkillResult` taxonomy. The tests exercise the adapter
behaviour with a fake email client and credential store, so the SMTP
transport is never touched here. Provider-level coverage (TLS upgrade,
credential lookup, allowlist enforcement) lives in
``tests/unit/automation/providers/test_email.py``.

Validates: Requirements 5.1, 5.2, 5.3, 5.6, 16.1, 16.2
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import Skill, SkillContext, SkillManifest
from jarvis.skills.builtin import send_email
from jarvis.skills.builtin.send_email import SCHEMA, SKILL, SendEmailSkill
from jarvis.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEmailClient:
    """Records every ``send`` call and replays a configured outcome."""

    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._result: dict[str, Any] | None = result
        self._raise: BaseException | None = raise_exc

    async def send(self, recipient: str, subject: str, body: str) -> dict[str, Any]:
        self.calls.append((recipient, subject, body))
        if self._raise is not None:
            raise self._raise
        return (
            self._result
            if self._result is not None
            else {
                "recipient": recipient,
                "subject": subject,
                "host": "smtp.example.com",
                "port": 587,
                "from": "alice@example.com",
            }
        )


class _FakeCredentialStore:
    """Minimal :class:`CredentialBackend`-shaped dictionary store.

    The Skill does not read credentials directly — that's the
    :class:`EmailClient`'s job — so this fake only exists to satisfy the
    Skill's "credential store is wired" sanity check.
    """

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


def _make_ctx(
    *,
    email_client: Any | None = None,
    credential_store: Any | None = ...,  # type: ignore[assignment]
) -> SkillContext:
    """Build a :class:`SkillContext` with the bits the Skill needs.

    Passing ``None`` for ``email_client`` simulates a misconfigured
    provider mapping (no ``"email"`` entry). Passing ``None`` for
    ``credential_store`` exercises the bootstrap-bug branch.
    """
    providers: dict[str, Any] = {}
    if email_client is not None:
        providers["email"] = email_client
    if credential_store is ...:
        credential_store = _FakeCredentialStore(
            {"email/smtp_user": "alice@example.com", "email/smtp_password": "pw"}
        )
    return SkillContext(
        providers=providers,
        credential_store=credential_store,
        run_id="send-email-test",
    )


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # Plugin discovery (registry._load_plugin_file) imports the module
    # and reads ``getattr(module, "SKILL", None)``; the constant must
    # exist and resolve to the singleton instance.
    assert isinstance(SKILL, SendEmailSkill)
    assert send_email.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable: confirm the
    # singleton would be accepted by the registry's ``isinstance`` gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_destructive_and_named_correctly() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "SendEmailSkill"
    # Requirement 16.1: SendEmailSkill is a hard-coded destructive skill.
    assert manifest.destructive is True
    assert manifest.source == "builtin"
    # The manifest's schema must be the same dict as the public ``SCHEMA``
    # constant so consumers (Mistral tool publishing, tests, docs) agree.
    assert manifest.json_schema is SCHEMA


def test_schema_requires_three_string_fields() -> None:
    # Requirement 5.1: the argument schema requires ``recipient``,
    # ``subject``, and ``body``.
    assert set(SCHEMA["required"]) == {"recipient", "subject", "body"}
    for name in ("recipient", "subject", "body"):
        assert SCHEMA["properties"][name]["type"] == "string"
    # ``additionalProperties: false`` keeps the LLM from smuggling cc /
    # bcc / attachments through this Skill.
    assert SCHEMA["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_dispatches_to_email_client_and_returns_payload() -> None:
    fake = _FakeEmailClient(
        result={
            "recipient": "bob@example.com",
            "subject": "Hi",
            "host": "smtp.example.com",
            "port": 587,
            "from": "alice@example.com",
        }
    )
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {
                "recipient": "bob@example.com",
                "subject": "Hi",
                "body": "Body of the message.",
            },
            ctx,
        )
    )

    assert result.ok is True
    assert result.error_code is None
    # The Skill forwards the provider's structured result verbatim so
    # the LLM can incorporate the confirmation into its spoken reply
    # (Requirement 5.3).
    assert result.value == {
        "recipient": "bob@example.com",
        "subject": "Hi",
        "host": "smtp.example.com",
        "port": 587,
        "from": "alice@example.com",
    }
    # The provider was invoked exactly once with the supplied args.
    assert fake.calls == [("bob@example.com", "Hi", "Body of the message.")]


def test_execute_strips_whitespace_from_recipient() -> None:
    # The schema's ``minLength: 1`` already rules out empty recipients
    # but does not strip surrounding whitespace; the Skill normalises
    # so the provider sees the same string the user expected.
    fake = _FakeEmailClient()
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {
                "recipient": "  bob@example.com  ",
                "subject": "Hi",
                "body": "Body",
            },
            ctx,
        )
    )

    assert result.ok is True
    assert fake.calls[0][0] == "bob@example.com"


# ---------------------------------------------------------------------------
# Error mappings
# ---------------------------------------------------------------------------


def test_missing_provider_returns_missing_credentials() -> None:
    # No ``"email"`` entry in providers — this is the operator-side
    # equivalent of "no credentials configured".
    ctx = _make_ctx(email_client=None)

    result = _run(
        SKILL.execute(
            {"recipient": "bob@example.com", "subject": "Hi", "body": "Body"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "missing_credentials"


def test_provider_missing_credentials_propagates_error_code() -> None:
    # The Skill must forward the provider's structured error to the
    # Dialog_Manager so the credential-setup flow is offered
    # (Requirement 5.6).
    fake = _FakeEmailClient(
        raise_exc=ProviderError(
            "missing_credentials",
            "credential 'email/smtp_password' is not set",
            provider="smtp",
        )
    )
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {"recipient": "bob@example.com", "subject": "Hi", "body": "Body"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "missing_credentials"
    assert "smtp_password" in (result.error_message or "")


def test_provider_unavailable_propagates_error_code() -> None:
    fake = _FakeEmailClient(
        raise_exc=ProviderError(
            "provider_unavailable",
            "SMTP transmission failed: connection refused",
            provider="smtp",
        )
    )
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {"recipient": "bob@example.com", "subject": "Hi", "body": "Body"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "provider_unavailable"


def test_network_policy_violation_maps_to_access_denied() -> None:
    # When the SMTP host is not on the allowlist the provider records
    # the audit row and raises :class:`NetworkPolicyViolation`. The
    # Skill must convert that to ``access_denied`` rather than
    # re-raising (which would cause the registry to record a *second*
    # audit row).
    fake = _FakeEmailClient(
        raise_exc=NetworkPolicyViolation(
            destination="smtp://blocked.invalid:587",
            host="blocked.invalid",
        )
    )
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {"recipient": "bob@example.com", "subject": "Hi", "body": "Body"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_value_error_from_provider_maps_to_schema_violation() -> None:
    # ``EmailClient.send`` raises ``ValueError`` for shape problems the
    # JSON Schema cannot express (e.g. comma-separated recipients).
    fake = _FakeEmailClient(
        raise_exc=ValueError(
            "recipient must be a single address; comma-separated lists " "are not supported"
        )
    )
    ctx = _make_ctx(email_client=fake)

    result = _run(
        SKILL.execute(
            {
                "recipient": "a@x.com,b@x.com",
                "subject": "Hi",
                "body": "Body",
            },
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Registry interop (cheap end-to-end via SkillRegistry)
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_through_registry() -> None:
    # End-to-end smoke test: the Skill must satisfy the registry's
    # manifest validation (draft-07 + Mistral subset) and dispatch
    # through the same code path that the production Dialog_Manager
    # uses.
    registry = SkillRegistry()
    registry.register(SKILL)

    fake = _FakeEmailClient()
    ctx = _make_ctx(email_client=fake)

    result = _run(
        registry.dispatch(
            "SendEmailSkill",
            {
                "recipient": "bob@example.com",
                "subject": "Hi",
                "body": "Body",
            },
            ctx,
        )
    )
    assert result.ok is True
    assert fake.calls == [("bob@example.com", "Hi", "Body")]


def test_registry_rejects_missing_required_fields() -> None:
    # Property 2 / CP2 gate: the registry validates args BEFORE
    # invoking ``execute``, so missing ``body`` must produce
    # ``schema_violation`` without ever calling the fake client.
    registry = SkillRegistry()
    registry.register(SKILL)

    fake = _FakeEmailClient()
    ctx = _make_ctx(email_client=fake)

    result = _run(
        registry.dispatch(
            "SendEmailSkill",
            {"recipient": "bob@example.com", "subject": "Hi"},
            ctx,
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []  # executor never reached


def test_registry_rejects_extra_fields() -> None:
    # ``additionalProperties: false`` keeps the LLM from smuggling
    # ``cc`` / ``bcc`` / attachments through the SendEmailSkill.
    registry = SkillRegistry()
    registry.register(SKILL)

    fake = _FakeEmailClient()
    ctx = _make_ctx(email_client=fake)

    result = _run(
        registry.dispatch(
            "SendEmailSkill",
            {
                "recipient": "bob@example.com",
                "subject": "Hi",
                "body": "Body",
                "cc": "carol@example.com",
            },
            ctx,
        )
    )
    assert result.ok is False
    assert result.error_code == "schema_violation"
    assert fake.calls == []
