"""Built-in ``SendEmailSkill``.

Implements the ``SendEmailSkill`` referenced from ``design.md §Built-in
Skills`` and Requirements 5.1, 5.2, 5.3, and 5.6. The Skill is a thin
wrapper around the :class:`~jarvis.automation.providers.email.EmailClient`
provider that the application bootstrap wires through
:attr:`SkillContext.providers` under the ``"email"`` key. The provider is
itself responsible for reading SMTP credentials out of the
:class:`~jarvis.security.credential_store.CredentialStore`, opening a
TLS-protected connection, and submitting the message; the Skill's job
here is to:

* Declare a Mistral-compatible JSON Schema for the three string arguments
  the LLM is expected to supply (``recipient``, ``subject``, ``body``).
* Mark itself ``destructive=True`` so the Authorization_Policy
  unconditionally requests confirmation before dispatch (Requirement
  5.2 / Requirement 16.1).
* Translate the provider's structured outcomes into the closed
  :class:`SkillResult` error taxonomy:

  * :class:`ProviderError` with code ``"missing_credentials"`` →
    :meth:`SkillResult.error` ``"missing_credentials"`` so the
    Dialog_Manager can guide the user through credential setup
    (Requirement 5.6).
  * :class:`ProviderError` with code ``"provider_unavailable"`` →
    :meth:`SkillResult.error` ``"provider_unavailable"`` (Requirement
    7.7).
  * :class:`NetworkPolicyViolation` raised when the configured SMTP host
    is not on ``security.network_destination_allowlist`` →
    :meth:`SkillResult.error` ``"access_denied"``. The provider client
    has already written the ``policy_violation`` audit row by the time
    we see the exception (Requirement 13.6), so we do **not** rewrite it
    as :class:`PolicyViolation`; doing so would double-count the audit
    entry through the registry's own catch-and-record path.

Confirmation flow
-----------------

The Skill itself never speaks to the user. The Authorization_Policy in
``src/jarvis/security/authorization.py`` consults the manifest's
:attr:`SkillManifest.destructive` flag, produces the "I'm about to send
an email to … OK?" read-back, and only invokes the Skill's
:meth:`execute` after the user assents. By the time we are running here,
that confirmation gate has already been cleared (Requirement 5.2).

Plugin discovery
----------------

The :class:`SkillRegistry` looks for a top-level ``SKILL`` attribute when
loading a plugin file (see ``registry._load_plugin_file``). The module
exposes the singleton instance under that exact name so the same module
can be discovered both as a built-in (registered programmatically) and,
in tests, as a generic plugin.

Validates: Requirements 5.1, 5.2, 5.3, 5.6, 16.1, 16.2
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = ["SCHEMA", "SKILL", "SendEmailSkill"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


# JSON Schema for the LLM-facing tool arguments. All three fields are
# required and constrained to non-empty strings so an obviously malformed
# tool call (e.g. ``{"recipient": ""}``) is rejected at the registry's
# argument-validation step (Property 2 / CP2) rather than reaching the
# SMTP transport.
#
# The schema is deliberately conservative: ``additionalProperties: false``
# means the LLM cannot smuggle ``cc``, ``bcc``, or attachment fields
# through this Skill — those will live behind their own dedicated tools
# if and when we add them.
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "SendEmail",
    "description": (
        "Send an email message via the configured SMTP provider. "
        "Requires explicit user confirmation before transmission."
    ),
    "properties": {
        "recipient": {
            "type": "string",
            "minLength": 1,
            "maxLength": 320,  # RFC 5321 maximum
            "description": (
                "Single email address of the recipient. Comma-separated "
                "lists are not supported by this tool."
            ),
        },
        "subject": {
            "type": "string",
            "minLength": 1,
            "maxLength": 998,  # RFC 5322 line length cap
            "description": "Subject line of the email.",
        },
        "body": {
            "type": "string",
            "minLength": 1,
            "description": "Plain-text body of the email.",
        },
    },
    "required": ["recipient", "subject", "body"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class SendEmailSkill:
    """Send an email via the wired :class:`EmailClient` provider.

    The Skill is intentionally stateless: a single instance is reused
    across invocations, with each ``execute`` call receiving the
    per-call :class:`SkillContext` produced by the
    :class:`SkillRegistry`. The provider lookup is deferred to
    :meth:`execute` rather than resolved at construction so the same
    instance can be registered before the providers are fully wired
    (the discovery path in :meth:`SkillRegistry.discover` runs at
    startup, before the run-loop in :func:`jarvis.app.main` populates
    every :class:`SkillContext`).
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name="SendEmailSkill",
        description=(
            "Send an email to a recipient with a subject and body. "
            "Confirms the action with the user before sending."
        ),
        json_schema=SCHEMA,
        destructive=True,
        timeout_seconds=30.0,
        # SMTP submission is OS-agnostic; the underlying ``smtplib`` module
        # is part of the standard library on every platform we ship.
        platforms=("windows", "macos", "linux"),
        source="builtin",
    )

    async def execute(
        self,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        # The registry already validated ``args`` against ``SCHEMA`` (and
        # would have returned ``schema_violation`` before reaching us if
        # any field were missing or empty), so we can trust the keys to
        # be present. We still pull them through ``str`` for type narrowing
        # so static analysis treats them as ``str`` rather than ``object``.
        recipient = str(args["recipient"]).strip()
        subject = str(args["subject"])
        body = str(args["body"])

        # ---- 1. Provider availability ----------------------------------
        # ``providers`` is a Mapping injected by the application
        # bootstrap; an absent ``"email"`` entry means the operator has
        # not enabled the SMTP provider in their config. From the user's
        # perspective this is indistinguishable from a missing
        # credential, so we surface ``missing_credentials`` to trigger
        # the documented credential-setup guidance (Requirement 5.6).
        email_client = ctx.providers.get("email")
        if email_client is None:
            return SkillResult.error(
                "missing_credentials",
                "Email provider is not configured. Configure "
                "providers.email in config.toml and store the SMTP "
                "credentials with `jarvis credentials set "
                "email/smtp_user` and `email/smtp_password`.",
            )

        # The credential store reference on the context is not used
        # directly here — :class:`EmailClient` consults its own copy of
        # the store at construction time — but we sanity-check that the
        # bootstrap wired one in. A missing store strongly implies a
        # misconfigured app, which we treat as ``internal_error`` rather
        # than masking the bug behind a credential error.
        if ctx.credential_store is None:  # pragma: no cover - defensive
            logger.error(
                "SendEmailSkill executed without a credential_store on "
                "the SkillContext; this indicates a bootstrap bug."
            )
            return SkillResult.error(
                "internal_error",
                "credential store is not wired into the skill context",
            )

        # ---- 2. Submit ------------------------------------------------
        try:
            sent = await email_client.send(recipient, subject, body)
        except ProviderError as exc:
            # ProviderError carries one of {"missing_credentials",
            # "provider_unavailable"}. Both map 1:1 onto the
            # SkillResult error taxonomy.
            return SkillResult.error(
                exc.error_code,
                str(exc),
            )
        except NetworkPolicyViolation as exc:
            # The provider client has already recorded the
            # ``policy_violation`` audit row before raising. We don't
            # re-raise as ``registry.PolicyViolation`` here because that
            # would cause the registry to write a *second* audit entry
            # for the same logical violation. ``access_denied`` is the
            # error code the registry's own PolicyViolation handler
            # uses, so the user-facing message remains consistent.
            return SkillResult.error(
                "access_denied",
                f"SMTP host blocked by network allowlist: {exc}",
            )
        except ValueError as exc:
            # ``EmailClient.send`` raises :class:`ValueError` for shape
            # problems the JSON Schema cannot express (e.g.
            # comma-separated recipients). Map to ``schema_violation``
            # so the LLM gets a chance to retry with a single recipient
            # (Requirement 14.5 caps retries at 2).
            return SkillResult.error(
                "schema_violation",
                f"invalid email arguments: {exc}",
            )

        # ---- 3. Success -----------------------------------------------
        # ``sent`` is the dict returned by ``EmailClient.send``; we
        # forward it verbatim to the LLM so it can incorporate the
        # confirmation into its spoken response (Requirement 5.3 — the
        # action is acknowledged after transmission). The ``EmailClient``
        # never includes secret material in this dict.
        return SkillResult.success(value=dict(sent))


#: Module-level singleton consumed by :meth:`SkillRegistry.discover`.
#: Typed as :class:`SendEmailSkill` rather than the :class:`Skill`
#: Protocol because the latter declares ``manifest`` as a writable
#: variable while we expose it as a :data:`Final` class attribute; the
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: SendEmailSkill = SendEmailSkill()
