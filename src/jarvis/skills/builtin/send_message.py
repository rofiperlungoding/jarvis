"""Built-in ``SendMessageSkill``.

Implements the ``SendMessageSkill`` referenced from ``design.md §Built-in
Skills`` and Requirements 5.4, 5.5, and 5.6. The Skill is the multi-channel
counterpart to :class:`~jarvis.skills.builtin.send_email.SendEmailSkill`:
where ``SendEmailSkill`` is bound to a single SMTP transport, this skill
dispatches the message to one of several pluggable *channel adapters*
(SMS, WhatsApp, Slack, IRC, …). Channels are looked up by name through a
:class:`MessageChannelDispatcher` exposed under
``ctx.providers["messaging"]``; each adapter is responsible for its own
transport, credentials, and audit accounting.

Design choices
--------------

* **Channel adapter Protocol.** :class:`MessageChannelAdapter` is a
  :class:`typing.Protocol` so concrete channels can live anywhere — in
  ``jarvis.automation.providers``, in user plugins, or as MCP servers —
  without inheriting from a base class. The Protocol mirrors
  :class:`~jarvis.automation.providers.email.EmailClient` in spirit but
  is intentionally narrower: a single ``send(recipient, body)`` coroutine
  returning a structured payload, plus a stable ``name`` identifying the
  channel.

* **Dispatcher pattern.** The Skill never reaches into a hard-coded
  channel registry of its own. Instead the application bootstrap builds
  a :class:`MessageChannelDispatcher` populated with whichever adapters
  the operator has configured and assigns it to
  ``ctx.providers["messaging"]``. The Skill calls
  ``dispatcher.send(channel, recipient, body)``, which raises
  :class:`UnknownChannelError` for an unconfigured channel name; the
  Skill maps that to ``"not_supported"`` per the closed error taxonomy.

* **Stub adapter ships in-tree.** :class:`InMemoryChannelAdapter` is the
  reference implementation used by the unit tests (and viable as a dev
  loopback while real adapters are still on the roadmap). It performs no
  network I/O, records each message in an in-memory list, and never asks
  the credential store for a secret — making it the simplest example of
  how a channel adapter integrates with the Skill.

Confirmation flow
-----------------

The Authorization_Policy reads the manifest's
:attr:`~SkillManifest.destructive` flag (``True``) and produces a spoken
read-back of the resolved channel, recipient, and body before invoking
the Skill, identical to the flow used for ``SendEmailSkill``
(Requirement 5.5). By the time :meth:`SendMessageSkill.execute` runs, the
user has already assented.

Error taxonomy mapping
----------------------

* :class:`UnknownChannelError` → ``"not_supported"``. The user asked for
  a channel the operator has not enabled.
* :class:`~jarvis.automation.providers.errors.ProviderError` with code
  ``"missing_credentials"`` → ``"missing_credentials"`` so the
  Dialog_Manager can guide credential setup (Requirement 5.6).
* :class:`~jarvis.automation.providers.errors.ProviderError` with code
  ``"provider_unavailable"`` → ``"provider_unavailable"`` (Requirement
  7.7).
* :class:`~jarvis.automation.providers.http.NetworkPolicyViolation` →
  ``"access_denied"``. The adapter has already recorded the
  ``policy_violation`` audit row (Requirement 13.6), so we surface the
  outcome without rewriting the audit trail.
* :class:`ValueError` raised by the adapter for shape problems the JSON
  Schema cannot express (e.g. recipient format) → ``"schema_violation"``
  to give the LLM one retry attempt (Requirement 14.5).

Validates: Requirements 5.4, 5.5, 5.6, 16.1, 16.2
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
from typing import Any, Final, Protocol, runtime_checkable

from jarvis.automation.providers.errors import ProviderError
from jarvis.automation.providers.http import NetworkPolicyViolation
from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MESSAGING_PROVIDER_KEY",
    "SCHEMA",
    "SKILL",
    "InMemoryChannelAdapter",
    "MessageChannelAdapter",
    "MessageChannelDispatcher",
    "SendMessageSkill",
    "UnknownChannelError",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Key under which the application bootstrap stores the configured
#: :class:`MessageChannelDispatcher` on :attr:`SkillContext.providers`.
#: Matches the convention documented in ``design.md`` for ``"weather"``,
#: ``"news"``, ``"email"``, ``"calendar"``, and ``"web_search"``.
MESSAGING_PROVIDER_KEY: Final[str] = "messaging"


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


# Mistral-compatible schema for the LLM-facing tool arguments. All three
# fields are required and constrained to non-empty strings. The
# ``channel`` field is intentionally *not* an enum: which channels are
# available is a deployment-time decision that the operator drives via
# config. Validating the channel name happens inside the Skill (against
# the live :class:`MessageChannelDispatcher`) so config changes do not
# require regenerating the manifest.
#
# ``additionalProperties: false`` is the gate that lets the registry
# return ``schema_violation`` for an LLM that smuggles unknown fields
# (e.g. ``cc``, ``subject``, attachments) — those will live behind their
# own dedicated tools if and when we add them.
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "SendMessage",
    "description": (
        "Send a chat message via a configured messaging channel. "
        "Requires explicit user confirmation before transmission."
    ),
    "properties": {
        "channel": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "description": (
                "Identifier of the messaging provider, for example "
                "'sms', 'whatsapp', 'slack'. Must match a channel "
                "configured by the operator."
            ),
        },
        "recipient": {
            "type": "string",
            "minLength": 1,
            "maxLength": 320,
            "description": (
                "Channel-specific recipient identifier (phone number, "
                "user handle, room id, …). Comma-separated lists are "
                "not supported by this tool."
            ),
        },
        "body": {
            "type": "string",
            "minLength": 1,
            "description": "Plain-text body of the message.",
        },
    },
    "required": ["channel", "recipient", "body"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Channel adapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MessageChannelAdapter(Protocol):
    """Plugin interface a messaging channel must satisfy.

    A channel adapter encapsulates everything specific to one transport:
    its provider client, its credential lookup, its allowlist accounting.
    The Skill itself remains transport-agnostic.

    Concrete adapters live alongside their provider clients (for
    example, ``jarvis.automation.providers.sms.SmsChannelAdapter``).
    The Protocol is :func:`runtime_checkable` so the
    :class:`MessageChannelDispatcher` can guard against accidentally
    registering a non-adapter object via ``isinstance`` rather than
    surfacing an :class:`AttributeError` at the first call.

    Attributes
    ----------
    name:
        Stable identifier of the channel as the LLM and the operator
        configuration spell it (e.g. ``"sms"``, ``"whatsapp"``,
        ``"slack"``). MUST match the dispatcher key under which the
        adapter was registered.
    """

    name: str

    async def send(self, recipient: str, body: str) -> Mapping[str, Any]:
        """Submit a message to ``recipient`` over this channel.

        Implementations MUST:

        * raise :class:`ProviderError` (with one of the documented
          provider error codes) for credential / transport failures;
        * raise :class:`NetworkPolicyViolation` for blocked destinations
          *after* the audit row has been written;
        * raise :class:`ValueError` for shape problems the JSON Schema
          cannot express; and
        * return a JSON-serialisable mapping describing the outcome on
          success. The Skill forwards the mapping verbatim to the LLM
          so it can phrase a confirmation, so adapters MUST NOT include
          secret material.
        """
        ...


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class UnknownChannelError(LookupError):
    """Raised when a channel name is not registered with the dispatcher.

    Inherits from :class:`LookupError` so callers that already treat
    "key not found" generically continue to work; the Skill maps the
    exception onto :class:`SkillResult` ``"not_supported"`` so the LLM
    is told the channel is unavailable rather than that the request
    itself was malformed.

    Attributes
    ----------
    channel:
        The unknown channel name as supplied by the LLM.
    available:
        Sorted tuple of channel names the dispatcher does have. The
        Skill includes this list in its error message so the LLM can
        reformulate a Tool_Call against a real channel on retry.
    """

    def __init__(self, channel: str, available: Iterable[str]) -> None:
        self.channel: str = channel
        self.available: tuple[str, ...] = tuple(sorted(available))
        if self.available:
            available_msg = ", ".join(self.available)
            message = (
                f"channel {channel!r} is not configured "
                f"(available: {available_msg})"
            )
        else:
            message = (
                f"channel {channel!r} is not configured "
                "(no messaging channels are configured)"
            )
        super().__init__(message)


class MessageChannelDispatcher:
    """Channel registry that fans Tool_Calls out to adapters by name.

    The dispatcher is the value the bootstrap stores under
    ``ctx.providers[MESSAGING_PROVIDER_KEY]``. It is intentionally a
    thin wrapper: register adapters once at startup, then call
    :meth:`send` (or :meth:`get`) for each Tool_Call. The dispatcher
    does *not* impose its own retry, audit, or allowlist policy — those
    are the adapter's responsibility — so adding a new channel never
    requires touching the dispatcher or the Skill.
    """

    def __init__(self, adapters: Iterable[MessageChannelAdapter] = ()) -> None:
        self._adapters: dict[str, MessageChannelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: MessageChannelAdapter) -> None:
        """Add ``adapter`` to the dispatcher.

        Raises
        ------
        TypeError
            ``adapter`` does not satisfy the
            :class:`MessageChannelAdapter` Protocol.
        ValueError
            ``adapter.name`` is not a non-empty string, or another
            adapter is already registered under the same name.
        """
        if not isinstance(adapter, MessageChannelAdapter):
            raise TypeError(
                "adapter must satisfy MessageChannelAdapter "
                f"(got {type(adapter).__name__})"
            )
        name = getattr(adapter, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("adapter.name must be a non-empty string")
        if name in self._adapters:
            raise ValueError(
                f"a messaging channel named {name!r} is already registered"
            )
        self._adapters[name] = adapter

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def channels(self) -> tuple[str, ...]:
        """Sorted tuple of registered channel names."""
        return tuple(sorted(self._adapters))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    def get(self, name: str) -> MessageChannelAdapter:
        """Return the adapter registered under ``name``.

        Raises :class:`UnknownChannelError` when no such adapter exists
        so callers who only need the lookup can rely on a structured
        exception (rather than a ``KeyError``).
        """
        try:
            return self._adapters[name]
        except KeyError:
            raise UnknownChannelError(name, self._adapters) from None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def send(
        self, channel: str, recipient: str, body: str
    ) -> Mapping[str, Any]:
        """Look up ``channel`` and forward the message to its adapter."""
        adapter = self.get(channel)
        return await adapter.send(recipient, body)


# ---------------------------------------------------------------------------
# Reference adapter — in-memory loopback
# ---------------------------------------------------------------------------


class InMemoryChannelAdapter:
    """Tiny stub :class:`MessageChannelAdapter` for tests and dev loopback.

    The adapter never opens a network connection: it appends each
    submitted ``(recipient, body)`` pair to :attr:`sent` and returns a
    JSON-serialisable confirmation dict. It is intentionally minimal —
    real channels (Twilio SMS, Slack, WhatsApp Cloud API) live in
    ``jarvis.automation.providers``; this adapter exists so the Skill
    has a working reference implementation from day one and so the test
    suite can exercise the dispatcher pattern without mocking a
    transport.

    Parameters
    ----------
    name:
        Stable channel identifier (e.g. ``"local"``, ``"console"``).
        Defaults to ``"local"`` to make the in-tree adapter
        self-describing.
    fail_with:
        Optional exception to raise from :meth:`send` instead of
        recording the message. Useful in tests that want to exercise
        the Skill's error mapping without writing a bespoke adapter.
    """

    def __init__(
        self,
        name: str = "local",
        *,
        fail_with: BaseException | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self.name: str = name
        self.sent: list[dict[str, str]] = []
        self._fail_with: BaseException | None = fail_with

    async def send(self, recipient: str, body: str) -> Mapping[str, Any]:
        if not isinstance(recipient, str) or not recipient.strip():
            raise ValueError("recipient must be a non-empty string")
        if "," in recipient:
            raise ValueError(
                "recipient must be a single identifier; comma-separated "
                "lists are not supported"
            )
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        if self._fail_with is not None:
            raise self._fail_with
        record = {"recipient": recipient, "body": body, "channel": self.name}
        self.sent.append(record)
        logger.debug(
            "InMemoryChannelAdapter[%s] queued message for %s (%d bytes)",
            self.name,
            recipient,
            len(body.encode("utf-8")),
        )
        return {"channel": self.name, "recipient": recipient, "delivered": True}


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class SendMessageSkill:
    """Send a chat message via the configured channel adapter.

    The Skill is stateless. Each ``execute`` call resolves the
    :class:`MessageChannelDispatcher` from the
    :class:`~jarvis.skills.base.SkillContext`, looks up the requested
    channel, and forwards the call. Channel-specific concerns
    (credentials, allowlists, retries) live entirely in the adapter.
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name="SendMessageSkill",
        description=(
            "Send a chat message to a recipient via a configured "
            "messaging channel (SMS, WhatsApp, Slack, …). Confirms the "
            "action with the user before sending."
        ),
        json_schema=SCHEMA,
        destructive=True,
        timeout_seconds=30.0,
        # Messaging channel adapters use OS-agnostic transports
        # (HTTP, WebSocket); the Skill itself does not require Windows.
        platforms=("windows", "macos", "linux"),
        source="builtin",
    )

    @staticmethod
    def _check_dispatcher(ctx: SkillContext) -> SkillResult | None:
        """Validate the messaging dispatcher wired into ``ctx``.

        Returns ``None`` when ``ctx.providers[MESSAGING_PROVIDER_KEY]``
        is a usable :class:`MessageChannelDispatcher`; otherwise returns
        the :class:`SkillResult` the Skill should surface to the LLM.
        Pulling this out of :meth:`execute` keeps the dispatch path's
        return-statement count under the configured Pylint budget while
        still expressing the two failure modes (no provider, wrong
        type) explicitly.
        """
        dispatcher = ctx.providers.get(MESSAGING_PROVIDER_KEY)
        if dispatcher is None:
            # No messaging dispatcher means the operator has not
            # configured any channels. Surface as ``missing_credentials``
            # so the Dialog_Manager runs the credential-setup flow that
            # Requirement 5.6 mandates — the practical user-facing
            # remedy is the same regardless of whether the dispatcher
            # itself is missing or merely empty.
            return SkillResult.error(
                "missing_credentials",
                "Messaging is not configured. Configure at least one "
                "channel under providers.messaging in config.toml and "
                "store the channel credentials in the credential store.",
            )

        if not isinstance(dispatcher, MessageChannelDispatcher):
            # Defensive: a smuggled-in object that is not a dispatcher
            # is a bootstrap bug, not a user error. Map to
            # ``internal_error`` so the audit log shows the bug rather
            # than masking it as a credential problem.
            logger.error(
                "SendMessageSkill received a providers[%r] that is not a "
                "MessageChannelDispatcher: %r",
                MESSAGING_PROVIDER_KEY,
                type(dispatcher).__name__,
            )
            return SkillResult.error(
                "internal_error",
                "messaging provider is not a MessageChannelDispatcher",
            )

        return None

    async def execute(
        self,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        # The registry already validated ``args`` against ``SCHEMA``,
        # so the keys are guaranteed present. We still narrow them to
        # ``str`` so static analysis treats the locals as ``str`` rather
        # than ``object``, and so a stray non-string slipping through
        # would surface here instead of inside the adapter.
        channel = str(args["channel"]).strip()
        recipient = str(args["recipient"]).strip()
        body = str(args["body"])

        # ---- 1. Dispatcher availability -------------------------------
        unavailable = self._check_dispatcher(ctx)
        if unavailable is not None:
            return unavailable
        # ``_check_dispatcher`` has already narrowed the type for us.
        dispatcher: MessageChannelDispatcher = ctx.providers[
            MESSAGING_PROVIDER_KEY
        ]

        # ---- 2. Submit ------------------------------------------------
        try:
            sent = await dispatcher.send(channel, recipient, body)
        except UnknownChannelError as exc:
            # The user asked for a channel that does not exist. The
            # closed error taxonomy entry that fits is ``not_supported``
            # — it tells the Dialog_Manager to inform the user rather
            # than retry the LLM.
            return SkillResult.error(
                "not_supported",
                str(exc),
                value={
                    "channel": exc.channel,
                    "available_channels": list(exc.available),
                },
            )
        except ProviderError as exc:
            # ProviderError carries one of {"missing_credentials",
            # "provider_unavailable"}; both map 1:1 onto the
            # SkillResult error taxonomy.
            return SkillResult.error(
                exc.error_code,
                str(exc),
            )
        except NetworkPolicyViolation as exc:
            # The adapter has already recorded the ``policy_violation``
            # audit row before raising. Avoid re-raising as
            # ``registry.PolicyViolation`` because that would write a
            # *second* audit entry for the same logical violation.
            return SkillResult.error(
                "access_denied",
                f"Messaging host blocked by network allowlist: {exc}",
            )
        except ValueError as exc:
            # Adapter-level shape failures the JSON Schema cannot
            # express (e.g. recipient format). ``schema_violation`` lets
            # the LLM retry once (Requirement 14.5 caps retries at 2).
            return SkillResult.error(
                "schema_violation",
                f"invalid messaging arguments: {exc}",
            )

        # ---- 3. Success ----------------------------------------------
        # ``sent`` is the mapping returned by the adapter. We materialise
        # it into a plain ``dict`` so :class:`SkillResult` can hold it
        # (the result type is ``dict | None``) and so any exotic Mapping
        # subclass cannot leak through to the Dialog_Manager.
        return SkillResult.success(value=dict(sent))


#: Module-level singleton consumed by :meth:`SkillRegistry.discover`.
#: Typed as :class:`SendMessageSkill` rather than the :class:`Skill`
#: Protocol because the latter declares ``manifest`` as a writable
#: variable while we expose it as a :data:`Final` class attribute; the
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: SendMessageSkill = SendMessageSkill()
