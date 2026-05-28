"""Built-in :class:`LaunchAppSkill` — open registered Windows applications.

The Skill_Registry exposes ``LaunchAppSkill`` to the LLM so the user can
ask JARVIS to open an app by spoken name ("open Chrome", "launch
Spotify"). When the model emits the corresponding Tool_Call, this Skill
resolves the requested ``application`` against the configured
``[automation.application_registry]`` table and, on a hit, delegates to
:meth:`jarvis.automation.platform.PlatformAdapter.launch_app` to spawn
the executable / URI handler.

Argument schema
---------------

A single required string field ``application`` (Requirement 2.1). The
schema declares ``additionalProperties: false`` so a hallucinated extra
field is rejected by the registry's :class:`Draft7Validator` before the
executor ever runs (Property 2 / CP2). The ``minLength: 1`` constraint
lets the registry surface the obvious "empty string" mistake as
``schema_violation`` rather than letting it reach the registry lookup
where the failure mode would be the more confusing "unknown
application" path.

Why ``ctx.extras["application_registry"]`` rather than ``ctx.providers``
-----------------------------------------------------------------------

The :class:`~jarvis.skills.base.SkillContext` data model offers
``providers`` (HTTP-shaped external services, e.g. ``"weather"``,
``"news"``, ``"web_search"``) and an open-ended ``extras`` mapping for
"MCP / test-injected fakes". The application registry is plain data, not
an HTTP-shaped service, so injecting it via ``extras`` under the
well-known :data:`APPLICATION_REGISTRY_EXTRAS_KEY` is the cleanest
forward-compatible choice. This mirrors the pattern already used by
:class:`~jarvis.skills.builtin.timer.TimerSkill` and
:class:`~jarvis.skills.builtin.reminder.ReminderSkill`. Task 19.x
(application wiring) installs the dict under that key when assembling
the context for every Tool_Call.

The Windows adapter *also* carries its own copy of the registry (so it
can resolve an already-validated name), but the Skill cannot rely on
that resolution alone because the adapter's ``_resolve_target`` falls
back to PATH when a name is unknown — that fallback would mask the
"unknown application" case that Requirement 2.4 asks us to surface
explicitly. Resolving in the Skill keeps Requirement 2.4 honest.

Error mapping
-------------

* ``schema_violation`` — caught by the Skill_Registry before
  ``execute`` runs (empty string, missing field, extra fields). This
  module never returns it directly.
* ``not_supported`` — the requested ``application`` is not present in
  the registry. The closed
  :data:`~jarvis.skills.base.SkillErrorCode` taxonomy has no dedicated
  "unknown enum value" code; ``not_supported`` is the most accurate
  fit (the value the user asked for is not supported by this
  installation's registry) and Requirement 2.4 asks the
  Dialog_Manager to "ask the user to clarify or register the
  application", so the Skill carries the registered names back in the
  result ``value`` to give the dialog layer everything it needs to
  phrase the clarification.
* ``platform_not_supported`` — :meth:`PlatformAdapter.launch_app`
  raised :class:`PlatformNotSupportedError`, e.g. because the active
  adapter is the no-op :class:`BasePlatformAdapter` from the test
  bench. Mirrors the pattern in
  :mod:`jarvis.skills.builtin.media_control`.
* ``internal_error`` — the adapter raised :class:`FileNotFoundError`
  (the resolved executable does not exist on disk) or the
  :class:`SkillContext` is misconfigured (no platform adapter, no
  application registry). Other unrelated exceptions propagate up to
  the registry's catch-all (Property 7 / CP10).

The Skill is non-destructive: launching an application has no
permanent side effect beyond starting a process, so
:attr:`SkillManifest.destructive` is ``False`` and the
Authorization_Policy will not request user confirmation before
dispatch (Requirement 16.1).

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    PlatformAdapter,
    PlatformNotSupportedError,
    ProcessHandle,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "APPLICATION_REGISTRY_EXTRAS_KEY",
    "SKILL",
    "LaunchAppSkill",
]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Key under which application wiring (Task 19.x) installs the
#: ``automation.application_registry`` dict into
#: :attr:`SkillContext.extras`. Exposed as a module-level constant so
#: tests, the application bootstrap, and any future caller share a
#: single source of truth.
APPLICATION_REGISTRY_EXTRAS_KEY: Final[str] = "application_registry"


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


# JSON-Schema draft-07 conforming to the Mistral function-calling subset
# enforced by ``MistralSchemaValidator``: no ``$ref``, no scalar/object
# ``oneOf`` mixing, no unsupported ``format`` keyword. The schema mirrors
# Requirement 2.1 exactly: a single required string field named
# ``application``. ``minLength: 1`` rejects the empty string at the
# schema gate so it never reaches the registry lookup, and
# ``additionalProperties: false`` is the gate that lets the registry
# return ``schema_violation`` for a Tool_Call carrying arguments the
# Skill does not understand.
_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "LaunchAppSkillArguments",
    "description": (
        "Arguments for launching a registered application. "
        "``application`` is the spoken / written name as registered in "
        "[automation.application_registry] (e.g. 'chrome', 'vscode', "
        "'spotify')."
    ),
    "properties": {
        "application": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Name of the application to launch, matching a key in "
                "the configured application registry."
            ),
        },
    },
    "required": ["application"],
    "additionalProperties": False,
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="LaunchAppSkill",
    description=(
        "Launch a registered application by name. The 'application' "
        "argument must match an entry in the configured application "
        "registry (e.g. 'chrome', 'vscode', 'spotify'). Unknown names "
        "are reported as not_supported so the assistant can ask the "
        "user to clarify or register the application."
    ),
    json_schema=_JSON_SCHEMA,
    destructive=False,
    timeout_seconds=15.0,
    platforms=("windows",),
    source="builtin",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_registry(ctx: SkillContext) -> Mapping[str, str] | None:
    """Return the application registry injected via ``ctx.extras``.

    Returns ``None`` when the key is missing or maps to a value that
    is not a :class:`Mapping` of strings to strings. The Skill surfaces
    the missing-key case as ``internal_error`` because — like a missing
    :class:`ReminderService` for :class:`TimerSkill` — it indicates a
    wiring bug at bootstrap, not a user-facing limitation.
    """
    candidate = ctx.extras.get(APPLICATION_REGISTRY_EXTRAS_KEY)
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        return None
    # Defence-in-depth: reject mappings that contain non-string entries.
    # The :class:`AutomationConfig` model already constrains the type to
    # ``dict[str, str]``, but a smuggled-in fake could carry anything.
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
    return candidate


def _registered_names(registry: Mapping[str, str]) -> list[str]:
    """Stable, sorted list of registry keys for clarification messages."""
    return sorted(registry.keys())


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class LaunchAppSkill:
    """Launch a registered application via the platform adapter.

    The Skill is a thin adapter: argument validation is owned by the
    Skill_Registry through the JSON Schema (Property 2 / CP2), so the
    executor only needs to (a) resolve the ``application`` against
    the registry injected via :attr:`SkillContext.extras`, (b) forward
    the resolved executable / URI to
    :meth:`PlatformAdapter.launch_app`, and (c) translate the resulting
    :class:`ProcessHandle` into a JSON-serialisable success payload the
    Dialog_Manager can speak back to the user (Requirement 2.5).

    The Skill never accepts arbitrary executable paths from the LLM.
    Resolution always goes through the configured registry, which is
    the security perimeter that lets the Authorization_Policy treat
    application launches as non-destructive: the user cannot weaponise
    the Skill to run arbitrary binaries by uttering a path.
    """

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Resolve ``args["application"]`` and dispatch the launch.

        ``args`` has already been validated against :data:`_JSON_SCHEMA`
        by the :class:`SkillRegistry` (Property 2 / CP2), so we can
        assume ``"application"`` is present and is a non-empty string.
        The executor still defends against context-level
        misconfiguration (no platform adapter, no application registry,
        smuggled-in non-Protocol objects) and against the
        unknown-application case (Requirement 2.4) by returning the
        appropriate :class:`SkillResult` error.
        """
        adapter = ctx.platform_adapter
        adapter_error = self._validate_adapter(adapter)
        if adapter_error is not None:
            return adapter_error
        # ``adapter`` is now guaranteed to satisfy the PlatformAdapter
        # Protocol (the validator above re-checks).
        assert isinstance(adapter, PlatformAdapter)

        registry = _resolve_registry(ctx)
        if registry is None:
            # A missing registry is a wiring bug at bootstrap, not a
            # user-facing limitation — surface as ``internal_error`` so
            # the Dialog_Manager apologises rather than steering the
            # user toward an irrelevant troubleshooting path.
            logger.error(
                "LaunchAppSkill invoked without an application registry on "
                "ctx.extras[%r]; check application bootstrap",
                APPLICATION_REGISTRY_EXTRAS_KEY,
            )
            return SkillResult.error(
                "internal_error",
                (
                    "LaunchAppSkill requires an application_registry under "
                    f"ctx.extras[{APPLICATION_REGISTRY_EXTRAS_KEY!r}]; none "
                    "was supplied. The dispatcher's run-context is "
                    "misconfigured."
                ),
            )

        application = args["application"]

        # Requirement 2.4: unknown applications must surface an error
        # AND give the Dialog_Manager the material it needs to ask the
        # user to clarify or register the application. Carrying the
        # known registry keys back in ``value`` lets the dialog layer
        # phrase a useful follow-up ("I know about chrome, vscode, and
        # spotify — which one did you mean?") without the LLM having
        # to guess.
        if application not in registry:
            known = _registered_names(registry)
            logger.info(
                "LaunchAppSkill: unknown application %r (known=%s)",
                application,
                known,
            )
            return SkillResult.error(
                "not_supported",
                (
                    f"application {application!r} is not registered. "
                    "Ask the user to clarify or to register the "
                    "application in [automation.application_registry]."
                ),
                value={
                    "application": application,
                    "known_applications": known,
                    "needs_clarification": True,
                },
            )

        target = registry[application]

        # Requirement 2.2: spawn the application as a new process via
        # the platform adapter. The adapter is responsible for
        # environment-variable expansion (``%USERNAME%``) and for
        # distinguishing path / URI / PATH-resolved targets; we pass
        # the registry value through verbatim. ``args=[]`` because the
        # Skill's argument schema only accepts ``application`` —
        # extending it to forward extra arguments would require an
        # explicit schema change and a re-evaluation of the
        # destructive-action contract.
        try:
            handle: ProcessHandle = await adapter.launch_app(target, [])
        except PlatformNotSupportedError as exc:
            # Mirrors :mod:`jarvis.skills.builtin.media_control`: the
            # platform layer's "I don't implement this capability" error
            # maps to ``platform_not_supported`` so the Dialog_Manager
            # can apologise with a platform-specific message.
            logger.info(
                "LaunchAppSkill: launch_app unsupported on platform %r: %s",
                exc.platform,
                exc.detail,
            )
            return SkillResult.error(
                PLATFORM_NOT_SUPPORTED,
                str(exc),
            )
        except FileNotFoundError as exc:
            # The registry entry resolves to a path that does not exist.
            # This is a configuration error rather than a Skill-layer
            # bug — surface ``internal_error`` so the user knows the
            # registry is stale rather than the request was unknown.
            # Carrying the resolved target in ``value`` helps the
            # Dialog_Manager point the user at the broken entry.
            logger.warning(
                "LaunchAppSkill: registered target %r for application %r "
                "is not found on disk: %s",
                target,
                application,
                exc,
            )
            return SkillResult.error(
                "internal_error",
                (
                    f"registered target for {application!r} could not be "
                    f"launched: {exc}"
                ),
                value={
                    "application": application,
                    "target": target,
                },
            )

        # Requirement 2.5: confirm the action by application name. The
        # Dialog_Manager reads ``application`` out of the success
        # payload to phrase a natural-language acknowledgement; the
        # additional fields are diagnostic.
        return SkillResult.success(
            value={
                "application": application,
                "target": handle.executable_or_uri,
                "pid": handle.pid,
                "detached": handle.detached,
            }
        )

    @staticmethod
    def _validate_adapter(adapter: Any) -> SkillResult | None:
        """Return an ``internal_error`` result if ``adapter`` is unusable.

        The :class:`SkillContext` field is typed ``Any`` to avoid an
        import cycle with :mod:`jarvis.automation.platform`, so a
        misconfigured context can smuggle in either ``None`` or an
        unrelated object. Both cases are wiring bugs, not user-facing
        limitations.
        """
        if adapter is None:
            return SkillResult.error(
                "internal_error",
                "LaunchAppSkill requires ctx.platform_adapter",
            )
        if not isinstance(adapter, PlatformAdapter):
            return SkillResult.error(
                "internal_error",
                "ctx.platform_adapter does not satisfy the PlatformAdapter "
                f"protocol (got {type(adapter).__name__})",
            )
        return None


#: Top-level export consumed by :meth:`SkillRegistry.discover`. Plugin
#: discovery looks for an attribute named exactly ``SKILL`` on the
#: loaded module, so we expose a single shared instance. Built-in
#: Skills are also registered explicitly during application bootstrap;
#: exposing ``SKILL`` here keeps the discovery contract uniform between
#: built-in and user-supplied modules.
SKILL: Skill = LaunchAppSkill()
