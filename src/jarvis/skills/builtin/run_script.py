"""Built-in :class:`RunScriptSkill` — execute a registered script.

The Skill_Registry exposes ``RunScriptSkill`` to the LLM so the user can
trigger pre-registered automation scripts ("run my backup", "deploy the
site"). When the model emits the corresponding Tool_Call, this Skill
resolves the requested ``script_id`` against the
:class:`~jarvis.automation.scripts.ScriptCatalog` injected into the
:class:`SkillContext` and forwards execution to the configured
interpreter via :meth:`PlatformAdapter.run_script`.

Why ``script_id`` rather than free-form script text
---------------------------------------------------

Requirement 9.5 forbids the assistant from executing arbitrary script
text supplied directly in a Tool_Call argument. The Skill therefore
accepts only a single string — the ``script_id`` — and relies on the
:class:`ScriptCatalog` to map it to a concrete interpreter / path pair
that the user has explicitly registered in
``[automation.script_catalog]``. The JSON Schema enforces this at the
gate (``additionalProperties: false`` rejects ``script``, ``command``,
``code`` and similar smuggled fields), and the executor enforces it
again at runtime by routing through the catalog.

Confirmation flow
-----------------

The Skill marks itself ``destructive=True`` so the
Authorization_Policy unconditionally requests confirmation before
dispatch (Requirement 9.2 / Requirement 16.1). The Skill itself never
talks to the user — by the time :meth:`execute` is called, the
confirmation gate has already been cleared by the Dialog_Manager.

Context contract
----------------

The Skill expects ``ctx.extras[SCRIPT_CATALOG_EXTRAS_KEY]`` to hold a
constructed :class:`~jarvis.automation.scripts.ScriptCatalog`. The
application bootstrap (``src/jarvis/app.py``, task 19.1) is
responsible for wiring it up when the run-loop assembles a context for
each Tool_Call. If the entry is missing — for example, in unit tests
or because the automation service crashed at boot — the executor
returns ``internal_error``: a missing dependency is a wiring bug, not
a user-facing limitation.

Error mapping (closed :class:`SkillResult` taxonomy)
----------------------------------------------------

* ``schema_violation`` — caught by the Skill_Registry before
  ``execute`` runs (missing ``script_id``, empty string, extra
  fields). This module never returns it directly.
* ``script_not_found`` — :meth:`ScriptCatalog.run` raised
  :class:`KeyError` because the supplied id is not registered
  (Requirement 9.4). The error result carries the offending id and
  the list of registered ids back to the Dialog_Manager so it can
  phrase a useful clarification ("I know about 'backup' and
  'deploy' — which one did you mean?").
* ``timeout`` — :class:`ScriptResult.timed_out` is ``True`` because the
  adapter killed the process for exceeding the configured budget
  (Requirement 9.8). The result payload preserves the captured
  stdout/stderr so the user can still see partial progress.
* ``platform_not_supported`` — the platform adapter raised
  :class:`PlatformNotSupportedError` (e.g., the active adapter is the
  no-op :class:`BasePlatformAdapter` from the test bench).
* ``internal_error`` — the catalog dependency is missing, the
  platform adapter is missing or shaped wrong, or the underlying
  ``run_script`` call raised a transient OS-level exception.

A non-zero exit code from a script that ran to completion within its
budget is reported as a success result with ``ok=True`` and the exit
code in the value payload. The closed error taxonomy has no
"non_zero_exit" code, and the script is best understood as having
"run successfully but reported failure" — a distinction the
Dialog_Manager can communicate to the user via the captured stderr.

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.8, 16.1, 16.2
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.automation.platform import (
    PLATFORM_NOT_SUPPORTED,
    PlatformAdapter,
    PlatformNotSupportedError,
    ScriptResult,
)
from jarvis.automation.scripts import (
    DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ScriptCatalog,
)
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA",
    "SCRIPT_CATALOG_EXTRAS_KEY",
    "SKILL",
    "RunScriptSkill",
]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: Key under which the application bootstrap (task 19.x) installs the
#: configured :class:`ScriptCatalog` into :attr:`SkillContext.extras`.
#: Exposed as a module-level constant so tests, the application
#: bootstrap, and any future caller share a single source of truth.
SCRIPT_CATALOG_EXTRAS_KEY: Final[str] = "script_catalog"


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


# JSON-Schema draft-07 conforming to the Mistral function-calling subset
# enforced by ``MistralSchemaValidator``: no ``$ref``, no ``oneOf``, no
# unsupported ``format`` keyword. The schema mirrors Requirement 9.1
# exactly: a single required string field named ``script_id``.
#
# ``minLength: 1`` rejects the empty string at the schema gate so it
# never reaches the catalog lookup (which would surface the more
# confusing "unknown id" path). ``additionalProperties: false`` is the
# guarantee that the LLM cannot smuggle ``script`` / ``command`` / ``code``
# / ``arguments`` fields through this Skill — those would let the model
# bypass the lookup-only contract that Requirement 9.5 establishes.
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "RunScript",
    "description": (
        "Arguments for executing a registered script. ``script_id`` is "
        "the catalog key declared in [automation.script_catalog]; the "
        "tool refuses to execute arbitrary script text."
    ),
    "properties": {
        "script_id": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Identifier of a script registered in the user's "
                "[automation.script_catalog]. The tool will not execute "
                "raw script text supplied in this argument."
            ),
        },
    },
    "required": ["script_id"],
    "additionalProperties": False,
}


_MANIFEST: Final[SkillManifest] = SkillManifest(
    name="RunScriptSkill",
    description=(
        "Run a script registered in the user's script catalog by its "
        "script_id. Supports PowerShell, Python, and batch scripts. "
        "Requires explicit user confirmation before execution. "
        "Refuses arbitrary script text — only registered ids are "
        "accepted."
    ),
    json_schema=SCHEMA,
    # Requirement 9.2 / 16.1 — every invocation is destructive and
    # requires user confirmation before dispatch. Flagged on the
    # manifest so the Authorization_Policy unconditionally classifies
    # the Tool_Call as Destructive_Action without having to consult the
    # config-driven ``destructive_skills`` list.
    destructive=True,
    # Requirement 9.8 — the manifest's wall-clock budget mirrors the
    # 60 s ceiling enforced by the platform adapter so the registry's
    # own timeout machinery and the adapter's interpreter timeout agree
    # on the same boundary. A value larger than
    # :data:`DEFAULT_SCRIPT_TIMEOUT_SECONDS` would let the adapter kill
    # the process before the registry gives up; smaller would do the
    # opposite. Aligning them keeps the user-visible behaviour single
    # sourced.
    timeout_seconds=DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    # Script execution is OS-agnostic at the Skill layer; the platform
    # adapter is the boundary that decides whether a given interpreter
    # is available on the host. Declaring the full matrix here lets the
    # Skill stay registered on platforms whose adapter implements
    # ``run_script`` (currently :class:`WindowsAdapter`); other adapters
    # surface :class:`PlatformNotSupportedError` which the Skill
    # translates into ``platform_not_supported``.
    platforms=("windows", "linux", "darwin"),
    source="builtin",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_catalog(ctx: SkillContext) -> ScriptCatalog | None:
    """Return the :class:`ScriptCatalog` injected via ``ctx.extras``.

    Returns ``None`` when the entry is missing or holds an object that
    is not a :class:`ScriptCatalog`. The Skill surfaces both as
    ``internal_error`` because — like a missing :class:`ReminderService`
    for :class:`TimerSkill` — they indicate a wiring bug at bootstrap,
    not a user-facing limitation.
    """
    candidate = ctx.extras.get(SCRIPT_CATALOG_EXTRAS_KEY)
    if candidate is None:
        return None
    if not isinstance(candidate, ScriptCatalog):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class RunScriptSkill:
    """Execute a registered script via the platform adapter.

    The Skill is a thin adapter: argument validation is owned by the
    Skill_Registry through :data:`SCHEMA` (Property 2 / CP2), so the
    executor only needs to (a) resolve the :class:`ScriptCatalog` and
    :class:`PlatformAdapter` from the :class:`SkillContext`,
    (b) forward the call to :meth:`ScriptCatalog.run`, and
    (c) translate the resulting :class:`ScriptResult` into a
    JSON-serialisable success payload (or one of the three documented
    error codes ``script_not_found`` / ``timeout`` /
    ``platform_not_supported``).

    The Skill never accepts arbitrary script text: the JSON Schema
    rejects unknown fields and the catalog rejects unknown ids. This
    is the security perimeter that lets the Authorization_Policy treat
    every invocation as a confirmed Destructive_Action without having
    to inspect the spawned process (Requirements 9.5, 16.1).
    """

    manifest: SkillManifest = _MANIFEST

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Resolve ``args["script_id"]`` and dispatch the script run.

        ``args`` has already been validated against :data:`SCHEMA` by
        the :class:`SkillRegistry` (Property 2 / CP2), so we can assume
        ``"script_id"`` is present and is a non-empty string. The
        executor still defends against context-level misconfiguration
        (no platform adapter, no script catalog, smuggled-in non-Protocol
        objects) and against the unknown-id case (Requirement 9.4) by
        returning the appropriate :class:`SkillResult` error.
        """
        adapter_error = self._validate_adapter(ctx.platform_adapter)
        if adapter_error is not None:
            return adapter_error

        catalog = _resolve_catalog(ctx)
        if catalog is None:
            # A missing catalog is a wiring bug at bootstrap. Surface as
            # ``internal_error`` so the Dialog_Manager apologises rather
            # than steering the user toward the credential-setup or
            # platform-troubleshooting flows that other error codes
            # would trigger.
            logger.error(
                "RunScriptSkill invoked without a ScriptCatalog on "
                "ctx.extras[%r]; check application bootstrap",
                SCRIPT_CATALOG_EXTRAS_KEY,
            )
            return SkillResult.error(
                "internal_error",
                (
                    "RunScriptSkill requires a ScriptCatalog under "
                    f"ctx.extras[{SCRIPT_CATALOG_EXTRAS_KEY!r}]; none "
                    "was supplied. The dispatcher's run-context is "
                    "misconfigured."
                ),
            )

        return await self._dispatch(catalog, args["script_id"])

    @staticmethod
    async def _dispatch(catalog: ScriptCatalog, script_id: str) -> SkillResult:
        """Resolve ``script_id`` against ``catalog`` and translate the outcome.

        Splitting this off the top-level :meth:`execute` keeps the
        coroutine's return-statement count under the linter's
        ``PLR0911`` budget while preserving the exception-translation
        table — every documented :meth:`ScriptCatalog.run` error is
        caught here and mapped onto the closed
        :class:`SkillResult` taxonomy. The same pattern is used by
        :mod:`jarvis.skills.builtin.brightness`.
        """
        # ``ScriptCatalog.run`` performs the lookup, the type guards
        # (``script_id`` non-empty string, timeout positive finite),
        # and forwards to ``PlatformAdapter.run_script`` with the
        # catalog-declared interpreter / path. Catching its documented
        # exceptions here keeps the error-translation table centralised
        # in this module so the Dialog_Manager only has to reason about
        # the closed :class:`SkillResult` taxonomy.
        try:
            outcome: ScriptResult = await catalog.run(script_id)
        except KeyError:
            # Requirement 9.4 — unknown ids surface as ``script_not_found``.
            # Carry the offending id and the list of registered ids back
            # in ``value`` so the Dialog_Manager can phrase a useful
            # follow-up clarification.
            known = catalog.list_ids()
            logger.info(
                "RunScriptSkill: unknown script_id %r (known=%s)",
                script_id,
                known,
            )
            return SkillResult.error(
                "script_not_found",
                (
                    f"script_id {script_id!r} is not registered. "
                    "Ask the user to clarify or to register the script "
                    "in [automation.script_catalog]."
                ),
                value={
                    "script_id": script_id,
                    "known_script_ids": known,
                    "needs_clarification": True,
                },
            )
        except PlatformNotSupportedError as exc:
            # The platform adapter does not implement ``run_script`` at
            # all (BasePlatformAdapter on a non-Windows host, or a
            # future stub). Mirrors the pattern used by
            # :mod:`jarvis.skills.builtin.media_control`.
            logger.info(
                "RunScriptSkill: run_script unsupported on platform %r: %s",
                exc.platform,
                exc.detail,
            )
            return SkillResult.error(
                PLATFORM_NOT_SUPPORTED,
                str(exc),
            )
        except (TypeError, ValueError) as exc:
            # The schema gate already rejects the obvious offenders, but
            # ``ScriptCatalog.run`` performs its own defensive checks
            # (``script_id`` empty, timeout non-finite) and may raise if
            # a future schema change loosens those constraints. Surface
            # as ``schema_violation`` so the LLM gets a chance to retry
            # with a better-shaped argument (Requirement 14.5 caps
            # retries at 2).
            logger.warning(
                "ScriptCatalog.run rejected script_id=%r: %s",
                script_id,
                exc,
            )
            return SkillResult.error(
                "schema_violation",
                f"invalid run_script arguments: {exc}",
            )
        except Exception as exc:
            # Adapter-level OS errors propagate through ``ScriptCatalog.run``
            # so the registry's exception barrier can classify them.
            # We catch them here instead of letting them bubble because
            # the Skill is the right place to attach the script id to
            # the diagnostic — the registry only sees the bare
            # exception. ``internal_error`` matches the design's
            # default classification for transient OS-level failures.
            logger.exception(
                "RunScriptSkill: adapter raised while executing script_id=%r",
                script_id,
            )
            return SkillResult.error(
                "internal_error",
                (
                    f"failed to execute script {script_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ),
                value={"script_id": script_id},
            )

        return RunScriptSkill._translate_outcome(script_id, outcome)

    @staticmethod
    def _translate_outcome(script_id: str, outcome: ScriptResult) -> SkillResult:
        """Translate a successful :meth:`ScriptCatalog.run` outcome.

        The result is either ``timeout`` (Requirement 9.8 — the adapter
        killed the process for exceeding its budget) or success: a
        non-zero exit code from a script that ran to completion is
        reported as ``ok=True`` with the exit code in ``value`` because
        the closed :class:`SkillResult` taxonomy has no dedicated
        "non_zero_exit" code (Requirement 9.3 — capture
        stdout/stderr/exit code; the Dialog_Manager decides how to
        communicate the script's own report of failure).
        """
        if outcome.timed_out:
            logger.info(
                "RunScriptSkill: script_id=%r timed out after %d ms",
                script_id,
                outcome.duration_ms,
            )
            return SkillResult.error(
                "timeout",
                (
                    f"script {script_id!r} exceeded its "
                    f"{int(DEFAULT_SCRIPT_TIMEOUT_SECONDS)}-second budget "
                    "and was terminated."
                ),
                value={
                    "script_id": script_id,
                    "timed_out": True,
                    "duration_ms": outcome.duration_ms,
                    "exit_code": outcome.exit_code,
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                },
            )
        return SkillResult.success(
            value={
                "script_id": script_id,
                "exit_code": outcome.exit_code,
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "duration_ms": outcome.duration_ms,
                "timed_out": False,
            }
        )

    @staticmethod
    def _validate_adapter(adapter: Any) -> SkillResult | None:
        """Return an ``internal_error`` result if ``adapter`` is unusable.

        The :class:`SkillContext` field is typed ``Any`` to avoid an
        import cycle with :mod:`jarvis.automation.platform`, so a
        misconfigured context can smuggle in either ``None`` or an
        unrelated object. Both are wiring bugs, not user-facing
        limitations.

        Note: we still validate the platform adapter even though the
        :class:`ScriptCatalog` carries its own reference. The catalog's
        constructor checks the Protocol shape, but a context-level
        misconfiguration where the catalog was wired but the adapter
        was not (or vice versa) would be confusing to debug — failing
        fast here surfaces the bootstrap bug clearly.
        """
        if adapter is None:
            return SkillResult.error(
                "internal_error",
                "RunScriptSkill requires ctx.platform_adapter",
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
SKILL: Skill = RunScriptSkill()
