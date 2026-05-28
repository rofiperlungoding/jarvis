"""Skill discovery, validation, and dispatch.

The :class:`SkillRegistry` is the single broker between the Dialog_Manager
and the heterogeneous set of Skills (built-in, user-defined, MCP). It owns
four responsibilities sketched in ``design.md §Skill_Registry``:

1. **Plugin discovery.** Scan a list of plugin directories at startup and
   load Python modules that expose a top-level ``SKILL: Skill`` attribute
   (Requirement 14.1).

2. **Manifest validation.** When a Skill is registered, the registry
   validates its ``json_schema`` against:

   * the JSON Schema draft-07 meta-schema (via
     :meth:`jsonschema.Draft7Validator.check_schema`); and
   * the Mistral function-calling subset (via
     :class:`jarvis.llm.mistral_schema.MistralSchemaValidator`).

   Failure refuses registration so the LLM_Backend never sees an unsafe
   tool definition (Requirement 14.3 / CP15).

3. **Tool-call dispatch.** On each ``dispatch(name, args, ctx)`` the
   registry validates ``args`` against the Skill's compiled
   :class:`jsonschema.Draft7Validator`. A failure short-circuits with
   ``schema_violation`` *without* invoking the executor — this is the
   exact contract Property 2 / CP2 verifies. On success the registry
   awaits the executor exactly once, catches any exception, and converts
   it to a structured :class:`SkillResult`:

   * :class:`PolicyViolation` (sandbox / network allowlist breach) →
     ``access_denied`` with a ``policy_violation`` entry recorded in the
     :class:`AuditLog` (Requirement 13.6).
   * Any other exception → ``internal_error`` carrying a short
     ``traceback_id`` so operators can correlate logs (Requirement 17.1
     / CP10, see also Property 7).

4. **Mistral tool publishing.** ``mistral_tool_definitions()`` projects
   every registered Skill into the dict shape Mistral's function-calling
   API expects (Requirement 19.4 / CP15) using
   :meth:`MistralSchemaValidator.to_mistral_tool`.

Design notes
------------

* The registry is backend-agnostic: it neither imports nor depends on the
  Mistral SDK. The only Mistral coupling is the schema-shape validator.
* Plugin loading uses :mod:`importlib.util` with explicit
  ``spec_from_file_location`` so we never mutate ``sys.path`` and never
  collide with installed packages.
* All error conditions are surfaced through :class:`SkillResult` rather
  than exceptions, with the single exception that ``register`` raises
  :class:`SkillRegistrationError` (loud-fail at startup is preferable to
  silently mis-configured tools).
* The audit-log write for policy violations is best-effort: if the audit
  log itself is misbehaving we still return the ``access_denied`` result
  to the Dialog_Manager rather than masking the original failure.

Validates: Requirements 13.6, 14.1, 14.2, 14.3, 14.4, 14.5, 17.1, 19.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import uuid

import jsonschema
from jsonschema import Draft7Validator

from jarvis.llm.mistral_schema import MistralSchemaError, MistralSchemaValidator
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillErrorCode,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NetworkPolicyViolation",
    "PolicyViolation",
    "SandboxViolation",
    "SkillRegistrationError",
    "SkillRegistry",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillRegistrationError(ValueError):
    """Raised when a Skill cannot be registered.

    Inherits from :class:`ValueError` so callers that already treat
    misconfigured skills as configuration errors continue to work, while
    type-checkers can still distinguish it from generic ``ValueError``
    via ``isinstance``.
    """


class PolicyViolation(Exception):  # noqa: N818 - public stable API name; renaming would break downstream Skills
    """Sandbox or network allowlist breach raised from a Skill executor.

    Skills MUST raise a :class:`PolicyViolation` (or subclass) when they
    detect that the requested operation would step outside the policies
    enforced by the application: for example, reading a path that
    escapes the allowed-directory list or contacting a host not in the
    network allowlist.

    The :class:`SkillRegistry` catches the exception, records a
    ``policy_violation`` audit entry, and converts it to a
    ``SkillResult.error("access_denied", ...)`` (Requirement 13.6).

    Attributes
    ----------
    justification:
        Human-readable reason the operation was denied. Stored in the
        audit log's ``justification`` column. Defaults to the exception
        message.
    """

    error_code: SkillErrorCode = "access_denied"

    def __init__(self, message: str, *, justification: str | None = None) -> None:
        super().__init__(message)
        self.justification = justification or message


class SandboxViolation(PolicyViolation):
    """Path / filesystem sandbox boundary breached."""


class NetworkPolicyViolation(PolicyViolation):
    """Network destination not in the configured allowlist."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_skill_like(obj: Any) -> bool:
    """Cheap structural check used by ``register``.

    The :class:`Skill` ``Protocol`` is :func:`runtime_checkable`, but its
    structural matching only inspects attribute presence — it cannot
    enforce that ``manifest`` is a real :class:`SkillManifest` or that
    ``execute`` is callable. We layer a stricter check on top so the
    registry can refuse obviously broken plugins with a clear error
    message instead of failing later with an :class:`AttributeError`.
    """
    if not hasattr(obj, "manifest") or not hasattr(obj, "execute"):
        return False
    return callable(obj.execute)


def _serialize_args(args: Any) -> str:
    """Render Tool_Call arguments as canonical JSON for audit logging.

    Falls back to ``repr`` if the arguments contain non-JSON values so
    the audit row still receives *something* useful for forensics. We do
    not raise here: a corrupt audit string is preferable to losing the
    fact that a policy violation occurred.
    """
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(args)


def _make_traceback_id() -> str:
    """Short, unique correlation id for ``internal_error`` failures."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Discover, validate, and dispatch Skills.

    Parameters
    ----------
    mistral_validator:
        Optional pre-built :class:`MistralSchemaValidator`. The default
        instance is sufficient for production use; tests inject a custom
        validator only when verifying registry behaviour around invalid
        schemas.
    monotonic:
        Injectable monotonic clock used to compute ``duration_ms`` for
        the dispatcher's synthetic results. Defaults to
        :func:`time.perf_counter`. Tests pass a callable returning a
        deterministic counter so timings are reproducible.
    """

    def __init__(
        self,
        *,
        mistral_validator: MistralSchemaValidator | None = None,
        monotonic: Any = None,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        # Pre-compiled validators, keyed by skill name. Compiling once at
        # registration time avoids per-call overhead in the hot dispatch
        # path (see Property 2 / CP2 — the validator must agree exactly
        # with ``Draft7Validator(S.json_schema).is_valid(A)``).
        self._validators: dict[str, Draft7Validator] = {}
        self._mistral_validator: MistralSchemaValidator = (
            mistral_validator or MistralSchemaValidator()
        )
        self._monotonic = monotonic or time.perf_counter

    # ------------------------------------------------------------------ public

    @property
    def names(self) -> list[str]:
        """Stable, sorted list of registered Skill names."""
        return sorted(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        """Return the Skill registered under ``name`` (or ``None``)."""
        return self._skills.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    # -- Registration ---------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Validate ``skill`` and add it to the registry.

        The Skill is rejected (and :class:`SkillRegistrationError` is
        raised) when:

        * ``skill`` is missing the ``manifest`` / ``execute`` attributes;
        * ``skill.manifest`` is not a :class:`SkillManifest`;
        * ``manifest.json_schema`` is not a valid JSON Schema (draft-07
          meta-schema check);
        * ``manifest.json_schema`` violates the Mistral function-calling
          subset; or
        * a Skill with the same ``manifest.name`` is already registered.

        On success the validator is compiled once and cached so subsequent
        ``dispatch`` calls do not re-parse the schema.
        """
        if not _is_skill_like(skill):
            raise SkillRegistrationError(
                "skill object must expose a SkillManifest 'manifest' "
                "attribute and a callable 'execute' coroutine"
            )

        manifest = skill.manifest
        if not isinstance(manifest, SkillManifest):
            raise SkillRegistrationError(
                "skill.manifest must be a SkillManifest instance, got "
                f"{type(manifest).__name__}"
            )

        # 1. JSON Schema meta-schema validation. ``check_schema`` raises
        #    ``jsonschema.SchemaError`` when the document itself is not a
        #    valid JSON Schema (e.g., ``required`` is not a list of
        #    strings, ``type`` is misspelled, ``properties`` is not an
        #    object). This is the first line of defence before we let
        #    the Mistral subset checker or the LLM see the schema.
        try:
            Draft7Validator.check_schema(manifest.json_schema)
        except jsonschema.SchemaError as exc:
            raise SkillRegistrationError(
                f"skill {manifest.name!r} has an invalid JSON Schema: {exc.message}"
            ) from exc

        # 2. Mistral subset validation. Rejects $ref to remote, mixed
        #    oneOf scalar/object branches, and unsupported `format`
        #    values. CP15 / Property 12 require this to be a hard fail.
        try:
            self._mistral_validator.validate(manifest.json_schema)
        except MistralSchemaError as exc:
            raise SkillRegistrationError(
                f"skill {manifest.name!r} JSON Schema is not Mistral-compatible: {exc}"
            ) from exc

        if manifest.name in self._skills:
            raise SkillRegistrationError(
                f"a skill named {manifest.name!r} is already registered"
            )

        # ``Draft7Validator`` is the validator stipulated by Property 2 /
        # CP2 ("SHALL return a schema_violation error iff
        # jsonschema.Draft7Validator(S.json_schema).is_valid(A) is
        # false"). We compile it from a deep-copy of the schema (via
        # ``dict(...)``) only if the manifest's schema is a non-dict
        # mapping, so the validator never sees a transient object.
        schema_for_validator: dict[str, Any]
        if isinstance(manifest.json_schema, dict):
            schema_for_validator = manifest.json_schema
        else:
            schema_for_validator = dict(manifest.json_schema)

        self._skills[manifest.name] = skill
        self._validators[manifest.name] = Draft7Validator(schema_for_validator)
        logger.debug("registered skill %r (source=%s)", manifest.name, manifest.source)

    # -- Discovery ------------------------------------------------------

    def discover(self, plugin_dirs: Iterable[Path]) -> None:
        """Load every plugin module under each directory in ``plugin_dirs``.

        A "plugin module" is any ``*.py`` file (excluding files whose
        name begins with ``_``) that, when imported, exposes a top-level
        attribute named ``SKILL`` whose value is a :class:`Skill`. Each
        plugin is registered individually; one bad plugin must not stop
        the others from loading, so import / registration failures are
        logged and discovery continues.

        Directories that do not exist or are not directories are ignored
        with a debug log. Files are processed in sorted order so the load
        sequence (and any "first one wins" behaviour for duplicate names)
        is deterministic.
        """
        for plugin_dir in plugin_dirs:
            try:
                resolved = Path(plugin_dir)
            except TypeError:
                logger.warning("invalid plugin directory entry: %r", plugin_dir)
                continue

            if not resolved.is_dir():
                logger.debug("skipping non-existent plugin dir %s", resolved)
                continue

            for file_path in sorted(resolved.glob("*.py")):
                if file_path.name.startswith("_"):
                    # ``__init__.py`` and private helper modules are not
                    # plugin entrypoints; the registry only consumes the
                    # top-level SKILL convention.
                    continue
                self._load_plugin_file(file_path)

    def _load_plugin_file(self, file_path: Path) -> None:
        """Import ``file_path`` and register its ``SKILL`` if present."""
        module_name = self._unique_module_name(file_path)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning("cannot build import spec for plugin %s", file_path)
                return
            module = importlib.util.module_from_spec(spec)
            # Register the module in ``sys.modules`` *before* executing it
            # so cyclic intra-plugin imports resolve correctly. We use a
            # private prefix to avoid colliding with installed packages.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Drop the partial module to avoid leaving a half-loaded
                # entry in ``sys.modules`` that other plugins might shadow.
                sys.modules.pop(module_name, None)
                raise
        except Exception:
            logger.exception("failed to import plugin %s", file_path)
            return

        skill = getattr(module, "SKILL", None)
        if skill is None:
            logger.warning(
                "plugin %s does not expose a top-level 'SKILL' attribute; skipping",
                file_path,
            )
            return

        try:
            self.register(skill)
        except SkillRegistrationError as exc:
            logger.error("plugin %s rejected during registration: %s", file_path, exc)
        except Exception:  # pragma: no cover - defensive
            logger.exception("plugin %s raised an unexpected error during registration", file_path)

    @staticmethod
    def _unique_module_name(file_path: Path) -> str:
        """Build a deterministic, namespaced module name for a plugin file."""
        # Hashing the absolute path keeps two plugins with the same
        # filename in different directories from clobbering each other in
        # ``sys.modules`` while still being readable in stack traces.
        digest = uuid.uuid5(uuid.NAMESPACE_URL, str(file_path.resolve())).hex[:8]
        return f"_jarvis_plugin_{file_path.stem}_{digest}"

    # -- Mistral tool definitions --------------------------------------

    def mistral_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Mistral function-calling tool definitions for all skills.

        The returned list mirrors the order produced by sorting
        ``names`` so callers can rely on a deterministic ordering when
        feeding the LLM (helpful for snapshot tests). Each entry is the
        exact dict shape Mistral's ``chat.complete`` / ``chat.stream``
        expects::

            {
                "type": "function",
                "function": {
                    "name": <skill name>,
                    "description": <description>,
                    "parameters": <JSON Schema dict>,
                },
            }

        The dict round-trips through ``json.dumps``/``json.loads``
        without information loss (Property 12 / CP15) because
        :meth:`MistralSchemaValidator.to_mistral_tool` already runs that
        round-trip internally.
        """
        return [
            self._mistral_validator.to_mistral_tool(self._skills[name].manifest)
            for name in self.names
        ]

    # -- Dispatch -------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        """Validate ``args`` and dispatch the named Skill.

        Contract (mirrors Property 2 / CP2 and Property 7 / CP10):

        * If no Skill is registered under ``name``, return
          ``internal_error`` without ever touching an executor.
        * If ``Draft7Validator(S.json_schema).is_valid(args)`` is false,
          return ``schema_violation`` and DO NOT invoke ``S.execute``.
        * Otherwise invoke ``S.execute(args, ctx)`` exactly once.
        * If the executor raises :class:`PolicyViolation`, record a
          ``policy_violation`` audit entry and return ``access_denied``.
        * If the executor raises any other exception, log the traceback
          and return ``internal_error`` with a short correlation id.
        * On a clean executor result, return it as-is, but back-fill
          ``duration_ms`` if the executor reported ``0``.
        """
        started = self._monotonic()

        skill = self._skills.get(name)
        if skill is None or name not in self._validators:
            # ``internal_error`` is the closest fit in the closed error
            # taxonomy: an "unknown skill" cannot be a schema violation
            # (no schema to violate) and is not a sandbox / platform
            # restriction. The Dialog_Manager surfaces this as a generic
            # "I don't know how to do that" message.
            return SkillResult.error(
                "internal_error",
                f"no skill registered under name {name!r}",
                duration_ms=self._elapsed_ms(started),
            )

        # ---- 1. Argument schema validation ---------------------------
        validator = self._validators[name]
        # ``iter_errors`` returns a generator we exhaust into a list so we
        # can both decide whether validation passed AND surface a
        # bounded set of details to the LLM for its retry attempt.
        errors = list(validator.iter_errors(args))
        if errors:
            details = [
                {
                    "path": list(err.absolute_path),
                    "message": err.message,
                }
                for err in errors[:5]
            ]
            summary = "; ".join(err.message for err in errors[:3])
            return SkillResult.error(
                "schema_violation",
                summary or "argument schema validation failed",
                value={"errors": details},
                duration_ms=self._elapsed_ms(started),
            )

        # ---- 2. Executor invocation ----------------------------------
        try:
            awaitable = skill.execute(args, ctx)
            # ``Skill.execute`` is typed as returning an Awaitable, but we
            # accept either an ``async def`` (the common case) or a
            # synchronous function that returns a coroutine — both are
            # awaitable, and ``inspect.isawaitable`` lets us reject a
            # plain ``SkillResult`` returned by mistake instead of
            # silently calling ``__await__`` on it.
            if not inspect.isawaitable(awaitable):
                raise TypeError(
                    f"skill {name!r}.execute returned a non-awaitable "
                    f"({type(awaitable).__name__}); expected coroutine"
                )
            result = await awaitable
        except PolicyViolation as exc:
            await self._record_policy_violation(name, args, exc, ctx)
            return SkillResult.error(
                exc.error_code,
                str(exc) or "operation blocked by policy",
                duration_ms=self._elapsed_ms(started),
            )
        except asyncio.CancelledError:
            # Never swallow cancellation: the dialog loop relies on it
            # to unwind cleanly during shutdown / barge-in.
            raise
        except Exception as exc:
            traceback_id = _make_traceback_id()
            logger.exception(
                "skill %r raised %s (traceback_id=%s)",
                name,
                type(exc).__name__,
                traceback_id,
            )
            tb_text = traceback.format_exc()
            return SkillResult.error(
                "internal_error",
                f"{type(exc).__name__}: {exc} (traceback_id={traceback_id})",
                value={
                    "traceback_id": traceback_id,
                    "exception_type": type(exc).__name__,
                    "traceback": tb_text,
                },
                duration_ms=self._elapsed_ms(started),
            )

        # ---- 3. Result shape sanity check ----------------------------
        if not isinstance(result, SkillResult):
            traceback_id = _make_traceback_id()  # type: ignore[unreachable]
            logger.error(
                "skill %r returned a non-SkillResult value: %r (traceback_id=%s)",
                name,
                type(result).__name__,
                traceback_id,
            )
            return SkillResult.error(
                "internal_error",
                f"skill returned {type(result).__name__}, expected SkillResult "
                f"(traceback_id={traceback_id})",
                value={"traceback_id": traceback_id},
                duration_ms=self._elapsed_ms(started),
            )

        # Back-fill duration when the executor reported zero so the
        # Dialog_Manager always sees a real wall-clock figure. We
        # rebuild the SkillResult because it is frozen.
        if result.duration_ms == 0:
            result = SkillResult(
                ok=result.ok,
                value=result.value,
                error_code=result.error_code,
                error_message=result.error_message,
                duration_ms=self._elapsed_ms(started),
            )
        return result

    # ------------------------------------------------------------------ helpers

    async def _record_policy_violation(
        self,
        skill_name: str,
        args: dict[str, Any],
        exc: PolicyViolation,
        ctx: SkillContext,
    ) -> None:
        """Best-effort audit log write for a policy violation."""
        audit_log = ctx.audit_log
        if audit_log is None:
            logger.warning(
                "policy violation in skill %r but no audit_log on context: %s",
                skill_name,
                exc,
            )
            return
        try:
            await audit_log.record_policy_violation(
                skill=skill_name,
                justification=exc.justification,
                args_json=_serialize_args(args),
                outcome=exc.error_code,
                run_id=ctx.run_id,
            )
        except Exception:
            # An audit-log failure must not mask the original policy
            # violation; we still return ``access_denied`` to the dialog
            # layer. Log so operators notice the audit subsystem is sick.
            logger.exception(
                "failed to record policy_violation audit entry for skill %r",
                skill_name,
            )

    def _elapsed_ms(self, started: float) -> int:
        """Compute milliseconds elapsed since ``started``, clamped at zero."""
        elapsed = self._monotonic() - started
        if elapsed < 0:  # pragma: no cover - defensive against clock skew
            return 0
        return int(elapsed * 1000)
