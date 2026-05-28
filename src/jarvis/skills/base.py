"""Skill plugin base interfaces.

This module defines the data models and the :class:`Skill` Protocol that the
:class:`~jarvis.skills.registry.SkillRegistry` and the Dialog_Manager rely on
to discover, validate, and dispatch Skills (built-in, user, MCP). The shapes
mirror ``design.md §Skill_Registry`` and ``design.md §Data Models``; the
closed set of error codes mirrors the Error Taxonomy table in ``design.md``.

What lives here
---------------

* :class:`SkillManifest` — frozen dataclass declaring the skill's name,
  description, JSON Schema, destructive flag, timeout, supported platforms,
  and provenance (``"builtin" | "user" | "mcp"``).
* :class:`SkillResult` — frozen dataclass describing the structured result
  every Skill executor returns. ``error_code`` is constrained to the
  documented 11-value enum so all error handling at the Dialog_Manager
  boundary can switch on a closed set (see Property 7 / CP10 and the Error
  Taxonomy table).
* :class:`SkillContext` — frozen dataclass bundling the dependencies the
  Skill_Registry passes into ``execute``: audit log, time source, platform
  adapter, credential store, LLM backend, provider clients, allowed
  directories, incognito flag, run id, and an open-ended ``extras``
  mapping for MCP/test injection.
* :class:`Skill` — runtime-checkable Protocol with a ``manifest`` attribute
  and an ``async execute(args, ctx) -> SkillResult`` coroutine.

Design notes
------------

* All public types are frozen so the Authorization_Policy and audit log can
  rely on stable hashes / equality when matching Tool_Calls against the
  trusted-action allowlist (see CP9 ordering invariants).
* JSON Schema dictionaries are intentionally declared as ``Mapping`` so we
  can persist them as ``dict`` without forcing callers to copy. Equality
  semantics match Python's structural ``Mapping`` comparison.
* Forward references for not-yet-implemented modules (``PlatformAdapter``,
  ``CredentialStore``, ``LLMBackend``) use ``typing.Any`` to avoid circular
  imports while later tasks (10.2, 10.3, 11.x, 13.x) materialise the
  concrete classes.

Validates: Requirements 14.2, 14.3, 17.1
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from jarvis.security.audit_log import AuditLog
    from jarvis.utils.time_source import TimeSource


__all__ = [
    "ERROR_CODES",
    "Skill",
    "SkillContext",
    "SkillErrorCode",
    "SkillManifest",
    "SkillResult",
    "SkillSource",
]


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

# The closed set of error codes documented in ``design.md §Error Taxonomy``.
# A Skill executor MUST set ``SkillResult.error_code`` to one of these eleven
# string values when ``ok`` is ``False``; the Dialog_Manager dispatches user
# messaging based on this enum (e.g., ``schema_violation`` triggers up to
# two LLM retries per Requirement 14.5, ``missing_credentials`` triggers
# the credential-setup flow per Requirement 5.6).
SkillErrorCode = Literal[
    "schema_violation",
    "missing_credentials",
    "not_supported",
    "access_denied",
    "file_too_large",
    "script_not_found",
    "timeout",
    "provider_unavailable",
    "internal_error",
    "platform_not_supported",
    "rate_limited",
]

# Runtime-accessible mirror of :data:`SkillErrorCode` for validation. The
# tuple is ``Final`` so static type-checkers flag accidental mutation.
ERROR_CODES: Final[tuple[str, ...]] = (
    "schema_violation",
    "missing_credentials",
    "not_supported",
    "access_denied",
    "file_too_large",
    "script_not_found",
    "timeout",
    "provider_unavailable",
    "internal_error",
    "platform_not_supported",
    "rate_limited",
)

_ERROR_CODE_SET: Final[frozenset[str]] = frozenset(ERROR_CODES)


# Provenance of a Skill, also matching the design data model.
SkillSource = Literal["builtin", "user", "mcp"]
_SKILL_SOURCES: Final[frozenset[str]] = frozenset({"builtin", "user", "mcp"})


# ---------------------------------------------------------------------------
# SkillManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillManifest:
    """Static metadata describing a Skill.

    The manifest is what the :class:`SkillRegistry` consults to:

    * map the Skill into a Mistral function-calling tool definition
      (``mistral_tool_definitions()``);
    * decide whether a Tool_Call requires confirmation
      (:attr:`destructive`, see Requirement 16.1);
    * enforce a per-call wall-clock budget (:attr:`timeout_seconds`);
    * gate cross-platform availability (:attr:`platforms`, Requirement
      15.4).

    Attributes
    ----------
    name:
        Stable identifier shared with the LLM and used as the
        ``function.name`` in the Mistral tool payload. MUST be a non-empty
        string. The registry additionally constrains the character set;
        we only enforce non-emptiness here so the dataclass remains a thin
        value type.
    description:
        Human-readable, model-facing description. Surfaces to the LLM as
        the ``function.description`` and to the user when the
        Authorization_Policy reads back the action summary.
    json_schema:
        JSON Schema (Draft-07, Mistral-compatible subset) describing the
        ``arguments`` object. Stored as a plain :class:`dict` so it
        survives ``json.dumps`` / ``json.loads`` round-trips required by
        Property 12 / CP15.
    destructive:
        ``True`` if every invocation is a Destructive_Action per
        Requirement 16.1. Operation-level destructive classification
        (e.g., ``CalendarSkill.create_event``) is configured separately by
        the Authorization_Policy.
    timeout_seconds:
        Maximum wall-clock duration the registry allows ``execute`` to
        run before returning ``SkillResult.error("timeout", ...)``.
        Defaults to 30 s per ``design.md``.
    platforms:
        Tuple of platform tags on which the Skill's underlying capability
        is implemented. The registry returns ``platform_not_supported``
        on platforms outside this set (Requirement 15.4).
    source:
        Provenance tag distinguishing built-in, user-defined, and MCP
        Skills. Used by the registry for sandboxing and for surfacing
        sources in the audit log.
    """

    name: str
    description: str
    json_schema: Mapping[str, Any]
    destructive: bool = False
    timeout_seconds: float = 30.0
    platforms: tuple[str, ...] = ("windows",)
    source: SkillSource = "builtin"

    def __post_init__(self) -> None:
        # Cheap structural validation. Heavier checks (Mistral subset
        # rejection, meta-schema validation) live in the Skill_Registry so
        # the manifest stays usable in pure-data contexts (e.g., tests
        # serialising/deserialising fixtures).
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SkillManifest.name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("SkillManifest.description must be a string")
        if not isinstance(self.json_schema, Mapping):
            raise TypeError(
                "SkillManifest.json_schema must be a Mapping[str, Any] "
                f"(got {type(self.json_schema).__name__!r})"
            )
        if not isinstance(self.destructive, bool):
            raise TypeError("SkillManifest.destructive must be a bool")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(
            self.timeout_seconds, bool
        ):
            raise TypeError("SkillManifest.timeout_seconds must be a number")
        if self.timeout_seconds <= 0:
            raise ValueError(
                "SkillManifest.timeout_seconds must be strictly positive"
            )
        if not isinstance(self.platforms, tuple) or not self.platforms:
            raise ValueError("SkillManifest.platforms must be a non-empty tuple")
        if not all(isinstance(p, str) and p for p in self.platforms):
            raise ValueError(
                "SkillManifest.platforms entries must be non-empty strings"
            )
        if self.source not in _SKILL_SOURCES:
            raise ValueError(
                "SkillManifest.source must be one of "
                f"{sorted(_SKILL_SOURCES)!r}, got {self.source!r}"
            )


# ---------------------------------------------------------------------------
# SkillResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillResult:
    """Structured outcome of a single Skill invocation.

    Every code path that can return data to the Dialog_Manager — including
    the registry's own validators — converges on this type so the dialog
    loop never has to handle bare exceptions (Property 7 / CP10).

    Attributes
    ----------
    ok:
        ``True`` for a successful execution; ``False`` otherwise. The
        ``__post_init__`` guard enforces that ``error_code`` is set iff
        ``ok`` is ``False`` and that successful results carry no error
        message, so downstream consumers can branch on either field
        interchangeably.
    value:
        Optional JSON-serialisable payload produced by the executor. By
        convention, successful results put structured data here for the
        LLM to incorporate; failure results may also carry diagnostic
        details (e.g., the offending JSON Schema path on
        ``schema_violation``).
    error_code:
        One of the eleven values in :data:`SkillErrorCode`, or ``None`` on
        success. The registry maps unknown error codes to
        ``"internal_error"`` rather than letting them propagate, but at
        this layer we still validate to catch programmer errors early.
    error_message:
        Human-readable description of the failure. Forwarded to the LLM
        and (in redacted form) to the user. ``None`` on success.
    duration_ms:
        Wall-clock execution time of the executor as observed by the
        registry. ``0`` is permitted for synthetic results (e.g., the
        registry's own ``schema_violation`` path before dispatch).
    """

    ok: bool
    value: dict[str, Any] | None
    error_code: SkillErrorCode | None
    error_message: str | None
    duration_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("SkillResult.ok must be a bool")
        if self.value is not None and not isinstance(self.value, dict):
            raise TypeError("SkillResult.value must be a dict or None")
        if self.error_code is not None and self.error_code not in _ERROR_CODE_SET:
            raise ValueError(
                "SkillResult.error_code must be one of "
                f"{sorted(_ERROR_CODE_SET)!r}, got {self.error_code!r}"
            )
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise TypeError("SkillResult.error_message must be a string or None")
        if not isinstance(self.duration_ms, int) or isinstance(self.duration_ms, bool):
            raise TypeError("SkillResult.duration_ms must be an int")
        if self.duration_ms < 0:
            raise ValueError("SkillResult.duration_ms must be non-negative")
        # Cross-field invariants: success and failure are disjoint.
        if self.ok:
            if self.error_code is not None:
                raise ValueError("SkillResult.error_code must be None when ok=True")
            if self.error_message is not None:
                raise ValueError("SkillResult.error_message must be None when ok=True")
        elif self.error_code is None:
            raise ValueError("SkillResult.error_code is required when ok=False")

    # -- Convenience constructors --------------------------------------------

    @classmethod
    def success(
        cls,
        value: dict[str, Any] | None = None,
        *,
        duration_ms: int = 0,
    ) -> SkillResult:
        """Build a successful result with optional payload and timing."""
        return cls(
            ok=True,
            value=value,
            error_code=None,
            error_message=None,
            duration_ms=duration_ms,
        )

    @classmethod
    def error(
        cls,
        error_code: SkillErrorCode,
        error_message: str | None = None,
        *,
        value: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> SkillResult:
        """Build a failure result.

        Mirrors the ``SkillResult.error("schema_violation", details)`` and
        ``SkillResult.error("internal_error", traceback_id)`` call sites
        sketched in ``design.md``. ``value`` is available for structured
        diagnostics (for example, attaching the JSON Schema validation
        path that failed).
        """
        return cls(
            ok=False,
            value=value,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillContext:
    """Bundle of dependencies passed to ``Skill.execute``.

    The registry instantiates one :class:`SkillContext` per Tool_Call so
    each invocation gets the right run-scoped state (audit log, run id,
    incognito flag, allowed directories) without the Skill code having to
    reach for module-level globals.

    All fields are optional and default to ``None`` / empty so tests can
    construct minimal contexts that exercise a single dependency. Skills
    that require a particular dependency (e.g., :class:`SendEmailSkill`
    needs ``credential_store`` and ``providers["email"]``) MUST validate
    its presence themselves and return ``missing_credentials`` /
    ``provider_unavailable`` accordingly.

    Attributes
    ----------
    audit_log:
        Append-only audit log for ``policy_violation``, ``network_egress``
        and ``error`` records. Skills typically delegate audit writes to
        the registry / authorization layer, but provider clients invoke
        ``record_network_egress`` directly.
    time_source:
        Injectable clock; Skills MUST prefer this over ``datetime.now`` /
        ``time.monotonic`` so tests stay deterministic (Property 5 / CP6).
    platform_adapter:
        :class:`PlatformAdapter` for OS-level side effects (launch,
        media keys, brightness, notifications, scripted UI). Typed as
        :class:`Any` here to avoid an import cycle with the not-yet-built
        ``jarvis.automation.platform`` module.
    credential_store:
        :class:`CredentialStore` for reading provider secrets. Same
        forward-reference rationale as ``platform_adapter``.
    llm_backend:
        Active :class:`LLMBackend` (Mistral primary, Ollama fallback) for
        Skills that need model output (``SummarizeFileSkill``).
    providers:
        Mapping of provider name (``"weather"``, ``"news"``, ``"email"``,
        ``"calendar"``, ``"web_search"``) to its HTTP client / adapter.
        Frozen dataclass holds the reference; callers SHOULD pass an
        immutable mapping (e.g., :class:`types.MappingProxyType`) to
        prevent mutation through the context.
    allowed_directories:
        Tuple of paths that the file-reading Skills are permitted to
        access (Requirements 8.2, 8.6). Empty tuple denies all.
    incognito:
        ``True`` while the user is in incognito mode (Requirement 13.3).
        Skills that persist data MUST honour this flag.
    run_id:
        Stable identifier of the current process run, propagated to audit
        entries. ``None`` is allowed for tests that do not exercise the
        audit log.
    extras:
        Open-ended mapping for MCP-injected dependencies and
        test-injected fakes. Avoid relying on this for built-in Skills;
        prefer adding a typed field above.
    """

    audit_log: AuditLog | None = None
    time_source: TimeSource | None = None
    platform_adapter: Any = None
    credential_store: Any = None
    llm_backend: Any = None
    providers: Mapping[str, Any] = field(default_factory=dict)
    allowed_directories: tuple[Path, ...] = ()
    incognito: bool = False
    run_id: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Skill protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Skill(Protocol):
    """Plugin interface every Skill must satisfy.

    A Skill is a value object exposing a static :class:`SkillManifest`
    (``manifest`` attribute) and a single ``async`` ``execute`` coroutine.
    The registry calls ``execute`` only after the supplied ``args`` have
    been validated against ``manifest.json_schema``; implementations may
    therefore assume well-formed input but MUST still defend against
    semantic errors (missing credentials, paths outside the sandbox, etc.)
    by returning the appropriate :class:`SkillResult` error code.

    The :func:`runtime_checkable` decorator lets the registry assert
    ``isinstance(obj, Skill)`` during plugin discovery, but the check is
    structural — it validates the *presence* of ``manifest`` and
    ``execute`` only, not the exact types. The registry performs deeper
    validation (Mistral subset, meta-schema) on the manifest separately.

    ``manifest`` is declared as a ``@property`` so concrete Skills may
    use a class-level ``Final[SkillManifest]`` attribute (the recommended
    style) without tripping mypy's "settable variable expected" check
    that fires on Protocols with bare attribute declarations.
    """

    @property
    def manifest(self) -> SkillManifest: ...

    def execute(
        self, args: dict[str, Any], ctx: SkillContext
    ) -> Awaitable[SkillResult]:
        """Run the Skill against pre-validated ``args``.

        The signature uses ``Awaitable[SkillResult]`` rather than
        ``async def`` because :class:`Protocol` does not let us declare an
        ``async`` method directly without forcing implementers to use the
        identical ``async def`` form. Returning an awaitable is sufficient
        and lets implementations choose between ``async def`` (the common
        case) and a hand-rolled coroutine.
        """
        ...
