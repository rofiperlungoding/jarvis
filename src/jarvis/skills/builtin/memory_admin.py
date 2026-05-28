"""Built-in ``MemoryAdminSkill``.

Implements Requirement 10.5 / 10.6 (the memory administration tool the
user invokes to inspect or forget records the Memory_Store has accrued)
and ties into Requirement 16.1 / 16.2 by classifying ``forget`` as a
Destructive_Action so the :class:`AuthorizationPolicy` requires user
confirmation before the registry dispatches it.

Three operations
----------------

The Skill multiplexes three distinct administration operations behind a
single ``operation`` discriminator so the LLM only has to discover one
tool. The shape mirrors :class:`~jarvis.skills.builtin.calendar.CalendarSkill`
so users (and prompt designers) can rely on a consistent convention:

* ``list`` (Requirement 10.5) — return every persisted record. Accepts
  an optional ``category`` filter (``chat`` / ``preference`` / ``fact``
  / ``summary``) so the LLM can keep the response focused. The
  underlying :meth:`MemoryStore.list_records` decrypts the content
  before returning, so the Skill can echo plaintext back to the user.
* ``search`` (Requirement 10.5) — semantic search across the store via
  :meth:`MemoryStore.retrieve`. Accepts a required ``query`` string and
  an optional integer ``k`` (defaulting to 5) capped at the
  configurable ``MEMORY_ADMIN_K_CAP`` (10) so a runaway LLM cannot
  pull the entire store in one go.
* ``forget`` (Requirements 10.6, 13.5) — remove the record whose
  ``record_id`` matches. Registered as a destructive operation in
  :data:`~jarvis.security.authorization.DEFAULT_DESTRUCTIVE_SKILLS` as
  ``"MemoryAdminSkill.forget"``, so the :class:`AuthorizationPolicy`
  reads the ``operation`` discriminator out of the Tool_Call arguments
  and gates the call on a user confirmation. The Skill itself does not
  re-prompt; by the time the registry dispatches us the user has
  already said "yes".

The manifest declares ``destructive=False`` because two of the three
operations are read-only. Operation-level destructive classification is
the right tool here: the policy's per-operation gate fires only on
``forget`` (and only ``forget``).

Context contract
----------------

The Skill expects ``ctx.extras["memory_store"]`` to hold the live
:class:`~jarvis.memory.store.MemoryStore`. The application bootstrap
(``src/jarvis/app.py``, task 19.1) injects the store there during
startup. If the entry is missing — for example, in unit tests or when
the store crashed at boot — the executor returns ``internal_error``:
a missing dependency is a wiring bug, not a platform limitation. The
extras-based injection follows the convention established by
:class:`~jarvis.skills.builtin.reminder.ReminderSkill`
(``"reminder_service"``); the typed :class:`SkillContext` does not (yet)
declare a ``memory_store`` field, and forward-compat refactors only
need to switch the lookup path in this module.

Privacy considerations
----------------------

``list`` and ``search`` return the *plaintext* content of every matched
record. That is the point of the Skill — users need to know what JARVIS
has memorised about them — but it means the result payload becomes part
of the LLM tool-result message and, transitively, of the assistant's
spoken response. The :class:`PIIRedactor` running at the
:meth:`MemoryStore.persist_turn` boundary already replaced
configured PII patterns with ``[REDACTED:<kind>]`` tokens before
encryption (Requirement 10.8), so what we read back is the redacted
representation. The Skill does not perform additional redaction on
read.

``forget`` returns ``{"forgotten": True}`` on success and
``{"forgotten": False}`` when the requested ``record_id`` was not
present in the collection. Returning a successful :class:`SkillResult`
in both cases (rather than ``internal_error`` for "missing record")
keeps the closed error taxonomy honest — there is no "not found" code
in the documented set, and the LLM's natural response to "forget the
record about my home address" should be an acknowledgement either way.

Validates: Requirements 10.5, 10.6, 13.5, 16.1, 16.2
"""

from __future__ import annotations

import logging
from typing import Any, Final

from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


__all__ = [
    "MEMORY_ADMIN_K_CAP",
    "MEMORY_ADMIN_K_DEFAULT",
    "MEMORY_ADMIN_LIST_CAP",
    "MEMORY_STORE_EXTRAS_KEY",
    "SCHEMA",
    "SKILL",
    "MemoryAdminSkill",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Key under which the application bootstrap installs the
#: :class:`~jarvis.memory.store.MemoryStore` instance into
#: :attr:`SkillContext.extras`. Pinned as a module-level constant so the
#: eventual ``app.py`` wiring (Task 19.1) and the unit tests share a
#: single source of truth.
MEMORY_STORE_EXTRAS_KEY: Final[str] = "memory_store"

#: Skill name surfaced to the LLM. Pinned because the
#: ``[authorization].destructive_skills`` config refers to the Skill by
#: ``"MemoryAdminSkill.forget"``; renaming would silently disable the
#: confirmation gate on the destructive operation.
_SKILL_NAME: Final[str] = "MemoryAdminSkill"

#: Closed set of operations the Skill exposes. Kept as a tuple so the
#: ``enum`` keyword in the JSON Schema and the runtime branching stay
#: in sync. Order is intentional: read-mostly ops come first to nudge
#: the LLM toward non-destructive responses.
_OPERATIONS: Final[tuple[str, ...]] = ("list", "search", "forget")

#: Closed set of memory categories the ``list`` operation can filter
#: on. Mirrors the literal in :class:`~jarvis.memory.store.MemoryRecord`.
#: Kept here (rather than imported) so this module's import graph stays
#: free of the heavyweight :mod:`jarvis.memory.store` dependency — that
#: module pulls in :mod:`chromadb`, which costs multiple seconds at
#: import time.
_CATEGORIES: Final[tuple[str, ...]] = ("chat", "preference", "fact", "summary")

#: Default ``k`` for ``search`` when the LLM does not supply one. Mirrors
#: the documented :meth:`MemoryStore.retrieve` default.
MEMORY_ADMIN_K_DEFAULT: Final[int] = 5

#: Hard cap on ``k`` for ``search``. The Skill clamps any larger value
#: at this ceiling so a runaway LLM cannot drag the entire store into a
#: single tool-result message. Matches the cap NewsSkill uses for
#: ``max_items`` (Requirement 7.4) for cross-skill consistency.
MEMORY_ADMIN_K_CAP: Final[int] = 10

#: Hard cap on the number of records the ``list`` operation will
#: include in its response. The store has no native limit; the Skill
#: trims the result client-side so the tool-result message stays
#: bounded. The LLM can re-issue ``list`` with a ``category`` filter to
#: page through more.
MEMORY_ADMIN_LIST_CAP: Final[int] = 50


_SKILL_DESCRIPTION: Final[str] = (
    "Inspect or remove records from the user's long-term memory. "
    "Operations: 'list' returns all stored records (optionally "
    "filtered by category), 'search' finds records semantically "
    "similar to a query, and 'forget' deletes a single record "
    "by id. Forgetting a record requires explicit user confirmation."
)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def _build_schema() -> dict[str, Any]:
    """Build the Mistral-compatible argument schema for the Skill.

    Uses ``allOf`` + ``if/then`` blocks so the LLM sees one tool with
    a closed ``operation`` enum while the per-operation requirements
    (``query`` for ``search``; ``record_id`` for ``forget``) are still
    enforced by :class:`Draft7Validator`. The same pattern is used by
    :class:`~jarvis.skills.builtin.calendar.CalendarSkill`; the Mistral
    subset validator accepts ``allOf`` / ``if`` / ``then`` / ``const``
    / ``enum`` (no ``$ref``, no scalar/object ``oneOf`` mixing) so the
    registry happily registers the Skill.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": (
                    "Memory administration operation to perform: "
                    "'list' returns every stored record (optionally "
                    "filtered by category), 'search' returns the top-k "
                    "records semantically similar to 'query', and "
                    "'forget' removes a single record by 'record_id' "
                    "(requires user confirmation)."
                ),
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": ("Search query (required when operation == 'search')."),
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MEMORY_ADMIN_K_CAP,
                "description": (
                    "Maximum number of records to return for 'search'. "
                    f"Defaults to {MEMORY_ADMIN_K_DEFAULT}; capped at "
                    f"{MEMORY_ADMIN_K_CAP}."
                ),
            },
            "record_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Identifier of the record to remove (required when "
                    "operation == 'forget')."
                ),
            },
            "category": {
                "type": "string",
                "enum": list(_CATEGORIES),
                "description": (
                    "Optional category filter for 'list'. Accepts: "
                    + ", ".join(_CATEGORIES)
                    + "."
                ),
            },
        },
        "required": ["operation"],
        "allOf": [
            {
                "if": {
                    "properties": {"operation": {"const": "search"}},
                    "required": ["operation"],
                },
                "then": {
                    "required": ["query"],
                },
            },
            {
                "if": {
                    "properties": {"operation": {"const": "forget"}},
                    "required": ["operation"],
                },
                "then": {
                    "required": ["record_id"],
                },
            },
        ],
    }


SCHEMA: Final[dict[str, Any]] = _build_schema()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_store(ctx: SkillContext) -> Any | None:
    """Return the :class:`MemoryStore` injected via ``ctx.extras``.

    Returns ``None`` when the key is missing or maps to a value that
    does not look like a :class:`MemoryStore` (i.e., a stale fake or
    placeholder). The duck-typed shape check — presence of the three
    coroutines we use — keeps the Skill testable with lightweight
    fakes that mimic only the surface area we touch.
    """

    candidate = ctx.extras.get(MEMORY_STORE_EXTRAS_KEY)
    if candidate is None:
        return None
    if not (
        hasattr(candidate, "list_records")
        and hasattr(candidate, "retrieve")
        and hasattr(candidate, "forget")
    ):
        return None
    return candidate


def _serialize_record(record: Any) -> dict[str, Any]:
    """Render a :class:`MemoryRecord` as a JSON-friendly dict.

    The Skill returns records inside :attr:`SkillResult.value`, which
    the dispatcher subsequently embeds in a tool-result message for the
    LLM. JSON serialisation must therefore be lossless and
    timezone-explicit — :meth:`datetime.isoformat` delivers the latter.

    Embeddings are intentionally omitted: they are large floating-point
    arrays that bloat the LLM context window and carry no information
    the model can usefully act on. ``MemoryAdminSkill`` is for human-
    facing administration, not vector debugging.
    """

    timestamp = getattr(record, "timestamp", None)
    timestamp_iso: str | None
    if timestamp is None:
        timestamp_iso = None
    else:
        try:
            timestamp_iso = timestamp.isoformat()
        except (AttributeError, TypeError):
            timestamp_iso = str(timestamp)

    provenance_raw = getattr(record, "provenance", None) or {}
    provenance = dict(provenance_raw) if isinstance(provenance_raw, dict) else {}

    return {
        "record_id": getattr(record, "record_id", ""),
        "category": getattr(record, "category", "chat"),
        "content": getattr(record, "content", ""),
        "timestamp": timestamp_iso,
        "redacted": bool(getattr(record, "redacted", False)),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# MemoryAdminSkill
# ---------------------------------------------------------------------------


class MemoryAdminSkill:
    """Skill that proxies to the configured :class:`MemoryStore`.

    Stateless: a single instance is reused across invocations, with
    each ``execute`` call receiving the per-call :class:`SkillContext`
    produced by the :class:`SkillRegistry`. The store lookup is
    deferred to :meth:`execute` rather than resolved at construction
    so the same instance can be registered before the store is fully
    wired (the discovery path in :meth:`SkillRegistry.discover` runs
    at startup, before the run-loop in :func:`jarvis.app.main`
    populates every :class:`SkillContext`).
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=SCHEMA,
        # Manifest-level destructive=False because two of three
        # operations are read-only. The :class:`AuthorizationPolicy`
        # reads ``[authorization].destructive_skills`` and the default
        # :data:`DEFAULT_DESTRUCTIVE_SKILLS` (which lists
        # ``"MemoryAdminSkill.forget"``) for the per-operation gate.
        destructive=False,
        timeout_seconds=30.0,
        # MemoryStore is OS-agnostic; declare every supported platform
        # so Requirement 15.4's gating does not block the Skill on
        # macOS / Linux builds.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Run the requested memory-admin operation.

        ``args`` has already been validated against
        :attr:`manifest.json_schema` by the
        :class:`~jarvis.skills.registry.SkillRegistry`, so structural
        checks here only need to defend against semantic errors the
        schema cannot fully express (e.g., a misconfigured
        :class:`SkillContext` without a memory store) and against
        smuggled-in operation values.
        """

        operation = args.get("operation")
        if operation not in _OPERATIONS:
            # Defence-in-depth: the JSON Schema's ``enum`` enforces
            # this, but the registry's validator runs *before* us. A
            # smuggled-in op should never reach the dispatch branches.
            return SkillResult.error(
                "schema_violation",
                f"operation must be one of {list(_OPERATIONS)!r}",
            )

        store = _resolve_store(ctx)
        if store is None:
            # Surfaced as ``internal_error`` (rather than
            # ``not_supported``) because a missing memory store is a
            # wiring bug, not a platform limitation. The Dialog_Manager
            # apologises rather than steering the user toward an
            # irrelevant troubleshooting path.
            logger.error(
                "MemoryAdminSkill invoked without a MemoryStore on "
                "ctx.extras[%r]; check application bootstrap",
                MEMORY_STORE_EXTRAS_KEY,
            )
            return SkillResult.error(
                "internal_error",
                "memory store is unavailable",
            )

        if operation == "list":
            return await self._list(store, args)
        if operation == "search":
            return await self._search(store, args)
        # ``forget`` is the only remaining branch; the
        # :class:`AuthorizationPolicy` has already obtained the user's
        # confirmation before the registry dispatched us here
        # (Requirement 16.2). The Skill does not re-prompt.
        return await self._forget(store, args)

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    @staticmethod
    async def _list(store: Any, args: dict[str, Any]) -> SkillResult:
        category = args.get("category")
        if category is not None and category not in _CATEGORIES:
            # Defence-in-depth: matches the JSON Schema enum.
            return SkillResult.error(
                "schema_violation",
                f"category must be one of {list(_CATEGORIES)!r}",
            )

        try:
            records = await store.list_records(category=category)
        except (ValueError, TypeError) as exc:
            return SkillResult.error("schema_violation", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MemoryStore.list_records raised unexpectedly")
            return SkillResult.error(
                "internal_error",
                f"failed to list memory records: {type(exc).__name__}: {exc}",
            )

        # Sort newest-first so the LLM sees the most recent records
        # before the cap kicks in. ``timestamp`` is already a tz-aware
        # datetime; sort handles ``None`` gracefully via the key
        # function's defensive default.
        sorted_records = sorted(
            records,
            key=lambda r: getattr(r, "timestamp", None) or 0,
            reverse=True,
        )
        truncated = sorted_records[:MEMORY_ADMIN_LIST_CAP]
        return SkillResult.success(
            {
                "operation": "list",
                "category": category,
                "total": len(sorted_records),
                "returned": len(truncated),
                "records": [_serialize_record(r) for r in truncated],
            }
        )

    @staticmethod
    async def _search(store: Any, args: dict[str, Any]) -> SkillResult:
        raw_query = args.get("query")
        if not isinstance(raw_query, str) or not raw_query.strip():
            return SkillResult.error(
                "schema_violation",
                "query must be a non-empty string",
            )
        query = raw_query

        raw_k = args.get("k")
        if raw_k is None:
            k = MEMORY_ADMIN_K_DEFAULT
        elif isinstance(raw_k, bool) or not isinstance(raw_k, int):
            # ``bool`` is a subclass of ``int``; reject explicitly so
            # ``True``/``False`` do not silently become ``1``/``0``.
            return SkillResult.error(
                "schema_violation",
                "k must be an integer",
            )
        elif raw_k < 1:
            return SkillResult.error(
                "schema_violation",
                "k must be greater than or equal to 1",
            )
        else:
            # Clamp at the ceiling rather than rejecting: the JSON
            # Schema already caps ``k`` at MEMORY_ADMIN_K_CAP, but
            # callers (tests, future plugins) constructing the Skill
            # context by hand may bypass the registry's validator.
            k = min(raw_k, MEMORY_ADMIN_K_CAP)

        try:
            records = await store.retrieve(query, k=k)
        except (ValueError, TypeError) as exc:
            return SkillResult.error("schema_violation", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MemoryStore.retrieve raised unexpectedly")
            return SkillResult.error(
                "internal_error",
                f"failed to search memory: {type(exc).__name__}: {exc}",
            )

        return SkillResult.success(
            {
                "operation": "search",
                "query": query,
                "k": k,
                "returned": len(records),
                "records": [_serialize_record(r) for r in records],
            }
        )

    @staticmethod
    async def _forget(store: Any, args: dict[str, Any]) -> SkillResult:
        raw_record_id = args.get("record_id")
        if not isinstance(raw_record_id, str) or not raw_record_id.strip():
            return SkillResult.error(
                "schema_violation",
                "record_id must be a non-empty string",
            )
        record_id = raw_record_id

        try:
            removed = await store.forget(record_id)
        except (ValueError, TypeError) as exc:
            return SkillResult.error("schema_violation", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MemoryStore.forget raised unexpectedly")
            return SkillResult.error(
                "internal_error",
                f"failed to forget record: {type(exc).__name__}: {exc}",
            )

        # Both ``True`` (removed) and ``False`` (no such id) are
        # successful outcomes from the Skill's perspective: the
        # closed error taxonomy has no "not_found" code, and the
        # natural user response in either case is an
        # acknowledgement. The boolean ``forgotten`` field lets the
        # LLM phrase the response correctly.
        return SkillResult.success(
            {
                "operation": "forget",
                "record_id": record_id,
                "forgotten": bool(removed),
            }
        )


# ---------------------------------------------------------------------------
# Plugin handle
# ---------------------------------------------------------------------------


#: Module-level singleton consumed by :meth:`SkillRegistry.discover`.
#: Typed as :class:`MemoryAdminSkill` rather than the :class:`Skill`
#: Protocol because the latter declares ``manifest`` as a writable
#: variable while we expose it as a :data:`Final` class attribute; the
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: MemoryAdminSkill = MemoryAdminSkill()
