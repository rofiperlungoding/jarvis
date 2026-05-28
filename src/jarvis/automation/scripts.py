"""Script catalog and runner for the Automation_Service.

This module implements :class:`ScriptCatalog`, the lookup-only runner that
sits between :class:`~jarvis.skills.builtin.run_script.RunScriptSkill` and
the platform-specific :meth:`PlatformAdapter.run_script` implementation.

Why "lookup-only"?
------------------

Requirement 9.5 forbids the assistant from executing arbitrary script text
supplied directly in a Tool_Call argument. The Skill therefore receives
only a ``script_id`` string; the actual interpreter and on-disk path are
resolved here against the user's ``[automation.script_catalog]`` config
section. If the id does not resolve, the runner raises :class:`KeyError`
and the Skill maps that to ``SkillResult.error("script_not_found", ...)``
(Requirement 9.4).

What the runner does *not* do
-----------------------------

* It does NOT validate that ``path`` exists on disk. The platform adapter
  is responsible for surfacing missing-file errors via its own exception
  taxonomy because adapter implementations may resolve paths differently
  (e.g., ``WindowsAdapter`` will resolve ``%USERPROFILE%`` segments).
* It does NOT classify the script as Destructive_Action — that is the
  job of the Authorization_Service via the ``destructive_skills`` config
  list (which includes ``"RunScriptSkill"`` by default).
* It does NOT enforce the 60 s timeout twice; it forwards the configured
  timeout to the adapter, which is responsible for actually killing the
  process and returning ``ScriptResult.timed_out=True`` (Requirement 9.8).

Validates: Requirements 9.1, 9.3, 9.4, 9.5, 9.8
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Final

from jarvis.automation.platform import (
    PlatformAdapter,
    ScriptInterpreter,
    ScriptResult,
)
from jarvis.config.schema import ScriptCatalogEntry

__all__ = [
    "DEFAULT_SCRIPT_TIMEOUT_SECONDS",
    "ScriptCatalog",
]


#: Maximum wall-clock seconds a registered script may run before the
#: platform adapter terminates it. Mirrors the ceiling in Requirement 9.8
#: ("WHEN a script execution exceeds 60 seconds, THE Automation_Service
#: SHALL terminate the process and SHALL return a 'timeout' error"). The
#: value is exposed as a module-level constant so the Skill layer and
#: tests can refer to it without re-encoding the literal.
DEFAULT_SCRIPT_TIMEOUT_SECONDS: Final[float] = 60.0


class ScriptCatalog:
    """Resolve ``script_id`` strings to registered scripts and run them.

    The catalog is constructed from the ``[automation.script_catalog]``
    section of the user's TOML config (already parsed into
    ``dict[str, ScriptCatalogEntry]`` by the config loader). At runtime,
    :class:`~jarvis.skills.builtin.run_script.RunScriptSkill` calls
    :meth:`run` with the user-supplied ``script_id`` and the configured
    timeout; this class looks the id up, then forwards to the
    :class:`PlatformAdapter` which actually spawns the interpreter.

    Parameters
    ----------
    entries:
        Mapping of ``script_id`` to its :class:`ScriptCatalogEntry`. The
        mapping is snapshot-copied into an internal ``dict`` so later
        mutations to the caller's dict do not retroactively change the
        catalog the runner sees — Skills that re-fetch the entry between
        the confirmation prompt and the dispatch should observe a stable
        view (Requirement 16.3).
    platform_adapter:
        The platform abstraction the runner forwards to. Typically a
        :class:`~jarvis.automation.windows_adapter.WindowsAdapter` in
        production and a fake adapter in tests. Required to satisfy the
        :class:`PlatformAdapter` Protocol so test stubs that only
        override ``run_script`` are accepted.

    Notes
    -----
    The catalog deliberately exposes a tiny surface (``list_ids``,
    ``get``, ``run``). It is *not* a registry; entries are immutable
    once the runner is built. Reload semantics are the loader's
    responsibility — at the time of writing the loader builds a fresh
    :class:`ScriptCatalog` whenever the user reloads the config.
    """

    def __init__(
        self,
        entries: Mapping[str, ScriptCatalogEntry],
        platform_adapter: PlatformAdapter,
    ) -> None:
        if not isinstance(entries, Mapping):
            raise TypeError(
                "ScriptCatalog.entries must be a Mapping[str, "
                "ScriptCatalogEntry]"
            )
        # Validate every entry up-front rather than lazily at run() time.
        # A malformed catalog is a configuration error and we want it to
        # surface during application startup, not on the first
        # ``run_script`` call (which would otherwise be observable to the
        # user as a confusing TypeError after a confirmation prompt).
        snapshot: dict[str, ScriptCatalogEntry] = {}
        for script_id, entry in entries.items():
            if not isinstance(script_id, str) or not script_id:
                raise ValueError(
                    "ScriptCatalog entry id must be a non-empty string; "
                    f"got {script_id!r}"
                )
            if not isinstance(entry, ScriptCatalogEntry):
                raise TypeError(
                    f"ScriptCatalog entry for {script_id!r} must be a "
                    f"ScriptCatalogEntry; got {type(entry).__name__}"
                )
            snapshot[script_id] = entry

        # ``runtime_checkable`` Protocol membership check. Done after the
        # entries validation so the most common configuration mistake
        # (a malformed entry) surfaces before the adapter-shape error.
        if not isinstance(platform_adapter, PlatformAdapter):
            raise TypeError(
                "ScriptCatalog.platform_adapter must implement the "
                "PlatformAdapter Protocol (missing run_script?)"
            )

        self._entries: dict[str, ScriptCatalogEntry] = snapshot
        self._platform_adapter: PlatformAdapter = platform_adapter

    # ------------------------------------------------------------------
    # Read-only catalog access
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Return the registered ``script_id`` values in insertion order.

        The returned list is a fresh copy; mutating it does not affect
        the catalog. Callers that want to surface the available scripts
        in a help message can rely on the order matching what the user
        wrote in their TOML file.
        """
        return list(self._entries.keys())

    def get(self, script_id: str) -> ScriptCatalogEntry | None:
        """Look up an entry by id, returning ``None`` if absent.

        This is the read-only sibling of :meth:`run`: the Skill layer
        uses :meth:`get` to compose the confirmation prompt
        (interpreter + description) without committing to execution,
        then calls :meth:`run` once the user has confirmed.
        """
        if not isinstance(script_id, str):
            # Defensive: the Skill layer validates the JSON Schema, but
            # callers internal to the project sometimes pass through
            # raw values. A clear TypeError beats a silent ``None``.
            raise TypeError(
                f"ScriptCatalog.get script_id must be a str; "
                f"got {type(script_id).__name__}"
            )
        return self._entries.get(script_id)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        script_id: str,
        timeout_seconds: float = DEFAULT_SCRIPT_TIMEOUT_SECONDS,
    ) -> ScriptResult:
        """Resolve ``script_id`` and execute the registered script.

        Parameters
        ----------
        script_id:
            The catalog key. MUST be a non-empty string registered via
            ``[automation.script_catalog]``. The Skill is responsible
            for validating the JSON Schema upstream; this method
            re-validates because the runner is also called directly
            from integration tests and from any future programmatic
            entry point that bypasses the registry.
        timeout_seconds:
            Wall-clock budget forwarded to
            :meth:`PlatformAdapter.run_script`. Defaults to
            :data:`DEFAULT_SCRIPT_TIMEOUT_SECONDS` (60 s) per
            Requirement 9.8. Negative or zero values are rejected
            because the adapter would otherwise have undefined
            behaviour ("kill immediately" is not a useful semantic).

        Returns
        -------
        :class:`ScriptResult`
            Captured stdout/stderr, exit code, measured duration, and a
            ``timed_out`` flag. The Skill maps ``timed_out=True`` to
            ``SkillResult.error("timeout", ...)``.

        Raises
        ------
        KeyError
            If ``script_id`` is not registered. The Skill maps this to
            ``SkillResult.error("script_not_found", ...)`` per
            Requirement 9.4.
        TypeError / ValueError
            If the inputs are malformed (non-string id, non-finite or
            non-positive timeout). These indicate programmer error and
            propagate to the registry's exception barrier, which
            classifies them as ``internal_error``.
        """
        if not isinstance(script_id, str):
            raise TypeError(
                f"ScriptCatalog.run script_id must be a str; "
                f"got {type(script_id).__name__}"
            )
        if not script_id:
            raise ValueError("ScriptCatalog.run script_id must be non-empty")
        # Reject ``bool`` explicitly: ``isinstance(True, int)`` is True in
        # Python, and ``True`` would otherwise sneak through the numeric
        # check below as ``timeout_seconds=1.0``.
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise TypeError(
                "ScriptCatalog.run timeout_seconds must be a real number; "
                f"got {type(timeout_seconds).__name__}"
            )
        timeout_value = float(timeout_seconds)
        # ``nan`` and ``-inf`` would survive a naive ``> 0`` test on some
        # platforms; explicit non-finite check keeps the error message
        # actionable.
        if not math.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError(
                "ScriptCatalog.run timeout_seconds must be a positive, "
                f"finite number; got {timeout_seconds!r}"
            )

        entry = self._entries.get(script_id)
        if entry is None:
            # Surface the missing id verbatim so the Skill can include it
            # in the user-facing error message ("I don't know a script
            # called 'backup_photos', sir.") without re-encoding the
            # literal here.
            raise KeyError(script_id)

        # ``ScriptCatalogEntry.interpreter`` is already constrained to
        # ``Literal["powershell", "python", "batch"]`` by the pydantic
        # schema, so this assignment is a static narrowing — no runtime
        # check needed beyond what the schema already enforces.
        interpreter: ScriptInterpreter = entry.interpreter
        script_path = Path(entry.path)

        return await self._platform_adapter.run_script(
            interpreter,
            script_path,
            timeout_value,
        )
