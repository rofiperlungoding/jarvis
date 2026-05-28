"""Built-in ``ReadFileSkill``.

Implements the file-reading half of Requirement 8 (the summarisation
half lives in :mod:`jarvis.skills.builtin.summarize_file`). The Skill
honours the path-sandbox contract documented as Property 9 / CP12 in
``design.md``:

    For any "path" argument P supplied to ReadFileSkill, if P
    (after canonicalisation, including resolution of symbolic links
    and "..") does not lie within the configured allowed-directory
    list, the Skill SHALL return ``access_denied`` and SHALL NOT
    open the file.

The test suite exercises the "no ``open()`` was called" half via a
monkey-patched ``builtins.open``; this module is therefore careful to
perform the sandbox check *before* any file-system call that might end
up issuing an ``open(2)`` (``os.path.getsize`` does not, ``pypdf`` and
``python-docx`` do).

Error mapping (closed ``SkillResult`` taxonomy)
-----------------------------------------------

* Path missing / not absolute / wrong type → ``schema_violation``. The
  registry's draft-07 validator already rejects these, but we belt-and-
  brace the runtime path so the Skill stays correct when invoked
  directly from tests.
* Path resolves outside ``ctx.allowed_directories`` → :class:`SandboxViolation`
  is raised so the registry records a single ``policy_violation`` audit
  entry (Requirement 13.6) and surfaces ``access_denied`` to the dialog
  layer (Requirement 8.6).
* Extension not in the supported set → ``not_supported`` (Requirement
  8.5 — only the documented formats are honoured).
* File larger than 5 MB → ``file_too_large`` (Requirement 8.7).
* File missing on disk → ``internal_error`` with a "file not found"
  message. The closed taxonomy has no dedicated "file_not_found"
  code, so this is the closest fit; the Dialog_Manager already reports
  ``internal_error`` clearly.
* PDF / DOCX parser raises → ``internal_error`` with a short
  correlation message.

Validates: Requirements 8.1, 8.2, 8.5, 8.6, 8.7, 13.6
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Final

import docx
from pypdf import PdfReader

from jarvis.skills.base import (
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.registry import SandboxViolation

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_FILE_SIZE_BYTES",
    "SCHEMA",
    "SKILL",
    "SUPPORTED_EXTENSIONS",
    "ReadFileSkill",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hard cap on file size the Skill will read (Requirement 8.7).
#: Anything strictly larger surfaces ``file_too_large``. Five megabytes
#: balances coverage of typical user documents against the time budget
#: of running the contents through the LLM downstream.
MAX_FILE_SIZE_BYTES: Final[int] = 5 * 1024 * 1024

#: Closed set of supported file extensions (Requirement 8.5). Stored
#: lower-cased and dot-prefixed because :func:`os.path.splitext` returns
#: the leading dot. The mapping is kept frozen so no caller can mutate
#: the global set at runtime.
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".py",
        ".js",
        ".ts",
        ".pdf",
        ".docx",
    }
)

#: Plain-text extensions read directly via :meth:`pathlib.Path.read_text`.
_TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts"}
)

# Encoding used for text extensions. UTF-8 is the documented baseline for
# every supported text format. Files with BOMs (e.g., ``utf-8-sig``) are
# tolerated by reading bytes first then decoding via ``utf-8`` with
# ``errors="replace"`` so a stray non-UTF-8 byte does not abort the read.
_TEXT_ENCODING: Final[str] = "utf-8"

#: Skill name surfaced to the LLM. Pinned because Requirement 8.1 anchors
#: the wording ("ReadFileSkill") and changing it would silently break
#: any configured trusted-action allowlist entries.
_SKILL_NAME: Final[str] = "ReadFileSkill"

_SKILL_DESCRIPTION: Final[str] = (
    "Read the contents of a file from one of the user's allowed "
    "directories. Supports text formats (.txt, .md, .csv, .json, "
    ".py, .js, .ts) and binary documents (.pdf, .docx). Files larger "
    "than 5 MB are refused with file_too_large."
)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "title": "ReadFile",
    "description": (
        "Read the contents of a file at an absolute path. The path "
        "MUST resolve inside one of the user's configured allowed "
        "directories."
    ),
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Absolute filesystem path to the file. Relative paths "
                "are rejected with a schema violation; the path is "
                "canonicalised (symlinks resolved, '..' collapsed) "
                "before the sandbox check."
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Pure-sync path helpers
# ---------------------------------------------------------------------------
#
# These helpers are intentionally synchronous and live at module scope so
# the ``execute`` coroutine can call them directly without tripping the
# ``ASYNC240`` lint rule (which forbids ``os.path`` calls *inside* async
# functions on the assumption they could block on a slow filesystem).
# ``realpath``, ``isfile``, ``getsize`` and friends are all very cheap
# ``stat()``-class operations on local disks; offloading them to a
# threadpool would add far more overhead than the calls themselves.


def _expand_path(raw: str) -> str:
    """Expand ``%VAR%``, ``$VAR``, and ``~`` tokens in ``raw``.

    The TOML loader already expands these for the configured
    allowed-directory list, but a defensive expansion here keeps the
    Skill correct when callers (e.g., tests, future plugins) construct
    a :class:`SkillContext` by hand.
    """
    return os.path.expanduser(os.path.expandvars(raw))


def _canonicalise(raw: str) -> str:
    """Return ``os.path.realpath`` of ``raw`` after env-var expansion.

    ``realpath`` resolves symbolic links and ``..`` segments even when
    the leaf does not exist. Combined with ``os.path.normcase`` at the
    comparison site, this gives the case-insensitive, symlink-safe
    identity Property 9 / CP12 demands.
    """
    return os.path.realpath(_expand_path(raw))


def _resolve_allowed_dirs(allowed_dirs: tuple[Path, ...]) -> tuple[str, ...]:
    """Canonicalise the configured allowed-directory list.

    Each directory is env-var-expanded and then run through
    :func:`os.path.realpath`. The result is normalised via
    :func:`os.path.normcase` so the per-call comparison is a cheap
    string prefix check rather than a repeated path-walk.
    """
    resolved: list[str] = []
    for entry in allowed_dirs:
        try:
            canonical = _canonicalise(str(entry))
        except (OSError, ValueError):
            # A malformed configured directory should not poison the
            # sandbox check; skip silently and let the remaining
            # entries provide coverage.
            logger.warning(
                "ReadFileSkill: skipping un-canonicalisable allowed_dir %r",
                entry,
            )
            continue
        resolved.append(os.path.normcase(canonical))
    return tuple(resolved)


def _is_within(canonical: str, allowed: tuple[str, ...]) -> bool:
    """Return ``True`` iff ``canonical`` lies within at least one allowed dir.

    The check is "starts-with allowed dir + sep", with both sides run
    through :func:`os.path.normcase` so Windows' case-insensitive
    semantics line up. An exact equality match also counts (a request to
    read the directory itself would still fail later — directories are
    not files — but the sandbox boundary itself is honoured).
    """
    target = os.path.normcase(canonical)
    for parent in allowed:
        if not parent:
            continue
        if target == parent:
            return True
        # Append the separator so ``/foo/barbaz`` is not classified as
        # being inside ``/foo/bar``.
        if target.startswith(parent + os.sep):
            return True
        # On Windows the loader may produce paths with mixed separators
        # (e.g., ``C:/Users/...``); ``os.sep`` is ``\\`` on Windows so
        # a fallback covers the alternate separator too.
        if os.altsep and target.startswith(parent + os.altsep):
            return True
    return False


# ---------------------------------------------------------------------------
# Sync read helpers (offloaded to a threadpool by ``execute``)
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    """Read a UTF-8 text file with replacement on decode errors.

    Stray non-UTF-8 bytes (common in CSV exports from legacy systems)
    must not break the Skill: replacement keeps the read total and lets
    the LLM still summarise the bulk of the document.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    return data.decode(_TEXT_ENCODING, errors="replace")


def _read_pdf(path: str) -> str:
    """Extract text from a PDF using :mod:`pypdf`.

    Pages without extractable text (scanned images, malformed XObjects)
    return an empty string from :meth:`pypdf.PageObject.extract_text`,
    which we keep as-is so the page boundaries remain countable.
    """
    reader = PdfReader(path)
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pragma: no cover - parser-dependent edge cases
            # Per-page failures should not abort the whole read; the LLM
            # can still summarise the surviving pages.
            logger.exception("pypdf failed to extract text from a PDF page")
            pages.append("")
    return "\n\n".join(pages)


def _read_docx(path: str) -> str:
    """Extract paragraph text from a ``.docx`` using :mod:`python-docx`."""
    document = docx.Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


# ---------------------------------------------------------------------------
# Sync orchestration helper
# ---------------------------------------------------------------------------


def _stat_and_read(canonical: str, ext_lower: str) -> dict[str, Any]:
    """Stat the file, enforce the size cap, and read its contents.

    Returns a dict shaped::

        {"ok": True, "size_bytes": int, "content": str}

    on success or ``{"ok": False, "code": str, "message": str, ...}``
    on a recoverable error. Raises :class:`OSError` for transient I/O
    failures the caller maps to ``internal_error``.

    Centralising the synchronous file-system work in one helper lets
    :meth:`ReadFileSkill.execute` offload the call via
    :func:`asyncio.to_thread`, which keeps the event loop responsive
    while large PDFs are being parsed and satisfies ``flake8-async``'s
    ASYNC240 rule (no ``os.path`` calls inside the async coroutine).
    """
    if not os.path.isfile(canonical):
        return {
            "ok": False,
            "code": "internal_error",
            "message": f"file not found: {canonical}",
            "value": {"path": canonical, "reason": "file_not_found"},
        }

    size_bytes = os.path.getsize(canonical)
    if size_bytes > MAX_FILE_SIZE_BYTES:
        return {
            "ok": False,
            "code": "file_too_large",
            "message": (
                f"file is {size_bytes} bytes which exceeds the "
                f"{MAX_FILE_SIZE_BYTES}-byte (5 MB) limit"
            ),
            "value": {
                "path": canonical,
                "size_bytes": size_bytes,
                "max_size_bytes": MAX_FILE_SIZE_BYTES,
            },
        }

    if ext_lower in _TEXT_EXTENSIONS:
        content = _read_text(canonical)
    elif ext_lower == ".pdf":
        content = _read_pdf(canonical)
    elif ext_lower == ".docx":
        content = _read_docx(canonical)
    else:  # pragma: no cover - guarded by the SUPPORTED_EXTENSIONS check
        return {
            "ok": False,
            "code": "not_supported",
            "message": f"unhandled supported extension {ext_lower!r}",
            "value": None,
        }

    return {"ok": True, "size_bytes": size_bytes, "content": content}


# ---------------------------------------------------------------------------
# Skill implementation
# ---------------------------------------------------------------------------


class ReadFileSkill:
    """Read a file from inside the configured sandbox.

    Stateless: a single instance is reused across invocations. The
    sandbox boundary is consulted on every call from
    :attr:`SkillContext.allowed_directories`, so a configuration change
    that adds or removes an allowed directory takes effect immediately
    without re-registering the Skill.
    """

    manifest: Final[SkillManifest] = SkillManifest(
        name=_SKILL_NAME,
        description=_SKILL_DESCRIPTION,
        json_schema=SCHEMA,
        destructive=False,
        timeout_seconds=30.0,
        # File reading is platform-agnostic; declare the full matrix so
        # Requirement 15.4 does not gate the Skill on a future build.
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(
        self,
        args: dict[str, Any],
        ctx: SkillContext,
    ) -> SkillResult:
        # ---- 1. Argument shape ----------------------------------------
        # The registry's draft-07 validator has already enforced these
        # rules; the runtime guard is for direct callers (tests,
        # future skill chains).
        validation = self._validate_path_arg(args)
        if isinstance(validation, SkillResult):
            return validation
        raw_path, expanded = validation

        # ---- 2. Sandbox check (Property 9 / CP12) ---------------------
        # The canonical path is computed BEFORE any ``open()`` so a
        # rejected request never reaches the filesystem. The registry
        # catches :class:`SandboxViolation` and records the
        # ``policy_violation`` audit row exactly once (Requirement 13.6).
        canonical = self._canonicalise_or_raise(raw_path, expanded)
        self._enforce_sandbox(raw_path, canonical, ctx.allowed_directories)

        # ---- 3. Extension gate (Requirement 8.5) ----------------------
        # Performed before ``getsize`` so an unsupported extension is
        # reported even when the file is missing — a friendlier failure
        # mode for misconfigured callers.
        ext_lower = os.path.splitext(canonical)[1].lower()
        if ext_lower not in SUPPORTED_EXTENSIONS:
            return SkillResult.error(
                "not_supported",
                (
                    f"file extension {ext_lower!r} is not supported; "
                    "ReadFileSkill accepts: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
                ),
                value={"extension": ext_lower},
            )

        # ---- 4. Stat + read (offloaded to a worker thread) ------------
        # ``_stat_and_read`` performs a ``stat()``, the size check, and
        # the format-specific parse. PDF and DOCX parsing can be
        # CPU-bound on large documents, so we hand the whole bundle to
        # a worker thread; small text reads pay a single context-switch
        # which is trivial compared to the LLM call that follows.
        try:
            outcome = await asyncio.to_thread(_stat_and_read, canonical, ext_lower)
        except OSError as exc:
            return SkillResult.error(
                "internal_error",
                f"failed to read {canonical!r}: {exc}",
            )
        except Exception as exc:
            # Parser-level failures (corrupt PDF, malformed DOCX) are
            # surfaced as ``internal_error`` rather than letting the
            # registry's catch-all path log a full traceback id; the
            # user sees the raw parser message which is usually
            # actionable ("EOF marker not found", etc.).
            return SkillResult.error(
                "internal_error",
                f"failed to parse {ext_lower} file: {exc}",
            )

        if not outcome["ok"]:
            return SkillResult.error(
                outcome["code"],
                outcome["message"],
                value=outcome.get("value"),
            )

        return SkillResult.success(
            value={
                "path": canonical,
                "extension": ext_lower,
                "size_bytes": outcome["size_bytes"],
                "content": outcome["content"],
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path_arg(args: dict[str, Any]) -> SkillResult | tuple[str, str]:
        """Return ``(raw_path, expanded_path)`` or a ``SkillResult`` failure.

        The split keeps :meth:`execute` shallow enough to satisfy the
        project's pylint return-count budget while still surfacing the
        documented error messages.
        """
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return SkillResult.error(
                "schema_violation",
                "ReadFileSkill 'path' must be a non-empty string",
            )
        expanded = _expand_path(raw_path)
        if not os.path.isabs(expanded):
            return SkillResult.error(
                "schema_violation",
                "ReadFileSkill 'path' must be an absolute filesystem path",
            )
        return raw_path, expanded

    @staticmethod
    def _canonicalise_or_raise(raw_path: str, expanded: str) -> str:
        """Run :func:`os.path.realpath` and raise :class:`SandboxViolation` on failure.

        ``realpath`` raises on truly malformed inputs (e.g., embedded
        NUL bytes). Treat those as sandbox violations so the request is
        denied without exposing the underlying error to the user.
        """
        try:
            return os.path.realpath(expanded)
        except (OSError, ValueError) as exc:
            raise SandboxViolation(
                f"path {raw_path!r} could not be canonicalised: {exc}",
                justification=(
                    "path canonicalisation failed; refusing to leave "
                    "the sandbox boundary unverified"
                ),
            ) from exc

    @staticmethod
    def _enforce_sandbox(
        raw_path: str,
        canonical: str,
        allowed_dirs: tuple[Path, ...],
    ) -> None:
        """Raise :class:`SandboxViolation` unless ``canonical`` is allowed.

        An empty allowed-directory list means file reading is disabled
        at the policy layer (the config validator rejects empty lists in
        user-supplied config, but tests and future code paths may
        construct one). Treat as a sandbox violation rather than
        ``not_supported`` so the audit log still records the attempt.
        """
        allowed = _resolve_allowed_dirs(allowed_dirs)
        if not allowed:
            raise SandboxViolation(
                "no allowed directories are configured; ReadFileSkill is disabled",
                justification="automation.allowed_directories.paths is empty",
            )
        if not _is_within(canonical, allowed):
            raise SandboxViolation(
                f"path {raw_path!r} resolves outside the allowed directories",
                justification=(
                    "canonical path is not within "
                    "automation.allowed_directories.paths"
                ),
            )


# ---------------------------------------------------------------------------
# Module-level singleton consumed by :meth:`SkillRegistry.discover`.
# ---------------------------------------------------------------------------


#: Typed as the concrete :class:`ReadFileSkill` rather than the
#: :class:`Skill` Protocol because the latter declares ``manifest`` as a
#: writable variable while we expose it as a :data:`Final` class
#: attribute (mirrors the convention used by :mod:`send_email`). The
#: registry's ``isinstance(obj, Skill)`` runtime check still validates
#: structural conformance at startup.
SKILL: ReadFileSkill = ReadFileSkill()
