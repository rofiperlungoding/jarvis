"""Unit tests for :mod:`jarvis.skills.builtin.read_file`.

Covers Requirement 8 acceptance criteria 1, 2, 5, 6, and 7 plus
Property 9 / CP12 (sandbox soundness — no ``open()`` syscall when the
sandbox check fails). Each supported extension exercises the read path
on a tiny fixture file produced inside the test's ``tmp_path`` (which
is also configured as the sole allowed directory for the call).

The traversal-blocked, symlink-escape, and oversized-file tests rely
on the path-canonicalisation contract documented in CP12: the Skill
must compute ``os.path.realpath`` and compare against the canonicalised
allowed-directory list before reaching for ``open()``. To assert the
"no ``open()`` syscall" half of the property we monkey-patch
``builtins.open`` and verify it was never called for blocked paths.

Validates: Requirements 8.1, 8.2, 8.5, 8.6, 8.7, 13.6
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Awaitable
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from jarvis.skills.base import Skill, SkillContext, SkillManifest, SkillResult
from jarvis.skills.builtin import read_file as read_file_module
from jarvis.skills.builtin.read_file import (
    MAX_FILE_SIZE_BYTES,
    SCHEMA,
    SKILL,
    SUPPORTED_EXTENSIONS,
    ReadFileSkill,
)
from jarvis.skills.registry import SandboxViolation, SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[Any]) -> Any:
    """Synchronously drive a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _make_ctx(*allowed_dirs: Path) -> SkillContext:
    """Build a :class:`SkillContext` with the supplied dirs as the sandbox.

    Every entry is canonicalised via ``realpath`` first so a symlinked
    ``tmp_path`` (common on macOS where ``/tmp`` is a symlink to
    ``/private/tmp``) does not accidentally fail the sandbox check
    inside the Skill — the production code does the same, so the test
    fixture must mirror it.
    """
    canonical = tuple(Path(os.path.realpath(str(d))) for d in allowed_dirs)
    return SkillContext(
        allowed_directories=canonical,
        run_id="read-file-test",
    )


def _make_text_file(parent: Path, name: str, content: str) -> Path:
    """Write ``content`` to ``parent/name`` and return the file path.

    Uses :meth:`Path.write_bytes` so newline characters survive intact
    on Windows (where text-mode writes translate ``\\n`` to ``\\r\\n``).
    The Skill reads the file in binary mode and decodes as UTF-8, so
    the bytes-on-disk are exactly what the test asserts against.
    """
    path = parent / name
    path.write_bytes(content.encode("utf-8"))
    return path


def _make_pdf_file(parent: Path, name: str, content: str) -> Path:
    """Build a tiny single-page PDF carrying ``content`` for the read test."""
    pypdf_module = pytest.importorskip("pypdf")
    writer = pypdf_module.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = parent / name
    with open(path, "wb") as handle:
        writer.write(handle)
    # ``content`` is intentionally unused by the bytes-on-disk because
    # ``add_blank_page`` produces a page with no extractable text. The
    # parameter exists for symmetry with ``_make_text_file`` and to
    # document the "blank PDF returns empty string content" expectation.
    del content
    return path


def _make_docx_file(parent: Path, name: str, content: str) -> Path:
    """Build a tiny ``.docx`` containing ``content`` as a single paragraph."""
    docx_module = pytest.importorskip("docx")
    document = docx_module.Document()
    document.add_paragraph(content)
    path = parent / name
    document.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Manifest / module surface
# ---------------------------------------------------------------------------


def test_module_exposes_skill_singleton() -> None:
    # Plugin discovery (registry._load_plugin_file) imports the module
    # and reads ``getattr(module, "SKILL", None)``; the constant must
    # exist and resolve to the singleton instance.
    assert isinstance(SKILL, ReadFileSkill)
    assert read_file_module.SKILL is SKILL


def test_skill_satisfies_skill_protocol() -> None:
    # The :class:`Skill` Protocol is runtime-checkable; confirm the
    # singleton would be accepted by the registry's isinstance gate.
    assert isinstance(SKILL, Skill)


def test_manifest_is_non_destructive_with_expected_name() -> None:
    manifest = SKILL.manifest
    assert isinstance(manifest, SkillManifest)
    # Requirement 8.1 anchors the wording.
    assert manifest.name == "ReadFileSkill"
    assert manifest.destructive is False
    assert manifest.source == "builtin"
    # The manifest's schema must be the same dict as the public ``SCHEMA``
    # constant so consumers (Mistral tool publishing, tests, docs) agree.
    assert manifest.json_schema is SCHEMA


def test_schema_requires_path_and_rejects_extras() -> None:
    # Requirement 8.1: argument schema requires an absolute "path".
    assert SCHEMA["required"] == ["path"]
    assert SCHEMA["properties"]["path"]["type"] == "string"
    # ``additionalProperties: false`` keeps the LLM from smuggling
    # encoding/range/format hints through the Skill.
    assert SCHEMA["additionalProperties"] is False


def test_supported_extensions_match_design() -> None:
    # Requirement 8.5: closed set of supported file types.
    assert (
        frozenset(
            {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".pdf", ".docx"}
        )
        == SUPPORTED_EXTENSIONS
    )


def test_max_size_is_five_megabytes() -> None:
    # Requirement 8.7: 5 MB cap.
    assert MAX_FILE_SIZE_BYTES == 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Happy-path reads (one per supported extension)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("notes.txt", "Hello, world!\nLine two."),
        ("readme.md", "# Heading\n\nSome **markdown** body."),
        ("rows.csv", "a,b,c\n1,2,3\n4,5,6"),
        ("blob.json", '{"key": "value", "n": 42}'),
        ("snippet.py", "def add(a, b):\n    return a + b\n"),
        ("snippet.js", "export const add = (a, b) => a + b;\n"),
        ("snippet.ts", "export const add = (a: number, b: number) => a + b;\n"),
    ],
)
def test_reads_each_text_extension(tmp_path: Path, name: str, payload: str) -> None:
    """Requirement 8.5: every text extension round-trips through the Skill."""
    path = _make_text_file(tmp_path, name, payload)
    ctx = _make_ctx(tmp_path)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True, result.error_message
    assert result.error_code is None
    assert result.value is not None
    assert result.value["content"] == payload
    assert result.value["extension"] == os.path.splitext(name)[1].lower()
    assert result.value["size_bytes"] == path.stat().st_size
    # ``path`` is reported as the canonical realpath so downstream
    # consumers can rely on it for caching / dedup.
    assert os.path.realpath(str(path)) == result.value["path"]


def test_reads_pdf_via_pypdf(tmp_path: Path) -> None:
    """Requirement 8.5: ``.pdf`` is handled via :mod:`pypdf`."""
    path = _make_pdf_file(tmp_path, "doc.pdf", "ignored")
    ctx = _make_ctx(tmp_path)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True, result.error_message
    assert result.value is not None
    assert result.value["extension"] == ".pdf"
    # ``add_blank_page`` produces no extractable text; we just assert
    # the read returned a string (possibly empty) without raising.
    assert isinstance(result.value["content"], str)


def test_reads_docx_via_python_docx(tmp_path: Path) -> None:
    """Requirement 8.5: ``.docx`` is handled via :mod:`python-docx`."""
    path = _make_docx_file(tmp_path, "doc.docx", "Hello DOCX world.")
    ctx = _make_ctx(tmp_path)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is True, result.error_message
    assert result.value is not None
    assert result.value["extension"] == ".docx"
    assert "Hello DOCX world." in result.value["content"]


# ---------------------------------------------------------------------------
# Sandbox enforcement (Requirement 8.6 / Property 9 / CP12)
# ---------------------------------------------------------------------------


def test_traversal_outside_sandbox_returns_access_denied(tmp_path: Path) -> None:
    """Requirement 8.6: a ``..``-traversal that escapes is blocked.

    The Skill raises :class:`SandboxViolation`, which the registry
    converts to ``access_denied`` plus a ``policy_violation`` audit
    entry. Direct ``execute`` invocations (this test) see the
    exception; the registry-level test below verifies the
    ``access_denied`` translation.
    """
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = _make_text_file(outside, "secret.txt", "leaked")

    # Build a traversal path: ``<inside>/../outside/secret.txt`` resolves
    # to ``<outside>/secret.txt`` after canonicalisation.
    traversal = str(inside / ".." / "outside" / "secret.txt")
    assert os.path.realpath(traversal) == os.path.realpath(str(secret))

    ctx = _make_ctx(inside)

    with pytest.raises(SandboxViolation):
        _run(SKILL.execute({"path": traversal}, ctx))


def test_sandbox_check_does_not_open_blocked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Property 9 / CP12: blocked paths produce no ``open()`` syscall."""
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = _make_text_file(outside, "secret.txt", "leaked")

    # Patch ``builtins.open`` to record every call. The Skill must
    # never reach this when the sandbox check fails — pypdf/docx
    # would otherwise stat or open the file.
    open_calls: list[tuple[Any, ...]] = []
    real_open = builtins.open

    def tracking_open(*args: Any, **kwargs: Any) -> Any:
        open_calls.append((args, tuple(sorted(kwargs.items()))))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    ctx = _make_ctx(inside)
    with pytest.raises(SandboxViolation):
        _run(SKILL.execute({"path": str(secret)}, ctx))

    # The Skill must not have touched the blocked file via ``open``.
    secret_real = os.path.realpath(str(secret))
    for args, _kwargs in open_calls:
        if not args:
            continue
        target = args[0]
        # Coerce path-like objects for a robust string comparison.
        try:
            target_real = os.path.realpath(os.fspath(target))
        except TypeError:
            continue
        assert target_real != secret_real, (
            "ReadFileSkill must not open a file outside the sandbox; "
            f"got open({target!r})"
        )


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason=(
        "creating filesystem symlinks on Windows requires Developer "
        "Mode or admin privileges; the symlink-escape semantics are "
        "still covered by the realpath-based traversal test"
    ),
)
def test_symlink_escape_is_blocked(tmp_path: Path) -> None:
    """Requirement 8.6: a symlink that points outside the sandbox is blocked.

    The Skill canonicalises through symlinks via :func:`os.path.realpath`
    so a symlink under the sandbox that points to a file outside still
    resolves to the outside path and fails the sandbox check.
    """
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _make_text_file(outside, "secret.txt", "leaked-via-symlink")

    link = inside / "link.txt"
    try:
        os.symlink(str(target), str(link))
    except OSError as exc:  # pragma: no cover - defensive on locked-down CI
        pytest.skip(f"symlink creation not permitted: {exc}")

    ctx = _make_ctx(inside)
    with pytest.raises(SandboxViolation):
        _run(SKILL.execute({"path": str(link)}, ctx))


def test_empty_allowed_directories_blocks_every_path(tmp_path: Path) -> None:
    """An empty allow list disables the Skill (defence in depth).

    The config validator already rejects empty lists, but the Skill
    code must remain correct when one slips through (e.g., a test
    that builds a :class:`SkillContext` directly).
    """
    path = _make_text_file(tmp_path, "notes.txt", "data")
    ctx = SkillContext(
        allowed_directories=(),
        run_id="read-file-test-empty-allowed",
    )

    with pytest.raises(SandboxViolation):
        _run(SKILL.execute({"path": str(path)}, ctx))


# ---------------------------------------------------------------------------
# File-too-large (Requirement 8.7)
# ---------------------------------------------------------------------------


def test_oversized_file_returns_file_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 8.7: > 5 MB → ``file_too_large``."""
    path = _make_text_file(tmp_path, "huge.txt", "x")  # tiny on disk

    # Avoid actually writing 5 MB to disk on every test run; patch
    # ``getsize`` to report a byte count above the cap. The Skill's
    # production code calls ``os.path.getsize`` exactly so this is a
    # faithful stand-in.
    real_getsize = os.path.getsize

    def fake_getsize(p: Any) -> int:
        if os.path.realpath(os.fspath(p)) == os.path.realpath(str(path)):
            return MAX_FILE_SIZE_BYTES + 1
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    ctx = _make_ctx(tmp_path)
    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "file_too_large"
    assert result.value is not None
    assert result.value["max_size_bytes"] == MAX_FILE_SIZE_BYTES
    assert result.value["size_bytes"] == MAX_FILE_SIZE_BYTES + 1


def test_oversized_file_is_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``file_too_large`` short-circuits before any ``open()`` call."""
    path = _make_text_file(tmp_path, "huge.txt", "x")
    real_getsize = os.path.getsize

    def fake_getsize(p: Any) -> int:
        if os.path.realpath(os.fspath(p)) == os.path.realpath(str(path)):
            return MAX_FILE_SIZE_BYTES + 1
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    open_calls: list[Any] = []
    real_open = builtins.open

    def tracking_open(*args: Any, **kwargs: Any) -> Any:
        open_calls.append(args[0] if args else None)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    ctx = _make_ctx(tmp_path)
    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.error_code == "file_too_large"
    target_real = os.path.realpath(str(path))
    for entry in open_calls:
        if entry is None:
            continue
        try:
            entry_real = os.path.realpath(os.fspath(entry))
        except TypeError:
            continue
        assert entry_real != target_real, (
            "Oversized files must not be opened; the size check has to "
            "short-circuit before the read"
        )


# ---------------------------------------------------------------------------
# Unsupported extension / missing file
# ---------------------------------------------------------------------------


def test_unsupported_extension_returns_not_supported(tmp_path: Path) -> None:
    """Requirement 8.5: only the documented formats are honoured."""
    path = _make_text_file(tmp_path, "image.exe", "binary-blob")
    ctx = _make_ctx(tmp_path)

    result = _run(SKILL.execute({"path": str(path)}, ctx))

    assert result.ok is False
    assert result.error_code == "not_supported"
    assert result.value is not None
    assert result.value["extension"] == ".exe"


def test_missing_file_returns_internal_error(tmp_path: Path) -> None:
    """A non-existent (but otherwise valid) file is reported clearly."""
    missing = tmp_path / "ghost.txt"
    assert not missing.exists()
    ctx = _make_ctx(tmp_path)

    result = _run(SKILL.execute({"path": str(missing)}, ctx))

    assert result.ok is False
    assert result.error_code == "internal_error"
    # The error message must mention the path so the user can fix it.
    assert (result.error_message or "").lower().count("not found") == 1


# ---------------------------------------------------------------------------
# Argument shape (defence in depth)
# ---------------------------------------------------------------------------


def test_relative_path_returns_schema_violation(tmp_path: Path) -> None:
    """Requirement 8.1: only absolute paths are accepted."""
    ctx = _make_ctx(tmp_path)
    result = _run(SKILL.execute({"path": "relative/notes.txt"}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_blank_path_returns_schema_violation(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    result = _run(SKILL.execute({"path": "   "}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_skill_registers_and_dispatches_through_registry(tmp_path: Path) -> None:
    """End-to-end: schema validation + dispatch through the real registry."""
    registry = SkillRegistry()
    registry.register(SKILL)
    assert "ReadFileSkill" in registry

    path = _make_text_file(tmp_path, "notes.txt", "Hello!")
    ctx = _make_ctx(tmp_path)
    result = _run(registry.dispatch("ReadFileSkill", {"path": str(path)}, ctx))

    assert isinstance(result, SkillResult)
    assert result.ok is True
    assert result.value is not None
    assert result.value["content"] == "Hello!"


def test_registry_translates_sandbox_violation_to_access_denied(
    tmp_path: Path,
) -> None:
    """Requirement 8.6 + 13.6: blocked path → ``access_denied`` via registry."""
    registry = SkillRegistry()
    registry.register(SKILL)

    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = _make_text_file(outside, "secret.txt", "leaked")

    ctx = _make_ctx(inside)
    result = _run(registry.dispatch("ReadFileSkill", {"path": str(secret)}, ctx))

    assert result.ok is False
    assert result.error_code == "access_denied"


def test_registry_rejects_extra_properties(tmp_path: Path) -> None:
    """``additionalProperties: false`` blocks LLM-smuggled fields."""
    registry = SkillRegistry()
    registry.register(SKILL)

    path = _make_text_file(tmp_path, "notes.txt", "Hi")
    ctx = _make_ctx(tmp_path)
    result = _run(
        registry.dispatch(
            "ReadFileSkill",
            {"path": str(path), "encoding": "utf-16"},
            ctx,
        )
    )

    assert result.ok is False
    assert result.error_code == "schema_violation"


def test_registry_rejects_missing_path(tmp_path: Path) -> None:
    registry = SkillRegistry()
    registry.register(SKILL)

    ctx = _make_ctx(tmp_path)
    result = _run(registry.dispatch("ReadFileSkill", {}, ctx))

    assert result.ok is False
    assert result.error_code == "schema_violation"
