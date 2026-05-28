"""Property 9 — Path sandbox soundness.

From ``design.md §Correctness Properties``:

    *For any* ``path`` argument ``P`` supplied to ``ReadFileSkill`` or
    ``SummarizeFileSkill``, if ``P`` (after canonicalisation, including
    resolution of symbolic links and ``..``) does not lie within the
    configured allowed-directory list, the Skill SHALL return
    ``access_denied`` and SHALL NOT open the file.

The "no ``open()`` syscall" half of the invariant is verified here by
monkey-patching every entry point that could plausibly issue an
``open(2)`` call:

* :data:`builtins.open` — covers the underlying open used by
  ``pathlib.Path.open``, the ``open()`` calls inside :mod:`pypdf`,
  :mod:`python-docx`, and :class:`ReadFileSkill._read_text`.
* :meth:`pathlib.Path.open` — belt-and-braces, in case a future
  implementation reaches for the bound-method form (which goes through
  :mod:`io.open` rather than :data:`builtins.open` on some Python
  versions).
* :meth:`io.open` — the third public entry point. On CPython 3.11+,
  :data:`builtins.open` *is* :meth:`io.open`, but some test harnesses
  (and a small number of third-party libraries) patch them
  independently, so we install our counter on both names to keep the
  property robust against that.

The strategy generates arbitrary "outside" paths in three flavours:

1. **Sibling absolute paths** — files under a sibling directory of the
   sandbox, with realistic-looking extensions. Catches the easiest
   regression: forgetting the sandbox check entirely.
2. **Traversal paths** — paths that *start* inside the sandbox but
   escape via ``..`` segments. The Skill must canonicalise via
   :func:`os.path.realpath` before the sandbox comparison.
3. **Drive-root / system paths** — Windows-style absolute paths that
   no test fixture would ever land on (``C:\\Windows\\...``). On Linux
   we substitute ``/etc/...`` and ``/root/...``. Both forms exercise
   the boundary check on paths that point at sensitive locations the
   sandbox is meant to protect.

For each generated path, the test calls both
:class:`ReadFileSkill` and :class:`SummarizeFileSkill` and asserts:

* the Skill raises :class:`SandboxViolation` directly (the registry
  translates this to ``access_denied`` — see the wrapper test below);
* the open-call counter remained at zero across all three patched
  entry points;
* dispatching the same arguments through the :class:`SkillRegistry`
  yields ``SkillResult(error_code='access_denied')`` while still
  observing zero ``open()`` calls.

A small companion property confirms the *positive* half of the
contract: paths *inside* the sandbox successfully read (and therefore
the open-counter MUST be non-zero), so the test does not pass
vacuously by patching ``open`` into a no-op.

Validates: Requirements 8.2, 8.6, 13.6 (CP12)
"""

from __future__ import annotations

import asyncio
import builtins
import io
import os
from pathlib import Path
import sys
from typing import Any

from hypothesis import HealthCheck, assume, given, settings, strategies as st
import pytest

from jarvis.skills.base import SkillContext, SkillResult
from jarvis.skills.builtin.read_file import SKILL as READ_FILE_SKILL
from jarvis.skills.builtin.summarize_file import SKILL as SUMMARIZE_FILE_SKILL
from jarvis.skills.registry import SandboxViolation, SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Synchronously drive a coroutine without depending on pytest-asyncio."""
    return asyncio.run(coro)


def _make_ctx(*allowed_dirs: Path) -> SkillContext:
    """Build a :class:`SkillContext` with the supplied dirs as the sandbox.

    Each entry is canonicalised via ``realpath`` so a symlinked
    ``tmp_path`` (common on macOS where ``/tmp`` -> ``/private/tmp``)
    does not accidentally fail the sandbox check inside the Skill —
    the production loader does the same, so the test fixture must
    mirror it.

    No ``llm_backend`` is provided: the property quantifies over
    *blocked* paths, which never reach the LLM call inside
    :class:`SummarizeFileSkill` because the sandbox check raises
    first.
    """
    canonical = tuple(Path(os.path.realpath(str(d))) for d in allowed_dirs)
    return SkillContext(
        allowed_directories=canonical,
        run_id="property-09-sandbox-soundness",
    )


class _OpenCounter:
    """Counts attempts to open ``target_paths`` via any patched entry point.

    The counter is targeted: it ignores ``open()`` calls that pytest
    itself, the test harness, or :class:`pypdf`/``python-docx`` make
    against unrelated files (e.g., site-packages metadata). We only
    care about the property's narrow claim: a *blocked* path is never
    opened.

    The counter records every observed call against ``target_paths``
    so failures can be diagnosed quickly. ``target_paths`` is a set of
    canonical (``os.path.realpath``-resolved) strings.
    """

    def __init__(self, target_paths: set[str]) -> None:
        self.targets = target_paths
        self.hits: list[str] = []

    def matches(self, candidate: Any) -> str | None:
        """Return the canonical path string if ``candidate`` is a target."""
        try:
            resolved: str = os.path.realpath(os.fspath(candidate))
        except TypeError:
            # ``int`` (file descriptor) or other non-path inputs cannot
            # name a target file, so they are uninteresting for the
            # property check.
            return None
        if resolved in self.targets:
            return resolved
        return None


def _install_open_counter(
    monkeypatch: pytest.MonkeyPatch, counter: _OpenCounter
) -> None:
    """Patch every public ``open()`` entry point to record target calls.

    We deliberately *do not* block the call — we only count it. The
    property test asserts the count remains at zero for blocked paths,
    so allowing the call to proceed (in the unlikely event one slips
    through) gives a more diagnostic failure ("the file was opened and
    read X bytes") than a synthetic IO error would.
    """
    real_builtins_open = builtins.open
    real_io_open = io.open
    real_path_open = Path.open

    def tracking_builtins_open(*args: Any, **kwargs: Any) -> Any:
        if args:
            hit = counter.matches(args[0])
            if hit is not None:
                counter.hits.append(hit)
        return real_builtins_open(*args, **kwargs)

    def tracking_io_open(*args: Any, **kwargs: Any) -> Any:
        if args:
            hit = counter.matches(args[0])
            if hit is not None:
                counter.hits.append(hit)
        return real_io_open(*args, **kwargs)

    def tracking_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        hit = counter.matches(self)
        if hit is not None:
            counter.hits.append(hit)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_builtins_open)
    monkeypatch.setattr(io, "open", tracking_io_open)
    monkeypatch.setattr(Path, "open", tracking_path_open)


# ---------------------------------------------------------------------------
# Strategies for "outside the sandbox" paths
# ---------------------------------------------------------------------------


# A realistic mix of file extensions: half are in
# :data:`SUPPORTED_EXTENSIONS` (so the sandbox check is the only thing
# standing between the call and a real read attempt) and half are not
# (so the property also covers paths that would otherwise fail the
# extension gate — the sandbox check must precede the extension check
# so there is still no ``open()`` call).
_extensions = st.sampled_from(
    [
        ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".pdf", ".docx",
        ".exe", ".bin", ".log", ".cfg", ".db",
    ]
)


# Filename component: simple ASCII identifiers so the assertions stay
# readable. Path-separator and reserved characters are excluded so the
# generated path is always a valid file path on every supported OS.
_filename_stems = st.from_regex(r"[A-Za-z0-9_\-]{1,16}", fullmatch=True)


@st.composite
def _outside_filename(draw: st.DrawFn) -> str:
    """``stem.ext`` with a Skill-relevant or Skill-irrelevant extension."""
    return draw(_filename_stems) + draw(_extensions)


# A small set of "system-like" absolute paths that the property should
# always block. These are *not* created on disk; the sandbox check
# rejects them on the canonical-prefix mismatch alone, so no IO ever
# happens. The OS-specific list is selected at module-import time so
# the same property test runs on Linux and Windows CI runners.
if sys.platform.startswith("win"):
    _system_paths: tuple[str, ...] = (
        r"C:\Windows\System32\config\SAM",
        r"C:\Windows\notepad.exe",
        r"C:\Users\Public\Documents\report.txt",
        r"D:\backups\secrets.json",
    )
else:
    _system_paths = (
        "/etc/passwd",
        "/etc/shadow",
        "/root/.ssh/id_rsa",
        "/var/log/auth.log",
    )


# ---------------------------------------------------------------------------
# Property 9 — blocked paths never trigger an open() syscall
# ---------------------------------------------------------------------------


@given(
    sibling_filename=_outside_filename(),
    traversal_filename=_outside_filename(),
    system_path=st.sampled_from(_system_paths),
    use_traversal=st.booleans(),
    use_system=st.booleans(),
)
@settings(
    # Inherit ``max_examples`` / ``deadline`` from the ``jarvis``
    # Hypothesis profile in ``tests/conftest.py``. ``tmp_path`` is a
    # function-scoped fixture; suppress the corresponding health check
    # so Hypothesis re-uses it across examples (this is safe because
    # we re-create the inside / outside subdirectories per call).
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
def test_blocked_paths_return_access_denied_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sibling_filename: str,
    traversal_filename: str,
    system_path: str,
    use_traversal: bool,
    use_system: bool,
) -> None:
    """Outside-sandbox paths → ``access_denied``; ``open()`` never fires.

    **Validates: Requirements 8.2, 8.6, 13.6 (CP12)**
    """

    # ---- Build the sandbox -----------------------------------------------
    inside = tmp_path / "inside"
    inside.mkdir(exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)

    # Materialise the sibling file on disk. We do NOT create the
    # traversal target (it shares a path with the sibling once
    # canonicalised), and we never create the system-path file.
    # Existence does not matter for the property — the sandbox check
    # short-circuits before any ``stat()`` — but creating the sibling
    # makes the failure mode obvious if a regression slips: the file
    # really would be readable, so a leak would actually leak data.
    sibling_path = outside / sibling_filename
    sibling_path.write_bytes(b"sensitive-bytes")

    # Build the three candidate paths: a plain absolute sibling path,
    # a traversal path that escapes via ``..``, and a system path
    # outside ``tmp_path`` entirely. ``use_traversal`` / ``use_system``
    # let Hypothesis explore each combination of the three flavours.
    candidates: list[str] = [str(sibling_path)]
    if use_traversal:
        # ``<inside>/../outside/<traversal_filename>`` resolves to
        # ``<outside>/<traversal_filename>`` after ``realpath``. We do
        # not require this file to exist — the sandbox check fails
        # purely on the canonical-prefix mismatch.
        traversal = inside / ".." / "outside" / traversal_filename
        candidates.append(str(traversal))
    if use_system:
        candidates.append(system_path)

    # ---- Install the open-call counter -----------------------------------
    target_set = {os.path.realpath(p) for p in candidates}
    counter = _OpenCounter(target_paths=target_set)
    _install_open_counter(monkeypatch, counter)

    ctx = _make_ctx(inside)

    # Each candidate must be rejected by both Skills.
    for candidate in candidates:
        # Property pre-condition: the candidate canonicalises *outside*
        # the sandbox. ``assume`` is cheap and lets Hypothesis discard
        # the (vanishingly rare) case where a generated traversal
        # happens to resolve back inside the sandbox.
        canonical = os.path.realpath(candidate)
        inside_canonical = os.path.realpath(str(inside))
        assume(
            canonical != inside_canonical
            and not canonical.startswith(inside_canonical + os.sep)
        )

        # ---- ReadFileSkill: raises SandboxViolation, no open ---------
        with pytest.raises(SandboxViolation):
            _run(READ_FILE_SKILL.execute({"path": candidate}, ctx))
        assert counter.hits == [], (
            f"ReadFileSkill opened a blocked path: hits={counter.hits!r} "
            f"(candidate={candidate!r}, canonical={canonical!r})"
        )

        # ---- SummarizeFileSkill: same contract -----------------------
        with pytest.raises(SandboxViolation):
            _run(SUMMARIZE_FILE_SKILL.execute({"path": candidate}, ctx))
        assert counter.hits == [], (
            f"SummarizeFileSkill opened a blocked path: hits={counter.hits!r} "
            f"(candidate={candidate!r}, canonical={canonical!r})"
        )

    # ---- Registry path: the SandboxViolation becomes access_denied -------
    # The registry is the surface the Dialog_Manager actually calls.
    # We confirm the same blocked paths produce
    # ``SkillResult(error_code='access_denied')`` and STILL never open
    # the file. This covers the half of CP12 that is observable to the
    # rest of the system (Requirement 13.6).
    registry = SkillRegistry()
    registry.register(READ_FILE_SKILL)  # type: ignore[arg-type]
    registry.register(SUMMARIZE_FILE_SKILL)  # type: ignore[arg-type]

    for candidate in candidates:
        canonical = os.path.realpath(candidate)
        inside_canonical = os.path.realpath(str(inside))
        if canonical == inside_canonical or canonical.startswith(
            inside_canonical + os.sep
        ):
            # Skip any candidate that, after canonicalisation, is
            # actually inside the sandbox; the property only quantifies
            # over outside paths.
            continue

        # ---- ReadFileSkill via registry ------------------------------
        result = _run(
            registry.dispatch("ReadFileSkill", {"path": candidate}, ctx)
        )
        assert isinstance(result, SkillResult)
        assert result.ok is False, (
            f"registry.dispatch(ReadFileSkill) succeeded for blocked path "
            f"{candidate!r}: {result!r}"
        )
        assert result.error_code == "access_denied", (
            f"expected access_denied for blocked path {candidate!r}; "
            f"got {result.error_code!r} (message={result.error_message!r})"
        )

        # ---- SummarizeFileSkill via registry -------------------------
        result = _run(
            registry.dispatch(
                "SummarizeFileSkill", {"path": candidate}, ctx
            )
        )
        assert isinstance(result, SkillResult)
        assert result.ok is False, (
            f"registry.dispatch(SummarizeFileSkill) succeeded for blocked "
            f"path {candidate!r}: {result!r}"
        )
        assert result.error_code == "access_denied", (
            f"expected access_denied for blocked path {candidate!r}; "
            f"got {result.error_code!r} (message={result.error_message!r})"
        )

    # Final invariant: across every candidate and both Skills (direct
    # + registry call paths) the open-counter never moved.
    assert counter.hits == [], (
        "no blocked-path open() call should ever happen, but "
        f"observed: {counter.hits!r}"
    )


# ---------------------------------------------------------------------------
# Companion: positive half — the open-counter DOES move for allowed reads
# ---------------------------------------------------------------------------


def test_allowed_path_actually_opens_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity check: the patched ``open`` IS observed for allowed reads.

    Without this guard, the property test above could pass vacuously
    if the patching machinery silently no-oped (e.g., if a future
    refactor moved the open call into a C extension that bypasses
    :data:`builtins.open`). Reading a sandboxed file MUST register on
    the counter.
    """
    inside = tmp_path / "inside"
    inside.mkdir()
    target = inside / "notes.txt"
    target.write_bytes(b"hello sandbox")

    counter = _OpenCounter(target_paths={os.path.realpath(str(target))})
    _install_open_counter(monkeypatch, counter)

    ctx = _make_ctx(inside)
    result = _run(READ_FILE_SKILL.execute({"path": str(target)}, ctx))

    assert result.ok is True, result.error_message
    assert result.value is not None
    assert result.value["content"] == "hello sandbox"
    # The patched open() must have observed the read. If this fails,
    # the property test above is no longer trustworthy — it would
    # silently pass even for genuine regressions.
    assert counter.hits, (
        "open-counter saw zero hits for an allowed read; the patching "
        "is no longer covering the Skill's I/O path"
    )
