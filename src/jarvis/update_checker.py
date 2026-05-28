"""Check GitHub Releases for a newer JARVIS build.

Design
------

The desktop app pings the GitHub Releases API on startup (background,
non-blocking) and compares the latest release tag to the bundled
:data:`jarvis.__version__`. If a newer version is available, the UI
surfaces a non-modal notification with two actions:

* **Open release page** — falls back to the user's browser. Useful
  when they want release notes or a manual install.
* **Update now** — JARVIS downloads ``JARVIS-Setup-<tag>.exe`` from
  the release's asset list, launches it with ``/SILENT /SUPPRESSMSGBOXES``
  flags, and exits the running process so the installer can replace
  the binaries in-place.

Why GitHub Releases
~~~~~~~~~~~~~~~~~~~

* Already paid for by the project's release CI workflow — no extra
  hosting, no S3 bucket, no signing infrastructure beyond what
  GitHub already provides.
* The unauthenticated REST endpoint allows 60 requests per IP per
  hour, far above any plausible startup-check budget.
* Tags follow PEP 440-compatible semver (``vMAJOR.MINOR.PATCH``),
  which keeps comparison logic trivial.

Auto-install flow
~~~~~~~~~~~~~~~~~

When the user clicks **Update now**, :func:`download_and_run_installer`
performs:

1. Streams the ``.exe`` asset to a temp file under ``%LOCALAPPDATA%``
   with a ``.partial`` suffix so a half-finished download can be
   resumed or cleaned up later.
2. Renames the temp file once the download completes — atomic on
   NTFS so the installer is never invoked against a partial blob.
3. Spawns the installer with
   ``/VERYSILENT /SUPPRESSMSGBOXES /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS``.
   Inno Setup's silent mode skips every page including the Finish
   page, but still respects the `[Run]` postinstall line that
   relaunches JARVIS — so the user just sees the window flicker
   away and come back.
4. ``os.execv`` would normally swap into the installer, but PyInstaller
   bundles don't support it. We use ``subprocess.Popen`` + ``sys.exit``
   instead: the installer becomes detached, JARVIS exits, the
   installer overwrites ``JARVIS.exe`` (the previous process is gone
   so there's no file lock), then the installer's `[Run]` step
   relaunches JARVIS.

Failure handling
~~~~~~~~~~~~~~~~

Every failure path returns ``None`` (for the check) or ``False`` (for
the install) rather than raising. The UI must treat the negative
outcome as "show a generic error toast and do nothing" — never as a
crash.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from jarvis import __version__

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GITHUB_REPO",
    "DEFAULT_TIMEOUT_S",
    "ReleaseInfo",
    "UpdateAvailable",
    "check_for_updates",
    "compare_versions",
    "download_and_run_installer",
    "parse_version",
]

#: Default repository slug. Override with the ``--update-repo`` CLI flag
#: or by passing ``repo=...`` to :func:`check_for_updates` when forking.
DEFAULT_GITHUB_REPO: Final[str] = "rofiperlungoding/jarvis"

#: Default request timeout for the latest-release lookup. Kept short
#: so a slow / unreachable GitHub never extends the cold-start window
#: by more than a couple of seconds.
DEFAULT_TIMEOUT_S: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """Subset of the GitHub Releases payload we actually care about."""

    #: The release tag. Stripped of any ``v`` prefix before being stored
    #: so consumers can compare directly against
    #: :data:`jarvis.__version__`.
    version: str

    #: Tag exactly as published (``v1.2.0``, ``1.2.0`` — whichever the
    #: maintainer used). Useful for clickable URLs and human display.
    tag_name: str

    #: HTML URL of the release page. Suitable for ``webbrowser.open``.
    html_url: str

    #: Optional human-readable release notes. May be empty.
    body: str

    #: Direct download URL of the Inno Setup installer asset, when
    #: present. ``None`` when the release exists but did not attach
    #: a setup ``.exe`` (e.g., source-only release).
    installer_url: str | None = None

    #: Size of the installer asset in bytes, when known. Useful for
    #: rendering progress bars.
    installer_size: int | None = None


@dataclass(frozen=True, slots=True)
class UpdateAvailable:
    """Result emitted when a newer release is found."""

    #: The locally-running version (``jarvis.__version__``).
    current: str

    #: The newest release on the configured repository.
    latest: ReleaseInfo


def parse_version(text: str) -> tuple[int, int, int, str]:
    """Parse a ``MAJOR.MINOR.PATCH[-suffix]`` string.

    Returns a 4-tuple ``(major, minor, patch, suffix)`` suitable for
    direct comparison via Python's tuple ordering. The suffix is the
    string after the third dot (or empty), kept as-is so pre-release
    tags sort *before* the corresponding final.

    Raises
    ------
    :class:`ValueError`
        If ``text`` is not in ``MAJOR.MINOR.PATCH`` form. The check is
        deliberately strict so a malformed tag (a date stamp,
        ``"latest"``, etc.) is detected as garbage rather than silently
        compared as ``(0, 0, 0, ...)``.
    """
    stripped = text.strip().lstrip("vV")
    parts = stripped.split(".")
    if len(parts) < 3:
        raise ValueError(
            f"version string {text!r} is not MAJOR.MINOR.PATCH"
        )
    try:
        major = int(parts[0])
        minor = int(parts[1])
        # The patch field may carry a pre-release suffix (``5-rc1``).
        # Split once on the first ``-``; the right side becomes the
        # tuple's fourth element.
        patch_text = parts[2]
        if "-" in patch_text:
            patch_str, suffix = patch_text.split("-", 1)
        else:
            patch_str, suffix = patch_text, ""
        patch = int(patch_str)
    except ValueError as exc:
        raise ValueError(
            f"version string {text!r} has non-integer components"
        ) from exc
    # Anything beyond the third dot collapses into the suffix so callers
    # never see a tuple longer than four elements.
    if len(parts) > 3:
        suffix = ".".join(parts[3:]) + (f"-{suffix}" if suffix else "")
    return major, minor, patch, suffix


def compare_versions(current: str, latest: str) -> int:
    """Return -1/0/1 if ``current`` is older / equal / newer than ``latest``.

    Wraps :func:`parse_version` so callers don't have to think about
    parsing failures. A malformed string on either side surfaces as a
    :class:`ValueError`; the public :func:`check_for_updates` catches
    it and reports "no update".

    Pre-release ordering: a non-empty suffix is treated as *older*
    than an empty suffix at the same ``(major, minor, patch)`` so
    ``1.0.0-rc1`` compares as older than ``1.0.0``. Among non-empty
    suffixes the lexicographic order is preserved (``rc1 < rc2``).
    """
    cur_major, cur_minor, cur_patch, cur_suffix = parse_version(current)
    new_major, new_minor, new_patch, new_suffix = parse_version(latest)
    cur_core = (cur_major, cur_minor, cur_patch)
    new_core = (new_major, new_minor, new_patch)
    if cur_core != new_core:
        return -1 if cur_core < new_core else 1
    # Same numeric core — compare suffixes. Empty (final) beats any
    # non-empty (pre-release) suffix.
    if cur_suffix == new_suffix:
        return 0
    if cur_suffix == "":
        return 1  # current is final, latest is pre-release → current newer
    if new_suffix == "":
        return -1  # current is pre-release, latest is final → current older
    return -1 if cur_suffix < new_suffix else 1


def check_for_updates(
    *,
    repo: str = DEFAULT_GITHUB_REPO,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    current_version: str = __version__,
) -> UpdateAvailable | None:
    """Return an :class:`UpdateAvailable` if the repo has a newer release.

    Returns ``None`` for every failure path (no network, 404, 403,
    malformed JSON, malformed tag, version equal-or-older). The UI
    can therefore treat any non-``None`` result as "show the
    notification", with no need to inspect the failure shape.

    Parameters
    ----------
    repo:
        ``owner/name`` slug. Defaults to :data:`DEFAULT_GITHUB_REPO`.
    timeout_s:
        Request timeout. Defaults to :data:`DEFAULT_TIMEOUT_S`.
    current_version:
        The locally-running version. Defaults to
        :data:`jarvis.__version__`. Tests pass an explicit value to
        exercise the comparison without monkey-patching the package.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    # The GitHub API requires a User-Agent on every request. Use the
    # project name + version so rate-limit reports can identify the
    # client population.
    headers = {
        "User-Agent": f"jarvis-app/{current_version}",
        "Accept": "application/vnd.github+json",
    }
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - URL is hard-coded above
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 404 = repo has no releases yet (newly published project).
        # 403 = rate limit hit. Either way: silently skip this run.
        logger.info(
            "Update check skipped (HTTP %d) on %s",
            exc.code,
            repo,
        )
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.info("Update check skipped (network error: %s)", exc)
        return None
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Update check skipped (malformed JSON: %s)", exc)
        return None

    try:
        tag_name = str(payload["tag_name"])
        html_url = str(payload.get("html_url") or f"https://github.com/{repo}/releases")
        body = str(payload.get("body") or "")
    except (KeyError, TypeError) as exc:
        logger.warning("Update check skipped (unexpected payload shape: %s)", exc)
        return None

    # Pull the .exe asset (if any). We pick the first asset whose name
    # ends with ``.exe`` — for Inno Setup releases there is exactly one,
    # named ``JARVIS-Setup-<version>.exe``. Future releases that attach
    # a portable .zip archive next to the installer will still match
    # the installer correctly.
    installer_url: str | None = None
    installer_size: int | None = None
    for asset in payload.get("assets", []) or []:
        try:
            name = str(asset.get("name") or "")
            if name.lower().endswith(".exe"):
                installer_url = str(asset.get("browser_download_url") or "")
                size = asset.get("size")
                installer_size = int(size) if isinstance(size, int) else None
                if installer_url:
                    break
        except (TypeError, ValueError):
            continue

    try:
        cmp = compare_versions(current_version, tag_name)
    except ValueError as exc:
        logger.warning(
            "Update check skipped (cannot compare versions %r vs %r: %s)",
            current_version,
            tag_name,
            exc,
        )
        return None

    if cmp >= 0:
        logger.info(
            "Update check: running %s, latest %s — up to date",
            current_version,
            tag_name,
        )
        return None

    info = ReleaseInfo(
        version=tag_name.lstrip("vV"),
        tag_name=tag_name,
        html_url=html_url,
        body=body,
        installer_url=installer_url,
        installer_size=installer_size,
    )
    logger.info(
        "Update available: %s → %s (page: %s, asset: %s)",
        current_version,
        info.tag_name,
        info.html_url,
        info.installer_url or "<no .exe>",
    )
    return UpdateAvailable(current=current_version, latest=info)



# ---------------------------------------------------------------------------
# Auto-install
# ---------------------------------------------------------------------------


#: Inno Setup silent flags.
#:
#: * ``/VERYSILENT`` — no progress dialog or any UI.
#: * ``/SUPPRESSMSGBOXES`` — auto-accepts every prompt the installer
#:   would otherwise show (e.g., overwrite confirmations).
#: * ``/CLOSEAPPLICATIONS`` — sends WM_CLOSE to apps using files the
#:   installer needs to overwrite. Inno Setup falls back to
#:   ``/RESTARTAPPLICATIONS`` if the user doesn't see them quit.
#: * ``/RESTARTAPPLICATIONS`` — relaunches the closed apps after install.
#:   Together with ``[Run] postinstall`` this gets us a clean restart.
#: * ``/NORESTART`` — never reboot the OS, even if requested by the
#:   installer (we never request reboots, but defensive).
#: * ``/LOG=...`` — tee the installer log to our own logs dir for
#:   post-mortem.
_SILENT_FLAGS: Final[tuple[str, ...]] = (
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/CLOSEAPPLICATIONS",
    "/RESTARTAPPLICATIONS",
    "/NORESTART",
)


def _download_dir() -> Path:
    """Return the per-user directory we drop downloaded installers into.

    Lives under ``%LOCALAPPDATA%\\Jarvis\\updates`` so it survives app
    restarts and is easy for the user to clear manually if a download
    gets stuck.
    """
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis" / "updates"
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_and_run_installer(
    info: ReleaseInfo,
    *,
    progress: object | None = None,
    timeout_s: float = 30.0,
) -> bool:
    """Download the installer for ``info`` and launch it silently.

    Returns ``True`` on success. The function never returns when it
    succeeds in launching the installer — the call site is expected
    to ``sys.exit(0)`` after a ``True`` so the running JARVIS goes
    away and the installer can overwrite the binaries. We document
    this contract here rather than calling ``sys.exit`` ourselves so
    the GUI thread has a chance to clean up Tk widgets first.

    Returns ``False`` for every failure path (no asset, network
    error, disk full, installer path missing, etc.). The caller
    should surface a generic toast and stay on the running version.

    Parameters
    ----------
    info:
        The :class:`ReleaseInfo` from :func:`check_for_updates`. Must
        have a non-``None`` ``installer_url``.
    progress:
        Optional callable invoked with ``(downloaded_bytes,
        total_bytes_or_None)`` each chunk. Use it to feed a UI
        progress bar. ``total`` is ``None`` when the server didn't
        send a Content-Length header.
    timeout_s:
        Per-request connect / read timeout. Defaults to 30 s — the
        download itself can take much longer; this only bounds how
        long we wait between bytes.
    """
    if not info.installer_url:
        logger.warning("Auto-install aborted: release %s has no .exe asset", info.tag_name)
        return False

    dest_dir = _download_dir()
    final_name = f"JARVIS-Setup-{info.version}.exe"
    final_path = dest_dir / final_name
    partial_path = dest_dir / (final_name + ".partial")

    # If a previous attempt left a half-finished file, clear it.
    for p in (partial_path, final_path):
        if p.is_file():
            try:
                p.unlink()
            except OSError as exc:
                logger.warning("Could not remove old %s: %s", p, exc)

    headers = {
        "User-Agent": f"jarvis-app/{__version__}",
        "Accept": "application/octet-stream",
    }
    request = urllib.request.Request(info.installer_url, headers=headers)  # noqa: S310 - URL came from GitHub API

    logger.info(
        "Downloading installer %s → %s",
        info.installer_url,
        partial_path,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            total_header = response.getheader("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KiB chunks; balances syscall count vs progress granularity

            with partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if callable(progress):
                        try:
                            progress(downloaded, total)  # type: ignore[misc]
                        except Exception:  # pragma: no cover - logged for diagnostics
                            logger.exception("Progress callback raised; ignoring")
    except urllib.error.HTTPError as exc:
        logger.error("Installer download failed (HTTP %d)", exc.code)
        _safe_unlink(partial_path)
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("Installer download failed (network/IO error: %s)", exc)
        _safe_unlink(partial_path)
        return False

    # Atomic on NTFS — once this succeeds the file is "ready to launch".
    try:
        partial_path.replace(final_path)
    except OSError as exc:
        logger.error("Could not rename %s to %s: %s", partial_path, final_path, exc)
        _safe_unlink(partial_path)
        return False

    logger.info(
        "Downloaded %d bytes; launching installer %s",
        downloaded,
        final_path,
    )

    log_target = dest_dir / f"install-{info.version}.log"
    try:
        # ``CREATE_NEW_PROCESS_GROUP`` + ``DETACHED_PROCESS`` (combined
        # value 0x208) so the installer survives JARVIS exiting in the
        # next breath. Without this the installer is a child of JARVIS
        # and its file handles get torn down when our process group
        # signals exit.
        creationflags = 0
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008  # noqa: N806 - Windows constant
            CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806 - Windows constant
            creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

        subprocess.Popen(  # noqa: S603 - launching an installer we just downloaded over HTTPS
            [str(final_path), *_SILENT_FLAGS, f"/LOG={log_target}"],
            creationflags=creationflags,
            close_fds=True,
        )
    except OSError as exc:
        logger.error("Failed to launch installer: %s", exc)
        return False

    logger.info(
        "Installer launched detached; the app should restart "
        "automatically once the upgrade finishes."
    )
    return True


def _safe_unlink(path: Path) -> None:
    """Best-effort delete; swallows OSError so cleanup never crashes."""
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover - logged for diagnostics
        logger.warning("Could not remove %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Test hook: shutil reference kept so monkeypatching works in unit tests.
# ---------------------------------------------------------------------------
_ = shutil  # noqa: F401 - reserved for future asset-verification hooks
_ = tempfile  # noqa: F401 - reserved for future temp-file fallbacks
