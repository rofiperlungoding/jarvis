"""Check GitHub Releases for a newer JARVIS build.

Design
------

The desktop app pings the GitHub Releases API on startup (background,
non-blocking) and compares the latest release tag to the bundled
:data:`jarvis.__version__`. If a newer version is available, the UI
surfaces a non-modal notification linking to the release page.

Why GitHub Releases
~~~~~~~~~~~~~~~~~~~

* Already paid for by the project's release CI workflow — no extra
  hosting, no S3 bucket, no signing infrastructure beyond what
  GitHub already provides.
* The unauthenticated REST endpoint allows 60 requests per IP per
  hour, far above any plausible startup-check budget.
* Tags follow PEP 440-compatible semver (``vMAJOR.MINOR.PATCH``),
  which keeps comparison logic trivial.

Why "notify, don't auto-install"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto-downloading and silently launching a 270 MB ``Setup.exe`` is a
UX antipattern: SmartScreen blocks unsigned installers by default,
the user loses control over when the upgrade happens (which is
disruptive for a voice assistant mid-conversation), and a partial
download leaves the previous installation in an undefined state.
A notification with a one-click open-release-page button gives the
user the same convenience without any of the failure modes.

Failure handling
~~~~~~~~~~~~~~~~

Every failure path returns ``None`` rather than raising:

* No network — ``urllib.error.URLError``.
* Repository not yet published — HTTP 404.
* Rate-limited — HTTP 403.
* Malformed release JSON — ``KeyError`` / ``ValueError``.

The UI must treat ``None`` as "no update available" and proceed
silently. This is critical because the check runs on the worker
thread during boot; a raised exception would surface as a red
error bubble and frighten users about a non-critical outcome.

Type stripping
~~~~~~~~~~~~~~

The tag prefix ``v`` is stripped before comparison, so both
``v1.2.0`` and ``1.2.0`` (the Inno Setup ``MyAppVersion`` style)
parse to the same tuple. Only PEP 440 ``MAJOR.MINOR.PATCH`` is
recognised; pre-release / build-metadata suffixes (``-rc1``,
``+sha1234``) are tolerated as the third component but compared as
strings, so ``1.0.0-rc1`` sorts *before* ``1.0.0`` (rc1 < empty
string lexicographically — fine for our use case where pre-release
tags should never beat a final).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    )
    logger.info(
        "Update available: %s → %s (page: %s)",
        current_version,
        info.tag_name,
        info.html_url,
    )
    return UpdateAvailable(current=current_version, latest=info)
