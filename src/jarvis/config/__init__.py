"""TOML config loader and resolver for the JARVIS AI Assistant.

This module is the single public entry point for loading the application
configuration. It:

1. Reads the package-shipped ``default.toml`` (next to this module) as the
   baseline.
2. Optionally deep-merges a user override TOML — either an explicit
   :class:`~pathlib.Path` supplied by the caller, or the well-known
   ``%APPDATA%/Jarvis/config.toml`` location when no path is given.
3. Expands Windows-style environment variables (``%APPDATA%``,
   ``%LOCALAPPDATA%``, ``%USERPROFILE%``, ``%USERNAME%``, and any other
   ``%NAME%`` token) in every string value of the merged tree.
4. Resolves ``${dotted.path}`` references — most importantly
   ``${app.data_dir}`` — against the (env-expanded) merged tree.
5. Validates the result through the pydantic :class:`Config` model in
   :mod:`jarvis.config.schema`.

The design intentionally separates these stages so each can be unit-tested
in isolation (task 2.3): :func:`deep_merge` is pure dict manipulation,
:func:`expand_environment_variables` and :func:`resolve_references` are
pure string transforms, and :func:`load_config` is the orchestrator.

Requirement traceability
------------------------

* Requirement 15.1 — Windows is the primary target; the loader resolves
  ``%APPDATA%`` based config locations and Windows-style environment
  variable tokens.
* Requirement 15.2 — Platform-specific path conventions are confined to
  this loader; the rest of the codebase consumes the validated
  :class:`Config` object only.
"""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import re
import tomllib
from typing import Any, cast

from jarvis.config.schema import Config

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "deep_merge",
    "expand_environment_variables",
    "load_config",
    "load_default_config_dict",
    "resolve_references",
    "user_config_path",
]


# ---------------------------------------------------------------------------
# Constants and compiled patterns
# ---------------------------------------------------------------------------

#: File name used both for the shipped default and the user override file.
DEFAULT_CONFIG_FILENAME = "config.toml"

# Internal name of the resource shipped inside the package.
_DEFAULT_RESOURCE_NAME = "default.toml"

# ``%NAME%`` — Windows-style environment variable reference. The character
# class mirrors the legal identifier shape accepted by ``cmd.exe`` and Python's
# :func:`os.path.expandvars` on Windows. We restrict to ASCII identifiers to
# avoid accidentally consuming literal percent signs around non-variable
# substrings.
_ENV_VAR_PATTERN = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")

# ``${dotted.path}`` — config-internal reference. Limited to a dotted
# identifier path so we never confuse it with shell-style ``${VAR:-default}``
# expressions, which TOML configuration is not expected to contain.
_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")


# ---------------------------------------------------------------------------
# Default config loading
# ---------------------------------------------------------------------------


def load_default_config_dict() -> dict[str, Any]:
    """Return the package-shipped ``default.toml`` parsed as a plain dict.

    The default TOML is bundled in the wheel via ``pyproject.toml``'s
    ``[tool.setuptools.package-data]`` entry. We use
    :func:`importlib.resources.files` so the loader works equally well from
    a source checkout, an installed wheel, and a zip-imported environment.
    """
    package_files = resources.files(__package__)
    text = package_files.joinpath(_DEFAULT_RESOURCE_NAME).read_text(
        encoding="utf-8"
    )
    return tomllib.loads(text)


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Merge semantics:

    * Nested tables (``dict`` values) are merged key-by-key so a user
      override for one nested key does not wipe out the sibling defaults
      (e.g., overriding ``voice.tts.voice`` keeps the default
      ``voice.tts.engine``).
    * Scalars and lists from ``override`` replace whatever was in ``base``.
      Lists are intentionally *replaced* rather than concatenated; merging
      lists by position or by value would produce surprising results
      (e.g., the ``destructive_skills`` allowlist must remain authoritative
      when the user overrides it).
    * Neither input is mutated; the returned dict is a fresh structure
      built by shallow-copying the base table at each level. Leaf values
      are passed through by reference, which is safe because the caller
      is expected to operate on the result as immutable data downstream.

    The function tolerates non-dict ``override`` shapes for nested keys
    only when the override explicitly intends to *replace* the entire
    sub-tree (matching the semantics above).
    """
    if not isinstance(base, dict):
        raise TypeError(f"deep_merge: 'base' must be a dict, got {type(base).__name__}")
    if not isinstance(override, dict):
        raise TypeError(
            f"deep_merge: 'override' must be a dict, got {type(override).__name__}"
        )

    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(existing, override_value)
        else:
            merged[key] = override_value
    return merged


# ---------------------------------------------------------------------------
# String expansion / reference resolution
# ---------------------------------------------------------------------------


def expand_environment_variables(value: str) -> str:
    """Expand ``%NAME%`` Windows-style environment variables in ``value``.

    Tokens whose variable is not present in :data:`os.environ` are left
    untouched. This is deliberate: keeping the original token in the
    rendered config makes misconfiguration easy to diagnose (the user sees
    the literal ``%APPDATA%`` in an error message rather than an empty
    string), and the downstream pydantic validators will still reject any
    placeholder that escapes into a path field requiring real content.

    The implementation is platform-neutral: it does not delegate to
    :func:`os.path.expandvars`, which on POSIX systems silently ignores
    the percent-style tokens that Windows users embed in their TOML.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return os.environ.get(name, match.group(0))

    return _ENV_VAR_PATTERN.sub(replace, value)


def resolve_references(value: str, root: dict[str, Any]) -> str:
    """Resolve ``${dotted.path}`` references in ``value`` against ``root``.

    The dotted path is walked through nested dicts in ``root``. References
    that fail to resolve (missing key, non-string leaf, traversal through a
    non-dict) are left as the original ``${...}`` token so callers see the
    misconfiguration during validation rather than during a quiet path-join
    much later in the pipeline.

    Resolution is single-pass: the substituted text is *not* re-scanned for
    further references. Configurations like ``${a.b}`` -> ``"${c.d}"`` are
    therefore not chained. This is by design — chained references make
    cycle detection necessary and the documented requirement only covers
    ``${app.data_dir}``-style direct lookups.
    """

    def replace(match: re.Match[str]) -> str:
        dotted = match.group(1)
        node: Any = root
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return match.group(0)
        # Only string leaves are substitutable; numeric/bool/list leaves
        # would corrupt the surrounding string template.
        if isinstance(node, str):
            return node
        return match.group(0)

    return _REF_PATTERN.sub(replace, value)


def _walk_strings(value: Any, transform: Any) -> Any:
    """Apply ``transform`` to every string leaf inside ``value``.

    Lists and dicts are reconstructed to avoid mutating the input. Non-string
    scalars (ints, floats, bools, ``None``) are returned unchanged.
    """
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {k: _walk_strings(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_strings(v, transform) for v in value]
    return value


def _expand_tree(merged: dict[str, Any]) -> dict[str, Any]:
    """Run the two-pass expansion described in the module docstring.

    Pass 1 expands ``%NAME%`` environment variables across the entire
    tree. This must happen before pass 2 so that ``app.data_dir`` — which
    itself contains ``%LOCALAPPDATA%`` — has been resolved to a real path
    by the time other strings reference it via ``${app.data_dir}``.

    Pass 2 walks the env-expanded tree again and resolves
    ``${dotted.path}`` references. The same env-expanded tree is passed
    in as the lookup root so references see the resolved values rather
    than the original templates.
    """
    env_expanded = _walk_strings(merged, expand_environment_variables)
    if not isinstance(env_expanded, dict):
        # Defensive: _walk_strings preserves shape, so this is unreachable
        # given a dict input, but the type checker cannot prove it.
        raise TypeError("merged config must be a table at the root")
    resolved = _walk_strings(
        env_expanded,
        lambda s: resolve_references(s, env_expanded),
    )
    # ``_walk_strings`` returns a recursively reconstructed dict here
    # because the input is a dict; ``cast`` simply tells mypy what the
    # body's invariants already guarantee.
    return cast("dict[str, Any]", resolved)


# ---------------------------------------------------------------------------
# User override path resolution
# ---------------------------------------------------------------------------


def user_config_path() -> Path | None:
    """Return the conventional user override path, or ``None`` if APPDATA is unset.

    On Windows ``%APPDATA%`` always resolves to the user's roaming AppData
    directory. On non-Windows hosts (test runners, Linux dev shells) the
    ``APPDATA`` environment variable may be unset; in that case we return
    ``None`` so :func:`load_config` falls through to defaults rather than
    raising an OS-specific path error.
    """
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Jarvis" / DEFAULT_CONFIG_FILENAME


def _read_toml_file(path: Path) -> dict[str, Any]:
    """Parse a TOML file from disk.

    Wrapped in a helper so :func:`load_config` keeps a single place where
    file IO happens, simplifying future additions (e.g., schema-version
    migration on read).
    """
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> Config:
    """Load and validate the JARVIS configuration.

    Parameters
    ----------
    path:
        Optional explicit path to a user override TOML.

        * ``None`` (the default) — the loader looks for
          ``%APPDATA%/Jarvis/config.toml``. If the file is missing, the
          shipped defaults are used as-is. This is the normal first-run
          behaviour.
        * A :class:`~pathlib.Path` — the loader requires the file to
          exist. Passing an explicit path signals intent to override, so
          a missing file is treated as a configuration error
          (:class:`FileNotFoundError`).

    Returns
    -------
    :class:`Config`
        The validated, fully-resolved configuration.

    Raises
    ------
    FileNotFoundError
        When ``path`` is given but does not point to an existing file.
    tomllib.TOMLDecodeError
        When either the default or the user TOML fails to parse.
    pydantic.ValidationError
        When the merged configuration violates the schema rules in
        :mod:`jarvis.config.schema`.
    """
    base = load_default_config_dict()

    user_data: dict[str, Any]
    if path is None:
        # Implicit lookup: silently fall through to defaults if the
        # well-known location does not exist.
        candidate = user_config_path()
        if candidate is not None and candidate.is_file():
            user_data = _read_toml_file(candidate)
        else:
            user_data = {}
    else:
        # Explicit path: caller expects this file to exist.
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        user_data = _read_toml_file(path)

    merged = deep_merge(base, user_data)
    resolved = _expand_tree(merged)
    return Config.model_validate(resolved)
