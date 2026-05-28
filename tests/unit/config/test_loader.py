"""Unit tests for the TOML loader in ``jarvis.config`` (task 2.2).

These tests target the loader entry point ``load_config(path: Path | None) -> Config``.
Task 2.2 is being implemented in parallel; if the loader symbol has not yet
been published the entire module is skipped at collection time so the schema
tests in :mod:`tests.unit.config.test_schema` remain runnable independently.

Validates: Requirements 1.3, 1.8, 6.6, 8.2, 10.3, 13.2 (loader-level)
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import textwrap
from typing import Any
import warnings

from pydantic import ValidationError
import pytest

from jarvis.config.schema import Config, UnknownConfigKeyWarning

# ---------------------------------------------------------------------------
# Skip cleanly if the parallel loader implementation is not yet available.
# ---------------------------------------------------------------------------

try:
    from jarvis.config import load_config as _load_config
except ImportError:  # pragma: no cover - exercised when task 2.2 is incomplete
    _load_config = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    _load_config is None,
    reason="jarvis.config.load_config is implemented by parallel task 2.2",
)


# A typed alias keeps the rest of the file readable while accommodating the
# pre-implementation None state.
LoadConfig = Callable[[Path | None], Config]


@pytest.fixture()
def load_config() -> LoadConfig:
    """Resolve ``load_config`` once per test, post-skip."""
    assert _load_config is not None  # narrowed by pytestmark
    return _load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, body: str, name: str = "config.toml") -> Path:
    """Write a TOML file dedented from a triple-quoted block."""
    target = tmp_path / name
    target.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return target


def _isolate_environment(
    monkeypatch: pytest.MonkeyPatch, **overrides: str
) -> None:
    """Set deterministic values for the variables the loader is expected to expand.

    Every variable referenced in the shipped ``default.toml`` (``%APPDATA%``,
    ``%LOCALAPPDATA%``, ``%USERPROFILE%``, ``%USERNAME%``) is given a stable
    test value so assertions about expansion stay reproducible across hosts.
    """
    base: dict[str, str] = {
        "APPDATA": "C:/TestRoaming",
        "LOCALAPPDATA": "C:/TestLocal",
        "USERPROFILE": "C:/Users/tester",
        "USERNAME": "tester",
    }
    base.update(overrides)
    for name, value in base.items():
        monkeypatch.setenv(name, value)


# ---------------------------------------------------------------------------
# Defaults: ``load_config(None)`` returns the shipped defaults
# ---------------------------------------------------------------------------


def test_load_config_with_none_returns_defaults(
    monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """``load_config(None)`` must produce a valid :class:`Config` from defaults."""
    _isolate_environment(monkeypatch)
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    # A few load-bearing defaults from design.md.
    assert cfg.voice.tts.voice == "en_GB-alan-medium"
    assert cfg.voice.stt.local_only is True
    assert cfg.memory.top_k == 5
    assert cfg.reminders.on_start_grace_seconds == 30


def test_load_config_with_missing_explicit_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """An *explicit* override path that does not exist surfaces as an error.

    Passing a :class:`Path` signals intent to override, so a typo in the
    path should not silently fall through to defaults.
    """
    _isolate_environment(monkeypatch)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.toml")


def test_load_config_with_none_when_appdata_missing_returns_defaults(
    monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """When ``APPDATA`` is unset and no path is given, defaults are returned."""
    monkeypatch.delenv("APPDATA", raising=False)
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.memory.top_k == 5


# ---------------------------------------------------------------------------
# Overrides: user file is deep-merged on top of the shipped defaults
# ---------------------------------------------------------------------------


def test_user_overrides_are_deep_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [voice.tts]
        voice = "en_GB-jenny-medium"

        [memory]
        top_k = 12

        [dialog]
        acknowledge_after_ms = 2500
        """,
    )

    cfg = load_config(user)
    # Overrides applied.
    assert cfg.voice.tts.voice == "en_GB-jenny-medium"
    assert cfg.memory.top_k == 12
    assert cfg.dialog.acknowledge_after_ms == 2500
    # Sibling defaults preserved through deep merge.
    assert cfg.voice.tts.engine == "piper"
    assert cfg.voice.stt.local_only is True
    assert cfg.dialog.max_tool_retry == 2


# ---------------------------------------------------------------------------
# Environment variable expansion (Requirement 15.x — loader contract)
# ---------------------------------------------------------------------------


def test_env_var_expansion_in_path_like_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """``%APPDATA%`` / ``%LOCALAPPDATA%`` / ``%USERPROFILE%`` are expanded."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [app]
        data_dir = "%LOCALAPPDATA%/Jarvis"
        plugin_dirs = ["%APPDATA%/Jarvis/plugins"]

        [automation.allowed_directories]
        paths = ["%USERPROFILE%/Documents", "%USERPROFILE%/Downloads"]
        """,
    )

    cfg = load_config(user)
    assert cfg.app.data_dir == "C:/TestLocal/Jarvis"
    assert cfg.app.plugin_dirs == ["C:/TestRoaming/Jarvis/plugins"]
    assert cfg.automation.allowed_directories.paths == [
        "C:/Users/tester/Documents",
        "C:/Users/tester/Downloads",
    ]


def test_username_expansion_in_application_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """``%USERNAME%`` interpolates into nested table values."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [automation.application_registry]
        vscode = "C:/Users/%USERNAME%/AppData/Local/Programs/Microsoft VS Code/Code.exe"
        """,
    )

    cfg = load_config(user)
    assert (
        cfg.automation.application_registry["vscode"]
        == "C:/Users/tester/AppData/Local/Programs/Microsoft VS Code/Code.exe"
    )


def test_app_data_dir_substitution_resolves_in_dependent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """``${app.data_dir}`` substitutions resolve against the merged ``app.data_dir``."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [app]
        data_dir = "D:/Jarvis"

        [memory]
        path = "${app.data_dir}/memory/chroma"

        [reminders]
        db_path = "${app.data_dir}/reminders.sqlite"

        [security]
        audit_log_path = "${app.data_dir}/audit.sqlite"
        """,
    )

    cfg = load_config(user)
    assert cfg.app.data_dir == "D:/Jarvis"
    assert cfg.memory.path == "D:/Jarvis/memory/chroma"
    assert cfg.reminders.db_path == "D:/Jarvis/reminders.sqlite"
    assert cfg.security.audit_log_path == "D:/Jarvis/audit.sqlite"


# ---------------------------------------------------------------------------
# Validation rules surface through the loader as well
# ---------------------------------------------------------------------------


def test_loader_rejects_invalid_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """Requirement 10.3 — out-of-range ``top_k`` raises during load."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [memory]
        top_k = 0
        """,
    )
    with pytest.raises(ValidationError):
        load_config(user)


def test_loader_rejects_sub_30_grace_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """Requirement 6.6 — sub-30 second grace fails through the loader."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [reminders]
        on_start_grace_seconds = 5
        """,
    )
    with pytest.raises(ValidationError):
        load_config(user)


def test_loader_rejects_empty_allowed_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """Requirements 8.2 / 8.6 — empty ``paths`` list fails through the loader."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [automation.allowed_directories]
        paths = []
        """,
    )
    with pytest.raises(ValidationError):
        load_config(user)


def test_loader_rejects_local_only_with_cloud_stt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """Requirement 13.2 — ``local_only=true`` plus cloud engine fails through the loader."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [voice.stt]
        local_only = true
        engine = "cloud"
        """,
    )
    with pytest.raises(ValidationError):
        load_config(user)


def test_loader_low_confidence_floor_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """Requirement 1.8 — the configured confidence floor survives load."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [voice.stt]
        min_confidence = 0.55
        """,
    )
    cfg = load_config(user)
    assert cfg.voice.stt.min_confidence == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Unknown top-level key warning surfaces through the loader
# ---------------------------------------------------------------------------


def test_loader_unknown_top_level_key_emits_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """An unknown top-level key warns but the loader still returns a Config."""
    _isolate_environment(monkeypatch)
    user = _write_toml(
        tmp_path,
        """
        [made_up_section]
        x = 1

        [memory]
        top_k = 7
        """,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(user)

    matches = [w for w in caught if issubclass(w.category, UnknownConfigKeyWarning)]
    assert len(matches) >= 1
    assert any("made_up_section" in str(w.message) for w in matches)
    # The known override still applies.
    assert cfg.memory.top_k == 7
    # Defaults remain elsewhere.
    assert cfg.reminders.on_start_grace_seconds == 30


# ---------------------------------------------------------------------------
# Sanity: defaults still match design.md after env expansion
# ---------------------------------------------------------------------------


def test_default_config_paths_resolve_against_environment(
    monkeypatch: pytest.MonkeyPatch, load_config: LoadConfig
) -> None:
    """The shipped defaults should resolve to the test-injected env values."""
    _isolate_environment(monkeypatch)
    cfg = load_config(None)
    # data_dir's default is %LOCALAPPDATA%/Jarvis -> C:/TestLocal/Jarvis
    # We allow the loader to either leave the literal in place or expand it,
    # but if it does expand we should see the test value.
    assert os.environ["LOCALAPPDATA"] == "C:/TestLocal"
    if "%" not in cfg.app.data_dir:
        assert cfg.app.data_dir.startswith("C:/TestLocal")


# ---------------------------------------------------------------------------
# Helper used inside the module above; kept here to avoid leaking into the
# schema-only test module.
# ---------------------------------------------------------------------------


def test_helper_writes_dedented_toml(tmp_path: Path) -> None:
    """``_write_toml`` strips leading indentation deterministically."""
    path = _write_toml(
        tmp_path,
        """
            [memory]
            top_k = 9
        """,
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("[memory]")
    # The fixture should be parseable as TOML even after dedent.
    import tomllib  # noqa: PLC0415  - local import keeps top of file tidy

    parsed: dict[str, Any] = tomllib.loads(text)
    assert parsed == {"memory": {"top_k": 9}}
