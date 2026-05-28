"""Unit tests for ``jarvis.config.schema``.

Covers the validation rules called out in section "Configuration Validation
Rules" of ``design.md``:

* Default :class:`Config` instantiates cleanly with sensible defaults
  (general robustness baseline behind Requirements 1.3 / 1.8).
* Overrides land correctly at every documented section.
* ``voice.stt.local_only=true`` + ``engine="cloud"`` is rejected
  (Requirement 13.2).
* ``memory.top_k`` outside ``[1, 50]`` is rejected (Requirement 10.3).
* ``reminders.on_start_grace_seconds < 30`` is rejected (Requirement 6.6).
* Empty ``automation.allowed_directories.paths`` is rejected
  (Requirements 8.2 / 8.6).
* Unknown top-level keys emit :class:`UnknownConfigKeyWarning` but the
  resulting :class:`Config` still loads with defaults for every known
  section.

Validates: Requirements 1.3, 1.8, 6.6, 8.2, 8.6, 10.3, 13.2
"""

from __future__ import annotations

from typing import Any
import warnings

from pydantic import ValidationError
import pytest

from jarvis.config.schema import (
    AllowedDirectoriesConfig,
    AppConfig,
    AuthorizationConfig,
    AutomationConfig,
    Config,
    DialogConfig,
    LlmConfig,
    LlmFallbackConfig,
    LlmMistralConfig,
    MemoryConfig,
    ProvidersConfig,
    RemindersConfig,
    SecurityConfig,
    SkillsConfig,
    TelemetryConfig,
    UnknownConfigKeyWarning,
    VoiceConfig,
    VoiceSttConfig,
    VoiceTtsConfig,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_config_instantiates_cleanly() -> None:
    """A bare :class:`Config()` call must succeed and populate every section."""
    cfg = Config()

    # Every documented top-level section is present and is the right type.
    assert isinstance(cfg.app, AppConfig)
    assert isinstance(cfg.voice, VoiceConfig)
    assert isinstance(cfg.dialog, DialogConfig)
    assert isinstance(cfg.llm, LlmConfig)
    assert isinstance(cfg.llm.mistral, LlmMistralConfig)
    assert isinstance(cfg.llm.fallback, LlmFallbackConfig)
    assert isinstance(cfg.memory, MemoryConfig)
    assert isinstance(cfg.reminders, RemindersConfig)
    assert isinstance(cfg.skills, SkillsConfig)
    assert isinstance(cfg.automation, AutomationConfig)
    assert isinstance(cfg.automation.allowed_directories, AllowedDirectoriesConfig)
    assert isinstance(cfg.providers, ProvidersConfig)
    assert isinstance(cfg.authorization, AuthorizationConfig)
    assert isinstance(cfg.security, SecurityConfig)
    assert isinstance(cfg.telemetry, TelemetryConfig)


def test_default_config_field_values_match_design_doc() -> None:
    """Critical defaults mirror the values documented in ``design.md``."""
    cfg = Config()

    # Requirement 1.8 — STT confidence floor.
    assert cfg.voice.stt.min_confidence == pytest.approx(0.4)
    # Requirement 13.2 — local-only STT is the default privacy-preserving mode.
    assert cfg.voice.stt.local_only is True
    assert cfg.voice.stt.engine == "faster_whisper"
    # Requirement 1.3 — VAD trailing-silence default.
    assert cfg.voice.vad.trailing_silence_ms == 700
    # Requirement 11.2 — JARVIS persona-matching voice.
    assert cfg.voice.tts.voice == "en_GB-alan-medium"
    # Requirement 10.3 — top_k retrieval default.
    assert cfg.memory.top_k == 5
    # Requirement 6.6 — reminders grace seconds floor and default.
    assert cfg.reminders.on_start_grace_seconds == 30
    # Requirement 12.3 — acknowledgement threshold.
    assert cfg.dialog.acknowledge_after_ms == 1500
    # Requirement 14.5 — schema retry cap.
    assert cfg.dialog.max_tool_retry == 2
    # Requirement 19.1 / 19.2 — Mistral defaults.
    assert cfg.llm.mistral.endpoint == "https://api.mistral.ai"
    assert cfg.llm.mistral.model == "mistral-large-latest"
    # Requirement 12.4 — fallback wired by default.
    assert cfg.llm.fallback.enabled is True
    assert cfg.llm.fallback.backend == "ollama"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_overrides_at_every_section_are_applied() -> None:
    """Each documented section accepts a partial override without altering siblings."""
    overrides: dict[str, dict[str, Any]] = {
        "app": {"log_level": "DEBUG", "incognito": True},
        "voice": {
            "tts": {"voice": "en_US-amy-medium", "speaking_rate": 1.2},
            "stt": {"min_confidence": 0.6, "language": "fr"},
        },
        "dialog": {"acknowledge_after_ms": 2000, "max_tool_retry": 5},
        "llm": {
            "mistral": {"model": "mistral-small-latest", "max_retries": 7},
            "fallback": {"enabled": False, "model": "llama3"},
        },
        "memory": {"top_k": 25, "redaction_enabled": False},
        "reminders": {"on_start_grace_seconds": 90, "toast_enabled": False},
        "automation": {
            "allowed_directories": {"paths": ["D:/Work", "D:/Notes"]},
        },
        "providers": {
            "weather": {"default_location": "Paris,FR", "timeout_seconds": 7.5},
            "search": {"max_results_default": 3, "max_results_cap": 8},
        },
        "authorization": {
            "destructive_skills": ["SendEmailSkill", "RunScriptSkill"],
        },
        "security": {
            "network_destination_allowlist": ["api.mistral.ai", "localhost"],
        },
        "telemetry": {"enabled": True, "crash_report_endpoint": "https://crash.example"},
    }

    cfg = Config.model_validate(overrides)

    # App
    assert cfg.app.log_level == "DEBUG"
    assert cfg.app.incognito is True
    # Voice — the override at one sub-section must not blow away siblings.
    assert cfg.voice.tts.voice == "en_US-amy-medium"
    assert cfg.voice.tts.speaking_rate == pytest.approx(1.2)
    assert cfg.voice.tts.engine == "piper"  # default preserved
    assert cfg.voice.stt.min_confidence == pytest.approx(0.6)
    assert cfg.voice.stt.language == "fr"
    assert cfg.voice.stt.engine == "faster_whisper"  # default preserved
    assert cfg.voice.vad.trailing_silence_ms == 700  # untouched section default
    # Dialog
    assert cfg.dialog.acknowledge_after_ms == 2000
    assert cfg.dialog.max_tool_retry == 5
    # LLM
    assert cfg.llm.mistral.model == "mistral-small-latest"
    assert cfg.llm.mistral.max_retries == 7
    assert cfg.llm.fallback.enabled is False
    assert cfg.llm.fallback.model == "llama3"
    # Memory
    assert cfg.memory.top_k == 25
    assert cfg.memory.redaction_enabled is False
    # Reminders
    assert cfg.reminders.on_start_grace_seconds == 90
    assert cfg.reminders.toast_enabled is False
    # Automation
    assert cfg.automation.allowed_directories.paths == ["D:/Work", "D:/Notes"]
    # Providers
    assert cfg.providers.weather.default_location == "Paris,FR"
    assert cfg.providers.weather.timeout_seconds == pytest.approx(7.5)
    assert cfg.providers.search.max_results_default == 3
    assert cfg.providers.search.max_results_cap == 8
    # Authorization
    assert cfg.authorization.destructive_skills == ["SendEmailSkill", "RunScriptSkill"]
    # Security
    assert cfg.security.network_destination_allowlist == ["api.mistral.ai", "localhost"]
    # Telemetry
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.crash_report_endpoint == "https://crash.example"


def test_voice_tts_voice_override_round_trips() -> None:
    """Spot-check ``voice.tts.voice`` per the task's explicit example."""
    cfg = Config.model_validate({"voice": {"tts": {"voice": "en_GB-jenny-medium"}}})
    assert cfg.voice.tts.voice == "en_GB-jenny-medium"


# ---------------------------------------------------------------------------
# Requirement 13.2 — local_only blocks cloud STT
# ---------------------------------------------------------------------------


def test_local_only_with_cloud_engine_raises_validation_error() -> None:
    """``voice.stt.local_only=true`` must veto any cloud STT engine."""
    with pytest.raises(ValidationError) as exc:
        VoiceSttConfig(local_only=True, engine="cloud")
    # The error message should make the conflict obvious to the user.
    assert "local_only" in str(exc.value)


def test_local_only_with_cloud_engine_at_top_level_raises() -> None:
    """The same rule applies when set via the top-level :class:`Config`."""
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"voice": {"stt": {"local_only": True, "engine": "cloud"}}}
        )


def test_local_only_false_allows_cloud_engine() -> None:
    """Opting out of ``local_only`` enables the cloud engine."""
    stt = VoiceSttConfig(local_only=False, engine="cloud")
    assert stt.local_only is False
    assert stt.engine == "cloud"


def test_local_only_true_with_local_engine_is_allowed() -> None:
    """The default combination remains valid."""
    stt = VoiceSttConfig(local_only=True, engine="faster_whisper")
    assert stt.local_only is True
    assert stt.engine == "faster_whisper"


# ---------------------------------------------------------------------------
# Requirement 10.3 — memory.top_k bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_top_k", [0, -1, -100, 51, 1000])
def test_invalid_memory_top_k_is_rejected(bad_top_k: int) -> None:
    """``top_k`` outside the documented ``[1, 50]`` window is rejected."""
    with pytest.raises(ValidationError):
        MemoryConfig(top_k=bad_top_k)


@pytest.mark.parametrize("good_top_k", [1, 5, 25, 50])
def test_valid_memory_top_k_is_accepted(good_top_k: int) -> None:
    cfg = MemoryConfig(top_k=good_top_k)
    assert cfg.top_k == good_top_k


def test_invalid_memory_top_k_at_top_level_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"memory": {"top_k": 0}})


# ---------------------------------------------------------------------------
# Requirement 6.6 — reminders grace seconds floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_seconds", [0, 1, 15, 29, -5])
def test_sub_30_grace_seconds_is_rejected(bad_seconds: int) -> None:
    """Any value below 30 seconds violates Requirement 6.6."""
    with pytest.raises(ValidationError):
        RemindersConfig(on_start_grace_seconds=bad_seconds)


@pytest.mark.parametrize("good_seconds", [30, 31, 60, 600])
def test_grace_seconds_at_or_above_30_is_accepted(good_seconds: int) -> None:
    cfg = RemindersConfig(on_start_grace_seconds=good_seconds)
    assert cfg.on_start_grace_seconds == good_seconds


def test_sub_30_grace_seconds_at_top_level_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"reminders": {"on_start_grace_seconds": 10}})


# ---------------------------------------------------------------------------
# Requirements 8.2 / 8.6 — non-empty allowed-directories
# ---------------------------------------------------------------------------


def test_empty_allowed_directories_is_rejected() -> None:
    """An empty ``paths`` list breaks every file Skill and is rejected."""
    with pytest.raises(ValidationError) as exc:
        AllowedDirectoriesConfig(paths=[])
    assert "at least one path" in str(exc.value)


def test_blank_only_allowed_directories_is_rejected() -> None:
    """Whitespace-only entries do not satisfy the non-empty constraint."""
    with pytest.raises(ValidationError):
        AllowedDirectoriesConfig(paths=["", "   ", "\t"])


def test_blank_entries_are_stripped_when_at_least_one_real_path_remains() -> None:
    cfg = AllowedDirectoriesConfig(paths=["", "D:/Work", "   "])
    assert cfg.paths == ["D:/Work"]


def test_empty_allowed_directories_at_top_level_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"automation": {"allowed_directories": {"paths": []}}}
        )


# ---------------------------------------------------------------------------
# Unknown top-level keys — warning, not error
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_emits_warning_but_loads_defaults() -> None:
    """Unknown top-level keys must surface a warning yet still build a Config."""
    raw = {
        "totally_made_up_section": {"hello": "world"},
        "voice": {"tts": {"voice": "en_GB-alan-medium"}},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = Config.model_validate(raw)

    matches = [w for w in caught if issubclass(w.category, UnknownConfigKeyWarning)]
    assert len(matches) == 1, [str(w.message) for w in caught]
    assert "totally_made_up_section" in str(matches[0].message)

    # The known section was applied; defaults remain available everywhere else.
    assert cfg.voice.tts.voice == "en_GB-alan-medium"
    assert cfg.app.log_level == "INFO"
    assert cfg.memory.top_k == 5
    # The unknown key did NOT become an attribute on the Config model.
    assert not hasattr(cfg, "totally_made_up_section")


def test_multiple_unknown_keys_each_emit_a_warning() -> None:
    raw = {
        "alpha_section": {},
        "beta_section": {"x": 1},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Config.model_validate(raw)

    messages = sorted(
        str(w.message)
        for w in caught
        if issubclass(w.category, UnknownConfigKeyWarning)
    )
    assert len(messages) == 2
    assert any("alpha_section" in m for m in messages)
    assert any("beta_section" in m for m in messages)


def test_known_top_level_keys_do_not_emit_warnings() -> None:
    """Sanity check — a fully valid config produces zero unknown-key warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Config()

    matches = [w for w in caught if issubclass(w.category, UnknownConfigKeyWarning)]
    assert matches == []


# ---------------------------------------------------------------------------
# Bonus: a couple of section-level guards exercised at the top level
# ---------------------------------------------------------------------------


def test_voice_tts_speaking_rate_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        VoiceTtsConfig(speaking_rate=0.0)
    with pytest.raises(ValidationError):
        VoiceTtsConfig(speaking_rate=-0.5)


def test_mistral_endpoint_must_be_https_or_localhost() -> None:
    """Requirement 13.4 / 19.3 — secrets should never travel cleartext."""
    with pytest.raises(ValidationError):
        LlmMistralConfig(endpoint="http://api.example.com")
    # https and http://localhost remain acceptable.
    LlmMistralConfig(endpoint="https://api.mistral.ai")
    LlmMistralConfig(endpoint="http://localhost:8080")
