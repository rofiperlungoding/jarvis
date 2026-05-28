"""Pydantic config models for the JARVIS AI Assistant.

This module defines the :class:`Config` pydantic v2 model that mirrors the
TOML structure described in ``design.md`` (``%APPDATA%/Jarvis/config.toml``)
and encodes the validation rules called out in section "Configuration
Validation Rules" of the design.

The model is intentionally pure-data: it does NOT load files, expand
environment variables, or substitute ``${app.data_dir}``-style references.
That work belongs in the TOML loader (task 2.2 / ``src/jarvis/config/__init__.py``).

Validation rules implemented here:

* ``voice.stt.local_only=true`` blocks cloud STT engines at startup
  (Requirement 13.2).
* ``memory.top_k`` is constrained to ``[1, 50]`` (Requirement 10.3).
* ``reminders.on_start_grace_seconds >= 30`` (Requirement 6.6 floor).
* ``automation.allowed_directories.paths`` must contain at least one path
  (Requirements 8.2 / 8.6).
* Unknown top-level keys trigger an :class:`UnknownConfigKeyWarning` rather
  than silently no-op'ing.

Requirement IDs referenced in field comments map directly to the acceptance
criteria in ``requirements.md``.
"""

from __future__ import annotations

from typing import Any, Literal
import warnings

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "AllowedDirectoriesConfig",
    "AppConfig",
    "AuthorizationConfig",
    "AutomationConfig",
    "Config",
    "DestructiveOperation",
    "DialogConfig",
    "LlmConfig",
    "LlmFallbackConfig",
    "LlmMistralConfig",
    "McpServerConfig",
    "MemoryConfig",
    "ProvidersCalendarConfig",
    "ProvidersConfig",
    "ProvidersEmailConfig",
    "ProvidersNewsConfig",
    "ProvidersSearchConfig",
    "ProvidersWeatherConfig",
    "RemindersConfig",
    "ScriptCatalogEntry",
    "SecurityConfig",
    "SkillsConfig",
    "TelemetryConfig",
    "TrustedAction",
    "UnknownConfigKeyWarning",
    "VoiceAudioConfig",
    "VoiceConfig",
    "VoiceSttConfig",
    "VoiceTtsConfig",
    "VoiceVadConfig",
    "VoiceWakeWordConfig",
]


# ---------------------------------------------------------------------------
# Shared base + helpers
# ---------------------------------------------------------------------------


class UnknownConfigKeyWarning(UserWarning):
    """Emitted when an unknown top-level config key is encountered.

    The design's "Configuration Validation Rules" section specifies that
    user-provided overrides at unknown top-level keys must surface a warning
    so misconfiguration does not silently no-op. Using a dedicated subclass
    of :class:`UserWarning` lets callers and tests filter precisely.
    """


class _Section(BaseModel):
    """Base model for nested config sections.

    ``extra="ignore"`` mirrors TOML's permissive nature for sub-sections;
    the top-level :class:`Config` is the only place we surface a warning
    for unknown keys, per the design rules.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        str_strip_whitespace=False,
    )


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# ---------------------------------------------------------------------------
# [app]
# ---------------------------------------------------------------------------


class AppConfig(_Section):
    """``[app]`` section."""

    data_dir: str = "%LOCALAPPDATA%/Jarvis"
    plugin_dirs: list[str] = Field(
        default_factory=lambda: ["%APPDATA%/Jarvis/plugins"]
    )
    log_level: LogLevel = "INFO"
    # Requirement 13.3: when true, the Dialog_Manager must not persist any
    # Memory_Record from the current session.
    incognito: bool = False


# ---------------------------------------------------------------------------
# [voice.*]
# ---------------------------------------------------------------------------


class VoiceWakeWordConfig(_Section):
    """``[voice.wake_word]`` section.

    Requirement 18.1 — the user can configure a custom wake phrase, including
    a custom Porcupine ``.ppn`` keyword file by absolute path.
    """

    engine: Literal["porcupine"] = "porcupine"
    phrase: str = "jarvis"
    custom_keyword_path: str = ""
    sensitivity: float = Field(0.55, ge=0.0, le=1.0)
    access_key_credential: str = "porcupine/access_key"


class VoiceAudioConfig(_Section):
    """``[voice.audio]`` section."""

    input_device: str = ""
    output_device: str = ""
    sample_rate_hz: int = Field(16000, ge=8000)
    frame_ms: int = Field(30, ge=10, le=120)


class VoiceVadConfig(_Section):
    """``[voice.vad]`` section.

    Requirement 1.3 — the trailing-silence threshold defaults to 700 ms and
    is the configurable knob that drives end-of-utterance detection.
    """

    engine: Literal["silero"] = "silero"
    trailing_silence_ms: int = Field(700, ge=100)
    speech_start_threshold: float = Field(0.5, ge=0.0, le=1.0)


class VoiceSttConfig(_Section):
    """``[voice.stt]`` section.

    Requirement 1.8 — ``min_confidence`` gates low-confidence transcripts
    away from the LLM_Backend.
    Requirement 13.2 — when ``local_only`` is true, cloud STT engines must
    be rejected at startup; the Voice_Pipeline must not transmit raw audio
    to any cloud service.
    """

    engine: Literal["faster_whisper", "cloud"] = "faster_whisper"
    model: str = "small.en"
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    min_confidence: float = Field(0.4, ge=0.0, le=1.0)
    local_only: bool = True

    @model_validator(mode="after")
    def _local_only_blocks_cloud_stt(self) -> VoiceSttConfig:
        # Requirement 13.2: a true local_only flag must veto cloud engines.
        # The current cloud engine alias is "cloud"; future cloud engines
        # should also be added to this exclusion list.
        cloud_engines: frozenset[str] = frozenset({"cloud"})
        if self.local_only and self.engine in cloud_engines:
            raise ValueError(
                "voice.stt.local_only=true forbids cloud STT engines "
                f"(got engine={self.engine!r}); set voice.stt.local_only=false "
                "to opt into a cloud engine."
            )
        return self


class VoiceTtsConfig(_Section):
    """``[voice.tts]`` section.

    Requirement 1.7 — barge-in toggle.
    Requirement 11.2 — JARVIS persona-matching default voice.
    """

    engine: Literal["piper", "elevenlabs", "openai"] = "piper"
    voice: str = "en_GB-alan-medium"
    speaking_rate: float = Field(1.0, gt=0.0)
    barge_in_enabled: bool = True


class VoiceConfig(_Section):
    """``[voice]`` aggregate section."""

    wake_word: VoiceWakeWordConfig = Field(default_factory=VoiceWakeWordConfig)
    audio: VoiceAudioConfig = Field(default_factory=VoiceAudioConfig)
    vad: VoiceVadConfig = Field(default_factory=VoiceVadConfig)
    stt: VoiceSttConfig = Field(default_factory=VoiceSttConfig)
    tts: VoiceTtsConfig = Field(default_factory=VoiceTtsConfig)


# ---------------------------------------------------------------------------
# [dialog]
# ---------------------------------------------------------------------------


class DialogConfig(_Section):
    """``[dialog]`` section.

    Requirement 12.3 — ``acknowledge_after_ms`` is the threshold (default
    1500 ms) above which the Dialog_Manager emits a "One moment, sir."
    acknowledgement utterance during long-running tool dispatch.
    Requirement 14.5 — ``max_tool_retry`` caps schema-violation retries.

    Requirement 11.1 / 11.4 — ``honorific`` overrides the active persona's
    honorific. When ``None`` (the default), the persona's own honorific
    is used; this lets a user switch to a custom persona whose author
    chose a different form of address (e.g., "boss") without forcing
    the user to also rewrite ``[dialog].honorific``. To pin the
    honorific regardless of persona, set this field explicitly.
    """

    persona_profile: str = "jarvis_default"
    honorific: str | None = None
    acknowledge_after_ms: int = Field(1500, ge=0)
    max_tool_retry: int = Field(2, ge=0)


# ---------------------------------------------------------------------------
# [llm.*]
# ---------------------------------------------------------------------------


class LlmMistralConfig(_Section):
    """``[llm.mistral]`` section.

    * Requirement 19.1 — default endpoint is the Mistral la Plateforme.
    * Requirement 19.2 / 19.6 — default model id and override path.
    * Requirement 19.3 — the API key is referenced indirectly through the
      Credential_Store and must never be stored inline in this file.
    * Requirement 12.4 — ``request_timeout_ms`` triggers fallback at 3 s.
    * Requirement 19.5 — ``streaming`` toggles Mistral's streaming API.
    * Requirement 19.8 — ``max_retries`` caps 429 backoff attempts.
    """

    endpoint: str = "https://api.mistral.ai"
    model: str = "mistral-large-latest"
    api_key_credential: str = "mistral/api_key"
    request_timeout_ms: int = Field(3000, ge=1)
    max_retries: int = Field(3, ge=0)
    retry_backoff_initial_ms: int = Field(200, ge=0)
    streaming: bool = True

    @field_validator("endpoint")
    @classmethod
    def _endpoint_is_https(cls, v: str) -> str:
        # Requirement 13.4 / 19.3 imply secrets should never travel in
        # cleartext. Reject obvious mistakes early; ``http://localhost``
        # remains usable through the fallback section.
        if not (v.startswith("https://") or v.startswith("http://localhost")):
            raise ValueError(
                "llm.mistral.endpoint must use https:// (or http://localhost "
                f"for local proxies); got {v!r}"
            )
        return v


class LlmFallbackConfig(_Section):
    """``[llm.fallback]`` section.

    Requirement 12.4 — when the cloud Mistral endpoint is unreachable for
    more than 3 s or returns 5xx errors, the Dialog_Manager falls back to
    the configured local backend (Ollama-hosted Mistral by default).
    """

    enabled: bool = True
    backend: Literal["ollama"] = "ollama"
    endpoint: str = "http://localhost:11434"
    model: str = "mistral"
    circuit_open_seconds: int = Field(30, ge=1)


class LlmConfig(_Section):
    """``[llm]`` aggregate section."""

    mistral: LlmMistralConfig = Field(default_factory=LlmMistralConfig)
    fallback: LlmFallbackConfig = Field(default_factory=LlmFallbackConfig)


# ---------------------------------------------------------------------------
# [memory]
# ---------------------------------------------------------------------------


class MemoryConfig(_Section):
    """``[memory]`` section.

    Requirement 10.3 — ``top_k`` is the configurable retrieval depth and is
    constrained to ``[1, 50]`` per the design's "Configuration Validation
    Rules".
    Requirement 10.7 — encryption-at-rest via DPAPI.
    Requirement 10.8 — PII redaction toggle and pattern set.
    """

    backend: Literal["chroma"] = "chroma"
    path: str = "${app.data_dir}/memory/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(5, ge=1, le=50)
    encrypt_at_rest: bool = True
    encrypt_embeddings: bool = False
    redaction_enabled: bool = True
    pii_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
            r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b",
        ]
    )


# ---------------------------------------------------------------------------
# [reminders]
# ---------------------------------------------------------------------------


class RemindersConfig(_Section):
    """``[reminders]`` section.

    Requirement 6.6 — when the application is not running, due reminders
    must fire on next launch within 30 seconds; the configured grace window
    is therefore floored at 30 seconds.
    """

    db_path: str = "${app.data_dir}/reminders.sqlite"
    toast_enabled: bool = True
    on_start_grace_seconds: int = Field(30, ge=30)


# ---------------------------------------------------------------------------
# [skills]
# ---------------------------------------------------------------------------


class McpServerConfig(_Section):
    """A single entry in ``[skills].mcp_servers``.

    Requirement 14.6 — external MCP servers contribute Skills via the
    :class:`MCPSkillAdapter`.
    """

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SkillsConfig(_Section):
    """``[skills]`` section.

    Requirement 14.3 — each Skill manifest's JSON Schema is validated
    against ``registry_meta_schema`` (defaulting to draft-07).
    """

    registry_meta_schema: Literal["draft-07"] = "draft-07"
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# [automation.*]
# ---------------------------------------------------------------------------


class ScriptCatalogEntry(_Section):
    """A single entry in ``[automation.script_catalog]``."""

    interpreter: Literal["powershell", "python", "batch"]
    path: str
    description: str = ""


class AllowedDirectoriesConfig(_Section):
    """``[automation.allowed_directories]`` section.

    Requirements 8.2 / 8.6 — file Skills must verify that resolved paths lie
    within at least one allowed directory; an empty list is rejected at
    configuration time so misconfigured installs do not silently lose file
    capabilities.
    """

    paths: list[str] = Field(
        default_factory=lambda: [
            "%USERPROFILE%/Documents",
            "%USERPROFILE%/Downloads",
        ]
    )

    @field_validator("paths")
    @classmethod
    def _paths_non_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError(
                "automation.allowed_directories.paths must contain at least "
                "one path (Requirement 8.2 / 8.6)."
            )
        # Strip pure-whitespace entries which the user almost certainly did
        # not intend; if the result is empty, also reject.
        cleaned = [p for p in v if p.strip()]
        if len(cleaned) == 0:
            raise ValueError(
                "automation.allowed_directories.paths contains only blank "
                "entries; provide at least one non-empty directory."
            )
        return cleaned


class AutomationConfig(_Section):
    """``[automation]`` aggregate section."""

    application_registry: dict[str, str] = Field(
        default_factory=lambda: {
            "chrome": (
                "C:/Program Files/Google/Chrome/Application/chrome.exe"
            ),
            "vscode": (
                "C:/Users/%USERNAME%/AppData/Local/Programs/"
                "Microsoft VS Code/Code.exe"
            ),
            "spotify": "spotify:",
        }
    )
    allowed_directories: AllowedDirectoriesConfig = Field(
        default_factory=AllowedDirectoriesConfig
    )
    script_catalog: dict[str, ScriptCatalogEntry] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# [providers.*]
# ---------------------------------------------------------------------------


class ProvidersWeatherConfig(_Section):
    """``[providers.weather]`` section.

    Requirement 7.7 — provider read timeout is 5 s by default.
    """

    provider: str = "openweather"
    api_key_credential: str = "weather/api_key"
    default_location: str = "Bandung,ID"
    timeout_seconds: float = Field(5.0, gt=0.0)


class ProvidersNewsConfig(_Section):
    """``[providers.news]`` section."""

    provider: str = "newsapi"
    api_key_credential: str = "news/api_key"
    default_topic: str = "technology"
    timeout_seconds: float = Field(5.0, gt=0.0)


class ProvidersCalendarConfig(_Section):
    """``[providers.calendar]`` section."""

    provider: str = "google"
    oauth_credential: str = "calendar/oauth_token"
    timeout_seconds: float = Field(5.0, gt=0.0)


class ProvidersEmailConfig(_Section):
    """``[providers.email]`` section."""

    provider: str = "smtp"
    host: str = "smtp.example.com"
    port: int = Field(587, ge=1, le=65535)
    username_credential: str = "email/smtp_user"
    password_credential: str = "email/smtp_password"


class ProvidersSearchConfig(_Section):
    """``[providers.search]`` section.

    Requirement 3.1 — default ``max_results`` is 5, capped at 10.
    """

    provider: Literal["tavily", "bing", "duckduckgo"] = "tavily"
    api_key_credential: str = "search/api_key"
    max_results_default: int = Field(5, ge=1, le=10)
    max_results_cap: int = Field(10, ge=1, le=50)

    @model_validator(mode="after")
    def _default_within_cap(self) -> ProvidersSearchConfig:
        if self.max_results_default > self.max_results_cap:
            raise ValueError(
                "providers.search.max_results_default "
                f"({self.max_results_default}) exceeds max_results_cap "
                f"({self.max_results_cap})."
            )
        return self


class ProvidersConfig(_Section):
    """``[providers]`` aggregate section."""

    weather: ProvidersWeatherConfig = Field(default_factory=ProvidersWeatherConfig)
    news: ProvidersNewsConfig = Field(default_factory=ProvidersNewsConfig)
    calendar: ProvidersCalendarConfig = Field(default_factory=ProvidersCalendarConfig)
    email: ProvidersEmailConfig = Field(default_factory=ProvidersEmailConfig)
    search: ProvidersSearchConfig = Field(default_factory=ProvidersSearchConfig)


# ---------------------------------------------------------------------------
# [authorization]
# ---------------------------------------------------------------------------


class TrustedAction(_Section):
    """An entry in ``[authorization].trusted_action_allowlist``.

    Requirement 16.3 — when the requested Tool_Call matches an entry with
    matching arguments, confirmation is bypassed for that single invocation.
    """

    skill: str
    args_subset: dict[str, Any] = Field(default_factory=dict)


class DestructiveOperation(_Section):
    """An entry in ``[authorization].destructive_operations``.

    Used for Skills whose destructiveness depends on a discriminator field
    (e.g., ``CalendarSkill.operation == "create_event"``).
    """

    skill: str
    op_field: str
    op_values: list[str]

    @field_validator("op_values")
    @classmethod
    def _op_values_non_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError(
                "authorization.destructive_operations[*].op_values must "
                "contain at least one value."
            )
        return v


class AuthorizationConfig(_Section):
    """``[authorization]`` section.

    Requirement 16.1 — ``destructive_skills`` lists Skills that are always
    classified as Destructive_Action.
    Requirement 16.3 — ``trusted_action_allowlist`` bypasses confirmation
    for matching arguments.
    """

    trusted_action_allowlist: list[TrustedAction] = Field(default_factory=list)
    destructive_skills: list[str] = Field(
        default_factory=lambda: [
            "SendEmailSkill",
            "SendMessageSkill",
            "RunScriptSkill",
            "MemoryAdminSkill.forget",
        ]
    )
    destructive_operations: list[DestructiveOperation] = Field(
        default_factory=lambda: [
            DestructiveOperation(
                skill="CalendarSkill",
                op_field="operation",
                op_values=["create_event"],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# [security]
# ---------------------------------------------------------------------------


class SecurityConfig(_Section):
    """``[security]`` section.

    Requirements 13.4 / 13.6 — outbound network destinations are constrained
    to the configured allowlist; the audit log captures destinations and
    justifications.
    """

    audit_log_path: str = "${app.data_dir}/audit.sqlite"
    network_destination_allowlist: list[str] = Field(
        default_factory=lambda: [
            "api.mistral.ai",
            "api.openweathermap.org",
            "newsapi.org",
            "www.googleapis.com",
            "smtp.example.com",
            "localhost",
        ]
    )


# ---------------------------------------------------------------------------
# [telemetry]
# ---------------------------------------------------------------------------


class TelemetryConfig(_Section):
    """``[telemetry]`` section.

    Telemetry is opt-in (defaults to disabled) per privacy requirements.
    """

    enabled: bool = False
    crash_report_endpoint: str = ""


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "app",
        "voice",
        "dialog",
        "llm",
        "memory",
        "reminders",
        "skills",
        "automation",
        "providers",
        "authorization",
        "security",
        "telemetry",
    }
)


class Config(BaseModel):
    """Top-level JARVIS configuration.

    Mirrors the TOML file documented in ``design.md`` and rejects malformed
    values via the field- and model-level validators on the nested sections.

    Unknown top-level keys do NOT cause validation to fail; instead they
    surface an :class:`UnknownConfigKeyWarning` so users notice typos
    before the override silently no-ops.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        # Surface field-level errors with sub-paths the user can map back to
        # the TOML file (e.g., ``voice.stt.engine``).
        loc_by_alias=False,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    dialog: DialogConfig = Field(default_factory=DialogConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    reminders: RemindersConfig = Field(default_factory=RemindersConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    authorization: AuthorizationConfig = Field(default_factory=AuthorizationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_top_level_keys(cls, data: Any) -> Any:
        # Pydantic invokes mode="before" validators with the raw input. The
        # caller may pass a dict (from ``tomllib.loads``) or another model
        # instance during ``model_copy``; we only inspect dict-shaped input.
        if isinstance(data, dict):
            unknown = set(data.keys()) - _KNOWN_TOP_LEVEL_KEYS
            for key in sorted(unknown):
                warnings.warn(
                    f"Unknown top-level config key {key!r}; setting will "
                    "be ignored. Check spelling against the documented "
                    "schema in design.md.",
                    UnknownConfigKeyWarning,
                    stacklevel=2,
                )
        return data
