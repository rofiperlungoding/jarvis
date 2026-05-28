"""Unit tests for :mod:`jarvis.app`.

These tests exercise the bootstrap orchestrator, the crash-detection
sentinel, and the wipe-all flow with lightweight fakes for every heavy
component (Mistral, Ollama, Whisper, Piper, Porcupine, ChromaDB,
APScheduler, the platform adapter). The goal is to validate the bootstrap
*ordering* and the public lifecycle contract without requiring OS audio,
network access, or the full set of audio extras.

Validates: Requirements 1.5, 13.5, 17.4
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from jarvis.app import (
    BOOTSTRAP_STEPS,
    SENTINEL_FILENAME,
    ComponentFactories,
    JarvisApp,
    _LLMStack,
    _read_last_run_sentinel,
    _write_last_run_sentinel,
    main,
)
from jarvis.config import load_config
from jarvis.config.schema import Config
from jarvis.security.credential_store import CredentialBackend
from jarvis.security.dpapi import DPAPI, NullDPAPI
from jarvis.security.log_redaction import LogRedactionFilter
from jarvis.skills.base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import FakeTimeSource, TimeSource

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCredentialStore:
    """In-memory :class:`CredentialBackend` compatible double."""

    def __init__(self, *, initial: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(initial or {})
        self.wipe_calls: int = 0

    def set(self, name: str, value: str) -> None:
        self._values[name] = value

    def get(self, name: str) -> str | None:
        return self._values.get(name)

    def delete(self, name: str) -> None:
        self._values.pop(name, None)

    def list_names(self) -> list[str]:
        return sorted(self._values)

    def wipe(self) -> None:
        self.wipe_calls += 1
        self._values.clear()


class _FakeMemoryStore:
    """Records ``wipe`` invocations and exposes a no-op ``persist_turn``."""

    def __init__(self) -> None:
        self.wipe_calls: int = 0

    async def wipe(self) -> None:
        self.wipe_calls += 1

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        del query, k
        return []

    async def persist_turn(self, turn: Any, persona: Any | None = None) -> list[Any]:
        del turn, persona
        return []


class _FakeReminderService:
    """Captures lifecycle calls so tests can assert ordering."""

    def __init__(self) -> None:
        self.started: bool = False
        self.stopped: bool = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _FakeBackend:
    """Trivial :class:`LLMBackend` compatible double — never streams."""

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("fake backend does not support stream()")


class _FakeSelector:
    """Selector that exposes ``_on_flip`` so wiring tests can verify the bind."""

    def __init__(self) -> None:
        self._on_flip: Callable[[], None] | None = None

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("fake selector does not support stream()")


class _FakeTTS:
    async def speak(self, text: str) -> None:
        del text

    async def stop(self) -> None:
        return None

    def is_playing(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None


class _FakeVoicePipeline:
    """Voice pipeline that records ``aclose`` so shutdown tests can verify it."""

    def __init__(self) -> None:
        self.tts: Any = _FakeTTS()
        self.stt: Any = object()
        self.vad: Any = object()
        self.wake_word: Any = object()
        self.aclose_calls: int = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _FakeDialogManager:
    """DialogManager double that simply echoes the transcript text."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def handle_turn(self, transcript: Any, state: Any) -> Any:
        self.calls.append((transcript, state))
        return ("response", transcript)


class _FakePlatformAdapter:
    pass


# ---------------------------------------------------------------------------
# A tiny no-op skill so the registry has at least one entry.
# ---------------------------------------------------------------------------


class _EchoSkill:
    manifest = SkillManifest(
        name="EchoSkill",
        description="Echo argument back; used by app.py tests.",
        json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        destructive=False,
        platforms=("windows", "linux", "darwin"),
        source="builtin",
    )

    async def execute(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        del ctx
        return SkillResult.success(value={"text": args["text"]})


# ---------------------------------------------------------------------------
# Factory builders
# ---------------------------------------------------------------------------


def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set deterministic env vars so config expansion is reproducible."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "userprofile"))
    monkeypatch.setenv("USERNAME", "tester")


def _build_factories(
    *,
    cred_store: _FakeCredentialStore | None = None,
    memory_store: _FakeMemoryStore | None = None,
    reminder_service: _FakeReminderService | None = None,
    voice: _FakeVoicePipeline | None = None,
    dialog_manager: _FakeDialogManager | None = None,
    selector: _FakeSelector | None = None,
    extra_skill: Skill | None = None,
) -> tuple[ComponentFactories, dict[str, Any]]:
    """Build a :class:`ComponentFactories` with lightweight test doubles."""
    state: dict[str, Any] = {
        "cred_store": cred_store or _FakeCredentialStore(),
        "memory_store": memory_store or _FakeMemoryStore(),
        "reminder_service": reminder_service or _FakeReminderService(),
        "voice": voice or _FakeVoicePipeline(),
        "dialog_manager": dialog_manager or _FakeDialogManager(),
        "selector": selector or _FakeSelector(),
        "platform": _FakePlatformAdapter(),
    }

    def dpapi_factory(_config: Config) -> DPAPI:
        return NullDPAPI(suppress_warning=True)  # type: ignore[return-value]

    def cred_factory(
        _config: Config, _dpapi: DPAPI, _filter: LogRedactionFilter
    ) -> CredentialBackend:
        cred_store: CredentialBackend = state["cred_store"]
        return cred_store

    def llm_factory(
        _config: Config,
        _credentials: CredentialBackend,
        _filter: LogRedactionFilter,
        _time: TimeSource,
    ) -> _LLMStack:
        primary = _FakeBackend()
        fallback = _FakeBackend()
        return _LLMStack(primary=primary, fallback=fallback, selector=state["selector"])

    def platform_factory(_config: Config) -> Any:
        return state["platform"]

    def memory_factory(_config: Config, _dpapi: DPAPI) -> Any:
        return state["memory_store"]

    async def reminder_factory(
        _config: Config, _platform: Any, _time: TimeSource
    ) -> Any:
        await state["reminder_service"].start()
        return state["reminder_service"]

    async def skills_factory(
        _config: Config,
        builtin_skills: Iterable[Skill],
        _hooks: Any,
    ) -> SkillRegistry:
        registry = SkillRegistry()
        for skill in builtin_skills:
            registry.register(skill)
        return registry

    async def voice_factory(_config: Config, _deps: Any) -> Any:
        return state["voice"]

    def dm_factory(_config: Config, _deps: Any) -> Any:
        return state["dialog_manager"]

    factories = ComponentFactories(
        dpapi=dpapi_factory,
        credential_store=cred_factory,
        llm_stack=llm_factory,
        platform_adapter=platform_factory,
        memory_store=memory_factory,
        reminder_service=reminder_factory,
        skills=skills_factory,
        voice_pipeline=voice_factory,
        dialog_manager=dm_factory,
    )
    return factories, state


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_jarvis_app_constructs_from_default_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``JarvisApp`` accepts a config produced by ``load_config(None)``."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    app = JarvisApp(config)
    assert isinstance(app.config, Config)
    assert app.bootstrap_steps == ()
    assert app.run_id


def test_jarvis_app_rejects_non_config() -> None:
    with pytest.raises(TypeError):
        JarvisApp(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bootstrap ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_records_steps_in_documented_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """:meth:`bootstrap` populates :attr:`bootstrap_steps` in the canonical order."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, _ = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        time_source=FakeTimeSource(),
        builtin_skills=[_EchoSkill()],
    )

    components = await app.bootstrap()

    assert app.bootstrap_steps == BOOTSTRAP_STEPS
    assert components.config is config
    assert components.skills.get("EchoSkill") is not None


@pytest.mark.asyncio
async def test_bootstrap_is_idempotent_per_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, _ = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        time_source=FakeTimeSource(),
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()
    with pytest.raises(RuntimeError):
        await app.bootstrap()


@pytest.mark.asyncio
async def test_bootstrap_starts_reminder_service_after_memory_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reminder service start must follow memory-store creation."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, state = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()
    assert state["reminder_service"].started is True
    # Memory_Store is created before reminder service in BOOTSTRAP_STEPS.
    assert BOOTSTRAP_STEPS.index("init_memory_store") < BOOTSTRAP_STEPS.index(
        "init_reminder_service"
    )


@pytest.mark.asyncio
async def test_bootstrap_wires_fallback_notice_onto_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback-notice TTS callback is bound onto ``BackendSelector._on_flip``."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    selector = _FakeSelector()
    factories, _ = _build_factories(selector=selector)
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()
    assert selector._on_flip is not None
    assert callable(selector._on_flip)


@pytest.mark.asyncio
async def test_bootstrap_seeds_redaction_with_known_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-existing credentials are registered with the redaction filter."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    cred_store = _FakeCredentialStore(initial={"mistral/api_key": "secret-token-XYZ"})
    factories, _ = _build_factories(cred_store=cred_store)
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    components = await app.bootstrap()
    # The secret should have been registered with the redaction filter.
    # ``registered_secret_count`` is at least 1 (the seeded credential).
    assert components.log_redaction_filter.registered_secret_count() >= 1


# ---------------------------------------------------------------------------
# Crash sentinel
# ---------------------------------------------------------------------------


def test_read_sentinel_returns_clean_when_missing(tmp_path: Path) -> None:
    sentinel = _read_last_run_sentinel(tmp_path / SENTINEL_FILENAME)
    assert sentinel.existed is False
    assert sentinel.clean_shutdown is True
    assert sentinel.run_id is None


def test_read_sentinel_returns_crash_when_unclean(tmp_path: Path) -> None:
    path = tmp_path / SENTINEL_FILENAME
    path.write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "started_at": "2024-01-01T12:00:00+00:00",
                "clean_shutdown": False,
            }
        ),
        encoding="utf-8",
    )
    sentinel = _read_last_run_sentinel(path)
    assert sentinel.existed is True
    assert sentinel.clean_shutdown is False
    assert sentinel.run_id == "abc123"
    assert sentinel.started_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_read_sentinel_treats_corrupt_file_as_crash(tmp_path: Path) -> None:
    path = tmp_path / SENTINEL_FILENAME
    path.write_text("not valid json{", encoding="utf-8")
    sentinel = _read_last_run_sentinel(path)
    assert sentinel.existed is True
    assert sentinel.clean_shutdown is False


def test_write_sentinel_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / SENTINEL_FILENAME
    started_at = datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC)
    _write_last_run_sentinel(
        path,
        run_id="run-1",
        started_at=started_at,
        clean_shutdown=False,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["clean_shutdown"] is False
    assert data["started_at"] == started_at.isoformat()


@pytest.mark.asyncio
async def test_bootstrap_records_crash_audit_when_prior_sentinel_unclean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale sentinel from the previous run produces a ``crash`` audit row."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)

    # Pre-seed a "previous run that crashed" sentinel.
    data_dir = Path(config.app.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    _write_last_run_sentinel(
        data_dir / SENTINEL_FILENAME,
        run_id="previous-run",
        started_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        clean_shutdown=False,
    )

    factories, _ = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    components = await app.bootstrap()

    assert app.prior_sentinel is not None
    assert app.prior_sentinel.existed is True
    assert app.prior_sentinel.clean_shutdown is False
    crash_entries = [e for e in components.audit_log.entries() if e.kind == "crash"]
    assert len(crash_entries) == 1
    # The diagnostics offer flow ran with no consent prompt wired
    # (the default test factories do not provide one), so the audit
    # row carries the ``no_prompt`` outcome from
    # :mod:`jarvis.diagnostics`.
    crash_outcome = crash_entries[0].outcome
    assert crash_outcome is not None
    assert crash_outcome.startswith("prior_run_did_not_shut_down_cleanly")
    assert "no_prompt" in crash_outcome
    # The orchestrator surfaces the diagnostics outcome on Components
    # so callers (e.g., a UI) can react to it.
    assert components.diagnostics_outcome is not None
    assert components.diagnostics_outcome.consented is False
    assert components.diagnostics_outcome.report_path is None


@pytest.mark.asyncio
async def test_aclose_marks_sentinel_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A graceful shutdown rewrites the sentinel with ``clean_shutdown=True``."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, state = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()
    assert app.sentinel_path is not None
    # Mid-run sentinel reports unclean.
    mid = _read_last_run_sentinel(app.sentinel_path)
    assert mid.existed is True
    assert mid.clean_shutdown is False

    await app.aclose()

    final = _read_last_run_sentinel(app.sentinel_path)
    assert final.existed is True
    assert final.clean_shutdown is True
    # Voice pipeline aclose must have been called.
    assert state["voice"].aclose_calls == 1
    # Reminder service stopped.
    assert state["reminder_service"].stopped is True


@pytest.mark.asyncio
async def test_aclose_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, state = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()
    await app.aclose()
    await app.aclose()
    # Voice aclose called exactly once.
    assert state["voice"].aclose_calls == 1


# ---------------------------------------------------------------------------
# Wipe-all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wipe_all_clears_three_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Memory, credential, and audit log wipes all run within the budget."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)

    cred_store = _FakeCredentialStore(initial={"weather/api_key": "wkey"})
    memory_store = _FakeMemoryStore()
    factories, _state = _build_factories(
        cred_store=cred_store, memory_store=memory_store
    )
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    components = await app.bootstrap()

    # Pre-seed an audit log entry so we can confirm wipe really clears it.
    await components.audit_log.record_executed(
        skill="EchoSkill", args_json={"text": "hi"}, outcome="ok"
    )
    assert components.audit_log.count() >= 1

    await app.wipe_all()

    assert memory_store.wipe_calls == 1
    assert cred_store.wipe_calls == 1
    assert components.audit_log.count() == 0
    # Filter cleared as part of credential wipe.
    assert components.log_redaction_filter.registered_secret_count() == 0

    await app.aclose()


@pytest.mark.asyncio
async def test_wipe_all_completes_within_5_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    factories, _ = _build_factories()
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()

    loop = asyncio.get_event_loop()
    start = loop.time()
    await app.wipe_all()
    elapsed = loop.time() - start
    assert elapsed < 5.0
    await app.aclose()


@pytest.mark.asyncio
async def test_wipe_all_requires_bootstrap_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)
    app = JarvisApp(config)
    with pytest.raises(RuntimeError):
        await app.wipe_all()


@pytest.mark.asyncio
async def test_wipe_all_times_out_when_a_wipe_is_slow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wipe that exceeds the budget surfaces :class:`TimeoutError`."""
    _isolate_environment(monkeypatch, tmp_path)
    config = load_config(None)

    class _SlowMemoryStore(_FakeMemoryStore):
        async def wipe(self) -> None:
            await asyncio.sleep(10.0)

    factories, _ = _build_factories(memory_store=_SlowMemoryStore())
    app = JarvisApp(
        config,
        factories=factories,
        builtin_skills=[_EchoSkill()],
    )
    await app.bootstrap()

    # Patch the budget down so the test does not actually wait 5s.
    with (
        patch("jarvis.app._WIPE_BUDGET_SECONDS", 0.1),
        pytest.raises((asyncio.TimeoutError, TimeoutError)),
    ):
        await app.wipe_all()
    await app.aclose()


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------


def test_main_exposes_help_without_running_event_loop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--help`` returns 0 without entering the run loop."""
    monkeypatch.setenv("APPDATA", "C:/test")
    monkeypatch.setenv("LOCALAPPDATA", "C:/test")
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "voice-driven AI assistant" in captured.out


def test_main_module_redirects_to_app_main() -> None:
    """``python -m jarvis`` resolves to :func:`jarvis.app.main`."""
    import jarvis.__main__ as module  # noqa: PLC0415

    assert module.main is main
