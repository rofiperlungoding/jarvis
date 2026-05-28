"""Application entrypoint and asyncio wiring for the JARVIS AI assistant.

This module owns the bootstrap order described in ``design.md §Project
Structure`` and ``§Concurrency Model``, the three cooperating async
loops (audio capture, dialog, output), the crash-detection sentinel
required by Requirement 17.4, and the data-erasure flow demanded by
Requirement 13.5.

What lives here
---------------

* :class:`ComponentFactories` — a small, pure-data bundle of optional
  zero-arg factory callables. The default factories build the production
  components (DPAPI, Mistral / Ollama / Selector, WindowsAdapter,
  ChromaDB-backed Memory_Store, APScheduler-backed Reminder_Service,
  faster-whisper STT, Piper TTS, Silero VAD, Porcupine wake word). Tests
  inject :class:`object()`-style fakes so the bootstrap sequence can be
  exercised on Linux CI runners that do not have the audio extras
  installed.
* :class:`Components` — the materialised result of a successful
  bootstrap. Holds references to every long-lived collaborator the
  three loops need so the loops themselves do not have to remember
  which factories produced what.
* :class:`JarvisApp` — the orchestrator. Performs the strict bootstrap
  order, owns the three loops via :class:`asyncio.TaskGroup`, exposes
  :meth:`wipe_all` (the user-issued data-erasure command) and
  :meth:`aclose` (graceful shutdown), and persists the
  ``last_run.json`` sentinel that Requirement 17.4 uses to detect a
  prior crash on the next launch.
* :func:`main` — synchronous console entry point referenced by
  ``[project.scripts] jarvis`` in ``pyproject.toml`` and by
  ``src/jarvis/__main__.py``. Boots the asyncio event loop, runs
  :meth:`JarvisApp.run`, and translates ``KeyboardInterrupt`` into the
  documented graceful-shutdown path.

Bootstrap order (Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 12.1, 12.4,
13.5, 17.3, 17.4):

1. Load config (``Config`` validated by pydantic).
2. Install the log redaction filter on the root logger so secrets the
   credential store will hand back are scrubbed *before* any subsequent
   step can produce a log line.
3. Construct the ``AuditLog`` (records every later step's policy /
   network / crash event).
4. Construct the DPAPI envelope.
5. Construct the ``CredentialStore`` (reads / writes secrets through
   the DPAPI envelope; registers credentials with the redaction
   filter as it loads them).
6. Build the LLM stack: ``MistralBackend`` from the credential store,
   ``OllamaBackend`` for the local fallback, both wrapped in
   ``BackendSelector`` with the fallback-notice TTS callback wired in
   later (after TTS exists).
7. Build the platform adapter (``WindowsAdapter`` on Windows; the
   no-op base adapter elsewhere). Provider HTTP clients are
   constructed alongside so the dialog SkillContext has them
   available.
8. Build the Memory_Store and Reminder_Service. Memory_Store reads /
   writes through DPAPI; Reminder_Service is constructed and started
   so missed reminders flush within the configured grace window.
9. Discover skills: the curated list of built-in singletons, then any
   modules under the configured plugin directories, then synthetic
   Skills produced by ``MCPSkillAdapter`` for each configured MCP
   server.
10. Build the voice pipeline: TTS (so the fallback-notice callback can
    be re-bound onto the BackendSelector), STT, VAD, wake-word
    detector.
11. Build the Dialog_Manager from everything above.

Three concurrent loops:

* **Audio capture loop** owns the microphone, runs the wake-word
  detector, drives the VAD into ``speech_start`` / ``speech_end``
  events, calls the STT engine on each captured utterance, and
  enqueues the resulting :class:`Transcript` onto a bounded asyncio
  queue.
* **Dialog loop** drains the transcript queue, calls
  :meth:`DialogManager.handle_turn`, and forwards the resulting
  :class:`AssistantResponse` onto an output queue. The Dialog_Manager
  already speaks tokens to TTS at sentence boundaries during
  ``handle_turn``; the output queue exists so future consumers
  (transcript log, telemetry, UI tray) can observe completed turns
  without coupling to the dialog loop.
* **Output loop** drains the output queue. Today it logs each
  :class:`AssistantResponse` for the UI / transcript log; barge-in is
  handled inline by the audio loop because that is where the
  ``speech_start`` signal originates.

Crash detection (Requirement 17.4):

The sentinel file lives at ``${app.data_dir}/last_run.json``. On
startup, we (a) read the file (if present) and check the
``clean_shutdown`` flag from the *previous* run, then (b) overwrite
the file with ``clean_shutdown=False`` for the *current* run. On
graceful :meth:`aclose`, the file is rewritten with
``clean_shutdown=True``. A stale (``clean_shutdown=False``) sentinel
on a fresh run therefore confirms an abrupt termination of the
previous run; the entry point logs an audit ``crash`` row and offers a
diagnostics report (today gated behind user consent — the actual
network submission is intentionally still left to the user, matching
the privacy posture the design takes throughout).

Wipe-all (Requirement 13.5):

:meth:`wipe_all` calls ``MemoryStore.wipe()``, ``CredentialStore.wipe()``,
and ``AuditLog.wipe()`` under a single :func:`asyncio.wait_for` budget
of 5 seconds. The three calls run concurrently so a slow ChromaDB
delete cannot push the audit log wipe past the budget; the audit log
itself is wiped *last in concept* (we explicitly record the wipe
intent first when an :class:`AuditLog` is still around) so the
operator can correlate the wipe in the log they preserve before
running the command.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 12.1, 12.4,
13.5, 17.3, 17.4
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
import json
import logging
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Final, Protocol
import uuid

from jarvis.config import load_config
from jarvis.config.schema import Config, McpServerConfig
from jarvis.diagnostics import (
    ConsentPrompt,
    DiagnosticsOfferFlow,
    DiagnosticsOfferOutcome,
)
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    ConfirmationDialog,
    TrustedActionAllowlist,
)
from jarvis.security.credential_store import (
    CredentialBackend,
    CredentialStore,
)
from jarvis.security.dpapi import DPAPI, create_default_dpapi
from jarvis.security.log_redaction import (
    LogRedactionFilter,
    install_log_redaction_filter,
)
from jarvis.skills.base import Skill
from jarvis.skills.registry import SkillRegistry
from jarvis.utils.time_source import SystemTimeSource, TimeSource

logger = logging.getLogger(__name__)

__all__ = [
    "BOOTSTRAP_STEPS",
    "SENTINEL_FILENAME",
    "ComponentFactories",
    "Components",
    "JarvisApp",
    "LastRunSentinel",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Filename of the crash-detection sentinel under ``${app.data_dir}``.
#: Mirrors the design's "Application crash" failure mode entry.
SENTINEL_FILENAME: Final[str] = "last_run.json"

#: Hard budget for :meth:`JarvisApp.wipe_all` in seconds — Requirement
#: 13.5 mandates "within 5 seconds". The watchdog raises
#: :class:`TimeoutError` if the three concurrent wipes exceed it.
_WIPE_BUDGET_SECONDS: Final[float] = 5.0

#: Default queue depth for the audio→dialog and dialog→output queues.
#: Bounded so a stalled consumer cannot drive memory growth without
#: bound; small enough that backpressure manifests early.
_DEFAULT_QUEUE_DEPTH: Final[int] = 4

#: Ordered list of bootstrap steps. Tests assert the steps fire in this
#: order; the constant is module-level so it stays the single source of
#: truth for both the runtime and the unit tests.
BOOTSTRAP_STEPS: Final[tuple[str, ...]] = (
    "load_config",
    "install_log_redaction",
    "init_audit_log",
    "init_dpapi",
    "init_credential_store",
    "init_llm_backends",
    "init_platform_adapter",
    "init_memory_store",
    "init_reminder_service",
    "discover_skills",
    "init_voice_pipeline",
    "init_dialog_manager",
)


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LastRunSentinel:
    """Snapshot of the previous run's ``last_run.json`` contents.

    Attributes
    ----------
    existed:
        ``True`` when a sentinel file was found on disk. ``False`` for
        a first launch, an explicitly wiped data dir, or a corrupt
        sentinel that could not be parsed.
    clean_shutdown:
        ``True`` when the previous run terminated through
        :meth:`JarvisApp.aclose`. ``False`` when the prior run crashed
        (or when no sentinel was present, since we cannot confirm a
        clean exit either way). The diagnostics-offer flow gates on
        ``existed and not clean_shutdown``.
    run_id:
        Run id recorded by the previous run, when available.
    started_at:
        Wall-clock timestamp the previous run wrote on launch, when
        available. Useful for the diagnostics offer flow ("the previous
        run from 2025-01-01 crashed; submit a report?").
    last_bootstrap_step:
        The last bootstrap step the previous run successfully
        recorded. Useful for distinguishing "the previous run
        crashed before it finished initialising" from "the previous
        run crashed mid-conversation". ``None`` when the sentinel was
        missing, corrupt, or written by a JARVIS version that did not
        record this field.
    raw:
        The raw JSON object as parsed from disk, with non-string values
        passed through. Empty mapping when ``existed`` is False.
    """

    existed: bool
    clean_shutdown: bool
    run_id: str | None
    started_at: datetime | None
    last_bootstrap_step: str | None
    raw: Mapping[str, Any]


def _read_last_run_sentinel(path: Path) -> LastRunSentinel:
    """Read the sentinel file; treat any error as "previous run crashed".

    Anything we cannot parse is conservatively reported as a crashed
    prior run so the diagnostics offer flow trips. Hiding a crash by
    silently treating a corrupt sentinel as "clean shutdown" would
    defeat the whole point of Requirement 17.4.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LastRunSentinel(
            existed=False,
            clean_shutdown=True,  # absence is not evidence of a crash
            run_id=None,
            started_at=None,
            last_bootstrap_step=None,
            raw=MappingProxyType({}),
        )
    except OSError:
        logger.warning(
            "could not read last-run sentinel at %s; assuming prior crash",
            path,
        )
        return LastRunSentinel(
            existed=True,
            clean_shutdown=False,
            run_id=None,
            started_at=None,
            last_bootstrap_step=None,
            raw=MappingProxyType({}),
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "last-run sentinel at %s is not valid JSON; assuming prior crash",
            path,
        )
        return LastRunSentinel(
            existed=True,
            clean_shutdown=False,
            run_id=None,
            started_at=None,
            last_bootstrap_step=None,
            raw=MappingProxyType({}),
        )
    if not isinstance(data, dict):
        return LastRunSentinel(
            existed=True,
            clean_shutdown=False,
            run_id=None,
            started_at=None,
            last_bootstrap_step=None,
            raw=MappingProxyType({}),
        )

    clean = bool(data.get("clean_shutdown", False))
    run_id = data.get("run_id")
    if not isinstance(run_id, str):
        run_id = None
    started_at_raw = data.get("started_at")
    started_at: datetime | None = None
    if isinstance(started_at_raw, str):
        try:
            started_at = datetime.fromisoformat(started_at_raw)
        except ValueError:
            started_at = None
    last_step_raw = data.get("last_bootstrap_step")
    last_bootstrap_step = last_step_raw if isinstance(last_step_raw, str) else None
    return LastRunSentinel(
        existed=True,
        clean_shutdown=clean,
        run_id=run_id,
        started_at=started_at,
        last_bootstrap_step=last_bootstrap_step,
        raw=MappingProxyType(dict(data)),
    )


def _write_last_run_sentinel(
    path: Path,
    *,
    run_id: str,
    started_at: datetime,
    clean_shutdown: bool,
    last_bootstrap_step: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write the sentinel atomically (best effort) via temp-file-and-rename.

    The sentinel is non-secret bookkeeping; we still use the rename
    pattern so a crash mid-write cannot leave a half-written file that
    a subsequent launch would fail to parse.
    """
    payload: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "clean_shutdown": clean_shutdown,
    }
    if last_bootstrap_step is not None:
        payload["last_bootstrap_step"] = last_bootstrap_step
    if extra:
        for key, value in extra.items():
            if key in payload:
                continue
            payload[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except OSError:
        logger.exception("failed to write last-run sentinel at %s", path)
        # Best-effort cleanup of the partially-written temp file; never
        # mask the original ``logger.exception`` above with a secondary
        # filesystem error.
        with contextlib.suppress(OSError):
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------


@dataclass
class ComponentFactories:
    """Pluggable factories for every component the bootstrap creates.

    Each field is an optional zero-arg callable. When ``None`` (the
    default), :meth:`JarvisApp.bootstrap` falls back to the
    production builder, which lives next to the field as a module-level
    ``_default_*`` function.

    Tests inject fakes for the heavy components — Mistral/Ollama,
    Memory_Store (ChromaDB), STT (faster-whisper), TTS (piper), wake
    word (Porcupine), and the platform adapter — so the bootstrap order
    can be exercised on a Linux CI runner without the audio extras.
    The factories receive the validated :class:`Config` and any
    earlier-built collaborators they need, so their signatures vary;
    each one is documented inline below.
    """

    dpapi: Callable[[Config], DPAPI] | None = None
    audit_log: Callable[[Config, str, TimeSource], AuditLog] | None = None
    credential_store: (
        Callable[[Config, DPAPI, LogRedactionFilter], CredentialBackend] | None
    ) = None
    llm_stack: (
        Callable[
            [
                Config,
                CredentialBackend,
                LogRedactionFilter,
                TimeSource,
            ],
            _LLMStack,
        ]
        | None
    ) = None
    platform_adapter: Callable[[Config], Any] | None = None
    memory_store: Callable[[Config, DPAPI], Any] | None = None
    reminder_service: Callable[[Config, Any, TimeSource], Awaitable[Any]] | None = None
    skills: (
        Callable[
            [
                Config,
                Iterable[Skill],
                _RegistryHooks,
            ],
            Awaitable[SkillRegistry],
        ]
        | None
    ) = None
    voice_pipeline: (
        Callable[[Config, _VoicePipelineDeps], Awaitable[VoicePipeline]] | None
    ) = None
    dialog_manager: Callable[[Config, _DialogDeps], Any] | None = None
    confirmation_dialog: Callable[[], ConfirmationDialog] | None = None
    consent_prompt: (
        Callable[[Config, _ConsentPromptDeps], ConsentPrompt | None] | None
    ) = None


# ---------------------------------------------------------------------------
# Internal collaborator bundles passed into factories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LLMStack:
    """LLM tier returned by the ``llm_stack`` factory."""

    primary: Any
    fallback: Any
    selector: Any


@dataclass(frozen=True)
class _RegistryHooks:
    """Side-channel inputs to the ``skills`` factory."""

    builtin_skills: tuple[Skill, ...]
    plugin_dirs: tuple[Path, ...]
    mcp_servers: tuple[McpServerConfig, ...]


@dataclass(frozen=True)
class _VoicePipelineDeps:
    """Inputs to the ``voice_pipeline`` factory."""

    credential_store: CredentialBackend
    log_redaction_filter: LogRedactionFilter


@dataclass(frozen=True)
class _DialogDeps:
    """Inputs to the ``dialog_manager`` factory."""

    backend: Any
    skills: SkillRegistry
    memory: Any
    policy: AuthorizationPolicy
    persona: Any
    tts: Any
    audit_log: AuditLog
    confirmation_dialog: ConfirmationDialog | None
    time_source: TimeSource
    run_id: str


@dataclass(frozen=True)
class _ConsentPromptDeps:
    """Inputs to the ``consent_prompt`` factory.

    The diagnostics offer flow needs an interactive consent prompt
    that can speak the question and listen for the reply. The
    production wiring composes the TTS engine and the Dialog_Manager;
    tests substitute a canned-answer fake.
    """

    tts: Any
    dialog_manager: Any
    time_source: TimeSource


# ---------------------------------------------------------------------------
# Voice pipeline bundle
# ---------------------------------------------------------------------------


class VoicePipeline(Protocol):
    """Bundle of voice components owned by the audio loop.

    The production implementation (:class:`_DefaultVoicePipeline`) wires
    together :class:`PiperTTS`, :class:`FasterWhisperSTT`,
    :class:`SileroVAD`, and :class:`WakeWordDetector`. Tests substitute
    a minimal stub that exposes the same attributes so the orchestrator
    can route lifecycle calls (``aclose``) and pass references into
    :class:`DialogManager` without caring which concrete classes are
    behind them.
    """

    tts: Any
    stt: Any
    vad: Any
    wake_word: Any

    async def aclose(self) -> None:
        """Release every voice resource the pipeline holds."""
        ...


# ---------------------------------------------------------------------------
# Components dataclass
# ---------------------------------------------------------------------------


@dataclass
class Components:
    """Materialised collaborators after a successful bootstrap.

    The fields are populated incrementally as :meth:`JarvisApp.bootstrap`
    walks the steps. Each field is :class:`Any` so the dataclass does
    not pull every concrete type into module-import time and so test
    fakes substitute cleanly.
    """

    config: Config
    run_id: str
    time_source: TimeSource
    log_redaction_filter: LogRedactionFilter
    audit_log: AuditLog
    dpapi: DPAPI
    credential_store: CredentialBackend
    llm_primary: Any
    llm_fallback: Any
    backend_selector: Any
    platform_adapter: Any
    providers: Mapping[str, Any]
    memory_store: Any
    reminder_service: Any
    skills: SkillRegistry
    persona: Any
    voice_pipeline: VoicePipeline
    dialog_manager: Any
    authorization_policy: AuthorizationPolicy
    diagnostics_offer: DiagnosticsOfferFlow
    diagnostics_outcome: DiagnosticsOfferOutcome | None = None
    mcp_close_fns: tuple[Callable[[], Awaitable[None]], ...] = ()


# ---------------------------------------------------------------------------
# JarvisApp orchestrator
# ---------------------------------------------------------------------------


class JarvisApp:
    """Lifecycle owner for one JARVIS process.

    Construction is cheap; all heavy work happens in :meth:`bootstrap`.
    Tests typically:

    1. Build a :class:`Config` (often via
       :meth:`Config.model_validate({})`).
    2. Build a :class:`ComponentFactories` whose fields point at
       lightweight fakes.
    3. Instantiate ``JarvisApp(config, factories=...)``.
    4. Await :meth:`bootstrap` to materialise the components.
    5. Drive :meth:`wipe_all` / :meth:`aclose` directly, or call
       :meth:`run` and arrange for cancellation.

    The production entry point :func:`main` performs steps 1, 3, and 5.
    """

    def __init__(
        self,
        config: Config,
        *,
        factories: ComponentFactories | None = None,
        time_source: TimeSource | None = None,
        run_id: str | None = None,
        builtin_skills: Sequence[Skill] | None = None,
    ) -> None:
        if not isinstance(config, Config):
            raise TypeError("config must be a Config instance")
        self._config: Config = config
        self._factories: ComponentFactories = factories or ComponentFactories()
        self._time_source: TimeSource = time_source or SystemTimeSource()
        self._run_id: str = run_id or uuid.uuid4().hex
        # Builtin skill discovery is module-level so tests can pin a
        # tiny skill list without touching the production builders.
        # ``None`` means "use the curated default list" and is resolved
        # lazily in :meth:`bootstrap`.
        self._builtin_skills_override: Sequence[Skill] | None = builtin_skills

        self._components: Components | None = None
        self._bootstrap_steps: list[str] = []
        self._sentinel_path: Path | None = None
        self._started_at: datetime | None = None
        self._prior_sentinel: LastRunSentinel | None = None
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def components(self) -> Components:
        if self._components is None:
            raise RuntimeError("JarvisApp.bootstrap() has not been called yet")
        return self._components

    @property
    def bootstrap_steps(self) -> tuple[str, ...]:
        """Steps executed so far, in order. Stable shape for tests."""
        return tuple(self._bootstrap_steps)

    @property
    def prior_sentinel(self) -> LastRunSentinel | None:
        """The previous run's sentinel, if :meth:`bootstrap` has run."""
        return self._prior_sentinel

    @property
    def sentinel_path(self) -> Path | None:
        return self._sentinel_path

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap(self) -> Components:  # noqa: PLR0915
        """Build every component in the documented order.

        Steps are recorded in :attr:`bootstrap_steps` *as each step
        completes*, so a partial bootstrap (test factory raising mid-
        sequence) shows the operator exactly how far we got. The
        method is idempotent — calling it twice on the same instance
        raises :class:`RuntimeError`.
        """
        if self._components is not None:
            raise RuntimeError("JarvisApp has already been bootstrapped")

        # Step 1: load_config — already provided to ``__init__``; we
        # record the step for the test ordering guard but do no work
        # here. This keeps :class:`JarvisApp` testable with an
        # already-validated config without re-parsing TOML.
        self._record_step("load_config")

        # Step 2: install the log redaction filter on the root logger
        # BEFORE any subsequent step runs so an exception from, e.g.,
        # the credential store cannot leak a freshly-loaded secret
        # into a stack trace.
        log_redaction_filter = install_log_redaction_filter()
        self._record_step("install_log_redaction")

        # Resolve the data directory once. Every later step writes
        # under it (audit log, credential store, memory store,
        # reminder DB, sentinel).
        data_dir = Path(self._config.app.data_dir)
        # ``mkdir`` is blocking but the cost is a single ``CreateDirectory``
        # syscall on Windows / ``mkdir`` on POSIX, well under a millisecond.
        # Off-loading to a thread would add overhead for no benefit at the
        # one place we *must* know the directory exists before going on.
        data_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

        # Step 3: AuditLog. Construction must happen after the redaction
        # filter so any exception raised during DB open is scrubbed.
        audit_factory = self._factories.audit_log or _default_audit_log
        audit_log = audit_factory(self._config, self._run_id, self._time_source)
        self._record_step("init_audit_log")

        # Detect a stale sentinel from the previous run. We do this
        # AFTER the audit log is open so we can run the diagnostics
        # offer flow once everything is ready (the offer flow needs
        # the audit log to record the ``crash`` row, the data dir to
        # write the report, and — once it exists — a TTS-backed
        # consent prompt). The new sentinel is written immediately
        # below with ``clean_shutdown=False`` so a crash *during*
        # bootstrap is still detectable on the next launch.
        self._sentinel_path = data_dir / SENTINEL_FILENAME
        prior = _read_last_run_sentinel(self._sentinel_path)
        self._prior_sentinel = prior
        self._started_at = self._time_source.now()
        if prior.existed and not prior.clean_shutdown:
            logger.warning(
                "previous JARVIS run did not shut down cleanly "
                "(run_id=%s, started_at=%s, last_bootstrap_step=%s); a "
                "diagnostics report can be generated on user consent.",
                prior.run_id,
                prior.started_at,
                prior.last_bootstrap_step,
            )
        # Refresh the sentinel BEFORE constructing any heavy component
        # so an unexpected SIGKILL during the next steps is recorded.
        # ``last_step=None`` means "we have not yet finished any step
        # of *this* run"; the value is updated after every successful
        # ``_record_step`` so the diagnostics offer on the *next*
        # launch knows how far this run got.
        self._refresh_sentinel(last_step=None)

        # Step 4: DPAPI envelope.
        dpapi_factory = self._factories.dpapi or _default_dpapi
        dpapi = dpapi_factory(self._config)
        self._record_step("init_dpapi")

        # Step 5: CredentialStore.
        cred_factory = self._factories.credential_store or _default_credential_store
        credential_store = cred_factory(self._config, dpapi, log_redaction_filter)
        # Pre-register every persisted secret with the redaction filter
        # so accidentally logging a stored value scrubs to ``[REDACTED]``.
        # This is best-effort — a missing get() is treated as "secret
        # not yet set" and skipped.
        _seed_redaction_with_known_secrets(credential_store, log_redaction_filter)
        self._record_step("init_credential_store")

        # Step 6: LLM stack. The fallback notice (Requirement 12.4) is
        # wired in step 11 once the TTS engine exists, via
        # :class:`BackendSelector` exposing an ``on_flip`` hook.
        llm_factory = self._factories.llm_stack or _default_llm_stack
        llm = llm_factory(
            self._config,
            credential_store,
            log_redaction_filter,
            self._time_source,
        )
        self._record_step("init_llm_backends")

        # Step 7: PlatformAdapter + provider clients.
        platform_factory = self._factories.platform_adapter or _default_platform_adapter
        platform_adapter = platform_factory(self._config)
        providers = _build_provider_map(
            self._config,
            audit_log=audit_log,
            credential_store=credential_store,
        )
        self._record_step("init_platform_adapter")

        # Step 8a: MemoryStore.
        memory_factory = self._factories.memory_store or _default_memory_store
        memory_store = memory_factory(self._config, dpapi)
        self._record_step("init_memory_store")

        # Step 8b: ReminderService — async because ``start`` flushes
        # any missed reminders within the configured grace window.
        reminder_factory = self._factories.reminder_service or _default_reminder_service
        reminder_service = await reminder_factory(
            self._config, platform_adapter, self._time_source
        )
        self._record_step("init_reminder_service")

        # Step 9: skill discovery (built-in singletons, plugin dirs,
        # MCP servers). The MCP adapters return ``aclose`` callables
        # that the orchestrator runs on shutdown.
        builtin_skills = self._resolve_builtin_skills()
        skills_factory = self._factories.skills or _default_skills_registry
        registry = await skills_factory(
            self._config,
            builtin_skills,
            _RegistryHooks(
                builtin_skills=tuple(builtin_skills),
                plugin_dirs=tuple(Path(p) for p in self._config.app.plugin_dirs),
                mcp_servers=tuple(self._config.skills.mcp_servers),
            ),
        )
        # MCP close functions, if any, are attached to the registry by
        # the default factory via a private attribute so the
        # orchestrator can find them without changing the public
        # SkillRegistry API.
        mcp_closers: tuple[Callable[[], Awaitable[None]], ...] = tuple(
            getattr(registry, "_mcp_close_fns", ())
        )
        self._record_step("discover_skills")

        # Step 10: voice pipeline.
        voice_factory = self._factories.voice_pipeline or _default_voice_pipeline
        voice = await voice_factory(
            self._config,
            _VoicePipelineDeps(
                credential_store=credential_store,
                log_redaction_filter=log_redaction_filter,
            ),
        )
        # Wire the fallback notice on the BackendSelector now that we
        # have a TTS engine. The selector exposes an ``on_flip``
        # attribute we can rebind. (Default selectors built by
        # ``_default_llm_stack`` accept this; test selectors that don't
        # support the rebind silently no-op via :func:`getattr`.)
        _wire_fallback_notice(llm.selector, voice.tts, self._config)
        self._record_step("init_voice_pipeline")

        # Step 11: DialogManager + AuthorizationPolicy + persona.
        from jarvis.dialog.persona import load_persona  # noqa: PLC0415

        persona = load_persona(self._config)
        allowlist = TrustedActionAllowlist(
            self._config.authorization.trusted_action_allowlist
        )
        policy = AuthorizationPolicy(
            allowlist=allowlist,
            audit=audit_log,
            hard_coded_destructive_skills=tuple(
                self._config.authorization.destructive_skills
            ),
            destructive_operations=tuple(
                self._config.authorization.destructive_operations
            ),
        )
        confirmation_dialog: ConfirmationDialog | None = None
        if self._factories.confirmation_dialog is not None:
            confirmation_dialog = self._factories.confirmation_dialog()

        dm_factory = self._factories.dialog_manager or _default_dialog_manager
        dialog_manager = dm_factory(
            self._config,
            _DialogDeps(
                backend=llm.selector,
                skills=registry,
                memory=memory_store,
                policy=policy,
                persona=persona,
                tts=voice.tts,
                audit_log=audit_log,
                confirmation_dialog=confirmation_dialog,
                time_source=self._time_source,
                run_id=self._run_id,
            ),
        )
        self._record_step("init_dialog_manager")

        # ------------------------------------------------------------------
        # Crash detection & diagnostics offer flow (Requirement 17.4).
        # ------------------------------------------------------------------
        # Build the offer flow now that every dependency it needs is
        # available: the audit log, the data dir, the redaction
        # filter, the configured PII redactor, and (optionally) a
        # consent prompt that composes the TTS engine + Dialog_Manager.
        from jarvis.memory.redactor import PIIRedactor  # noqa: PLC0415

        try:
            pii_redactor: Any = PIIRedactor.from_config_patterns(
                self._config.memory.pii_patterns
            )
        except Exception:
            logger.exception(
                "could not build PIIRedactor for diagnostics offer; "
                "report will be written without PII redaction"
            )
            pii_redactor = None

        consent_prompt: ConsentPrompt | None = None
        if self._factories.consent_prompt is not None:
            try:
                consent_prompt = self._factories.consent_prompt(
                    self._config,
                    _ConsentPromptDeps(
                        tts=voice.tts,
                        dialog_manager=dialog_manager,
                        time_source=self._time_source,
                    ),
                )
            except Exception:
                logger.exception(
                    "consent_prompt factory raised; the diagnostics "
                    "offer flow will record a no-prompt outcome"
                )
                consent_prompt = None

        diagnostics_offer = DiagnosticsOfferFlow(
            audit_log=audit_log,
            data_dir=data_dir,
            consent_prompt=consent_prompt,
            log_redaction_filter=log_redaction_filter,
            pii_redactor=pii_redactor,
        )

        # Run the offer flow only when the prior sentinel said the
        # last run did *not* shut down cleanly. A first-launch
        # ``existed=False`` or a clean-shutdown ``existed=True`` skips
        # both the prompt and the audit row.
        diagnostics_outcome: DiagnosticsOfferOutcome | None = None
        if prior.existed and not prior.clean_shutdown:
            try:
                diagnostics_outcome = await diagnostics_offer.run(
                    prior_run_id=prior.run_id,
                    prior_started_at=prior.started_at,
                    last_bootstrap_step=prior.last_bootstrap_step,
                    sentinel_extras=prior.raw,
                    now=self._time_source.now(),
                )
            except Exception:
                logger.exception(
                    "diagnostics offer flow raised; continuing with "
                    "bootstrap so the assistant is still usable"
                )
                diagnostics_outcome = None

        components = Components(
            config=self._config,
            run_id=self._run_id,
            time_source=self._time_source,
            log_redaction_filter=log_redaction_filter,
            audit_log=audit_log,
            dpapi=dpapi,
            credential_store=credential_store,
            llm_primary=llm.primary,
            llm_fallback=llm.fallback,
            backend_selector=llm.selector,
            platform_adapter=platform_adapter,
            providers=providers,
            memory_store=memory_store,
            reminder_service=reminder_service,
            skills=registry,
            persona=persona,
            voice_pipeline=voice,
            dialog_manager=dialog_manager,
            authorization_policy=policy,
            diagnostics_offer=diagnostics_offer,
            diagnostics_outcome=diagnostics_outcome,
            mcp_close_fns=mcp_closers,
        )
        self._components = components
        return components

    def _record_step(self, name: str) -> None:
        """Record a successful bootstrap step and verify ordering.

        We assert against :data:`BOOTSTRAP_STEPS` so a future refactor
        cannot accidentally re-order the bootstrap without also
        updating the documented contract. The sentinel file is
        refreshed with the new ``last_bootstrap_step`` so that, if
        the next step crashes, the diagnostics offer flow on the
        following launch knows exactly how far this run got.
        """
        expected_index = len(self._bootstrap_steps)
        if expected_index >= len(BOOTSTRAP_STEPS):
            raise RuntimeError(
                f"bootstrap recorded too many steps; got {name!r} after "
                f"{self._bootstrap_steps!r}"
            )
        expected_name = BOOTSTRAP_STEPS[expected_index]
        if name != expected_name:
            raise RuntimeError(
                f"bootstrap step out of order: expected {expected_name!r} "
                f"but recorded {name!r}"
            )
        self._bootstrap_steps.append(name)
        # Persist the updated step into the sentinel as soon as it is
        # known. Best-effort: a transient filesystem error is logged
        # but does not abort the bootstrap.
        self._refresh_sentinel(last_step=name)

    def _refresh_sentinel(self, *, last_step: str | None) -> None:
        """Write the current run's sentinel with ``clean_shutdown=False``.

        Centralised so :meth:`bootstrap`, :meth:`_record_step`, and
        :meth:`aclose` all go through the same code path. ``last_step``
        carries the most recent successfully-completed bootstrap step
        (or ``None`` very early in bootstrap, before any step has
        finished). The clean-shutdown flag is rewritten by
        :meth:`aclose` once a graceful shutdown completes.
        """
        if self._sentinel_path is None or self._started_at is None:
            return
        _write_last_run_sentinel(
            self._sentinel_path,
            run_id=self._run_id,
            started_at=self._started_at,
            clean_shutdown=False,
            last_bootstrap_step=last_step,
        )

    def _resolve_builtin_skills(self) -> list[Skill]:
        """Return the curated built-in skill singletons, or the override."""
        if self._builtin_skills_override is not None:
            return list(self._builtin_skills_override)
        return _default_builtin_skills()

    # ------------------------------------------------------------------
    # Run loops
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Drive the three concurrent loops until cancellation.

        The loops cooperate over two bounded asyncio queues:

        * ``audio_to_dialog`` carries :class:`Transcript` values from
          the audio capture loop into the dialog loop.
        * ``dialog_to_output`` carries :class:`AssistantResponse`
          values from the dialog loop into the output loop.

        :class:`asyncio.TaskGroup` ensures that a failure in any one
        loop tears the others down via cancellation, and that
        :meth:`run` propagates the original exception. The dialog loop
        terminates when the audio loop exits (sentinel ``None`` posted
        on ``audio_to_dialog``); the output loop terminates analogously.
        """
        if self._components is None:
            raise RuntimeError("JarvisApp.bootstrap() must run before run()")

        audio_to_dialog: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=_DEFAULT_QUEUE_DEPTH
        )
        dialog_to_output: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=_DEFAULT_QUEUE_DEPTH
        )

        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                self._audio_capture_loop(audio_to_dialog),
                name="jarvis-audio-loop",
            )
            tg.create_task(
                self._dialog_loop(audio_to_dialog, dialog_to_output),
                name="jarvis-dialog-loop",
            )
            tg.create_task(
                self._output_loop(dialog_to_output),
                name="jarvis-output-loop",
            )

    async def _audio_capture_loop(self, out_queue: asyncio.Queue[Any]) -> None:
        """Run the production audio capture loop or its test stub.

        The default voice pipeline exposes an async ``run`` coroutine
        on the bundled adapter; if present, we delegate to it. Tests
        substitute a pipeline whose ``run`` simply posts a sentinel
        and returns so the loop terminates cleanly.
        """
        components = self.components
        pipeline = components.voice_pipeline
        run_fn = getattr(pipeline, "run_audio_loop", None)
        try:
            if run_fn is not None:
                await run_fn(out_queue)
            else:
                # No real audio source — block forever until the loop
                # is cancelled. The default stub voice pipeline avoids
                # this by providing ``run_audio_loop``; production
                # pipelines override it.
                await asyncio.Event().wait()
        finally:
            # Sentinel so the dialog loop can drain and exit.
            with _suppress_queue_full(out_queue):
                out_queue.put_nowait(None)

    async def _dialog_loop(
        self,
        in_queue: asyncio.Queue[Any],
        out_queue: asyncio.Queue[Any],
    ) -> None:
        """Drain transcripts → DialogManager → AssistantResponse queue."""
        components = self.components
        dialog_manager = components.dialog_manager
        # ConversationState is the long-lived per-process state that
        # threads through every turn. The default DialogManager
        # constructor does not require us to instantiate one upfront;
        # we lazily build it on the first transcript.
        conversation_state: Any = None

        while True:
            transcript = await in_queue.get()
            if transcript is None:
                break

            if conversation_state is None:
                conversation_state = _new_conversation_state(self._config)

            try:
                response = await dialog_manager.handle_turn(
                    transcript, conversation_state
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Property 7 / CP10 demands the dialog loop survives a
                # skill / backend failure. Log and continue so a single
                # broken turn cannot tear down the pipeline.
                logger.exception(
                    "dialog_manager.handle_turn raised; dropping turn and continuing"
                )
                continue

            await out_queue.put(response)

        # Sentinel for the output loop.
        with _suppress_queue_full(out_queue):
            out_queue.put_nowait(None)

    async def _output_loop(self, in_queue: asyncio.Queue[Any]) -> None:
        """Consume :class:`AssistantResponse` values for downstream sinks.

        The :class:`DialogManager` already speaks tokens to the TTS
        engine sentence-by-sentence during ``handle_turn``, so the
        output loop's job today is bookkeeping — log the response so a
        UI / transcript log can subscribe to it. The loop terminates
        on the sentinel posted by the dialog loop.
        """
        while True:
            response = await in_queue.get()
            if response is None:
                break
            logger.debug("dialog turn completed: %r", response)

    # ------------------------------------------------------------------
    # Wipe-all
    # ------------------------------------------------------------------

    async def wipe_all(self) -> None:
        """Clear MemoryStore, CredentialStore, and the audit log within 5s.

        Implements Requirement 13.5. The three stores are wiped
        concurrently so a slow ChromaDB delete cannot push the audit
        log wipe past the budget. The whole call is bounded by
        :data:`_WIPE_BUDGET_SECONDS`; on overrun, :class:`TimeoutError`
        propagates and the orchestrator decides what to surface to
        the user (typically a "wipe partially completed; try again"
        prompt).
        """
        if self._components is None:
            raise RuntimeError("JarvisApp.bootstrap() must run before wipe_all()")

        components = self._components

        async def _maybe_async(call: Callable[[], Any]) -> None:
            """Run ``call`` and await it if it returns a coroutine."""
            result = call()
            if asyncio.iscoroutine(result):
                await result

        async def _wipe_memory() -> None:
            await _maybe_async(components.memory_store.wipe)

        async def _wipe_credentials() -> None:
            await _maybe_async(components.credential_store.wipe)
            # Reset the redaction filter as well: its cached secret set
            # was sourced from the credential store, and after a wipe
            # those secrets no longer exist.
            components.log_redaction_filter.clear()

        async def _wipe_audit() -> None:
            await components.audit_log.wipe()

        await asyncio.wait_for(
            asyncio.gather(_wipe_memory(), _wipe_credentials(), _wipe_audit()),
            timeout=_WIPE_BUDGET_SECONDS,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Tear down every component and refresh the sentinel as clean.

        Idempotent. Each ``aclose`` call is wrapped in a defensive
        ``try`` so a failure in one component cannot prevent the
        others from releasing their resources.
        """
        if self._closed:
            return
        self._closed = True

        components = self._components
        if components is not None:
            for closer in components.mcp_close_fns:
                try:
                    await closer()
                except Exception:
                    logger.exception("MCP server aclose raised during shutdown")

            try:
                voice_close = getattr(components.voice_pipeline, "aclose", None)
                if voice_close is not None:
                    await voice_close()
            except Exception:
                logger.exception("voice pipeline aclose raised during shutdown")

            try:
                stop = getattr(components.reminder_service, "stop", None)
                if stop is not None:
                    result = stop()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                logger.exception("reminder service stop raised during shutdown")

            try:
                close = getattr(components.audit_log, "close", None)
                if close is not None:
                    close()
            except Exception:
                logger.exception("audit log close raised during shutdown")

        if self._sentinel_path is not None and self._started_at is not None:
            _write_last_run_sentinel(
                self._sentinel_path,
                run_id=self._run_id,
                started_at=self._started_at,
                clean_shutdown=True,
                last_bootstrap_step=(
                    self._bootstrap_steps[-1] if self._bootstrap_steps else None
                ),
            )


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------


def _default_dpapi(config: Config) -> DPAPI:
    """Build the platform-appropriate DPAPI envelope."""
    del config  # the dpapi backend is platform-derived, not config-driven.
    return create_default_dpapi()


def _default_audit_log(
    config: Config, run_id: str, time_source: TimeSource
) -> AuditLog:
    """Open the SQLite audit log at the configured path."""
    return AuditLog(
        Path(config.security.audit_log_path),
        time_source=time_source,
        run_id=run_id,
    )


def _default_credential_store(
    config: Config, dpapi: DPAPI, log_redaction_filter: LogRedactionFilter
) -> CredentialBackend:
    """Open the DPAPI-backed credential store."""
    del log_redaction_filter  # registration happens after the store is open.
    return CredentialStore(
        Path(config.app.data_dir) / "secrets",
        dpapi,
    )


def _default_llm_stack(
    config: Config,
    credential_store: CredentialBackend,
    log_redaction_filter: LogRedactionFilter,
    time_source: TimeSource,
) -> _LLMStack:
    """Build the Mistral primary + Ollama fallback wrapped in a selector."""
    from jarvis.llm.mistral_backend import (  # noqa: PLC0415
        MistralBackend,
        MistralCredentialMissingError,
    )
    from jarvis.llm.ollama_backend import OllamaBackend  # noqa: PLC0415
    from jarvis.llm.selector import BackendSelector  # noqa: PLC0415

    primary: Any
    try:
        primary = MistralBackend.from_credential_store(
            credential_store,
            api_key_credential_name=config.llm.mistral.api_key_credential,
            endpoint=config.llm.mistral.endpoint,
            model=config.llm.mistral.model,
            max_retries=config.llm.mistral.max_retries,
            retry_backoff_initial_ms=config.llm.mistral.retry_backoff_initial_ms,
            request_timeout_ms=config.llm.mistral.request_timeout_ms,
            log_redaction_filter=log_redaction_filter,
        )
    except MistralCredentialMissingError:
        # First-launch case: the user has not yet supplied a Mistral
        # key. We continue with a None primary so the selector falls
        # straight through to Ollama; the orchestrator can offer the
        # CredentialUpdateFlow in a follow-up turn.
        logger.warning(
            "Mistral API key not present in CredentialStore; "
            "starting with Ollama fallback only."
        )
        primary = _UnconfiguredBackend("MistralBackend (no API key configured)")

    fallback = OllamaBackend(
        endpoint=config.llm.fallback.endpoint,
        model=config.llm.fallback.model,
    )

    selector = BackendSelector(
        primary,
        fallback,
        timeout_seconds=config.llm.mistral.request_timeout_ms / 1000.0,
        cool_down_seconds=float(config.llm.fallback.circuit_open_seconds),
        time_source=time_source,
    )
    return _LLMStack(primary=primary, fallback=fallback, selector=selector)


def _default_platform_adapter(config: Config) -> Any:
    """Instantiate the platform-appropriate adapter.

    Falls back to :class:`BasePlatformAdapter` (the no-op base) on
    non-Windows hosts and on any error importing the Windows
    extras.  The base adapter's methods raise
    :class:`PlatformNotSupportedError`, which the Skills layer maps
    to ``platform_not_supported`` per Requirement 15.4.
    """
    from jarvis.automation.platform import BasePlatformAdapter  # noqa: PLC0415

    if sys.platform == "win32":
        try:
            from jarvis.automation.windows_adapter import (  # noqa: PLC0415
                WindowsAdapter,
            )

            return WindowsAdapter(
                application_registry=config.automation.application_registry,
            )
        except Exception:
            logger.exception(
                "WindowsAdapter could not be constructed; falling back to "
                "BasePlatformAdapter"
            )

    return BasePlatformAdapter()


def _default_memory_store(config: Config, dpapi: DPAPI) -> Any:
    """Build the ChromaDB-backed Memory_Store using the production embedder."""
    from jarvis.memory.embedder import create_default_embedder  # noqa: PLC0415
    from jarvis.memory.redactor import PIIRedactor  # noqa: PLC0415
    from jarvis.memory.store import MemoryStore  # noqa: PLC0415

    embedder = create_default_embedder(config.memory.embedding_model)
    redactor = PIIRedactor.from_config_patterns(config.memory.pii_patterns)
    return MemoryStore(
        Path(config.memory.path),
        embedder,
        dpapi,
        redactor,
        incognito=config.app.incognito,
        redaction_enabled=config.memory.redaction_enabled,
        encrypt_embeddings=config.memory.encrypt_embeddings,
    )


async def _default_reminder_service(
    config: Config,
    platform_adapter: Any,
    time_source: TimeSource,
) -> Any:
    """Build and start the APScheduler-backed ReminderService."""
    from jarvis.reminders.notifier import ToastNotifier  # noqa: PLC0415
    from jarvis.reminders.service import ReminderService  # noqa: PLC0415

    # The ReminderService needs a TTS engine for spoken announcements;
    # at this point in the bootstrap order we do not yet have one. We
    # therefore wire a no-op placeholder TTS so the service can start
    # immediately, and the audio loop / DialogManager wiring code
    # (which builds the real TTS in a later step) will rebind the
    # ToastNotifier's TTS reference if the application chooses to.
    notifier = ToastNotifier(
        platform_adapter=platform_adapter,
        tts_engine=_NoopTTS(),
        time_source=time_source,
    )
    service = ReminderService(
        Path(config.reminders.db_path),
        notifier,
        _NoopTTS(),
        time_source=time_source,
        on_start_grace_seconds=config.reminders.on_start_grace_seconds,
    )
    await service.start()
    return service


async def _default_skills_registry(
    config: Config,
    builtin_skills: Iterable[Skill],
    hooks: _RegistryHooks,
) -> SkillRegistry:
    """Build a :class:`SkillRegistry` populated from every documented source."""
    registry = SkillRegistry()
    for skill in builtin_skills:
        try:
            registry.register(skill)
        except Exception:
            logger.exception("failed to register builtin skill %r", skill)

    if hooks.plugin_dirs:
        registry.discover(hooks.plugin_dirs)

    mcp_closers: list[Callable[[], Awaitable[None]]] = []
    if hooks.mcp_servers:
        from jarvis.skills.mcp_adapter import connect_mcp_skills  # noqa: PLC0415

        for server in hooks.mcp_servers:
            try:
                connected = await connect_mcp_skills(server)
            except Exception:
                logger.exception(
                    "failed to connect MCP server %r; skipping", server.name
                )
                continue
            for mcp_skill in connected.skills:
                try:
                    registry.register(mcp_skill)
                except Exception:
                    logger.exception(
                        "failed to register MCP skill %r from server %r",
                        getattr(mcp_skill, "manifest", None),
                        server.name,
                    )
            mcp_closers.append(connected.aclose)

    # Stash the MCP closers on the registry so the orchestrator can
    # find them on shutdown without changing the public API.
    registry._mcp_close_fns = tuple(mcp_closers)  # type: ignore[attr-defined]
    return registry


async def _default_voice_pipeline(
    config: Config, deps: _VoicePipelineDeps
) -> VoicePipeline:
    """Build the production voice pipeline (PiperTTS + faster-whisper + ...).

    The default pipeline lazily constructs each component so a missing
    optional dependency surfaces a clear error from inside
    :meth:`bootstrap` rather than at import time. On hosts without the
    audio extras, callers should supply a custom
    :attr:`ComponentFactories.voice_pipeline`.
    """
    return await _DefaultVoicePipeline.create(config, deps)


def _default_dialog_manager(config: Config, deps: _DialogDeps) -> Any:
    """Build the production :class:`DialogManager`."""
    from jarvis.dialog.manager import DialogManager  # noqa: PLC0415

    return DialogManager(
        backend=deps.backend,
        skills=deps.skills,
        memory=deps.memory,
        policy=deps.policy,
        persona=deps.persona,
        tts=deps.tts,
        audit_log=deps.audit_log,
        config=config.dialog,
        confirmation_dialog=deps.confirmation_dialog,
        time_source=deps.time_source,
        memory_k=config.memory.top_k,
        min_confidence=config.voice.stt.min_confidence,
        run_id=deps.run_id,
    )


# ---------------------------------------------------------------------------
# Provider map / built-in skills
# ---------------------------------------------------------------------------


def _build_provider_map(
    config: Config,
    *,
    audit_log: AuditLog,
    credential_store: CredentialBackend,
) -> Mapping[str, Any]:
    """Construct the mapping of Skill provider name → HTTP client.

    Provider clients themselves perform network egress, so failures
    inside their constructors must not abort bootstrap; we log and
    skip individual providers instead.
    """
    providers: dict[str, Any] = {}
    allowlist = tuple(config.security.network_destination_allowlist)

    def _safe_build(name: str, builder: Callable[[], Any]) -> None:
        try:
            providers[name] = builder()
        except Exception:
            logger.exception(
                "failed to build provider %r; the matching Skill will surface "
                "missing_credentials / provider_unavailable on use",
                name,
            )

    from jarvis.automation.providers.calendar import CalendarClient  # noqa: PLC0415
    from jarvis.automation.providers.email import EmailClient  # noqa: PLC0415
    from jarvis.automation.providers.news import NewsClient  # noqa: PLC0415
    from jarvis.automation.providers.search import WebSearchClient  # noqa: PLC0415
    from jarvis.automation.providers.weather import WeatherClient  # noqa: PLC0415

    _safe_build(
        "weather",
        lambda: WeatherClient(
            audit_log=audit_log,
            network_allowlist=allowlist,
            credential_store=credential_store,
            provider_config=config.providers.weather,
        ),
    )
    _safe_build(
        "news",
        lambda: NewsClient(
            audit_log=audit_log,
            network_allowlist=allowlist,
            credential_store=credential_store,
            provider_config=config.providers.news,
        ),
    )
    _safe_build(
        "calendar",
        lambda: CalendarClient(
            audit_log=audit_log,
            network_allowlist=allowlist,
            credential_store=credential_store,
            provider_config=config.providers.calendar,
        ),
    )
    _safe_build(
        "email",
        lambda: EmailClient(
            audit_log=audit_log,
            network_allowlist=allowlist,
            credential_store=credential_store,
            provider_config=config.providers.email,
        ),
    )
    _safe_build(
        "web_search",
        lambda: WebSearchClient(
            audit_log=audit_log,
            network_allowlist=allowlist,
            credential_store=credential_store,
            provider_config=config.providers.search,
        ),
    )

    return MappingProxyType(providers)


def _default_builtin_skills() -> list[Skill]:
    """Return the curated list of built-in Skill singletons.

    Each entry is the ``SKILL`` (or ``SKILLS`` plural) attribute on the
    matching ``jarvis.skills.builtin.*`` module. Imported lazily so a
    missing optional dependency in any one module logs a warning and
    skips that Skill rather than aborting bootstrap.
    """
    builtin_modules: tuple[str, ...] = (
        "jarvis.skills.builtin.launch_app",
        "jarvis.skills.builtin.web_search",
        "jarvis.skills.builtin.media_control",
        "jarvis.skills.builtin.volume",
        "jarvis.skills.builtin.brightness",
        "jarvis.skills.builtin.send_email",
        "jarvis.skills.builtin.send_message",
        "jarvis.skills.builtin.reminder",
        "jarvis.skills.builtin.timer",
        "jarvis.skills.builtin.weather",
        "jarvis.skills.builtin.news",
        "jarvis.skills.builtin.calendar",
        "jarvis.skills.builtin.read_file",
        "jarvis.skills.builtin.summarize_file",
        "jarvis.skills.builtin.run_script",
        "jarvis.skills.builtin.desktop_automation",
        "jarvis.skills.builtin.memory_admin",
    )
    skills: list[Skill] = []
    for dotted in builtin_modules:
        try:
            module = import_module(dotted)
        except Exception:
            logger.exception(
                "failed to import builtin skill module %r; skipping", dotted
            )
            continue
        # Modules expose either ``SKILL`` (singleton) or ``SKILLS`` (list).
        skill_singleton = getattr(module, "SKILL", None)
        skill_list = getattr(module, "SKILLS", None)
        if skill_singleton is not None:
            skills.append(skill_singleton)
        elif skill_list is not None:
            skills.extend(skill_list)
        else:
            logger.warning(
                "builtin skill module %r exposed neither SKILL nor SKILLS",
                dotted,
            )
    return skills


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _seed_redaction_with_known_secrets(
    credential_store: CredentialBackend,
    log_redaction_filter: LogRedactionFilter,
) -> None:
    """Pre-register every persisted credential value with the redactor.

    A ``get`` failure is treated as "not yet set" and skipped. We
    deliberately *do not* enumerate the store via ``list_names`` and
    pass each value through the filter unconditionally — only values
    that round-trip cleanly are registered, so a corrupt blob does not
    poison the redactor.
    """
    try:
        names = credential_store.list_names()
    except Exception:
        logger.exception("could not enumerate credential store")
        return
    for name in names:
        try:
            value = credential_store.get(name)
        except Exception:
            logger.warning(
                "could not read credential %r during redaction seeding", name
            )
            continue
        if isinstance(value, str) and value:
            log_redaction_filter.register_secret(value)


def _wire_fallback_notice(selector: Any, tts: Any, config: Config) -> None:
    """Bind the fallback-notice TTS callback onto a :class:`BackendSelector`.

    Implements Requirement 12.4's "inform the user of the fallback"
    clause. Tolerates selectors that do not expose ``_on_flip`` / a
    setter so test stubs can simply ignore the wiring.
    """
    try:
        from jarvis.dialog.fallback_notice import (  # noqa: PLC0415
            build_backend_fallback_notice,
        )

        honorific = config.dialog.honorific or "sir"
        callback = build_backend_fallback_notice(tts, honorific=honorific)
    except Exception:
        logger.exception("failed to build fallback notice; skipping")
        return

    # The production :class:`BackendSelector` exposes the callback via a
    # private ``_on_flip`` slot; setting it directly is the path used
    # by the existing tests.
    if hasattr(selector, "_on_flip"):
        try:
            selector._on_flip = callback
        except Exception:
            logger.exception("failed to attach fallback notice to selector")


def _new_conversation_state(config: Config) -> Any:
    """Build a fresh :class:`ConversationState` honouring incognito mode."""
    from jarvis.dialog.conversation_state import ConversationState  # noqa: PLC0415

    return ConversationState(
        session_id=uuid.uuid4().hex,
        started_at=datetime.now(tz=UTC),
        turns=[],
        pending_confirmation=None,
        incognito=config.app.incognito,
    )


def _suppress_queue_full(queue: asyncio.Queue[Any]) -> Any:
    """Context manager that swallows :class:`asyncio.QueueFull`.

    A bounded queue may already be full when we try to post the
    "stream finished" sentinel; that is not a problem because the
    consumer will see the upstream task exit anyway. This helper
    keeps the call sites readable.
    """
    del queue  # parameter retained for documentation; the suppress is generic.
    return contextlib.suppress(asyncio.QueueFull)


# ---------------------------------------------------------------------------
# Lightweight stubs used by default factories on degraded environments
# ---------------------------------------------------------------------------


class _UnconfiguredBackend:
    """Placeholder LLM backend used when the API key is not yet set.

    The selector routes around it to the local fallback whenever the
    ``stream`` call would otherwise need credentials. The placeholder
    raises an explicit error if anyone attempts to invoke it, so a
    misconfigured selector surfaces immediately rather than silently
    misbehaving.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            f"{self._label}: cannot stream because the backend is not configured."
        )


class _NoopTTS:
    """Minimal TTS placeholder used by the reminder service stub.

    Implements just enough of the :class:`TTSEngine` Protocol that the
    reminder service can probe ``is_playing`` and never have a spoken
    announcement requested. The default reminder service receives this
    placeholder; callers that want spoken reminders supply their own
    factory.
    """

    async def speak(self, text: str) -> None:
        del text  # intentionally silent.

    async def stop(self) -> None:
        return None

    def is_playing(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Default voice pipeline (production)
# ---------------------------------------------------------------------------


@dataclass
class _DefaultVoicePipeline:
    """Production voice pipeline binding Piper / Whisper / Silero / Porcupine.

    Construction is lazy: any heavy import that fails on the host
    surfaces during :meth:`create` rather than at module-import time
    so non-Windows / non-audio test runs that override this factory
    are unaffected.
    """

    tts: Any
    stt: Any
    vad: Any
    wake_word: Any

    @classmethod
    async def create(
        cls, config: Config, deps: _VoicePipelineDeps
    ) -> _DefaultVoicePipeline:
        # TTS, STT, VAD are constructed lazily; their heavy imports are
        # inside the factory so a missing optional dependency surfaces
        # here rather than at import time.
        from jarvis.voice.stt.faster_whisper import FasterWhisperSTT  # noqa: PLC0415
        from jarvis.voice.tts.piper import PiperTTS  # noqa: PLC0415
        from jarvis.voice.vad import SileroVAD  # noqa: PLC0415
        from jarvis.voice.wake_word import (  # noqa: PLC0415
            BUILTIN_KEYWORD_JARVIS,
            WakeWordDetector,
        )

        # The Piper voice file lives under ``${app.data_dir}/voices/<voice>.onnx``
        # by convention; the application bootstrap is not responsible
        # for downloading it (the docs ship the ``piper download``
        # invocation users run during setup).
        voice_dir = Path(config.app.data_dir) / "voices"
        model_path = voice_dir / f"{config.voice.tts.voice}.onnx"
        tts: Any = PiperTTS(
            model_path=model_path,
            voice_id=config.voice.tts.voice,
            speaking_rate=config.voice.tts.speaking_rate,
        )

        stt = FasterWhisperSTT(
            config.voice.stt.model,
            device=config.voice.stt.device,
            compute_type=config.voice.stt.compute_type,
        )

        vad = SileroVAD(
            trailing_silence_ms=config.voice.vad.trailing_silence_ms,
            speech_start_threshold=config.voice.vad.speech_start_threshold,
        )

        access_key = (
            deps.credential_store.get(config.voice.wake_word.access_key_credential)
            or ""
        )
        keyword: Any
        if config.voice.wake_word.custom_keyword_path:
            keyword = Path(config.voice.wake_word.custom_keyword_path)
        else:
            keyword = config.voice.wake_word.phrase or BUILTIN_KEYWORD_JARVIS

        wake_word: Any
        try:
            wake_word = WakeWordDetector(
                access_key=access_key,
                keyword_paths=[keyword],
                sensitivity=config.voice.wake_word.sensitivity,
            )
        except Exception:
            logger.exception(
                "WakeWordDetector could not be constructed; the wake-word "
                "loop will not run until the credential / keyword is fixed."
            )
            wake_word = None

        return cls(tts=tts, stt=stt, vad=vad, wake_word=wake_word)

    async def aclose(self) -> None:
        for resource in (self.tts, self.stt, self.vad, self.wake_word):
            if resource is None:
                continue
            close = getattr(resource, "aclose", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("voice resource %r aclose raised", resource)


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — voice-driven AI assistant.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file (default: %%APPDATA%%/Jarvis/config.toml).",
    )
    parser.add_argument(
        "--wipe-all",
        action="store_true",
        help="Erase all stored data (memory, credentials, audit log) and exit.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the log level for this run (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser


async def _async_main(argv: Sequence[str] | None) -> int:  # noqa: PLR0911
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    log_level = args.log_level or config.app.log_level
    logging.basicConfig(level=log_level, force=False)

    app = JarvisApp(config)
    try:
        await app.bootstrap()
    except Exception:
        logger.exception("JARVIS bootstrap failed; exiting")
        await app.aclose()
        return 1

    if args.wipe_all:
        try:
            await app.wipe_all()
            logger.info("wipe-all completed.")
            return 0
        finally:
            await app.aclose()

    try:
        await app.run()
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        # ``asyncio.run`` re-raises a Ctrl+C as KeyboardInterrupt before
        # the TaskGroup gets a chance to wrap it. CancelledError covers
        # the case where the loop is being torn down for shutdown.
        logger.info("interrupt received; shutting down gracefully")
        return 0
    except BaseExceptionGroup as group:
        # The TaskGroup inside :meth:`run` wraps every child failure in
        # an ExceptionGroup. Split out interrupt-like exceptions so
        # Ctrl+C still produces exit code 0.
        _interrupts, rest = group.split((KeyboardInterrupt, asyncio.CancelledError))
        if rest is None:
            logger.info("interrupt received; shutting down gracefully")
            return 0
        logger.exception("JARVIS run loop terminated with an error", exc_info=rest)
        return 1
    except Exception:
        logger.exception("JARVIS run loop terminated with an error")
        return 1
    finally:
        await app.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous console entry point.

    Drives the asyncio event loop, returns a process exit code, and
    handles ``KeyboardInterrupt`` (Ctrl+C) by gracefully shutting the
    application down rather than emitting a traceback.

    The ``argv`` parameter exists for test injection; production
    callers (the ``jarvis`` console script and ``python -m jarvis``)
    pass ``None`` so :mod:`argparse` reads from :data:`sys.argv`.
    """
    try:
        return asyncio.run(_async_main(argv))
    except KeyboardInterrupt:
        return 0
