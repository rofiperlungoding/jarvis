"""JARVIS desktop app — modern CustomTkinter UI with sidebar nav.

Pages:
  * Chat       — primary surface; segmented mode toggle + transcript + input
  * Skills     — list of registered skills with status
  * Settings   — voice / device / theme / persona / API keys

Threading model is identical to ``jarvis_gui.py``:
  * Main thread runs Tk + CustomTkinter event loop
  * One worker thread owns an asyncio event loop running ``DialogManager``,
    ``FasterWhisperSTT``, ``PiperTTS``, and the cloud Mistral backend
  * Communication via ``WORK_QUEUE`` (UI → async) and ``UI_QUEUE`` (async → UI)

First-run detection: shown automatically if no Mistral key is registered.
Also reachable via Settings → "Re-run setup wizard".
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import re
import sys
import threading
import tkinter as tk
import uuid
from collections import deque
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk
import numpy as np
import sounddevice as sd

# Lazy heavy imports happen in the worker thread
from jarvis.config import load_config
from jarvis.dialog.conversation_state import ConversationState
from jarvis.dialog.manager import DialogManager
from jarvis.dialog.persona import load_persona
from jarvis.llm.mistral_backend import MistralBackend
from jarvis.memory.embedder import HashEmbedder
from jarvis.memory.redactor import PIIRedactor
from jarvis.memory.store import MemoryStore
from jarvis.reminders.service import ReminderService
from jarvis.security.audit_log import AuditLog
from jarvis.security.authorization import (
    AuthorizationPolicy,
    TrustedActionAllowlist,
)
from jarvis.security.credential_store import CredentialStore
from jarvis.security.dpapi import create_default_dpapi
from jarvis.skills.base import Skill, SkillContext
from jarvis.skills.builtin.brightness import SKILL as BRIGHTNESS_SKILL
from jarvis.skills.builtin.launch_app import (
    APPLICATION_REGISTRY_EXTRAS_KEY,
    SKILL as LAUNCH_APP_SKILL,
)
from jarvis.skills.builtin.media_control import SKILL as MEDIA_CONTROL_SKILL
from jarvis.skills.builtin.read_file import SKILL as READ_FILE_SKILL
from jarvis.skills.builtin.reminder import (
    REMINDER_SERVICE_EXTRAS_KEY,
    ListReminderSkill,
    ReminderSkill,
)
from jarvis.skills.builtin.summarize_file import SKILL as SUMMARIZE_FILE_SKILL
from jarvis.skills.builtin.timer import SKILL as TIMER_SKILL
from jarvis.skills.builtin.volume import SKILL as VOLUME_SKILL
from jarvis.skills.registry import SkillRegistry
from jarvis.voice.audio_io import (
    PORCUPINE_SAMPLE_RATE_HZ,
    VAD_FRAME_SAMPLES,
    AudioReframer,
)
from jarvis.voice.stt.base import Transcript
from jarvis.voice.stt.faster_whisper import FasterWhisperSTT
from jarvis.voice.tts.piper import PiperTTS
from jarvis.voice.vad import (
    SileroVAD,
    VADEventKind,
    load_default_silero_probability_fn,
)

def _setup_logging() -> Path:
    """Configure logging: WARNING to stderr (if any), INFO+ to a rotating file.

    The file lives at ``%LOCALAPPDATA%\\Jarvis\\logs\\jarvis.log`` and rotates
    at 2 MB with 5 backups. Critical for the bundled ``console=False`` build
    where stdout/stderr are discarded — without this users have NO visibility
    when something goes wrong (Mistral auth fails, skill crashes, TTS device
    dies, etc.).
    """
    from logging.handlers import RotatingFileHandler  # noqa: PLC0415

    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Idempotent: don't double-add when re-entering on hot reload.
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(file_handler)

    # stderr is discarded in the bundled GUI build, but useful when running
    # `python jarvis_app.py` directly during development.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(fmt)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(stream_handler)

    logging.getLogger("jarvis").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    return log_path


_LOG_FILE = _setup_logging()
logger = logging.getLogger("jarvis.app")
logger.info("=" * 60)
logger.info("JARVIS app starting; log file: %s", _LOG_FILE)

SAMPLE_RATE = PORCUPINE_SAMPLE_RATE_HZ
FRAME_SAMPLES = VAD_FRAME_SAMPLES
SAMPLE_WIDTH = 2

# Color palette
ACCENT = "#4fc3f7"
ACCENT_HOVER = "#3aa8d8"
ACCENT_DIM = "#2a6a8a"
BG_DARK = "#1a1a2e"
SIDEBAR_BG = "#16162a"
PANEL_BG = "#252535"
TEXT_FG = "#e8e8e8"
SUBTLE_FG = "#9aa0a6"
USER_BUBBLE = "#2c5d8f"
ASSISTANT_BUBBLE = "#2a3548"
SUCCESS = "#7ed957"
ERROR = "#ff6b6b"
WARNING = "#f0c674"

# ---------------------------------------------------------------------------
# Markdown stripper for TTS (shared with onboarding)
# ---------------------------------------------------------------------------
_MARKDOWN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"```[\s\S]*?```", re.MULTILINE), " "),
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"\*\*\*([^*]+)\*\*\*"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*\n]+)\*"), r"\1"),
    (re.compile(r"___([^_]+)___"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)"), r"\1"),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
    (re.compile(r"!\[([^\]]*)\]\([^)]+\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^\s{0,3}>\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),
    (re.compile(r"(?<!\w)[*_`~](?!\w)"), ""),
]


def strip_markdown_for_speech(text: str) -> str:
    cleaned = text
    for pattern, repl in _MARKDOWN_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class _SpeechFilteringTTS:
    def __init__(self, inner: PiperTTS) -> None:
        self._inner = inner

    @property
    def inner(self) -> PiperTTS:
        return self._inner

    async def speak(self, text: str) -> None:
        cleaned = strip_markdown_for_speech(text)
        if cleaned:
            await self._inner.speak(cleaned)

    async def stop(self) -> None:
        await self._inner.stop()

    def is_playing(self) -> bool:
        return self._inner.is_playing()

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Cross-thread queues
# ---------------------------------------------------------------------------
WORK_QUEUE: queue.Queue[Callable[..., Awaitable[None]] | None] = queue.Queue()
UI_QUEUE: queue.Queue[tuple[str, Any]] = queue.Queue()


def post_ui(kind: str, payload: Any) -> None:
    UI_QUEUE.put((kind, payload))


def _resolve_piper_voice(voice_id: str = "en_GB-alan-medium") -> Path:
    cache = Path.home() / ".cache" / "jarvis" / "piper"
    cache.mkdir(parents=True, exist_ok=True)
    onnx = cache / f"{voice_id}.onnx"
    json_cfg = cache / f"{voice_id}.onnx.json"
    if not (onnx.is_file() and json_cfg.is_file()):
        post_ui("status", f"Downloading Piper voice {voice_id}…")
        from piper.download_voices import download_voice  # noqa: PLC0415

        download_voice(voice_id, cache)
    return onnx


# ---------------------------------------------------------------------------
# Backend worker
# ---------------------------------------------------------------------------
class JarvisWorker:
    """Owns the heavy components and exposes coroutines for the UI thread."""

    def __init__(self) -> None:
        self.manager: DialogManager | None = None
        self.state: ConversationState | None = None
        self.stt: FasterWhisperSTT | None = None
        self.tts: _SpeechFilteringTTS | None = None
        self.audit: AuditLog | None = None
        self.persona_name: str = "JARVIS"
        self.skill_names: list[str] = []
        self._voice_path: Path | None = None
        self._output_device: int | None = None
        self._voice_id: str = "en_GB-alan-medium"
        self._reminder_service: ReminderService | None = None
        self._backend: MistralBackend | None = None
        self._cred_store: CredentialStore | None = None

    async def boot(self) -> None:
        post_ui("status", "Loading config…")
        cfg = load_config()
        secrets_root = Path(cfg.app.data_dir) / "secrets"
        secrets_root.mkdir(parents=True, exist_ok=True)
        self._cred_store = CredentialStore(secrets_root, create_default_dpapi())

        audit_path = Path(cfg.app.data_dir) / "audit.sqlite"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(audit_path, run_id=f"app-{uuid.uuid4().hex[:8]}")

        post_ui("status", "Connecting to Mistral cloud…")
        backend = MistralBackend.from_credential_store(
            self._cred_store,
            api_key_credential_name=cfg.llm.mistral.api_key_credential,
            endpoint=cfg.llm.mistral.endpoint,
            model=cfg.llm.mistral.model,
        )
        self._backend = backend

        memory_dir = Path(cfg.app.data_dir) / "memory" / "app"
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory = MemoryStore(
            db_path=memory_dir,
            embedder=HashEmbedder(dimension=64),
            dpapi=create_default_dpapi(),
            redactor=PIIRedactor.with_defaults(),
            collection_name="jarvis-app",
            redaction_enabled=False,
        )

        post_ui("status", "Wiring Windows skills…")
        platform_adapter = self._build_platform_adapter(cfg)
        reminder_service = await self._build_reminder_service(cfg)
        skills = self._build_skills()
        self.skill_names = list(skills.names)
        application_registry = dict(cfg.automation.application_registry)
        allowed_dirs = tuple(
            Path(os.path.expandvars(p)).expanduser()
            for p in cfg.automation.allowed_directories.paths
        )

        policy = AuthorizationPolicy(
            allowlist=TrustedActionAllowlist(),
            audit=self.audit,
        )
        persona = load_persona(cfg)
        self.persona_name = persona.name

        post_ui("status", "Loading neural voice…")
        voice_path = _resolve_piper_voice(self._voice_id)
        self._voice_path = voice_path
        piper = PiperTTS(
            model_path=voice_path,
            output_device=self._output_device,
            speaking_rate=1.18,
        )
        self.tts = _SpeechFilteringTTS(piper)

        post_ui("status", "Loading speech-to-text model…")
        self.stt = FasterWhisperSTT(
            model_size="small.en", device="cpu", compute_type="int8"
        )
        # Warm Whisper so the first transcription doesn't pay model load.
        await self.stt.transcribe(b"\x00" * (SAMPLE_RATE * SAMPLE_WIDTH // 4),
                                  language="en")

        self.manager = DialogManager(
            backend=backend,
            skills=skills,
            memory=memory,
            policy=policy,
            persona=persona,
            tts=self.tts,
            audit_log=self.audit,
            memory_k=5,
        )
        self._reminder_service = reminder_service
        original_run_id = self.manager._run_id  # noqa: SLF001
        audit_ref = self.audit
        time_ref = self.manager._time  # noqa: SLF001
        backend_ref = backend
        cred_ref = self._cred_store

        def _enriched_context(state: ConversationState) -> SkillContext:
            return SkillContext(
                audit_log=audit_ref,
                time_source=time_ref,
                platform_adapter=platform_adapter,
                credential_store=cred_ref,
                llm_backend=backend_ref,
                allowed_directories=allowed_dirs,
                incognito=state.incognito,
                run_id=original_run_id,
                extras={
                    APPLICATION_REGISTRY_EXTRAS_KEY: application_registry,
                    REMINDER_SERVICE_EXTRAS_KEY: reminder_service,
                },
            )

        self.manager._build_skill_context = _enriched_context  # type: ignore[method-assign]  # noqa: SLF001

        self.state = ConversationState(
            session_id=f"app-{uuid.uuid4().hex[:8]}",
            started_at=datetime.now(tz=UTC),
        )
        post_ui("ready", self.persona_name)
        post_ui("skills", self.skill_names)
        post_ui("status", "Ready")

        # Fire-and-forget update check. Runs in a daemon thread so a
        # slow / unreachable GitHub never extends the boot path.
        # Failures inside ``check_for_updates`` already collapse to
        # ``None`` per its contract, so the only thing that can go
        # wrong here is a programming error in the post_ui call.
        def _update_thread() -> None:
            try:
                from jarvis.update_checker import check_for_updates  # noqa: PLC0415

                result = check_for_updates()
                if result is not None:
                    post_ui("update_available", result)
            except Exception:  # pragma: no cover - logged for diagnostics
                logger.exception("Update check thread crashed")

        threading.Thread(
            target=_update_thread,
            name="jarvis-update-check",
            daemon=True,
        ).start()

    def _build_platform_adapter(self, cfg: Any) -> Any:
        try:
            from jarvis.automation.windows_adapter import WindowsAdapter  # noqa: PLC0415

            return WindowsAdapter(
                application_registry=dict(cfg.automation.application_registry),
            )
        except Exception as exc:
            post_ui("status", f"WindowsAdapter unavailable: {exc}")
            from jarvis.automation.platform import BasePlatformAdapter  # noqa: PLC0415

            return BasePlatformAdapter()

    async def _build_reminder_service(self, cfg: Any) -> ReminderService | None:
        try:
            db_path = Path(os.path.expandvars(str(cfg.reminders.db_path)))
            db_path.parent.mkdir(parents=True, exist_ok=True)

            class _NoopToast:
                async def notify(self, title: str, body: str) -> None:
                    return None

            class _NoopTTSGate:
                async def speak(self, text: str) -> None:
                    return None

                def is_playing(self) -> bool:
                    return False

            service = ReminderService(
                db_path=db_path,
                toast=_NoopToast(),
                tts=_NoopTTSGate(),
            )
            await service.start()
            return service
        except Exception as exc:
            post_ui("status", f"ReminderService disabled: {exc}")
            return None

    def _build_skills(self) -> SkillRegistry:
        registry = SkillRegistry()
        candidates: list[Skill] = [
            LAUNCH_APP_SKILL,
            MEDIA_CONTROL_SKILL,
            VOLUME_SKILL,
            BRIGHTNESS_SKILL,
            TIMER_SKILL,
            ReminderSkill(),
            ListReminderSkill(),
            READ_FILE_SKILL,
            SUMMARIZE_FILE_SKILL,
        ]
        for skill in candidates:
            try:
                registry.register(skill)
            except Exception as exc:
                post_ui("status", f"Skipped {type(skill).__name__}: {exc}")
        return registry

    async def transcribe_and_reply(self, pcm: bytes) -> None:
        if self.stt is None or self.manager is None or self.state is None:
            post_ui("status", "Not ready yet")
            return
        if len(pcm) < SAMPLE_RATE * SAMPLE_WIDTH // 4:
            post_ui("status", "Too short, ignored")
            return
        post_ui("status", f"Transcribing {len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH):.1f}s…")
        transcript = await self.stt.transcribe(pcm, language="en")
        if not transcript.text.strip():
            post_ui("status", "No speech detected")
            return
        post_ui("user", transcript.text)
        await self._reply(transcript)

    async def reply_to_text(self, text: str) -> None:
        if self.manager is None or self.state is None:
            post_ui("status", "Not ready yet")
            return
        post_ui("user", text)
        transcript = Transcript(
            text=text,
            confidence=1.0,
            started_at=datetime.now(tz=UTC),
            duration_ms=0,
            language="en",
        )
        await self._reply(transcript)

    async def set_output_device(self, device: int | None) -> None:
        if self._voice_path is None:
            self._output_device = device
            return
        if device == self._output_device and self.tts is not None:
            return
        post_ui("status", f"Switching output device → {device}")
        old_tts = self.tts
        self._output_device = device
        new_piper = PiperTTS(
            model_path=self._voice_path,
            output_device=device,
            speaking_rate=1.18,
        )
        new_tts = _SpeechFilteringTTS(new_piper)
        self.tts = new_tts
        if self.manager is not None:
            self.manager._tts = new_tts  # noqa: SLF001
        if old_tts is not None:
            try:
                await old_tts.aclose()
            except Exception:
                pass
        post_ui("status", "Output device changed")

    async def set_voice(self, voice_id: str) -> None:
        if voice_id == self._voice_id and self.tts is not None:
            return
        post_ui("status", f"Switching voice → {voice_id}")
        try:
            voice_path = _resolve_piper_voice(voice_id)
        except Exception as exc:
            post_ui("status", f"Voice {voice_id} unavailable: {exc}")
            return
        old_tts = self.tts
        self._voice_id = voice_id
        self._voice_path = voice_path
        new_piper = PiperTTS(
            model_path=voice_path,
            output_device=self._output_device,
            speaking_rate=1.18,
        )
        new_tts = _SpeechFilteringTTS(new_piper)
        self.tts = new_tts
        if self.manager is not None:
            self.manager._tts = new_tts  # noqa: SLF001
        if old_tts is not None:
            try:
                await old_tts.aclose()
            except Exception:
                pass
        post_ui("status", f"Voice: {voice_id}")

    async def _reply(self, transcript: Transcript) -> None:
        assert self.manager is not None and self.state is not None
        post_ui("status", "Thinking…")
        logger.info("handle_turn START: text=%r conf=%.2f", transcript.text, transcript.confidence)
        try:
            response = await asyncio.wait_for(
                self.manager.handle_turn(transcript, self.state),
                timeout=90.0,
            )
        except TimeoutError:
            logger.exception("handle_turn timed out after 90s")
            post_ui("status", "Timeout — backend took >90s to respond", )
            post_ui("assistant_error", "Timeout: the LLM backend did not respond within 90 seconds. Check your network and Mistral API key.")
            return
        except Exception as exc:
            logger.exception("handle_turn raised: %s", exc)
            post_ui("status", f"Error: {exc.__class__.__name__}")
            post_ui("assistant_error", f"{exc.__class__.__name__}: {exc}\n\nLog: %LOCALAPPDATA%\\Jarvis\\logs\\jarvis.log")
            return
        logger.info(
            "handle_turn END: reply_len=%d tool_calls=%d",
            len(response.text),
            len(response.tool_calls),
        )
        if not response.text.strip():
            logger.warning("handle_turn returned empty text — surfacing placeholder")
            post_ui("assistant_error", "(Empty response from backend — see jarvis.log)")
        else:
            post_ui("assistant", response.text)
        if response.tool_calls:
            names = ", ".join(tc.skill_name for tc in response.tool_calls)
            post_ui("toast", f"Ran: {names}")
        post_ui("status", "Ready")


def worker_thread_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker = JarvisWorker()

    async def runner() -> None:
        try:
            await worker.boot()
        except Exception as exc:
            logger.exception("Boot failed: %s", exc)
            post_ui("status", f"Boot failed: {exc.__class__.__name__}")
            post_ui("assistant_error", f"Boot failed: {exc.__class__.__name__}: {exc}\n\nLog: %LOCALAPPDATA%\\Jarvis\\logs\\jarvis.log")
            post_ui("error", str(exc))
            return
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        getter_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-getter")
        try:
            while True:
                job = await loop.run_in_executor(getter_pool, WORK_QUEUE.get)
                if job is None:
                    return
                try:
                    await job(worker)  # type: ignore[misc]
                except Exception as exc:
                    logger.exception("Worker job failed: %s", exc)
                    post_ui("status", f"Job failed: {exc.__class__.__name__}")
                    post_ui("assistant_error", f"Internal error: {exc.__class__.__name__}: {exc}")
        finally:
            getter_pool.shutdown(wait=False, cancel_futures=True)

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
class PushToTalkRecorder:
    def __init__(self, device: int | None) -> None:
        self.device = device
        self._stream: sd.RawInputStream | None = None
        self._buf: bytearray = bytearray()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._buf = bytearray()

        def _cb(indata: Any, frames: int, t: Any, status: Any) -> None:
            with self._lock:
                self._buf.extend(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=_cb,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is None:
            return b""
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            return bytes(self._buf)


class AlwaysListeningCapture:
    def __init__(
        self,
        device: int | None,
        on_segment: Callable[[bytes], None],
        on_event: Callable[[str], None],
    ) -> None:
        self.device = device
        self._on_segment = on_segment
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        vad = SileroVAD(
            trailing_silence_ms=700,
            speech_start_threshold=0.5,
            probability_fn=load_default_silero_probability_fn(),
        )
        reframer = AudioReframer.for_vad()
        q: queue.Queue[bytes] = queue.Queue()

        def _cb(indata: Any, frames: int, t: Any, status: Any) -> None:
            q.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                callback=_cb,
                device=self.device,
            )
            stream.start()
        except Exception as exc:
            self._on_event(f"Mic error: {exc}")
            return

        speaking = False
        pre_roll: deque[bytes] = deque(maxlen=8)
        utterance = bytearray()
        max_bytes = 20 * SAMPLE_RATE * SAMPLE_WIDTH

        try:
            while not self._stop.is_set():
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                for frame in reframer.feed(chunk):
                    events = vad.process(frame)
                    if speaking:
                        utterance.extend(frame)
                        if len(utterance) >= max_bytes:
                            self._on_segment(bytes(utterance))
                            utterance = bytearray()
                            speaking = False
                            pre_roll.clear()
                            continue
                    else:
                        pre_roll.append(frame)

                    for ev in events:
                        if ev.kind is VADEventKind.SPEECH_START and not speaking:
                            speaking = True
                            for pre in pre_roll:
                                utterance.extend(pre)
                            pre_roll.clear()
                            utterance.extend(frame)
                            self._on_event("speaking")
                        elif ev.kind is VADEventKind.SPEECH_END and speaking:
                            speaking = False
                            self._on_event("silence")
                            if utterance:
                                self._on_segment(bytes(utterance))
                            utterance = bytearray()
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
class JarvisApp(ctk.CTk):
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("JARVIS")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARK)

        self.mode_var = ctk.StringVar(value="ptt")
        self.input_device_var = ctk.IntVar(value=self._default_input_device())
        self.output_device_var = ctk.IntVar(value=self._default_output_device())
        self.voice_var = ctk.StringVar(value="en_GB-alan-medium")
        self.theme_var = ctk.StringVar(value="dark")
        self.text_var = ctk.StringVar(value="")
        self.ready = False
        self.current_page = "chat"

        self.ptt_recorder: PushToTalkRecorder | None = None
        self.always_capture: AlwaysListeningCapture | None = None

        self._build_layout()
        self._poll_id = self.after(50, self._poll_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<KeyPress-space>", self._space_press)
        self.bind("<KeyRelease-space>", self._space_release)
        self._space_held = False

    # ----------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR_BG, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo + title
        logo_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_row.pack(fill="x", padx=20, pady=(20, 24))
        logo_canvas = tk.Canvas(logo_row, width=36, height=36, bg=SIDEBAR_BG, highlightthickness=0)
        logo_canvas.create_oval(2, 2, 34, 34, fill="#0a0a18", outline=ACCENT, width=2)
        logo_canvas.create_text(18, 18, text="J", fill="white", font=("Segoe UI", 14, "bold"))
        logo_canvas.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(logo_row, text="JARVIS", font=("Segoe UI", 16, "bold"),
                     text_color=TEXT_FG).pack(side="left")

        # Nav items
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, icon, label in [
            ("chat", "💬", "Chat"),
            ("skills", "⚡", "Skills"),
            ("settings", "⚙", "Settings"),
        ]:
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {icon}   {label}", height=40, anchor="w",
                font=("Segoe UI", 12),
                fg_color="transparent", text_color=TEXT_FG,
                hover_color="#2a2a3a",
                command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav_buttons[key] = btn

        # Bottom: status + version
        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=20, pady=14)
        self.status_dot = tk.Canvas(bottom, width=10, height=10, bg=SIDEBAR_BG, highlightthickness=0)
        self.status_dot.create_oval(1, 1, 9, 9, fill=WARNING, outline="")
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_label = ctk.CTkLabel(bottom, text="Booting…", font=("Segoe UI", 11),
                                          text_color=SUBTLE_FG, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(self.sidebar, text="v1.0.0", font=("Segoe UI", 10),
                     text_color=SUBTLE_FG).pack(side="bottom", pady=(0, 8))

        # Pages container
        self.pages_frame = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.pages_frame.grid(row=0, column=1, sticky="nsew")
        self.pages_frame.grid_columnconfigure(0, weight=1)
        self.pages_frame.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {
            "chat": ChatPage(self.pages_frame, self),
            "skills": SkillsPage(self.pages_frame, self),
            "settings": SettingsPage(self.pages_frame, self),
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self._show_page("chat")

        # Toast container (bottom right of pages_frame)
        self._toast_frame: ctk.CTkFrame | None = None
        # Sticky update-available banner (top of pages_frame)
        self._update_banner: ctk.CTkFrame | None = None

    # ----------------------------------------------------------------- nav
    def _show_page(self, key: str) -> None:
        page = self.pages.get(key)
        if page is None:
            return
        page.tkraise()
        self.current_page = key
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color=ACCENT_DIM, text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_FG)

    # ----------------------------------------------------------------- helpers
    def _default_input_device(self) -> int:
        try:
            return int(sd.default.device[0])
        except Exception:
            return 0

    def _default_output_device(self) -> int:
        try:
            return int(sd.default.device[1])
        except Exception:
            return 0

    def list_input_devices(self) -> list[tuple[int, str]]:
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append((i, d["name"].strip()))
        return out

    def list_output_devices(self) -> list[tuple[int, str]]:
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                out.append((i, d["name"].strip()))
        return out

    def set_status(self, text: str, color: str = SUBTLE_FG, dot_color: str | None = None) -> None:
        self.status_label.configure(text=text, text_color=color)
        if dot_color is not None:
            self.status_dot.delete("all")
            self.status_dot.create_oval(1, 1, 9, 9, fill=dot_color, outline="")

    def show_toast(self, message: str) -> None:
        # destroy existing
        if self._toast_frame is not None:
            try:
                self._toast_frame.destroy()
            except Exception:
                pass
        toast = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=8,
                              border_width=1, border_color=ACCENT)
        ctk.CTkLabel(toast, text=f"  {message}  ", font=("Segoe UI", 11),
                     text_color=TEXT_FG).pack(padx=12, pady=8)
        toast.place(relx=0.98, rely=0.95, anchor="se")
        self._toast_frame = toast
        self.after(3000, lambda: toast.destroy() if toast.winfo_exists() else None)

    def _show_update_banner(self, payload: Any) -> None:
        """Render a sticky banner advertising a newer GitHub release.

        Unlike :meth:`show_toast`, this banner does NOT auto-dismiss —
        the user has to click "Open release page" or the close button.
        We place it at the top of the pages frame so it's visible
        regardless of which page (Chat / Skills / Settings) is
        currently selected, without affecting the grid layout.
        """
        # Defensive: only one banner at a time.
        if self._update_banner is not None:
            try:
                self._update_banner.destroy()
            except Exception:
                pass
            self._update_banner = None

        latest = getattr(payload, "latest", None)
        current = getattr(payload, "current", "?")
        tag = getattr(latest, "tag_name", "?") if latest is not None else "?"
        url = (
            getattr(latest, "html_url", "https://github.com/")
            if latest is not None
            else "https://github.com/"
        )

        banner = ctk.CTkFrame(
            self.pages_frame,
            fg_color="#1c2a44",
            corner_radius=8,
            border_width=1,
            border_color=ACCENT,
            height=44,
        )

        ctk.CTkLabel(
            banner,
            text=f"  ⬆  Update available: {current} → {tag}",
            font=("Segoe UI", 12, "bold"),
            text_color=ACCENT,
            anchor="w",
        ).pack(side="left", padx=12, pady=8)

        def _open() -> None:
            import webbrowser  # noqa: PLC0415

            webbrowser.open(url)

        ctk.CTkButton(
            banner,
            text="Open release page",
            command=_open,
            width=160,
            height=28,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#000000",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right", padx=(4, 8), pady=8)

        def _dismiss() -> None:
            try:
                banner.destroy()
            except Exception:
                pass
            self._update_banner = None

        ctk.CTkButton(
            banner,
            text="✕",
            command=_dismiss,
            width=28,
            height=28,
            fg_color="transparent",
            hover_color=PANEL_BG,
            text_color=TEXT_FG,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="right", padx=(4, 4), pady=8)

        # Float the banner near the top of the pages frame; tk's
        # ``place`` keeps it out of the grid manager's way so the
        # underlying page layout is unaffected.
        banner.place(relx=0.5, rely=0, y=12, anchor="n", relwidth=0.95)
        self._update_banner = banner

    # ----------------------------------------------------------------- ptt
    def _space_press(self, _e: Any) -> None:
        if self.current_page != "chat" or self.mode_var.get() != "ptt" or self._space_held:
            return
        self._space_held = True
        self.pages["chat"].ptt_press()

    def _space_release(self, _e: Any) -> None:
        if not self._space_held:
            return
        self._space_held = False
        self.pages["chat"].ptt_release()

    # ----------------------------------------------------------------- queue poll
    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = UI_QUEUE.get_nowait()
                if kind == "status":
                    self.set_status(str(payload))
                elif kind == "user":
                    self.pages["chat"].append_user(str(payload))
                elif kind == "assistant":
                    self.pages["chat"].append_assistant(str(payload))
                elif kind == "assistant_error":
                    self.pages["chat"].append_assistant_error(str(payload))
                elif kind == "ready":
                    self.ready = True
                    self.set_status("Ready", TEXT_FG, SUCCESS)
                    self.pages["chat"].on_ready()
                    if self.mode_var.get() == "vad":
                        self.pages["chat"].start_vad()
                elif kind == "skills":
                    self.pages["skills"].update_skills(payload)
                elif kind == "toast":
                    self.show_toast(str(payload))
                elif kind == "update_available":
                    self._show_update_banner(payload)
                elif kind == "error":
                    self.set_status(f"Error: {payload}", ERROR, ERROR)
        except queue.Empty:
            pass
        self._poll_id = self.after(50, self._poll_ui_queue)

    # ----------------------------------------------------------------- close
    def _on_close(self) -> None:
        try:
            if self.ptt_recorder is not None:
                self.ptt_recorder.stop()
            self.pages["chat"].stop_vad()
        finally:
            WORK_QUEUE.put(None)
            self.destroy()


# =============================================================================
# Pages
# =============================================================================
class ChatPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, app: JarvisApp) -> None:
        super().__init__(parent, fg_color=BG_DARK, corner_radius=0)
        self.app = app

        # Top bar with title + mode selector
        top = ctk.CTkFrame(self, fg_color="transparent", height=64)
        top.pack(fill="x", padx=24, pady=(20, 12))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="Conversation", font=("Segoe UI", 22, "bold"),
                     text_color=TEXT_FG).pack(side="left")

        # Mode segmented control (right side)
        mode_frame = ctk.CTkFrame(top, fg_color=PANEL_BG, corner_radius=8)
        mode_frame.pack(side="right")
        self._mode_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in [("ptt", "🎙  Hold to talk"), ("vad", "👂  Auto"), ("text", "⌨  Text")]:
            btn = ctk.CTkButton(mode_frame, text=label, height=34, width=130,
                                font=("Segoe UI", 11),
                                fg_color="transparent", text_color=TEXT_FG,
                                hover_color="#3a3a4a", corner_radius=6,
                                command=lambda k=key: self.set_mode(k))
            btn.pack(side="left", padx=2, pady=2)
            self._mode_buttons[key] = btn

        # Transcript scrollable frame
        self.transcript = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.transcript.pack(fill="both", expand=True, padx=24, pady=(0, 12))

        # Welcome message until ready
        self._welcome_label = ctk.CTkLabel(
            self.transcript, text="Booting JARVIS — first launch may take 30–60 seconds while\n"
            "we load the speech model and connect to Mistral.",
            font=("Segoe UI", 12), text_color=SUBTLE_FG, justify="center")
        self._welcome_label.pack(pady=80)

        # Input area
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent", height=120)
        self.input_frame.pack(fill="x", padx=24, pady=(0, 20))
        self.input_frame.pack_propagate(False)

        # PTT button (visible when mode == ptt)
        self.ptt_button = ctk.CTkButton(
            self.input_frame, text="🎙   Hold to talk  (or hold Spacebar)",
            height=80, font=("Segoe UI", 14, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#ffffff",
            corner_radius=12,
        )
        self.ptt_button.bind("<ButtonPress-1>", lambda _e: self.ptt_press())
        self.ptt_button.bind("<ButtonRelease-1>", lambda _e: self.ptt_release())

        # VAD indicator
        self.vad_frame = ctk.CTkFrame(self.input_frame, fg_color=PANEL_BG, corner_radius=12,
                                       height=80)
        self.vad_indicator = ctk.CTkLabel(
            self.vad_frame, text="👂  Always listening — start speaking any time",
            font=("Segoe UI", 13), text_color=ACCENT)
        self.vad_indicator.place(relx=0.5, rely=0.5, anchor="center")

        # Text entry
        self.entry_frame = ctk.CTkFrame(self.input_frame, fg_color=PANEL_BG, corner_radius=12,
                                         height=80)
        self.entry = ctk.CTkEntry(
            self.entry_frame, textvariable=self.app.text_var,
            font=("Segoe UI", 13), height=44, fg_color="transparent",
            border_width=0, placeholder_text="Type a message and press Enter…")
        self.entry.bind("<Return>", lambda _e: self._send_text())
        self.entry.pack(side="left", fill="x", expand=True, padx=(16, 8), pady=18)
        ctk.CTkButton(self.entry_frame, text="Send", width=80, height=36,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color="#ffffff", font=("Segoe UI", 12, "bold"),
                      command=self._send_text).pack(side="right", padx=(8, 16), pady=18)

        self.set_mode("ptt")

    def on_ready(self) -> None:
        try:
            self._welcome_label.destroy()
        except Exception:
            pass

    def set_mode(self, mode: str) -> None:
        self.app.mode_var.set(mode)
        for k, btn in self._mode_buttons.items():
            if k == mode:
                btn.configure(fg_color=ACCENT, text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_FG)

        # Tear down old mode
        if self.app.ptt_recorder is not None:
            self.app.ptt_recorder.stop()
            self.app.ptt_recorder = None
        self.stop_vad()

        for w in (self.ptt_button, self.vad_frame, self.entry_frame):
            w.pack_forget()

        if mode == "ptt":
            self.ptt_button.pack(fill="x")
        elif mode == "vad":
            self.vad_frame.pack(fill="x")
            if self.app.ready:
                self.start_vad()
        elif mode == "text":
            self.entry_frame.pack(fill="x")
            self.entry.focus_set()

    def ptt_press(self) -> None:
        if not self.app.ready or self.app.mode_var.get() != "ptt":
            return
        if self.app.ptt_recorder is not None:
            return
        self.app.ptt_recorder = PushToTalkRecorder(self.app.input_device_var.get())
        try:
            self.app.ptt_recorder.start()
        except Exception as exc:
            self.app.set_status(f"Mic error: {exc}", ERROR, ERROR)
            self.app.ptt_recorder = None
            return
        self.ptt_button.configure(fg_color="#d63838", hover_color="#b02020",
                                   text="🔴   Recording — release to send")
        self.app.set_status("Recording…", ACCENT, ACCENT)

    def ptt_release(self) -> None:
        if self.app.ptt_recorder is None:
            return
        pcm = self.app.ptt_recorder.stop()
        self.app.ptt_recorder = None
        self.ptt_button.configure(fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                   text="🎙   Hold to talk  (or hold Spacebar)")
        if not pcm:
            self.app.set_status("No audio captured", WARNING, WARNING)
            return
        self.app.set_status("Sending…", ACCENT, ACCENT)
        WORK_QUEUE.put(lambda w: w.transcribe_and_reply(pcm))

    def start_vad(self) -> None:
        if self.app.always_capture is not None:
            return
        self.app.always_capture = AlwaysListeningCapture(
            device=self.app.input_device_var.get(),
            on_segment=self._on_vad_segment,
            on_event=lambda kind: self.after(
                0, lambda: self.app.set_status(f"VAD: {kind}", ACCENT, ACCENT)),
        )
        self.app.always_capture.start()
        self.app.set_status("Listening…", ACCENT, ACCENT)

    def stop_vad(self) -> None:
        if self.app.always_capture is None:
            return
        self.app.always_capture.stop()
        self.app.always_capture = None

    def _on_vad_segment(self, pcm: bytes) -> None:
        WORK_QUEUE.put(lambda w: w.transcribe_and_reply(pcm))

    def _send_text(self) -> None:
        if not self.app.ready:
            return
        text = self.app.text_var.get().strip()
        if not text:
            return
        self.app.text_var.set("")
        WORK_QUEUE.put(lambda w: w.reply_to_text(text))

    def append_user(self, text: str) -> None:
        bubble = ctk.CTkFrame(self.transcript, fg_color=USER_BUBBLE, corner_radius=12)
        bubble.pack(anchor="e", padx=(60, 0), pady=4, fill="x")
        ctk.CTkLabel(bubble, text="You", font=("Segoe UI", 9, "bold"),
                     text_color="#bdd9f5", anchor="e").pack(anchor="e", padx=12, pady=(8, 0))
        ctk.CTkLabel(bubble, text=text, font=("Segoe UI", 12),
                     text_color="#ffffff", wraplength=560, justify="left",
                     anchor="w").pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        self.after(50, lambda: self.transcript._parent_canvas.yview_moveto(1.0))

    def append_assistant(self, text: str) -> None:
        bubble = ctk.CTkFrame(self.transcript, fg_color=ASSISTANT_BUBBLE, corner_radius=12)
        bubble.pack(anchor="w", padx=(0, 60), pady=4, fill="x")
        ctk.CTkLabel(bubble, text="JARVIS", font=("Segoe UI", 9, "bold"),
                     text_color=ACCENT, anchor="w").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(bubble, text=text, font=("Segoe UI", 12),
                     text_color=TEXT_FG, wraplength=580, justify="left",
                     anchor="w").pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        self.after(50, lambda: self.transcript._parent_canvas.yview_moveto(1.0))

    def append_assistant_error(self, text: str) -> None:
        """Render a backend error as a red bubble so the user notices.

        Status-bar errors disappear after the next status update; users
        scrolling chat history would never see them. A visible bubble
        with the error class plus a pointer to the log file gives the
        user something to act on (re-run wizard, check network, etc.).
        """
        bubble = ctk.CTkFrame(self.transcript, fg_color="#3a1a1a", corner_radius=12,
                              border_color=ERROR, border_width=1)
        bubble.pack(anchor="w", padx=(0, 60), pady=4, fill="x")
        ctk.CTkLabel(bubble, text="⚠ JARVIS — error", font=("Segoe UI", 9, "bold"),
                     text_color=ERROR, anchor="w").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(bubble, text=text, font=("Segoe UI", 12),
                     text_color="#ffcccc", wraplength=580, justify="left",
                     anchor="w").pack(anchor="w", padx=12, pady=(0, 8), fill="x")
        self.after(50, lambda: self.transcript._parent_canvas.yview_moveto(1.0))


class SkillsPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, app: JarvisApp) -> None:
        super().__init__(parent, fg_color=BG_DARK, corner_radius=0)
        self.app = app

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 12))
        ctk.CTkLabel(top, text="Skills", font=("Segoe UI", 22, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(top, text="Built-in capabilities JARVIS can run on your behalf. "
                     "More can be added by dropping plugins into your config plugin_dirs.",
                     font=("Segoe UI", 11), text_color=SUBTLE_FG,
                     wraplength=700, justify="left").pack(anchor="w", pady=(4, 0))

        self.list_frame = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.list_frame.pack(fill="both", expand=True, padx=24, pady=(8, 24))

        self._render_placeholder()

    def _render_placeholder(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        ctk.CTkLabel(self.list_frame, text="Loading skills…",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG).pack(pady=40)

    def update_skills(self, names: list[str]) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        descriptions = {
            "LaunchAppSkill": ("📱", "Launch applications", "Open Chrome, Notepad, VS Code, "
                                "Spotify, Calculator, and any app registered in config."),
            "MediaControlSkill": ("⏯", "Media transport", "Play, pause, next, previous, stop "
                                   "for any media player listening to media keys."),
            "VolumeSkill": ("🔊", "Master volume", "Set, increase, decrease, mute the system "
                             "volume."),
            "BrightnessSkill": ("☀", "Display brightness", "Adjust your monitor brightness "
                                 "(WMI-based; works on most laptops)."),
            "TimerSkill": ("⏱", "Countdown timers", "'Set a 10 minute timer' — fires a toast "
                            "notification + spoken alert."),
            "ReminderSkill": ("🔔", "Scheduled reminders", "Schedule a reminder for any future "
                              "ISO timestamp. Persists across restarts."),
            "ListReminderSkill": ("📋", "List reminders", "Read back all pending reminders, "
                                  "alarms, and timers."),
            "ReadFileSkill": ("📄", "Read files", "Read documents inside your sandboxed "
                              "allowed directories. Supports txt/md/csv/json/pdf/docx."),
            "SummarizeFileSkill": ("📝", "Summarize files", "Run a file through the LLM and "
                                    "get back a short summary."),
        }
        for name in names:
            emoji, label, desc = descriptions.get(name, ("⚙", name, ""))
            card = ctk.CTkFrame(self.list_frame, fg_color=PANEL_BG, corner_radius=10)
            card.pack(fill="x", pady=4)
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=14)
            ctk.CTkLabel(row, text=emoji, font=("Segoe UI", 20)).pack(side="left", padx=(0, 14))
            text_block = ctk.CTkFrame(row, fg_color="transparent")
            text_block.pack(side="left", fill="x", expand=True)
            top_row = ctk.CTkFrame(text_block, fg_color="transparent")
            top_row.pack(anchor="w", fill="x")
            ctk.CTkLabel(top_row, text=label, font=("Segoe UI", 13, "bold"),
                         text_color=TEXT_FG).pack(side="left")
            ctk.CTkLabel(top_row, text="  ✓ active", font=("Segoe UI", 10),
                         text_color=SUCCESS).pack(side="left", padx=(8, 0))
            ctk.CTkLabel(text_block, text=desc, font=("Segoe UI", 11),
                         text_color=SUBTLE_FG, wraplength=600, justify="left",
                         anchor="w").pack(anchor="w", pady=(2, 0), fill="x")
            ctk.CTkLabel(text_block, text=f"  {name}", font=("Consolas", 9),
                         text_color="#5a6068", anchor="w").pack(anchor="w", pady=(2, 0))


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, app: JarvisApp) -> None:
        super().__init__(parent, fg_color=BG_DARK, corner_radius=0)
        self.app = app

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 12))
        ctk.CTkLabel(top, text="Settings", font=("Segoe UI", 22, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(top, text="Customize JARVIS to your environment.",
                     font=("Segoe UI", 11), text_color=SUBTLE_FG).pack(anchor="w", pady=(4, 0))

        body = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=(8, 24))

        # Voice section
        self._section(body, "Voice", "How JARVIS sounds.")
        self._add_combobox(
            body, "Voice", values=[
                "en_GB-alan-medium",
                "en_GB-cori-high",
                "en_GB-northern_english_male-medium",
                "en_US-ryan-high",
            ], variable=self.app.voice_var,
            on_change=lambda v: WORK_QUEUE.put(lambda w: w.set_voice(v)))

        # Audio devices
        self._section(body, "Audio", "Microphone and speaker selection.")
        mics = [f"[{i}] {name}" for i, name in self.app.list_input_devices()]
        outs = [f"[{i}] {name}" for i, name in self.app.list_output_devices()]
        self._add_device_combo(body, "Microphone", mics, self.app.input_device_var, "input")
        self._add_device_combo(body, "Output", outs, self.app.output_device_var, "output")

        # Theme
        self._section(body, "Appearance", "Light or dark mode.")
        self._add_combobox(
            body, "Theme", values=["dark", "light", "system"],
            variable=self.app.theme_var,
            on_change=lambda v: ctk.set_appearance_mode(v))

        # API keys
        self._section(body, "Credentials", "Encrypted with Windows DPAPI.")
        re_run = ctk.CTkButton(body, text="🪄  Re-run setup wizard", width=240, height=36,
                                fg_color=PANEL_BG, hover_color="#3a3a4a", text_color=TEXT_FG,
                                anchor="w", command=self._launch_wizard)
        re_run.pack(anchor="w", pady=(0, 12))

        # About
        self._section(body, "About", "")
        ctk.CTkLabel(body, text="JARVIS v1.0.0  •  MIT License", font=("Segoe UI", 11),
                     text_color=SUBTLE_FG, anchor="w").pack(anchor="w")

    def _section(self, parent: tk.Widget, title: str, subtitle: str) -> None:
        spacer = ctk.CTkFrame(parent, fg_color="transparent", height=12)
        spacer.pack(fill="x")
        ctk.CTkLabel(parent, text=title, font=("Segoe UI", 14, "bold"),
                     text_color=TEXT_FG, anchor="w").pack(anchor="w", pady=(8, 0))
        if subtitle:
            ctk.CTkLabel(parent, text=subtitle, font=("Segoe UI", 10),
                         text_color=SUBTLE_FG, anchor="w").pack(anchor="w", pady=(0, 6))

    def _add_combobox(self, parent: tk.Widget, label: str, values: list[str],
                       variable: ctk.StringVar,
                       on_change: Callable[[str], None]) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=SUBTLE_FG,
                     width=120, anchor="w").pack(side="left")
        combo = ctk.CTkComboBox(row, values=values, variable=variable,
                                  width=320, height=32, font=("Segoe UI", 11),
                                  command=on_change)
        combo.pack(side="left", padx=10)

    def _add_device_combo(self, parent: tk.Widget, label: str, values: list[str],
                          variable: ctk.IntVar, kind: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color=SUBTLE_FG,
                     width=120, anchor="w").pack(side="left")

        def _on_change(sel: str) -> None:
            try:
                idx = int(sel.split("]", 1)[0].lstrip("["))
                variable.set(idx)
                if kind == "output":
                    WORK_QUEUE.put(lambda w: w.set_output_device(idx))
                # Input change just updates state; mode restart picks it up
            except Exception:
                pass

        combo = ctk.CTkComboBox(row, values=values, width=420, height=32,
                                  font=("Segoe UI", 11), command=_on_change)
        # Pre-select current
        for v in values:
            try:
                if int(v.split("]", 1)[0].lstrip("[")) == variable.get():
                    combo.set(v)
                    break
            except Exception:
                continue
        combo.pack(side="left", padx=10)

    def _launch_wizard(self) -> None:
        try:
            import subprocess
            subprocess.Popen([sys.executable, "-m", "onboarding"])
        except Exception as exc:
            self.app.show_toast(f"Could not launch wizard: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _check_first_run() -> bool:
    """Return True if onboarding wizard should be shown."""
    try:
        cfg = load_config()
        secrets = Path(cfg.app.data_dir) / "secrets"
        if not secrets.exists():
            return True
        store = CredentialStore(secrets, create_default_dpapi())
        return store.get("mistral/api_key") is None
    except Exception:
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="JARVIS desktop app")
    parser.add_argument("--first-run", action="store_true",
                        help="Force the onboarding wizard")
    parser.add_argument("--skip-onboarding", action="store_true",
                        help="Skip onboarding even if no API key is set")
    args = parser.parse_args()

    if args.first_run or (_check_first_run() and not args.skip_onboarding):
        from onboarding import run_onboarding  # noqa: PLC0415
        ok = run_onboarding()
        if not ok:
            print("Onboarding cancelled or no API key registered. Exiting.", file=sys.stderr)
            return 1

    # Verify Mistral key is now present
    try:
        cfg = load_config()
        secrets = Path(cfg.app.data_dir) / "secrets"
        store = CredentialStore(secrets, create_default_dpapi())
        if store.get("mistral/api_key") is None:
            print("No Mistral API key registered. Run with --first-run to set one up.",
                  file=sys.stderr)
            return 2
    except Exception:
        pass

    worker = threading.Thread(target=worker_thread_main, daemon=True)
    worker.start()

    app = JarvisApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
