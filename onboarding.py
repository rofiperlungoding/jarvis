"""First-run onboarding wizard for JARVIS.

Multi-page CustomTkinter dialog that walks a brand-new user through:

1. Welcome
2. Mistral API key registration (validated against the live API)
3. Optional provider keys (Tavily / OpenWeather / NewsAPI)
4. Microphone test (record + play back)
5. Speaker / voice test (synthesise + play)
6. Quick tour of the three modes
7. Finish

Returns ``True`` from :func:`run_onboarding` when the user completes the
flow and the Mistral key is registered, ``False`` if they cancel.

The wizard is launched from :mod:`jarvis_app` whenever
``CredentialStore.get("mistral/api_key")`` is ``None`` or whenever the
``--first-run`` CLI flag is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import tkinter as tk
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import customtkinter as ctk
import numpy as np
import sounddevice as sd

logger = logging.getLogger("jarvis.onboarding")

ACCENT = "#4fc3f7"
ACCENT_HOVER = "#3aa8d8"
BG_DARK = "#1e1e2e"
SIDEBAR_BG = "#2a2a3a"
TEXT_FG = "#e8e8e8"
SUBTLE_FG = "#9aa0a6"
ERROR_FG = "#ff6b6b"
SUCCESS_FG = "#7ed957"


class OnboardingWizard(ctk.CTk):
    """Top-level CustomTkinter wizard window.

    Pages live as dedicated frames stacked in the same parent and shown
    one at a time via ``tkraise``. Each page exposes ``can_advance`` and
    ``on_enter`` / ``on_leave`` hooks the controller calls.
    """

    PAGES = [
        "welcome",
        "mistral",
        "providers",
        "mic_test",
        "speaker_test",
        "tour",
        "finish",
    ]

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("JARVIS Setup")
        self.geometry("780x560")
        self.minsize(720, 520)
        self.configure(fg_color=BG_DARK)

        self._completed: bool = False
        self._current_index: int = 0
        self._state: dict[str, Any] = {
            "mistral_key_registered": False,
            "mic_ok": False,
            "speaker_ok": False,
            "providers": {},
            "selected_input": None,
            "selected_output": None,
            "selected_voice": "en_GB-alan-medium",
        }

        # Lazily-created CredentialStore + voice cache
        self._cred_store: Any | None = None

        self._build_layout()
        self._show_page("welcome")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        # Header strip with logo + title
        header = ctk.CTkFrame(self, fg_color=SIDEBAR_BG, height=72, corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Simple "logo": a circle with J inside, drawn on a small canvas
        logo = tk.Canvas(header, width=44, height=44, bg=SIDEBAR_BG, highlightthickness=0)
        logo.create_oval(2, 2, 42, 42, fill="#1a1a2e", outline=ACCENT, width=2)
        logo.create_text(22, 22, text="J", fill="white", font=("Segoe UI", 18, "bold"))
        logo.pack(side="left", padx=(20, 12), pady=14)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", fill="y")
        ctk.CTkLabel(title_frame, text="JARVIS Setup", font=("Segoe UI", 16, "bold"),
                     text_color=TEXT_FG).pack(anchor="w", pady=(14, 0))
        ctk.CTkLabel(title_frame, text="Welcome — let's get you up and running",
                     font=("Segoe UI", 11), text_color=SUBTLE_FG).pack(anchor="w")

        # Step indicator
        self._step_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 11),
                                        text_color=SUBTLE_FG)
        self._step_label.pack(side="right", padx=20)

        # Body: container that swaps page frames in/out
        self._body = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self._body.pack(fill="both", expand=True, side="top")

        # Footer with Back / Skip / Next buttons
        footer = ctk.CTkFrame(self, fg_color=SIDEBAR_BG, height=72, corner_radius=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self._back_btn = ctk.CTkButton(footer, text="Back", width=100, height=36,
                                       fg_color="transparent", border_width=1,
                                       border_color=SUBTLE_FG, text_color=TEXT_FG,
                                       hover_color="#3a3a4a", command=self._go_back)
        self._back_btn.pack(side="left", padx=(20, 10), pady=18)

        self._skip_btn = ctk.CTkButton(footer, text="Skip", width=80, height=36,
                                       fg_color="transparent", text_color=SUBTLE_FG,
                                       hover_color="#3a3a4a", command=self._go_next)
        self._skip_btn.pack(side="left", padx=4, pady=18)

        self._next_btn = ctk.CTkButton(footer, text="Next →", width=140, height=36,
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                       text_color="#ffffff", font=("Segoe UI", 12, "bold"),
                                       command=self._go_next)
        self._next_btn.pack(side="right", padx=(10, 20), pady=18)

        # Build all pages once; show only the active one
        self._pages: dict[str, ctk.CTkFrame] = {}
        self._pages["welcome"] = WelcomePage(self._body, self)
        self._pages["mistral"] = MistralKeyPage(self._body, self)
        self._pages["providers"] = ProvidersPage(self._body, self)
        self._pages["mic_test"] = MicTestPage(self._body, self)
        self._pages["speaker_test"] = SpeakerTestPage(self._body, self)
        self._pages["tour"] = TourPage(self._body, self)
        self._pages["finish"] = FinishPage(self._body, self)
        for page in self._pages.values():
            page.place(x=0, y=0, relwidth=1, relheight=1)

    # ----------------------------------------------------------------- nav
    def _show_page(self, key: str) -> None:
        page = self._pages[key]
        page.tkraise()
        self._current_index = self.PAGES.index(key)
        self._step_label.configure(text=f"Step {self._current_index + 1} of {len(self.PAGES)}")

        # Adjust footer buttons per page
        self._back_btn.configure(state="normal" if self._current_index > 0 else "disabled")

        last = self._current_index == len(self.PAGES) - 1
        self._next_btn.configure(text="Finish ✓" if last else "Next →")

        # Skip allowed on optional pages
        skip_pages = {"providers", "mic_test", "speaker_test", "tour"}
        if key in skip_pages:
            self._skip_btn.pack(side="left", padx=4, pady=18)
        else:
            self._skip_btn.pack_forget()

        if hasattr(page, "on_enter"):
            page.on_enter()

    def _go_next(self) -> None:
        page_key = self.PAGES[self._current_index]
        page = self._pages[page_key]
        if hasattr(page, "can_advance"):
            ok, msg = page.can_advance()
            if not ok:
                page.show_error(msg)
                return
        if hasattr(page, "on_leave"):
            page.on_leave()
        if self._current_index >= len(self.PAGES) - 1:
            self._completed = True
            self.destroy()
            return
        self._show_page(self.PAGES[self._current_index + 1])

    def _go_back(self) -> None:
        if self._current_index <= 0:
            return
        self._show_page(self.PAGES[self._current_index - 1])

    def _on_close(self) -> None:
        self._completed = self._state.get("mistral_key_registered", False)
        self.destroy()

    # --- exposed to pages
    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @property
    def cred_store(self) -> Any:
        if self._cred_store is None:
            from jarvis.config import load_config
            from jarvis.security.credential_store import CredentialStore
            from jarvis.security.dpapi import create_default_dpapi
            cfg = load_config()
            secrets = Path(cfg.app.data_dir) / "secrets"
            secrets.mkdir(parents=True, exist_ok=True)
            self._cred_store = CredentialStore(secrets, create_default_dpapi())
        return self._cred_store

    @property
    def completed(self) -> bool:
        return self._completed


# =============================================================================
# Page: Welcome
# =============================================================================
class WelcomePage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(wrapper, text="Welcome to JARVIS", font=("Segoe UI", 28, "bold"),
                     text_color=TEXT_FG).pack(pady=(0, 12))

        ctk.CTkLabel(wrapper,
                     text=("Your private, voice-driven AI assistant for Windows.\n"
                           "Powered by Mistral cloud LLM, local Whisper speech-to-text,\n"
                           "and neural Piper text-to-speech."),
                     font=("Segoe UI", 13), text_color=SUBTLE_FG, justify="center").pack(pady=4)

        ctk.CTkLabel(wrapper, text="This setup wizard will get you running in about 60 seconds.",
                     font=("Segoe UI", 12, "italic"), text_color=ACCENT).pack(pady=(28, 0))

        bullets = [
            ("🔑", "Connect your Mistral API key (free tier works)"),
            ("🎤", "Pick and test your microphone"),
            ("🔊", "Pick a voice and test your speakers"),
            ("⚡", "Learn the three ways to talk to JARVIS"),
        ]
        for emoji, text in bullets:
            row = ctk.CTkFrame(wrapper, fg_color="transparent")
            row.pack(anchor="w", pady=4)
            ctk.CTkLabel(row, text=emoji, font=("Segoe UI", 16)).pack(side="left", padx=(0, 12))
            ctk.CTkLabel(row, text=text, font=("Segoe UI", 12), text_color=TEXT_FG).pack(side="left")

    def can_advance(self) -> tuple[bool, str]:
        return True, ""

    def show_error(self, msg: str) -> None:
        pass


# =============================================================================
# Page: Mistral key
# =============================================================================
class MistralKeyPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        self._validating = False

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=60, pady=40)

        ctk.CTkLabel(wrapper, text="Connect your Mistral AI account",
                     font=("Segoe UI", 22, "bold"), text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(wrapper,
                     text="JARVIS uses Mistral as its primary language model. The free "
                          "tier is more than enough to get started.",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG, wraplength=620,
                     justify="left").pack(anchor="w", pady=(8, 24))

        # Step-by-step guidance
        steps_frame = ctk.CTkFrame(wrapper, fg_color="#252535", corner_radius=8)
        steps_frame.pack(fill="x", pady=(0, 20))
        for i, text in enumerate([
            "1. Go to console.mistral.ai and sign up (free).",
            "2. Open the API Keys section in your dashboard.",
            "3. Click 'Create new key', copy the value, paste it below.",
        ], 1):
            ctk.CTkLabel(steps_frame, text=text, font=("Segoe UI", 12),
                         text_color=TEXT_FG).pack(anchor="w", padx=16, pady=(10 if i == 1 else 4,
                                                                              10 if i == 3 else 4))

        link_btn = ctk.CTkButton(wrapper, text="🌐  Open console.mistral.ai", width=240, height=34,
                                 fg_color="#252535", hover_color="#303040", text_color=ACCENT,
                                 anchor="w", command=lambda: self._open_url("https://console.mistral.ai"))
        link_btn.pack(anchor="w", pady=(0, 16))

        ctk.CTkLabel(wrapper, text="API Key", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        self.key_entry = ctk.CTkEntry(wrapper, height=40, font=("Consolas", 13), show="•",
                                      placeholder_text="paste your key here…")
        self.key_entry.pack(fill="x", pady=(6, 8))

        self.show_key_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(wrapper, text="Show key", variable=self.show_key_var,
                        command=self._toggle_show, font=("Segoe UI", 11)).pack(anchor="w")

        self.status_label = ctk.CTkLabel(wrapper, text="", font=("Segoe UI", 11),
                                          text_color=SUBTLE_FG)
        self.status_label.pack(anchor="w", pady=(16, 0))

    def on_enter(self) -> None:
        # Pre-fill if a key is already registered
        try:
            existing = self.controller.cred_store.get("mistral/api_key")
            if existing:
                self.key_entry.delete(0, "end")
                self.key_entry.insert(0, existing)
                self.status_label.configure(
                    text="✓ A key is already registered. You can keep it or replace it.",
                    text_color=SUCCESS_FG)
                self.controller.state["mistral_key_registered"] = True
        except Exception as exc:  # nosec - pre-fill is best-effort
            logger.debug("Could not preload Mistral key: %s", exc)

    def _toggle_show(self) -> None:
        self.key_entry.configure(show="" if self.show_key_var.get() else "•")

    def _open_url(self, url: str) -> None:
        import webbrowser
        webbrowser.open(url)

    def can_advance(self) -> tuple[bool, str]:
        if self.controller.state["mistral_key_registered"]:
            return True, ""
        key = self.key_entry.get().strip()
        if not key:
            return False, "Please paste your Mistral API key first."
        if self._validating:
            return False, "Validating…"
        # Synchronously validate (small request, < 3s)
        self._validating = True
        self.status_label.configure(text="⏳ Validating key…", text_color=SUBTLE_FG)
        self.update_idletasks()
        ok, msg = self._validate_key(key)
        self._validating = False
        if ok:
            try:
                self.controller.cred_store.set("mistral/api_key", key)
                self.controller.state["mistral_key_registered"] = True
                self.status_label.configure(text="✓ Key validated and saved securely.",
                                            text_color=SUCCESS_FG)
                return True, ""
            except Exception as exc:
                return False, f"Could not save key: {exc}"
        else:
            self.status_label.configure(text=f"✗ {msg}", text_color=ERROR_FG)
            return False, msg

    def show_error(self, msg: str) -> None:
        self.status_label.configure(text=f"✗ {msg}", text_color=ERROR_FG)

    def _validate_key(self, key: str) -> tuple[bool, str]:
        """Make a tiny chat call to confirm the key works."""
        try:
            import asyncio
            from jarvis.config import load_config
            from jarvis.llm.mistral_backend import MistralBackend
            from jarvis.llm.base import ContentDeltaEvent

            cfg = load_config()

            async def _check() -> tuple[bool, str]:
                backend = MistralBackend(api_key=key, endpoint=cfg.llm.mistral.endpoint,
                                         model=cfg.llm.mistral.model)
                try:
                    async with backend.stream(
                        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                        tools=[],
                    ) as events:
                        async for ev in events:
                            if isinstance(ev, ContentDeltaEvent):
                                return True, "ok"
                    return True, "ok"
                except Exception as exc:
                    return False, str(exc)

            return asyncio.run(_check())
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "403" in msg or "Unauthorized" in msg:
                return False, "The key was rejected by Mistral. Double-check it."
            return False, f"Could not reach Mistral: {msg[:80]}"


# =============================================================================
# Page: Optional providers
# =============================================================================
class ProvidersPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=60, pady=40)

        ctk.CTkLabel(wrapper, text="Optional: Enable extra abilities",
                     font=("Segoe UI", 22, "bold"), text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(wrapper,
                     text="Add API keys for these services to unlock more skills. "
                          "Leave blank to skip — you can add them later in Settings.",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG, wraplength=620,
                     justify="left").pack(anchor="w", pady=(8, 16))

        self.entries: dict[str, ctk.CTkEntry] = {}

        for cred_name, label, desc, url in [
            ("search/api_key", "Tavily (web search)",
             "Lets JARVIS search the web", "https://tavily.com"),
            ("weather/api_key", "OpenWeather",
             "Lets JARVIS report weather", "https://openweathermap.org/api"),
            ("news/api_key", "NewsAPI",
             "Lets JARVIS read headlines", "https://newsapi.org/register"),
        ]:
            block = ctk.CTkFrame(wrapper, fg_color="#252535", corner_radius=8)
            block.pack(fill="x", pady=6)
            top = ctk.CTkFrame(block, fg_color="transparent")
            top.pack(fill="x", padx=14, pady=(10, 4))
            ctk.CTkLabel(top, text=label, font=("Segoe UI", 13, "bold"),
                         text_color=TEXT_FG).pack(side="left")
            ctk.CTkLabel(top, text=desc, font=("Segoe UI", 10),
                         text_color=SUBTLE_FG).pack(side="left", padx=(8, 0))
            link = ctk.CTkLabel(top, text=f"  ↗ {url}", font=("Segoe UI", 10),
                                text_color=ACCENT, cursor="hand2")
            link.pack(side="right")
            link.bind("<Button-1>", lambda _e, u=url: __import__("webbrowser").open(u))

            entry = ctk.CTkEntry(block, height=32, font=("Consolas", 11), show="•",
                                 placeholder_text=f"{label} key (optional)")
            entry.pack(fill="x", padx=14, pady=(0, 12))
            self.entries[cred_name] = entry

    def on_enter(self) -> None:
        for cred_name, entry in self.entries.items():
            try:
                existing = self.controller.cred_store.get(cred_name)
                if existing:
                    entry.delete(0, "end")
                    entry.insert(0, existing)
            except Exception:
                pass

    def can_advance(self) -> tuple[bool, str]:
        # Save anything that was entered
        for cred_name, entry in self.entries.items():
            value = entry.get().strip()
            if value:
                try:
                    self.controller.cred_store.set(cred_name, value)
                    self.controller.state["providers"][cred_name] = True
                except Exception as exc:
                    return False, f"Could not save {cred_name}: {exc}"
        return True, ""

    def show_error(self, msg: str) -> None:
        pass


# =============================================================================
# Page: Mic test
# =============================================================================
class MicTestPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        self._stream: sd.RawInputStream | None = None
        self._buf = bytearray()
        self._recording = False

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=60, pady=40)

        ctk.CTkLabel(wrapper, text="Microphone test", font=("Segoe UI", 22, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(wrapper,
                     text="Pick your microphone, then click Record. Speak for 3 seconds. "
                          "We'll play it back so you can confirm we heard you.",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG, wraplength=620,
                     justify="left").pack(anchor="w", pady=(8, 16))

        ctk.CTkLabel(wrapper, text="Microphone", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        self.mic_combo = ctk.CTkComboBox(wrapper, height=36, font=("Segoe UI", 11),
                                          values=self._mic_choices(),
                                          command=self._on_mic_change)
        self.mic_combo.pack(fill="x", pady=(6, 16))
        if self.mic_combo.cget("values"):
            self.mic_combo.set(self.mic_combo.cget("values")[0])
            self._on_mic_change(self.mic_combo.get())

        # Big record button
        self.record_btn = ctk.CTkButton(wrapper, text="🎤   Record 3 seconds", height=56,
                                        font=("Segoe UI", 14, "bold"),
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                        command=self._record)
        self.record_btn.pack(fill="x", pady=(8, 12))

        self.status_label = ctk.CTkLabel(wrapper, text="", font=("Segoe UI", 12),
                                          text_color=SUBTLE_FG, wraplength=620)
        self.status_label.pack(anchor="w", pady=(4, 0))

        self.confirm_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
        self.confirm_frame.pack(anchor="w", pady=(8, 0))

    def _mic_choices(self) -> list[str]:
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append(f"[{i}] {d['name'].strip()}")
        return out

    def _on_mic_change(self, sel: str) -> None:
        try:
            idx = int(sel.split("]", 1)[0].lstrip("["))
            self.controller.state["selected_input"] = idx
        except Exception:
            self.controller.state["selected_input"] = None

    def _record(self) -> None:
        if self._recording:
            return
        self.record_btn.configure(state="disabled", text="🔴   Recording…")
        self.status_label.configure(text="Speak now (3 seconds)…", text_color=ACCENT)
        self.update_idletasks()

        device = self.controller.state.get("selected_input")
        self._buf = bytearray()
        try:
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1,
                           dtype="int16", device=device, blocking=True)
        except Exception as exc:
            self.status_label.configure(text=f"✗ Mic error: {exc}", text_color=ERROR_FG)
            self.record_btn.configure(state="normal", text="🎤   Record 3 seconds")
            return

        # Quick level check
        peak = int(np.max(np.abs(audio)))
        if peak < 200:
            self.status_label.configure(
                text=f"⚠  Recorded but audio was very quiet (peak={peak}). "
                     "Try a different mic, or speak louder.",
                text_color="#f0c674")
        else:
            self.status_label.configure(
                text=f"✓ Recorded {3.0:.1f}s, peak level {peak}. Now playing back…",
                text_color=SUCCESS_FG)

        try:
            sd.play(audio, samplerate=16000, blocking=True)
        except Exception as exc:
            self.status_label.configure(text=f"Playback error: {exc}", text_color=ERROR_FG)

        self.record_btn.configure(state="normal", text="🎤   Record again")

        # Show confirm buttons
        for child in self.confirm_frame.winfo_children():
            child.destroy()
        ctk.CTkLabel(self.confirm_frame, text="Could you hear yourself clearly?",
                     font=("Segoe UI", 12, "bold"), text_color=TEXT_FG).pack(anchor="w", pady=(8, 4))
        btn_row = ctk.CTkFrame(self.confirm_frame, fg_color="transparent")
        btn_row.pack(anchor="w", pady=4)
        ctk.CTkButton(btn_row, text="✓  Yes, that's me", width=160, fg_color=SUCCESS_FG,
                      hover_color="#6cbf4a", text_color="#0a2810",
                      command=lambda: self._mark_ok(True)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="✗  No, try again", width=160, fg_color="transparent",
                      border_width=1, border_color=SUBTLE_FG, text_color=TEXT_FG,
                      command=lambda: self._mark_ok(False)).pack(side="left")

    def _mark_ok(self, ok: bool) -> None:
        self.controller.state["mic_ok"] = ok
        if ok:
            self.status_label.configure(text="✓ Microphone configured.", text_color=SUCCESS_FG)
            for child in self.confirm_frame.winfo_children():
                child.destroy()

    def can_advance(self) -> tuple[bool, str]:
        return True, ""

    def show_error(self, msg: str) -> None:
        self.status_label.configure(text=msg, text_color=ERROR_FG)


# =============================================================================
# Page: Speaker test
# =============================================================================
class SpeakerTestPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=60, pady=40)

        ctk.CTkLabel(wrapper, text="Speaker & voice test", font=("Segoe UI", 22, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(wrapper,
                     text="Pick where JARVIS should speak, choose a voice, "
                          "and click Test to hear it.",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG, wraplength=620,
                     justify="left").pack(anchor="w", pady=(8, 16))

        ctk.CTkLabel(wrapper, text="Output device", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        self.out_combo = ctk.CTkComboBox(wrapper, height=36, font=("Segoe UI", 11),
                                          values=self._output_choices(),
                                          command=self._on_out_change)
        self.out_combo.pack(fill="x", pady=(6, 12))
        if self.out_combo.cget("values"):
            self.out_combo.set(self.out_combo.cget("values")[0])
            self._on_out_change(self.out_combo.get())

        ctk.CTkLabel(wrapper, text="Voice", font=("Segoe UI", 12, "bold"),
                     text_color=TEXT_FG).pack(anchor="w")
        self.voice_combo = ctk.CTkComboBox(wrapper, height=36, font=("Segoe UI", 11),
                                            values=[
                                                "en_GB-alan-medium",
                                                "en_GB-cori-high",
                                                "en_GB-northern_english_male-medium",
                                                "en_US-ryan-high",
                                            ], command=self._on_voice_change)
        self.voice_combo.set("en_GB-alan-medium")
        self.voice_combo.pack(fill="x", pady=(6, 16))

        self.test_btn = ctk.CTkButton(wrapper, text="🔊   Test voice", height=48,
                                       font=("Segoe UI", 13, "bold"),
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                       command=self._test_voice)
        self.test_btn.pack(fill="x", pady=4)

        self.status_label = ctk.CTkLabel(wrapper, text="", font=("Segoe UI", 12),
                                          text_color=SUBTLE_FG, wraplength=620)
        self.status_label.pack(anchor="w", pady=(8, 0))

    def _output_choices(self) -> list[str]:
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                out.append(f"[{i}] {d['name'].strip()}")
        return out

    def _on_out_change(self, sel: str) -> None:
        try:
            idx = int(sel.split("]", 1)[0].lstrip("["))
            self.controller.state["selected_output"] = idx
        except Exception:
            self.controller.state["selected_output"] = None

    def _on_voice_change(self, sel: str) -> None:
        self.controller.state["selected_voice"] = sel

    def _test_voice(self) -> None:
        self.test_btn.configure(state="disabled", text="🔊   Loading voice…")
        self.status_label.configure(text="Loading voice (first time may take ~30s)…",
                                    text_color=ACCENT)
        self.update_idletasks()

        # Run synthesis in a thread to keep UI responsive
        def _do_play() -> None:
            try:
                from pathlib import Path as _P
                from piper.download_voices import download_voice  # noqa: PLC0415
                from jarvis.voice.tts.piper import PiperTTS  # noqa: PLC0415

                voice_id = self.controller.state["selected_voice"]
                cache = _P.home() / ".cache" / "jarvis" / "piper"
                cache.mkdir(parents=True, exist_ok=True)
                voice_path = cache / f"{voice_id}.onnx"
                cfg_path = cache / f"{voice_id}.onnx.json"
                if not voice_path.exists() or not cfg_path.exists():
                    download_voice(voice_id, cache)

                async def _speak() -> None:
                    tts = PiperTTS(model_path=voice_path, output_device=self.controller.state.get("selected_output"),
                                   speaking_rate=1.18)
                    await tts.speak("Good evening, sir. JARVIS is online and ready.")
                    while tts.is_playing():
                        await asyncio.sleep(0.1)
                    await asyncio.sleep(0.5)

                asyncio.run(_speak())
                self.after(0, lambda: self._on_test_done(True, ""))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_test_done(False, str(exc)))

        threading.Thread(target=_do_play, daemon=True).start()

    def _on_test_done(self, ok: bool, err: str) -> None:
        self.test_btn.configure(state="normal", text="🔊   Test again")
        if ok:
            self.status_label.configure(text="✓ If you heard JARVIS, you're all set.",
                                        text_color=SUCCESS_FG)
            self.controller.state["speaker_ok"] = True
        else:
            self.status_label.configure(text=f"✗ Could not play voice: {err[:120]}",
                                        text_color=ERROR_FG)

    def can_advance(self) -> tuple[bool, str]:
        return True, ""

    def show_error(self, msg: str) -> None:
        self.status_label.configure(text=msg, text_color=ERROR_FG)


# =============================================================================
# Page: Tour
# =============================================================================
class TourPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="both", expand=True, padx=60, pady=30)

        ctk.CTkLabel(wrapper, text="Three ways to talk to JARVIS",
                     font=("Segoe UI", 22, "bold"), text_color=TEXT_FG).pack(anchor="w")
        ctk.CTkLabel(wrapper,
                     text="Pick whatever fits the moment. You can switch any time.",
                     font=("Segoe UI", 12), text_color=SUBTLE_FG).pack(anchor="w", pady=(8, 16))

        cards_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
        cards_frame.pack(fill="both", expand=True)

        for emoji, title, desc, hint in [
            ("🎙️", "Push-to-talk",
             "Hold the big button (or Spacebar) while you speak. "
             "Most reliable — no misfires from background noise.",
             "Best for: focused commands."),
            ("👂", "Always listening",
             "Voice activity detection runs continuously. Just talk — "
             "JARVIS detects when you start and stop.",
             "Best for: hands-free use."),
            ("⌨️", "Text chat",
             "Type your message, press Enter. Replies are spoken AND printed.",
             "Best for: quiet rooms, complex queries."),
        ]:
            card = ctk.CTkFrame(cards_frame, fg_color="#252535", corner_radius=10)
            card.pack(fill="x", pady=6)
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=14)
            ctk.CTkLabel(row, text=emoji, font=("Segoe UI", 24)).pack(side="left", padx=(0, 16))
            text_block = ctk.CTkFrame(row, fg_color="transparent")
            text_block.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(text_block, text=title, font=("Segoe UI", 14, "bold"),
                         text_color=TEXT_FG).pack(anchor="w")
            ctk.CTkLabel(text_block, text=desc, font=("Segoe UI", 11),
                         text_color=SUBTLE_FG, wraplength=520, justify="left").pack(anchor="w")
            ctk.CTkLabel(text_block, text=hint, font=("Segoe UI", 10, "italic"),
                         text_color=ACCENT).pack(anchor="w", pady=(2, 0))

    def can_advance(self) -> tuple[bool, str]:
        return True, ""

    def show_error(self, msg: str) -> None:
        pass


# =============================================================================
# Page: Finish
# =============================================================================
class FinishPage(ctk.CTkFrame):
    def __init__(self, parent: tk.Widget, controller: OnboardingWizard) -> None:
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(wrapper, text="🎉", font=("Segoe UI", 60)).pack()
        ctk.CTkLabel(wrapper, text="You're all set!", font=("Segoe UI", 28, "bold"),
                     text_color=TEXT_FG).pack(pady=(8, 4))
        ctk.CTkLabel(wrapper,
                     text="JARVIS is ready to use. Click Finish below to launch the app.",
                     font=("Segoe UI", 13), text_color=SUBTLE_FG).pack(pady=(0, 20))

        tips = ctk.CTkFrame(wrapper, fg_color="#252535", corner_radius=8)
        tips.pack(pady=8)
        ctk.CTkLabel(tips, text="💡 Quick tips", font=("Segoe UI", 12, "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=14, pady=(10, 4))
        for tip in [
            "Try saying: 'Open Chrome'  •  'Set timer for 5 minutes'  •  'Mute volume'",
            "Settings, voice picker, and API keys can all be changed later in-app.",
            "JARVIS keeps your API keys encrypted with Windows DPAPI — never in plain text.",
        ]:
            ctk.CTkLabel(tips, text="• " + tip, font=("Segoe UI", 11),
                         text_color=TEXT_FG, wraplength=520, justify="left",
                         anchor="w").pack(anchor="w", padx=22, pady=2)
        ctk.CTkLabel(tips, text="", font=("Segoe UI", 4)).pack()

    def can_advance(self) -> tuple[bool, str]:
        return True, ""

    def show_error(self, msg: str) -> None:
        pass


def run_onboarding() -> bool:
    """Show the wizard. Returns ``True`` when the user completed it."""
    try:
        wiz = OnboardingWizard()
        wiz.mainloop()
        return wiz.completed
    except Exception:
        logger.exception("Onboarding wizard crashed")
        return False


if __name__ == "__main__":
    ok = run_onboarding()
    sys.exit(0 if ok else 1)
