# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for JARVIS AI Assistant.

Produces a one-folder bundle at dist/JARVIS/ with JARVIS.exe as the entry.
Bundles the Piper voice model so first-run doesn't need to download it.
Whisper model is downloaded on first launch (too large to bundle).
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
ICON = ROOT / "installer" / "jarvis.ico"
PIPER_VOICE = Path.home() / ".cache" / "jarvis" / "piper"

# Collect Piper voice if available
datas = [
    (str(SRC / "jarvis" / "config" / "default.toml"), "jarvis/config"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "dist" / "JARVIS" / "README.txt"), "."),
]
if PIPER_VOICE.is_dir():
    for f in PIPER_VOICE.glob("en_GB-alan-medium*"):
        datas.append((str(f), "piper_voices"))

# chromadb has plugin-style submodules (telemetry providers, embedding
# functions, etc.) that PyInstaller's static analysis can't see. Pull
# every submodule + data file in.
chromadb_hidden = collect_submodules("chromadb")
chromadb_data = collect_data_files("chromadb")
datas.extend(chromadb_data)

# Same story for tokenizers / faster_whisper / customtkinter assets
ctk_data = collect_data_files("customtkinter")
datas.extend(ctk_data)

a = Analysis(
    [str(ROOT / "jarvis_app.py")],
    pathex=[str(SRC), str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "onboarding",
        "customtkinter",
        "jarvis",
        "jarvis.config",
        "jarvis.config.schema",
        "jarvis.dialog",
        "jarvis.dialog.manager",
        "jarvis.dialog.persona",
        "jarvis.dialog.persona_guard",
        "jarvis.dialog.conversation_state",
        "jarvis.llm",
        "jarvis.llm.base",
        "jarvis.llm.mistral_backend",
        "jarvis.memory",
        "jarvis.memory.store",
        "jarvis.memory.embedder",
        "jarvis.memory.redactor",
        "jarvis.reminders",
        "jarvis.reminders.service",
        "jarvis.security",
        "jarvis.security.audit_log",
        "jarvis.security.authorization",
        "jarvis.security.credential_store",
        "jarvis.security.dpapi",
        "jarvis.skills",
        "jarvis.skills.base",
        "jarvis.skills.registry",
        "jarvis.skills.builtin",
        "jarvis.skills.builtin.launch_app",
        "jarvis.skills.builtin.media_control",
        "jarvis.skills.builtin.volume",
        "jarvis.skills.builtin.brightness",
        "jarvis.skills.builtin.timer",
        "jarvis.skills.builtin.reminder",
        "jarvis.skills.builtin.read_file",
        "jarvis.skills.builtin.summarize_file",
        "jarvis.automation",
        "jarvis.automation.platform",
        "jarvis.automation.windows_adapter",
        "jarvis.voice",
        "jarvis.voice.audio_io",
        "jarvis.voice.vad",
        "jarvis.voice.stt",
        "jarvis.voice.stt.base",
        "jarvis.voice.stt.faster_whisper",
        "jarvis.voice.tts",
        "jarvis.voice.tts.base",
        "jarvis.voice.tts.piper",
        "jarvis.utils",
        "jarvis.utils.time_source",
        # Third-party hidden imports PyInstaller often misses
        "piper",
        "piper.voice",
        "faster_whisper",
        "ctranslate2",
        "sounddevice",
        "numpy",
        "chromadb",
        "chromadb.config",
        "sqlite3",
        "mistralai",
        "httpx",
        "tenacity",
        "pydantic",
        "jsonschema",
        "win32crypt",
        "win32api",
        "pywintypes",
        "comtypes",
        "pyautogui",
        "onnxruntime",
    ] + chromadb_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "hypothesis",
        "mypy",
        "ruff",
        "black",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="JARVIS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed app, no console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
    version_info=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="JARVIS",
)
