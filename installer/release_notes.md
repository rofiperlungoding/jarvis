# JARVIS 1.0.0 — Initial public release

A voice-driven AI desktop assistant for Windows 10 / 11.

## Highlights

- **Three input modes** — push-to-talk, always-listening (Silero VAD), text chat.
- **Cloud Mistral** for reasoning + function calling, with optional local
  Ollama fallback.
- **Local Whisper STT** — your voice never leaves your machine.
- **Neural Piper TTS** — five voices, live device swap.
- **9 built-in skills** — launch app, media control, volume, brightness,
  timer, reminder, list reminder, read file, summarize file.
- **Plugin system** — drop a `Skill` Protocol into any directory.
- **MCP integration** — register external Model Context Protocol servers.
- **Encrypted Memory_Store** — ChromaDB sealed with Windows DPAPI.
- **Confirmation flow** for destructive actions.
- **7-page first-run wizard** with mic test, voice picker, and three-mode tour.
- **Auto-update notifier** — non-intrusive banner when a new release ships.
- **Rotating log file** at `%LOCALAPPDATA%\Jarvis\logs\jarvis.log`.

## Install

1. Download `JARVIS-Setup-1.0.0.exe` below.
2. Run it. Per-user install, no admin needed.
3. Launch JARVIS from Start Menu — first-run wizard guides setup.

You'll need a free Mistral API key from <https://console.mistral.ai>.

See the [setup guide](https://github.com/rofiperlungoding/jarvis/blob/main/docs/setup.md)
for advanced configuration.

## What's tested

- 1666 unit + integration + property-based tests passing.
- Property tests cover serialisation, schema validation, memory determinism,
  audit ordering, persona invariance.

## Known limitations

- Windows-only (the platform adapter binds pywin32, pycaw, pywinauto).
- The installer is unsigned; SmartScreen will prompt the first time. Click
  **More info** → **Run anyway**.
- Whisper `small.en` (~250 MB) downloads on first launch.

Full changelog: [CHANGELOG.md](https://github.com/rofiperlungoding/jarvis/blob/main/CHANGELOG.md).
