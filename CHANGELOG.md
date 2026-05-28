# Changelog

All notable changes to JARVIS are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-28

### Added

- Initial public release.
- Desktop GUI built with CustomTkinter, with sidebar navigation
  (Chat / Skills / Settings).
- Three input modes: push-to-talk, always-listening (Silero VAD), text chat.
- Cloud LLM via Mistral with optional local Ollama fallback.
- Local Whisper STT (`small.en`, INT8 CPU).
- Neural Piper TTS with five British / American voices and live
  output-device swap.
- 9 built-in Skills: launch app, media control, volume, brightness,
  timer, reminder, list reminder, read file, summarize file.
- Plugin system: drop a `*.py` with a `SKILL` symbol into
  `[app].plugin_dirs` and the registry picks it up at startup.
- MCP server bridge: external Model Context Protocol servers
  contribute their tools to the registry via `[skills].mcp_servers`.
- Encrypted Memory_Store backed by ChromaDB and DPAPI.
- Confirmation flow for destructive actions.
- Wipe-all command (`jarvis --wipe-all`) clears Memory_Store,
  Credential_Store, and audit log under a 5-second budget.
- 7-page first-run onboarding wizard with live API key validation,
  microphone test with playback, speaker / voice picker, and a
  three-mode tour.
- Auto-update notification: app pings GitHub Releases on startup
  and surfaces a sticky banner when a newer version is available.
- Rotating log file at `%LOCALAPPDATA%\Jarvis\logs\jarvis.log`.
- Multi-page Inno Setup installer with welcome banner, license,
  pre-flight info, components, and tasks pages.

### Tests

- 1666 unit + integration + property tests passing.
- 16 property-based tests via Hypothesis covering serialisation
  round-trips, schema validation, memory determinism, authorization
  audit ordering, and persona invariance.
