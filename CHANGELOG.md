# Changelog

All notable changes to JARVIS are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] — 2026-05-28

### Added

- **One-click in-app updates**. The "Update available" banner now
  shows two buttons:
  - **Update now** — JARVIS downloads the installer from the GitHub
    release, launches it silently, and exits so the installer can
    overwrite the binaries in place. The installer's postinstall
    step relaunches JARVIS automatically; to the user it looks like
    the window vanishes for a moment and comes back on the new
    version. Progress is rendered in the banner.
  - **Release page** — falls back to the browser. Useful if the
    user wants release notes first.
- New module ``jarvis.update_checker.download_and_run_installer``
  handles the streaming download (with progress callback), atomic
  rename to a final path, and detached silent launch with
  ``/VERYSILENT /SUPPRESSMSGBOXES /CLOSEAPPLICATIONS
  /RESTARTAPPLICATIONS /NORESTART``.
- Installer logs now go to ``%LOCALAPPDATA%\Jarvis\updates\install-<version>.log``
  for post-mortem when an upgrade misbehaves.

## [1.0.3] — 2026-05-28

### Fixed

- **Critical**: A turn that died mid-stream (Mistral rate limit,
  network drop, etc.) left a dangling user-only Turn in
  ``ConversationState`` whose assistant text was empty. The next
  turn's history replay then emitted an invalid
  ``{role: assistant, content: ""}`` payload, which Mistral rejects
  with HTTP 400 / code 3240
  (``"Assistant message must have either content or tool_calls,
  but not none."``). Two layers of defence:
  1. ``handle_turn`` now seals the turn with whatever text was
     accumulated so far before re-raising, so state stays
     consistent across errors.
  2. ``_render_messages`` defensively skips past assistant
     messages with empty content during replay, so even malformed
     state from older builds doesn't corrupt subsequent requests.

### Improved

- **Speech-to-text quality**. Tuned ``faster-whisper`` decode
  parameters to suppress the most common hallucination modes:
  - ``condition_on_previous_text=False`` — stops the
    "Thank you for watching" / "Bye!" regenerations on fresh
    utterances.
  - ``vad_filter=True`` with ``min_silence_duration_ms=500`` —
    strips Whisper-internal silence segments.
  - ``temperature=(0.0, 0.2, 0.4)`` — deterministic decode for
    the common case, fallback rungs for noisy edge cases.
  - ``no_speech_threshold=0.6`` — suppresses the
    "you" / "Thank you." silence false-positives.
  - ``log_prob_threshold=-1.0`` — accept lower-confidence tokens
    when the temperature ladder needs them.

## [1.0.2] — 2026-05-28

### Fixed

- **Critical**: Multi-turn conversations broke after the first tool
  call. Cause: the message renderer replayed past `tool_calls` on
  assistant messages without the matching `tool` response messages,
  so Mistral rejected the second turn with HTTP 400 (error code 3230,
  "Not the same number of function calls and responses"). The
  assistant's prose reply already summarises what the tool did, so
  we now omit the raw tool-call payload from history replay.
  Visible symptom in 1.0.1: ask JARVIS to open Calculator → succeeds;
  follow-up question → red error bubble.

## [1.0.1] — 2026-05-28

### Fixed

- **Critical**: Bundled installer crashed silently the moment Piper TTS
  tried to phonemize the first sentence. Root cause: the
  ``piper/espeak-ng-data`` directory (espeak's phoneme dictionaries)
  was missing from the PyInstaller bundle because PyInstaller's static
  analysis can't see data-only directories. Now collected explicitly.
- Added global ``sys.excepthook`` and ``threading.excepthook`` so
  unhandled exceptions on background threads land in the log file
  instead of being swallowed by the windowed PyInstaller bundle.
- Extended ``hiddenimports`` for ``piper.config``, ``piper.const``,
  ``piper.phoneme_ids``, ``piper.phonemize_espeak``,
  ``piper.phonemize_chinese``, and ``piper.audio_playback``.

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
