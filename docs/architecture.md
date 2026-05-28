# JARVIS Architecture

This is a high-level tour of how the bits fit together. For the
authoritative spec see `.kiro/specs/jarvis-ai-assistant/`.

---

## Threading model

JARVIS is split across two threads plus a few helpers:

```
┌─────────────────────┐         ┌──────────────────────────┐
│ Main thread (Tk)    │  WORK   │ Worker thread (asyncio)  │
│                     │ ─────►  │                          │
│ CustomTkinter UI    │  QUEUE  │ DialogManager            │
│ User input          │         │ MistralBackend           │
│ Render bubbles      │ ◄─────  │ MemoryStore              │
│ Render error bubbles│   UI    │ FasterWhisperSTT         │
│ Status pill         │  QUEUE  │ PiperTTS (own threads)   │
│                     │         │ SkillRegistry            │
└─────────────────────┘         └──────────────────────────┘
        │                                    │
        ▼                                    ▼
   tk.mainloop()                    asyncio event loop
```

- **Main thread** owns the Tk / CustomTkinter event loop. It is the
  only thread allowed to touch widgets directly. It polls the
  `UI_QUEUE` every 50 ms via `self.after(50, _poll_ui_queue)`.
- **Worker thread** owns one `asyncio` event loop. Heavy work
  (Mistral streaming, Whisper transcription, MemoryStore retrieval,
  Skill dispatch) lives here so the UI never blocks.
- **PiperTTS** spawns its own dedicated synthesis + playback threads
  internally.

Cross-thread communication happens via two `queue.Queue` instances:

- `WORK_QUEUE: queue.Queue[Callable[[Worker], Awaitable[None]] | None]`
  — main → worker. The UI puts coroutine factories; the worker pulls
  one at a time and `await`s it.
- `UI_QUEUE: queue.Queue[tuple[str, Any]]`
  — worker → main. The worker posts `("status", "Thinking…")`,
  `("user", text)`, `("assistant", text)`, `("assistant_error",
  text)`, `("toast", text)`, `("ready", None)`. The UI poll
  loop dispatches each kind to the right widget.

A dedicated single-thread `ThreadPoolExecutor` named `jarvis-getter`
runs the blocking `WORK_QUEUE.get()` so it doesn't compete with
asyncio's default executor (which Piper also uses for its
`stream.write()` calls — they would deadlock).

---

## Event flow: a single turn

For a text-mode turn the call graph looks like:

```
User types "Open Chrome"
  └─ ChatPage._send_text()
      └─ WORK_QUEUE.put(lambda w: w.reply_to_text(text))

[worker thread]
  └─ JarvisWorker.reply_to_text(text)
      ├─ post_ui("user", text)           → main renders blue bubble
      └─ JarvisWorker._reply(transcript)
          ├─ post_ui("status", "Thinking…")
          └─ DialogManager.handle_turn(transcript, state)
              ├─ Empty/low-confidence gate
              ├─ MemoryStore.retrieve(query, k=5)
              ├─ Render messages = [system, memories?, ...turns]
              ├─ MistralBackend.stream(messages, tools=tool_defs)
              │   └─ for each event:
              │       ├─ content_delta → SentenceAccumulator
              │       │   └─ each complete sentence → PiperTTS.speak()
              │       └─ tool_call → SkillRegistry.dispatch()
              ├─ PersonaGuard.check(final_text)
              └─ MemoryStore.persist_turn(state)  (skipped if incognito)

  ├─ post_ui("assistant", response.text)  → main renders gray bubble
  └─ post_ui("status", "Ready")
```

For voice mode the prefix is:

```
SileroVAD speech_start → AudioReframer → captured PCM
  └─ WORK_QUEUE.put(lambda w: w.transcribe_and_reply(pcm))

[worker thread]
  └─ JarvisWorker.transcribe_and_reply(pcm)
      ├─ FasterWhisperSTT.transcribe(pcm) → Transcript
      ├─ post_ui("user", transcript.text)
      └─ JarvisWorker._reply(transcript)  [same as text mode]
```

Push-to-talk replaces the VAD with explicit start/stop on the PTT
button.

---

## Boot sequence

`JarvisWorker.boot()` runs once on worker startup, in this order:

1. **Config** — `load_config()` deep-merges
   `default.toml + %APPDATA%\Jarvis\config.toml`.
2. **Credential_Store** — DPAPI-backed; root at
   `<data_dir>\secrets\`.
3. **Audit log** — append-only SQLite at `<data_dir>\audit.sqlite`.
4. **Mistral backend** — pulls API key from Credential_Store; raises
   `MistralCredentialMissingError` if not set.
5. **Memory_Store** — ChromaDB at `<data_dir>\memory\app\`.
6. **Platform_Adapter** — `WindowsAdapter` wired to media keys,
   brightness (WMI), volume (pycaw), notifications (win10toast).
7. **Reminder_Service** — APScheduler in-process.
8. **Skill registry** — registers 9 builtins
   (LaunchApp, MediaControl, Volume, Brightness, Timer, Reminder,
   ListReminder, ReadFile, SummarizeFile) plus any plugin-discovered
   user Skills.
9. **Authorization_Policy** — TrustedActionAllowlist + audit hook.
10. **Persona** — loads from `[dialog].persona_profile`; default is
    `jarvis_default`.
11. **Piper voice** — downloads `en_GB-alan-medium` if needed.
12. **PiperTTS** — wrapped in `_SpeechFilteringTTS` (strips Markdown).
13. **FasterWhisperSTT** — `small.en`, INT8 quantised, CPU. Warmed
    with a silent buffer so the first real transcription doesn't pay
    the model-load cost.
14. **DialogManager** — composes everything above.

Total cold-start: ~6 s on an SSD with the models already cached.

---

## Persistence layout

```
%LOCALAPPDATA%\Jarvis\                    (default data_dir)
├── audit.sqlite                          append-only audit log
├── reminders.sqlite                      APScheduler job store
├── secrets\                              DPAPI Credential_Store
│   └── mistral%2Fapi_key.bin
├── memory\app\                           ChromaDB (per app instance)
│   └── chroma.sqlite3
└── logs\
    ├── jarvis.log                        rotating, 2 MB × 5
    ├── jarvis.log.1
    └── ...

%APPDATA%\Jarvis\                         user config + plugins
├── config.toml                           override layer
└── plugins\
    └── *.py                              user-authored Skills

~\.cache\jarvis\piper\                    Piper voice models
└── en_GB-alan-medium.{onnx,onnx.json}
```

---

## LLM tool loop

`DialogManager._run_llm_tool_loop` iterates until the LLM emits a
content-only response (no tool calls), or until the per-turn schema
violation retry budget is exhausted.

Per iteration:

1. Open a streaming request via `MistralBackend.stream(messages,
   tools=tools)`.
2. Forward `content_delta` events through `SentenceAccumulator` to
   PiperTTS, sentence-by-sentence (so playback starts as soon as the
   first sentence is complete, before the full response arrives).
3. Reassemble `tool_call` events into `ToolCall` objects.
4. After the stream closes, if there are tool calls, dispatch each
   via `SkillRegistry.dispatch(name, args, ctx)` while a 1.5-second
   "One moment, sir." acknowledgement timer races in parallel.
5. Append the assistant message + tool results back to `messages`
   and loop.

If a Skill returns `schema_violation` (LLM emitted bad arguments),
the manager retries up to `max_schema_violation_retries` times
before terminating the turn with a logged warning.

---

## Backend selection / fallback

`MistralBackend` is the default LLM. Mistral API is consulted first
on every turn. If it returns 5xx or times out, a `BackendSelector`
(when configured) opens a circuit breaker and routes subsequent
turns to a local Ollama instance (`http://localhost:11434` by
default) until the breaker half-opens and Mistral becomes available
again.

The fallback is **off by default** in the bundled GUI to keep the
install lean. Configure it in your override:

```toml
[llm]
fallback = "ollama"

[llm.fallback]
endpoint = "http://localhost:11434"
model = "mistral"
```

---

## Security boundaries

- **Credentials** — every secret goes through `CredentialStore`. The
  store wraps DPAPI so blobs are bound to the user account on disk.
- **Allowed directories** — file Skills compare canonicalised
  realpaths against `[automation.allowed_directories].paths`.
  Symlink escapes are blocked.
- **Audit log** — append-only SQLite. Records `tool_call`,
  `tool_result`, `network_egress`, `policy_violation`, `error`. Used
  by the wipe-all command for fast deletion.
- **Network egress** — every outbound HTTP call from a Skill records
  `network_egress(url, status_code)` to the audit log. Audited
  domains are listed in `[security].audited_domains`.
- **Wipe-all** — `JarvisApp.wipe_all()` concurrently clears
  Memory_Store, Credential_Store, and the audit log under a
  5-second budget.

---

## Spec correspondence

| Component | Spec section |
|---|---|
| `DialogManager` | §1, §11, §12, §17 |
| `MistralBackend` / `BackendSelector` | §12, §19 |
| `MemoryStore` | §13.3, §17 |
| `Skill` / `SkillRegistry` | §14, §15 |
| `Authorization_Policy` | §16 |
| `CredentialStore` | §13.1, §13.2 |
| `AuditLog` | §13.4, §13.6 |
| `ReminderService` | §10 |
| `WindowsAdapter` | §8 |
| `FasterWhisperSTT` / `PiperTTS` / `SileroVAD` | §1, §3 |
| `JarvisApp` (GUI) | §19 (app lifecycle), §22 |

The spec lives at `.kiro/specs/jarvis-ai-assistant/`. All 123 tasks
in `tasks.md` are implemented and covered by 1666 passing tests.
