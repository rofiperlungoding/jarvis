# JARVIS Setup Guide (Windows)

This guide walks through getting JARVIS running on Windows 10 (1903+) or
Windows 11. There are two paths:

- **Installer (recommended for end users)** — a signed `.exe` that bundles
  Python, all dependencies, and the desktop GUI. No build tools, no `pip`,
  no virtual environment.
- **Source / dev install** — clone the repo, install in editable mode, and
  run the GUI or the CLI directly. Use this if you plan to author Skills
  or hack on the code.

By the end of either path you will have:

- The desktop app launchable from Start Menu / Desktop.
- A user override at `%APPDATA%\Jarvis\config.toml` pointing at your data
  directory and persona.
- A Mistral API key stored in the DPAPI-backed Credential_Store — never on
  disk in plaintext.
- A configured allowed-directories sandbox for file Skills.

> If a step fails, check [Troubleshooting](#troubleshooting) and the
> rotating log file at `%LOCALAPPDATA%\Jarvis\logs\jarvis.log`.

---

## Path A — Installer

### A.1 Download the installer

Grab the latest `JARVIS-Setup-<version>.exe` from your build output
(`installer/output/`) or from the GitHub Releases page once published.

### A.2 Run the installer

Double-click the `.exe`. The wizard walks through:

1. **Welcome** — splash screen.
2. **License** — MIT.
3. **Pre-flight info** — Windows 10 1903+ requirement, ~1 GB of disk
   space, optional internet for the Mistral cloud LLM.
4. **Install location** — defaults to
   `%LocalAppData%\Programs\JARVIS` (per-user, no admin rights).
5. **Components** — Core (required) + Voice models (optional, ~25 MB).
6. **Tasks** — Desktop shortcut, Start Menu entry, autostart on login.
7. **Ready** — review, confirm.
8. **Install** — progress bar.
9. **Finish** — checkbox to launch JARVIS immediately.

### A.3 First launch

On the first launch the **Setup Wizard** appears (7 pages):

1. Welcome.
2. **Mistral API key** — paste your key; the wizard validates it live
   against `https://api.mistral.ai`.
3. **Optional providers** — Whisper model size, MCP servers.
4. **Microphone test** — record 3 seconds and play back so you can verify
   the right device is picked up.
5. **Speaker / voice picker** — pick the output device and Piper voice.
6. **Three-mode tour** — Push-to-Talk / Always-Listening / Text Chat.
7. **Finish** — drops you on the Chat page.

You can re-run the wizard later via Settings → "Re-run setup wizard" or
by launching `JARVIS.exe --first-run`.

### A.4 Uninstall

Use *Settings → Apps* (or the entry in *Start Menu → JARVIS →
Uninstall*). The uninstaller removes the application, the desktop /
start-menu shortcuts, and the autostart entry. Your data
(`%LOCALAPPDATA%\Jarvis\`) is preserved by default — delete that folder
manually if you want a clean wipe.

---

## Path B — Source / dev install

Use this if you want to author your own Skills, run the test suite, or
contribute changes upstream.

### B.1 Prerequisites

Install in this order; restart PowerShell after each so new `PATH`
entries take effect.

#### Python 3.11 or newer

JARVIS targets Python `>=3.11` (declared in `pyproject.toml`). Download
from [python.org](https://www.python.org/downloads/windows/), tick
**Add python.exe to PATH**, then verify:

```powershell
python --version
# Python 3.11.x  (or newer)
```

#### Visual C++ Build Tools

Several runtime dependencies (`faster-whisper` / CTranslate2,
`chromadb`, `pywin32`) ship native extensions that need the MSVC
toolchain when a prebuilt wheel is not available.

Install **Build Tools for Visual Studio 2022** from
[Microsoft's downloads page](https://visualstudio.microsoft.com/downloads/)
and select the *Desktop development with C++* workload. Reboot when
done.

### B.2 Install the package

```powershell
git clone <repository-url> jarvis
cd jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev]"
```

Verify the entry point resolves:

```powershell
jarvis --help
```

### B.3 Run the desktop GUI

```powershell
python jarvis_app.py
```

Pass `--first-run` to force the onboarding wizard:

```powershell
python jarvis_app.py --first-run
```

### B.4 Run the test suite

```powershell
pytest                       # unit + integration + property
ruff check .                 # lint
mypy src tests               # type check
```

---

## Configuration

JARVIS reads its configuration in two layers:

1. The package-shipped defaults at `src/jarvis/config/default.toml`
   (always loaded).
2. Your override at `%APPDATA%\Jarvis\config.toml` (deep-merged on top),
   or any explicit path passed via the `--config` CLI flag.

Create the override directory and seed it from defaults:

```powershell
$dest = Join-Path $env:APPDATA 'Jarvis\config.toml'
New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
Copy-Item .\src\jarvis\config\default.toml $dest
notepad $dest
```

Common keys you might edit:

- `[app].data_dir` — where Memory_Store, audit log, reminders live
  (defaults to `%LOCALAPPDATA%\Jarvis`).
- `[app].plugin_dirs` — extra directories scanned for user Skills.
- `[dialog].persona_profile` — see [Persona customization](#persona-customization).
- `[automation.allowed_directories].paths` — see [Allowed directories](#allowed-directories).

Tokens like `%APPDATA%`, `%LOCALAPPDATA%`, `%USERPROFILE%`,
`%USERNAME%` are expanded by the loader.

---

## Mistral API key

The key is **never** stored in `config.toml`. The file references a
credential name (`[llm.mistral].api_key_credential = "mistral/api_key"`)
and JARVIS reads the secret from the DPAPI-backed `CredentialStore` at
startup.

The onboarding wizard (Path A) registers it for you. If you need to do
it manually:

```powershell
python -c "from pathlib import Path; from jarvis.config import load_config; from jarvis.security.dpapi import create_default_dpapi; from jarvis.security.credential_store import CredentialStore; cfg = load_config(); root = Path(cfg.app.data_dir) / 'secrets'; CredentialStore(root, create_default_dpapi()).set('mistral/api_key', input('Mistral API key: '))"
```

The blob lands at `<data_dir>\secrets\mistral%2Fapi_key.bin` (the `/`
is URL-encoded).

If Mistral returns HTTP 401/403 at runtime, JARVIS surfaces a red error
bubble in the chat with the status code. Re-run the snippet above (or
re-run the wizard) to refresh the key.

> Never echo the key value back from PowerShell or commit it to a file.

---

## Allowed directories

File-touching Skills (`ReadFileSkill`, `SummarizeFileSkill`) only
operate on paths under `[automation.allowed_directories].paths`.
Anything outside is rejected with `access_denied`.

```toml
[automation.allowed_directories]
paths = [
  "%USERPROFILE%/Documents",
  "%USERPROFILE%/Downloads",
  "%USERPROFILE%/Documents/Codes",
]
```

The list must be non-empty. Each entry is canonicalised through
`os.path.realpath`, so symlink escapes are blocked automatically.

---

## Persona customization

The default persona is `jarvis_default` — witty, formal, mildly
sarcastic, with the `en_GB-alan-medium` Piper voice.

### Light tweaks

```toml
[dialog]
persona_profile = "jarvis_default"
honorific = "boss"

[voice.tts]
voice = "en_US-amy-medium"
```

### Full custom persona

Drop a Python module under `[app].plugin_dirs` that registers a factory:

```python
# %APPDATA%/Jarvis/plugins/my_persona.py
from jarvis.dialog.persona import PersonaProfile, register_persona


def _friday() -> PersonaProfile:
    return PersonaProfile(
        name="FRIDAY",
        honorific="boss",
        system_prompt=(
            "You are FRIDAY, a precise and upbeat assistant. "
            "Address the user as 'boss'. Keep replies short and actionable."
        ),
        tts_voice="en_US-amy-medium",
        forbidden_self_refs=("ChatGPT", "Claude", "as an AI language model"),
    )


register_persona("friday", _friday)
```

Point the config at it:

```toml
[dialog]
persona_profile = "friday"
```

The post-generation `PersonaGuard` rewrites any output that leaks
`forbidden_self_refs`, so list every phrase you want avoided.

---

## Incognito mode

Disables Memory_Store persistence for the active session. Turns are
still spoken, Skills still execute, the audit log still records
destructive actions, but nothing is committed to ChromaDB.

```toml
[app]
incognito = true
```

---

## Wipe everything

To erase locally-stored secrets and memory:

```powershell
jarvis --wipe-all
```

Concurrently clears Memory_Store, Credential_Store, and the audit log
under a 5-second budget. Reminders and `last_run.json` are deliberately
preserved — delete `%LOCALAPPDATA%\Jarvis\` for a truly clean slate.

---

## Logs

JARVIS writes a rotating log to:

```
%LOCALAPPDATA%\Jarvis\logs\jarvis.log
```

Format: `<ISO timestamp> [LEVEL] <module>: <message>`. Rotation: 2 MB
per file, 5 backups.

Useful when something goes wrong:

```powershell
Get-Content "$env:LOCALAPPDATA\Jarvis\logs\jarvis.log" -Tail 50
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Installer says "App already running" | Close JARVIS from the system tray and retry. |
| App launches, then closes immediately | Check `%LOCALAPPDATA%\Jarvis\logs\jarvis.log` for the boot exception. |
| Red bubble: `MistralAuthError: status_code=401` | Wrong / expired API key. Re-run the wizard or the Credential_Store snippet above. |
| Red bubble: `MistralRateLimitError` | Hit the Mistral rate limit. Wait, or upgrade the plan. |
| Red bubble: `Timeout: backend did not respond within 90 seconds` | Network issue, or Mistral slow. Check connectivity. |
| Microphone never picks up speech | Settings → Microphone — pick the right device. The wizard's mic test page is the fastest way to verify. |
| `pip install -e .` fails on `Building wheel for ctranslate2` / `tokenizers` | VC++ Build Tools missing or *Desktop development with C++* workload not selected. |
| `access_denied` reading a file | Path outside `[automation.allowed_directories].paths`. Add it (or a parent) to the list. |
| Audio plays through wrong device | Settings → Output device. The dropdown live-swaps Piper without a restart. |
| Asterisks read aloud | Should not happen — `_SpeechFilteringTTS` strips Markdown, and the persona prompt forbids it. If you see this, file an issue with the `jarvis.log` excerpt. |

---

_See `docs/plugins.md` for authoring your own Skills, `docs/architecture.md`
for the threading model, and `docs/troubleshooting.md` for deeper diagnostic
recipes._
