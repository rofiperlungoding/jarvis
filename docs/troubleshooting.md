# Troubleshooting JARVIS

This is the deep-dive companion to the brief table in `docs/setup.md`.
When something is wrong, work top-to-bottom: open the log file first,
then match the symptom against the sections below.

---

## 1. Read the log first

Every JARVIS process — bundled or dev — writes to:

```
%LOCALAPPDATA%\Jarvis\logs\jarvis.log
```

Tail the latest entries:

```powershell
Get-Content "$env:LOCALAPPDATA\Jarvis\logs\jarvis.log" -Tail 100
```

Or watch live while you reproduce a bug:

```powershell
Get-Content "$env:LOCALAPPDATA\Jarvis\logs\jarvis.log" -Tail 0 -Wait
```

The format is `YYYY-MM-DD HH:MM:SS [LEVEL] <module>: <message>`.

Rotation: 2 MB per file, 5 backups. Older logs sit next to the active
one as `jarvis.log.1`, `.2`, etc.

---

## 2. Boot failures

Symptom: app launches, sidebar appears, but the chat shows a red
"Boot failed" bubble or the status pill stays on "Booting…" forever.

### 2.1 No Mistral key registered

Log line:

```
[ERROR] jarvis.app: Boot failed: MistralCredentialMissingError: no Mistral API key found ...
```

Fix: re-run the onboarding wizard.

```powershell
"%LocalAppData%\Programs\JARVIS\JARVIS.exe" --first-run
```

Or, dev install:

```powershell
python jarvis_app.py --first-run
```

### 2.2 Mistral key invalid

Log line:

```
[WARNING] jarvis.llm.mistral_backend: Mistral stream error: status=401 ...
```

The wizard registered a key, but it is wrong / expired / revoked. Open
[console.mistral.ai](https://console.mistral.ai/) → API keys → create
a fresh one and re-run the wizard.

### 2.3 Whisper model download stalls

First run downloads `small.en` (~250 MB) into the HuggingFace cache.
On a slow connection, or if a download manager (IDM, etc.) intercepts
the request, this can hang.

Workaround: run once on the CLI to confirm the download completes:

```powershell
python -c "from jarvis.voice.stt.faster_whisper import FasterWhisperSTT; FasterWhisperSTT(model_size='small.en', device='cpu', compute_type='int8')"
```

Or pre-cache manually with `huggingface-cli download`.

### 2.4 Piper voice download fails

The default `en_GB-alan-medium` voice (~25 MB) is fetched on first
launch into `~\.cache\jarvis\piper\`. If the download fails, the app
won't play any audio.

Manual download:

```powershell
python -c "from piper.download_voices import download_voice; from pathlib import Path; download_voice('en_GB-alan-medium', Path.home() / '.cache' / 'jarvis' / 'piper')"
```

---

## 3. "I talk but JARVIS doesn't reply"

You see your input as a blue bubble, but no JARVIS bubble appears.

Open the log and look for one of:

| Log line | Meaning | Fix |
|---|---|---|
| `[INFO] jarvis.app: handle_turn START: text='...'` then nothing | Backend hung. The 90-second timeout will fire. | Check network. |
| `[ERROR] jarvis.app: handle_turn timed out after 90s` | Mistral did not respond. | Network or rate limit. |
| `[WARNING] jarvis.llm.mistral_backend: Mistral stream error: status=401` | Auth failed. | Re-register API key. |
| `[WARNING] ... status=429` | Rate limited. | Wait or upgrade plan. |
| `[INFO] ... handle_turn END: reply_len=0` | Backend returned empty text. | Usually transient — retry the turn. |
| `[ERROR] ... handle_turn raised: <exception>` | Skill or memory error mid-turn. | See the traceback. |

Errors also show as red bubbles in chat now, so you should see them
even without the log.

---

## 4. Voice / audio issues

### 4.1 Microphone doesn't pick up speech

Check (in order):

1. *Settings → Microphone* in JARVIS — is the right device selected?
2. *Windows Settings → System → Sound → Input* — is the device listed
   as the default and showing a level when you talk?
3. Run the wizard's mic test page (Settings → "Re-run setup wizard")
   — it records and plays back so you can verify end-to-end.
4. Check the log for `silero_vad` lines. If frames keep coming but no
   `speech_start` event fires, the VAD is rejecting your audio level.
   Move closer to the mic or switch device.

### 4.2 No sound from JARVIS

1. *Settings → Output device* — pick the right speaker / headphone.
   The dropdown live-swaps Piper without restarting the app.
2. Confirm the Piper voice download finished (see §2.4).
3. Log line `[ERROR] jarvis.voice.tts.piper: ...` indicates a Piper
   crash; capture the traceback and file an issue.

### 4.3 Audio plays through old device after switch

Should not happen — the output dropdown rebuilds the `PiperTTS`
instance with the new device. If it does, restart JARVIS as a
workaround and capture the log.

### 4.4 Asterisks / Markdown read aloud

Should not happen. The `_SpeechFilteringTTS` wrapper strips Markdown
before the text reaches Piper, and the persona system prompt forbids
Markdown output.

If you hear it: file an issue with the offending sentence and
the surrounding `jarvis.log` excerpt — likely a regex gap.

---

## 5. Skill failures

### 5.1 "Open Chrome" doesn't launch the browser

1. Check the `[automation.application_registry]` config block — is
   `chrome` defined?
2. Log line: `[INFO] jarvis.skills.builtin.launch_app: ...` should
   show the matched alias and the path used.
3. If `LaunchAppSkill.execute` returns `script_not_found`, the path
   in the registry is wrong. Edit your override config.

### 5.2 "Set timer for 30 seconds" — no timer fires

The TimerSkill is fire-and-forget; it returns immediately and prints
the timer ID in the chat. The actual notification fires from the
`ReminderService`. Check `%LOCALAPPDATA%\Jarvis\reminders.sqlite`
exists, and that the win10toast import succeeds in the log.

### 5.3 "Mute volume" — no effect

`VolumeSkill` uses pycaw under the hood. If the log shows
`provider_unavailable`, the COM registration for the default audio
endpoint failed. Reboot, or pin the device explicitly via Settings.

### 5.4 ReadFile / SummarizeFile says `access_denied`

Path is outside `[automation.allowed_directories].paths`. Add the
parent directory of your file to the list and restart.

### 5.5 Skill returns `schema_violation`

The LLM emitted arguments that don't match the Skill's JSON Schema.
Check the log — the validator's error message names the offending
field. Usually a model regression; rephrase your request to be more
explicit, or file an issue.

---

## 6. Performance

### 6.1 First reply is slow (~5–10 s)

Expected. Whisper loads the model (~1 s), Piper warms up
(~500 ms), and the first Mistral request opens a fresh HTTPS
connection (~500 ms RTT). Subsequent turns should land in
~1.5 s.

### 6.2 Voice mode latency feels high

Look at `speech_start` → `tts.speak` timestamps in the log:

```
HH:MM:SS [INFO] jarvis.voice.vad: speech_start
HH:MM:SS [INFO] jarvis.app: handle_turn START
HH:MM:SS [INFO] jarvis.app: handle_turn END
HH:MM:SS [INFO] jarvis.voice.tts.piper: speak chunk 1
```

If the gap between START and END exceeds 3 s, the bottleneck is
Mistral. If START → first `speak chunk` exceeds 4 s, look at memory
retrieval (set `[dialog].memory_k = 0` to disable).

### 6.3 Bundle launches slowly

The PyInstaller bundle decompresses ~900 MB on first launch into
`%TEMP%\_MEI*`. SSDs handle this in ~3 s; spinning rust takes
~30 s. Subsequent launches reuse the cache.

---

## 7. Reset everything

When in doubt:

```powershell
# 1. Quit JARVIS (system tray right-click → Exit, or Task Manager).
# 2. Wipe all stored data:
jarvis --wipe-all
# Or, with the bundle:
"%LocalAppData%\Programs\JARVIS\JARVIS.exe" --wipe-all
# 3. Delete reminders and config (optional, if --wipe-all isn't enough):
Remove-Item "$env:LOCALAPPDATA\Jarvis" -Recurse -Force
Remove-Item "$env:APPDATA\Jarvis" -Recurse -Force
# 4. Re-run the wizard:
"%LocalAppData%\Programs\JARVIS\JARVIS.exe" --first-run
```

---

## 8. Filing a useful issue

A good issue includes:

1. JARVIS version (Help → About, or `JARVIS.exe --version`).
2. Windows build (`winver`).
3. Steps to reproduce, in order.
4. The last 100 lines of `%LOCALAPPDATA%\Jarvis\logs\jarvis.log`
   captured **right after** the bug fires.
5. Screenshot of any red bubble, if applicable.

The log file already redacts the Mistral API key (the redaction
filter rewrites known secret values to `***`), so it is safe to share.
