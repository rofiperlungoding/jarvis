# JARVIS 1.0.1 — Critical bugfix

## Fixed

- **Critical**: 1.0.0 crashed silently the moment Piper TTS tried to
  phonemize the first reply. Cause: the
  ``piper/espeak-ng-data`` directory (espeak's phoneme dictionaries —
  365 files, 17.6 MB) was missing from the bundle because
  PyInstaller's static analysis can't see data-only directories.
  Now collected explicitly. **Install this version if 1.0.0 closed
  itself the moment you said anything.**

- Added `sys.excepthook` and `threading.excepthook` so background-thread
  crashes always land in the log file. The previous silent-close
  symptom is now impossible — you'll see a stack trace in
  `%LOCALAPPDATA%\Jarvis\logs\jarvis.log` and the app stays alive.

- Extended PyInstaller `hiddenimports` for `piper.config`, `piper.const`,
  `piper.phoneme_ids`, `piper.phonemize_espeak`,
  `piper.phonemize_chinese`, `piper.audio_playback`.

## Upgrade

If you have 1.0.0 installed, JARVIS will show an "Update available"
banner the next time you launch it. Click **Open release page**, grab
`JARVIS-Setup-1.0.1.exe` below, and run it. The installer overwrites
1.0.0 in place; your settings, memory, and reminders are preserved.

Or download manually: [JARVIS-Setup-1.0.1.exe](https://github.com/rofiperlungoding/jarvis/releases/download/v1.0.1/JARVIS-Setup-1.0.1.exe).

## Bundle size

279 MB installer (+ 6 MB vs 1.0.0 because of the espeak data files).
