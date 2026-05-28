# JARVIS 1.0.3

## Fixed

- **Critical**: HTTP 400 / code 3240
  (`"Assistant message must have either content or tool_calls,
  but not none"`) after a turn died mid-stream (rate limit, network
  drop). Cause: the failed turn left a dangling user-only Turn in
  conversation state with empty assistant text; the next request
  replayed it as an invalid empty-content assistant message.
  Two layers of defence:
  1. `handle_turn` now seals every turn before re-raising so state
     stays consistent across errors.
  2. `_render_messages` defensively skips past assistant messages
     with empty content during replay.

## Improved

- **Transcription quality**. Tuned `faster-whisper` decode parameters
  to suppress the most common hallucination modes:
  - `condition_on_previous_text=False` — stops the
    "Thank you for watching" / "Bye!" regenerations on fresh
    utterances.
  - `vad_filter=True` with `min_silence_duration_ms=500` — strips
    Whisper-internal silence segments.
  - `temperature=(0.0, 0.2, 0.4)` — deterministic decode for the
    common case, fallback rungs for noisy edge cases.
  - `no_speech_threshold=0.6` (up from 0.45) — suppresses the
    "you" / "Thank you." silence false-positives.
  - `log_prob_threshold=-1.0` — accept lower-confidence tokens.

If your voice still gets transcribed inaccurately:
1. Run the setup wizard again (Settings → Re-run setup wizard) and
   check the mic test page that you've selected the right input.
2. Speak closer to the mic, in a relatively quiet room.
3. The bundled model is `small.en`. A future release may switch to
   `medium.en` for higher accuracy at the cost of ~1.3 GB extra
   bundle size.

## Upgrade

If you have any earlier 1.0.x installed, JARVIS will show an
"Update available" banner the next time you launch it.

Or download manually:
[JARVIS-Setup-1.0.3.exe](https://github.com/rofiperlungoding/jarvis/releases/download/v1.0.3/JARVIS-Setup-1.0.3.exe).
