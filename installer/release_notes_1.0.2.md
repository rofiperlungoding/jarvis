# JARVIS 1.0.2 — Multi-turn fix

## Fixed

- **Critical**: Multi-turn conversations broke after the first tool
  call. Symptom: ask JARVIS to open Calculator (or any app) — the
  tool fires successfully, reply is spoken — then any follow-up
  question produces a red error bubble with HTTP 400.

  Root cause: `_render_messages` replayed past assistant `tool_calls`
  without the matching `tool` response messages, violating Mistral's
  invariant that function-calls and responses must come in pairs.
  Mistral rejected the second turn with error code 3230,
  `"Not the same number of function calls and responses"`.

  Fix: omit `tool_calls` from replayed assistant messages. The
  assistant's prose reply already summarises what the tool did,
  which is sufficient context for follow-up turns.

  Verified end-to-end against real Mistral:
  - Turn 1: "Open Chrome" → tool fires, reply: "Chrome is now at your
    service, sir."
  - Turn 2: "What is two plus two?" → reply: "Four, sir."

## Upgrade

If you have 1.0.0 or 1.0.1 installed, JARVIS will show an "Update
available" banner the next time you launch it.

Or download manually:
[JARVIS-Setup-1.0.2.exe](https://github.com/rofiperlungoding/jarvis/releases/download/v1.0.2/JARVIS-Setup-1.0.2.exe).
