"""Text-to-Speech subsystem for the JARVIS voice pipeline.

Re-exports the :class:`TTSEngine` Protocol and
:class:`SentenceAccumulator` from :mod:`jarvis.voice.tts.base`, plus the
default :class:`PiperTTS` adapter from :mod:`jarvis.voice.tts.piper`, so
callers can ``from jarvis.voice.tts import PiperTTS`` without needing to
know the submodule layout.
"""

from __future__ import annotations

from jarvis.voice.tts.base import SentenceAccumulator, TTSEngine
from jarvis.voice.tts.piper import DEFAULT_VOICE_ID, PiperTTS

__all__ = [
    "DEFAULT_VOICE_ID",
    "PiperTTS",
    "SentenceAccumulator",
    "TTSEngine",
]
