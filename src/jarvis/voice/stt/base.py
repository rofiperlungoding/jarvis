"""STT engine contract for the JARVIS voice pipeline.

This module defines the public interface shared by every Speech-to-Text
backend (faster-whisper local default, optional cloud) and the
:class:`Transcript` data model emitted by the engine. Concrete engines
(``faster_whisper.py``, ``cloud.py``) implement :class:`STTEngine` and
return :class:`Transcript` instances.

The shape mirrors the design document's "STT_Engine" and "Data Models"
sections and the project structure note that this file owns the
``STTEngine`` Protocol and the ``Transcript`` dataclass.

Requirement IDs referenced below map to ``requirements.md``:

* Requirement 1.3 — when the user finishes speaking, as determined by a
  voice activity detector with a 700 ms trailing-silence threshold, the
  STT_Engine produces a Transcript of the captured audio. This module
  defines the Transcript record and the ``transcribe`` entry point that
  the audio capture loop calls once the VAD signals end-of-utterance.
* Requirement 1.8 — the Dialog_Manager gates on empty text or
  ``confidence < 0.4``. ``Transcript.confidence`` therefore carries the
  per-utterance confidence (computed in faster-whisper as
  ``mean(exp(token_logprob))`` per the design's STT_Engine notes) so the
  Dialog_Manager can apply that threshold without re-deriving it from
  log-probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

__all__ = [
    "AudioBuffer",
    "STTEngine",
    "Transcript",
]


# ---------------------------------------------------------------------------
# Audio buffer alias
# ---------------------------------------------------------------------------

# The captured utterance handed to the STT engine. The audio capture loop
# (``src/jarvis/voice/audio_io.py``, task 4.1) is responsible for delivering
# 16 kHz / 16-bit / mono PCM as raw ``bytes``. Declaring ``AudioBuffer`` as
# a type alias here keeps the STT contract self-contained: faster-whisper
# accepts a bytes-like buffer directly, and a future ``AudioReframer`` can
# narrow the alias (e.g., to a numpy ``ndarray``) without touching every
# engine implementation. ``bytes`` is chosen over ``bytearray`` so the
# buffer is hashable and immutable across the STT call boundary.
AudioBuffer = bytes


# ---------------------------------------------------------------------------
# Transcript data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transcript:
    """Result of transcribing a single user utterance.

    Field shape matches the design's "Data Models" section verbatim. The
    dataclass is frozen so transcripts can be safely shared across the
    audio capture, dialog, and output loops without defensive copies, and
    so they participate in equality / hashing for property-based tests
    (Property 13: STT empty / low-confidence gating).

    Attributes:
        text: Raw transcribed text. May be the empty string when the
            engine detected speech but produced no decodable tokens; the
            Dialog_Manager treats empty text as a re-prompt trigger
            (Requirement 1.8).
        confidence: Engine-reported confidence in ``[0.0, 1.0]``. For
            faster-whisper this is ``mean(exp(token_logprob))`` across the
            decoded segments. The Dialog_Manager gates at ``< 0.4``
            (Requirement 1.8).
        started_at: Timezone-aware UTC timestamp of when audio capture
            for this utterance began (i.e., when the VAD emitted
            ``speech_start``). Naive datetimes are rejected at
            construction time so downstream consumers can rely on aware
            semantics for serialization and audit logging.
        duration_ms: Length of the captured audio in milliseconds, as
            integer. Includes only the speech window passed to the
            engine, not the trailing 700 ms silence the VAD used to
            decide end-of-utterance (Requirement 1.3).
        language: BCP-47 / ISO-639-1 language tag the engine decoded with
            (e.g., ``"en"``). Matches the ``language`` argument passed
            into :meth:`STTEngine.transcribe`; engines that perform
            language detection should echo back the detected tag here.
    """

    text: str
    confidence: float
    started_at: datetime
    duration_ms: int
    language: str

    def __post_init__(self) -> None:
        # Reject obviously malformed inputs so producers cannot quietly
        # emit a Transcript that violates the gating logic in
        # Requirement 1.8 or the timestamp contract relied on by the
        # audit log. Frozen dataclasses run __post_init__ exactly once
        # after the auto-generated __init__ assigns the fields.
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Transcript.confidence must be in [0.0, 1.0]; got "
                f"{self.confidence!r}"
            )
        if self.duration_ms < 0:
            raise ValueError(
                "Transcript.duration_ms must be non-negative; got "
                f"{self.duration_ms!r}"
            )
        if self.started_at.tzinfo is None:
            raise ValueError(
                "Transcript.started_at must be timezone-aware; got naive "
                f"datetime {self.started_at!r}"
            )
        if self.language == "":
            raise ValueError(
                "Transcript.language must be a non-empty BCP-47 / "
                "ISO-639-1 tag (e.g., 'en')."
            )


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class STTEngine(Protocol):
    """Speech-to-Text engine contract.

    Implementations transcribe a captured utterance and return a
    :class:`Transcript`. The interface is deliberately minimal: VAD
    end-of-utterance detection, audio reframing, and confidence gating
    happen *outside* this protocol (in ``vad.py``, ``audio_io.py``, and
    the Dialog_Manager respectively), per the design's STT_Engine notes.

    Concrete classes:

    * ``FasterWhisperSTT`` — default, local CPU/GPU, satisfies the
      privacy-mode requirement (Requirement 13.2).
    * ``CloudSTT`` — optional, e.g., OpenAI Whisper API; rejected at
      startup when ``voice.stt.local_only=true``.

    Streaming partial transcripts (``transcribe_stream``) are intentionally
    not part of this protocol: the design notes they exist for future
    "live caption" features but are not on the wake-to-response path, so
    keeping them off the core contract avoids forcing every backend to
    implement streaming.
    """

    async def transcribe(
        self,
        audio: AudioBuffer,
        language: str,
    ) -> Transcript:
        """Transcribe a single captured utterance.

        Args:
            audio: Raw 16 kHz / 16-bit / mono PCM frames for the utterance,
                as produced by the audio capture loop after the VAD
                signals ``speech_end`` (Requirement 1.3). The buffer
                contains only the speech window; trailing silence is
                trimmed by the VAD.
            language: BCP-47 / ISO-639-1 language tag to decode in
                (e.g., ``"en"``). Engines that perform automatic language
                detection should still honor this argument when supplied
                non-empty, falling back to detection only when callers
                pass an explicit auto-detect sentinel agreed at the
                engine level.

        Returns:
            A :class:`Transcript` whose ``text`` may be empty if the
            engine produced no decodable tokens. The Dialog_Manager —
            not this method — is responsible for translating empty text
            or low confidence into a re-prompt (Requirement 1.8).
        """
        ...
