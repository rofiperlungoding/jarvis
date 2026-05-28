"""Unit tests for ``jarvis.voice.stt.base``.

Covers the :class:`Transcript` dataclass field shape and validators, and
verifies that the :class:`STTEngine` Protocol is :func:`runtime_checkable`
so the bootstrap in ``app.py`` can ``isinstance``-check engine plugins.

Tests reference:
* Requirement 1.3 — Transcript carries ``started_at`` / ``duration_ms``
  alongside text/confidence so the audit log and latency telemetry can
  reason about each captured utterance.
* Requirement 1.8 — ``confidence`` is a float in ``[0, 1]``; values
  outside that range are rejected at construction time so the gating
  threshold (``< 0.4``) cannot be silently bypassed.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime

import pytest

from jarvis.voice.stt.base import AudioBuffer, STTEngine, Transcript

# ---------------------------------------------------------------------------
# Transcript shape
# ---------------------------------------------------------------------------


def _valid_transcript(**overrides: object) -> Transcript:
    base: dict[str, object] = {
        "text": "turn on the lights",
        "confidence": 0.92,
        "started_at": datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC),
        "duration_ms": 1234,
        "language": "en",
    }
    base.update(overrides)
    return Transcript(**base)  # type: ignore[arg-type]


def test_transcript_fields_match_design_data_model() -> None:
    """Field set must be exactly: text, confidence, started_at, duration_ms, language."""
    field_names = tuple(f.name for f in dataclasses.fields(Transcript))
    assert field_names == (
        "text",
        "confidence",
        "started_at",
        "duration_ms",
        "language",
    )


def test_transcript_is_frozen() -> None:
    t = _valid_transcript()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.text = "mutated"  # type: ignore[misc]


def test_transcript_is_hashable_and_equatable() -> None:
    a = _valid_transcript()
    b = _valid_transcript()
    assert a == b
    assert hash(a) == hash(b)
    assert a is not b


def test_transcript_accepts_empty_text() -> None:
    """Empty text is valid; the Dialog_Manager handles re-prompting (Req 1.8)."""
    t = _valid_transcript(text="", confidence=0.0)
    assert t.text == ""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, -1.0, 2.0])
def test_transcript_rejects_out_of_range_confidence(bad_confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_transcript(confidence=bad_confidence)


@pytest.mark.parametrize("good_confidence", [0.0, 0.4, 0.5, 1.0])
def test_transcript_accepts_in_range_confidence(good_confidence: float) -> None:
    t = _valid_transcript(confidence=good_confidence)
    assert t.confidence == good_confidence


def test_transcript_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        _valid_transcript(duration_ms=-1)


def test_transcript_accepts_zero_duration() -> None:
    """A zero-length utterance is structurally valid (e.g., immediate VAD cut)."""
    t = _valid_transcript(duration_ms=0)
    assert t.duration_ms == 0


def test_transcript_rejects_naive_datetime() -> None:
    naive = datetime(2024, 5, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        _valid_transcript(started_at=naive)


def test_transcript_rejects_empty_language() -> None:
    with pytest.raises(ValueError, match="language"):
        _valid_transcript(language="")


# ---------------------------------------------------------------------------
# STTEngine Protocol
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal in-memory engine used to exercise the Protocol contract."""

    async def transcribe(self, audio: AudioBuffer, language: str) -> Transcript:
        return Transcript(
            text="hello",
            confidence=0.95,
            started_at=datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC),
            duration_ms=len(audio),
            language=language,
        )


def test_sttengine_is_runtime_checkable() -> None:
    """The bootstrap (``app.py``) relies on isinstance() against STTEngine."""
    assert isinstance(_FakeEngine(), STTEngine)


class _MissingTranscribe:
    pass


def test_sttengine_rejects_implementations_without_transcribe() -> None:
    assert not isinstance(_MissingTranscribe(), STTEngine)


def test_fake_engine_returns_transcript_with_requested_language() -> None:
    engine: STTEngine = _FakeEngine()
    result = asyncio.run(engine.transcribe(b"\x00\x00" * 8000, "en"))
    assert isinstance(result, Transcript)
    assert result.language == "en"
    assert result.text == "hello"


# ---------------------------------------------------------------------------
# AudioBuffer alias
# ---------------------------------------------------------------------------


def test_audio_buffer_alias_is_bytes() -> None:
    """AudioBuffer is a structural alias for bytes; downstream code may rely on this."""
    buf: AudioBuffer = b"\x00\x01"
    assert isinstance(buf, bytes)
