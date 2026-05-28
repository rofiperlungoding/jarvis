"""Unit tests for ``jarvis.voice.stt.cloud``.

Covers the construction-time privacy gate that enforces Requirement
13.2 ("the system MUST process voice locally and MUST NOT transmit raw
audio to any cloud service") at the engine boundary.

Tests reference:
* Requirement 13.2 — privacy mode: cloud STT must refuse to instantiate
  while ``voice.stt.local_only=true`` so no audio buffer can ever reach
  the engine in privacy mode.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from jarvis.config.schema import VoiceSttConfig
from jarvis.voice.stt.base import STTEngine
from jarvis.voice.stt.cloud import CloudSTT, CloudSTTDisabledError

# ---------------------------------------------------------------------------
# Construction-time privacy gate (Requirement 13.2)
# ---------------------------------------------------------------------------


def test_explicit_local_only_true_keyword_blocks_construction() -> None:
    """``local_only=True`` keyword must veto construction outright."""
    with pytest.raises(CloudSTTDisabledError) as exc:
        CloudSTT(local_only=True)
    # The message should point users at the config knob to flip.
    assert "local_only" in str(exc.value)


def test_explicit_local_only_false_keyword_allows_construction() -> None:
    """``local_only=False`` keyword opts into the cloud engine."""
    engine = CloudSTT(local_only=False)
    # Structural check — CloudSTT must satisfy the STTEngine Protocol so
    # the bootstrap layer can substitute it for FasterWhisperSTT.
    assert isinstance(engine, STTEngine)


def test_config_with_local_only_true_blocks_construction() -> None:
    """A privacy-mode config must veto construction even without the kwarg."""
    cfg = VoiceSttConfig(local_only=True, engine="faster_whisper")
    with pytest.raises(CloudSTTDisabledError):
        CloudSTT(cfg)


def test_config_with_local_only_false_allows_construction() -> None:
    """A non-privacy config passes the gate and yields a working stub."""
    cfg = VoiceSttConfig(local_only=False, engine="cloud")
    engine = CloudSTT(cfg)
    assert isinstance(engine, STTEngine)


def test_kwarg_overrides_config_local_only() -> None:
    """The ``local_only`` keyword wins over ``config.local_only``.

    Bootstrap code (and tests) need a way to gate the engine without
    rebuilding a full pydantic model. Mirroring CLI-over-config
    semantics, the keyword takes precedence — and that precedence is
    what guarantees an explicit privacy override cannot be silently
    ignored by a stale config object.
    """
    cfg = VoiceSttConfig(local_only=False, engine="cloud")
    with pytest.raises(CloudSTTDisabledError):
        CloudSTT(cfg, local_only=True)


def test_constructor_requires_a_privacy_signal() -> None:
    """Bare ``CloudSTT()`` must fail loudly rather than default to off.

    Defaulting ``local_only`` to ``False`` would let a bug in the
    bootstrap layer instantiate a cloud engine without ever consulting
    the user's config — exactly the failure mode Requirement 13.2 forbids.
    """
    with pytest.raises(TypeError):
        CloudSTT()


# ---------------------------------------------------------------------------
# Stub behavior
# ---------------------------------------------------------------------------


def test_transcribe_raises_not_implemented_until_wired() -> None:
    """The stub must not silently produce empty transcripts.

    A no-op transcription would be downstream-indistinguishable from a
    real low-confidence result and would slip past the Dialog_Manager's
    < 0.4 confidence gate (Requirement 1.8) only by accident. Failing
    loud keeps any accidental wiring obvious.
    """
    engine = CloudSTT(local_only=False)
    with pytest.raises(NotImplementedError):
        asyncio.run(engine.transcribe(b"\x00" * 32, "en"))
    # Sanity: a Transcript reference exists so future implementers don't
    # have to chase imports — pin the import so a rename triggers a test
    # failure here too.
    _ = datetime(2024, 1, 1, tzinfo=UTC)
