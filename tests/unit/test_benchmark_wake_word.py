"""Tests for the wake-word benchmark harness (task 23.2).

These tests exercise the *harness plumbing* — corpus loading, frame
slicing, threshold logic, JSON shape, and CLI exit codes — without
loading the native :mod:`pvporcupine` engine. The actual FAR/FRR
measurement is exercised in release certification (see design.md
§Wake-Word Validation), not per-PR CI.

Validates: Requirements 18.2, 18.3 (harness-level only)
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import struct
import wave

from benchmarks import wake_word
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(
    path: Path,
    *,
    duration_seconds: float = 0.5,
    sample_rate: int = wake_word._REQUIRED_SAMPLE_RATE_HZ,
    channels: int = wake_word._REQUIRED_CHANNELS,
    sample_width: int = wake_word._REQUIRED_SAMPLE_WIDTH_BYTES,
) -> None:
    n_samples = round(duration_seconds * sample_rate)
    silent = b"\x00\x00" * n_samples * channels
    if sample_width != 2:
        silent = b"\x00" * n_samples * channels * sample_width
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(silent)


# ---------------------------------------------------------------------------
# Frame slicing
# ---------------------------------------------------------------------------


def test_iter_porcupine_frames_yields_fixed_size_chunks() -> None:
    # 3.5 frames worth of audio: the harness must drop the partial tail.
    full_frames = 3
    pcm = b"\x00" * (
        full_frames * wake_word._FRAME_BYTES + wake_word._FRAME_BYTES // 2
    )
    chunks = list(wake_word._iter_porcupine_frames(pcm))
    assert len(chunks) == full_frames
    assert all(len(c) == wake_word._FRAME_BYTES for c in chunks)


def test_iter_porcupine_frames_empty_input() -> None:
    assert list(wake_word._iter_porcupine_frames(b"")) == []


# ---------------------------------------------------------------------------
# WAV reader
# ---------------------------------------------------------------------------


def test_read_wav_pcm_accepts_well_formed_file(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    _write_wav(wav, duration_seconds=1.0)
    pcm, duration = wake_word._read_wav_pcm(wav)
    # 1.0 s * 16000 Hz * 2 bytes/sample == 32000 bytes
    assert len(pcm) == 32000
    assert duration == pytest.approx(1.0, rel=1e-6)


def test_read_wav_pcm_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    wav = tmp_path / "bad_rate.wav"
    _write_wav(wav, sample_rate=44100)
    with pytest.raises(wake_word.CorpusError, match="16000 Hz"):
        wake_word._read_wav_pcm(wav)


def test_read_wav_pcm_rejects_stereo(tmp_path: Path) -> None:
    wav = tmp_path / "stereo.wav"
    _write_wav(wav, channels=2)
    with pytest.raises(wake_word.CorpusError, match="mono"):
        wake_word._read_wav_pcm(wav)


def test_iter_wav_files_sorts_recursively(tmp_path: Path) -> None:
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.wav").write_bytes(b"")
    found = [p.name for p in wake_word._iter_wav_files(tmp_path)]
    # Sorted by full path under rglob; the harness only guarantees a
    # stable ordering (sorted) so we check that property.
    assert found == sorted(found)
    assert set(found) == {"a.wav", "b.wav", "c.wav"}


def test_iter_wav_files_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(wake_word.CorpusError, match="does not exist"):
        list(wake_word._iter_wav_files(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Result types and threshold logic
# ---------------------------------------------------------------------------


def test_far_result_passed_at_threshold() -> None:
    # Exactly at the threshold counts as passing per "≤ 0.5".
    far = wake_word.FARResult(
        activations=12,
        duration_seconds=24 * 3600.0,
        per_hour=wake_word.WAKE_WORD_FAR_THRESHOLD,
    )
    assert far.passed


def test_far_result_fails_above_threshold() -> None:
    far = wake_word.FARResult(
        activations=100,
        duration_seconds=3600.0,
        per_hour=100.0,
    )
    assert not far.passed


def test_frr_result_passed_requires_minimum_sample_size() -> None:
    # Under the 200-utterance floor, even a perfect rate is not "passed".
    frr_few = wake_word.FRRResult(total=10, missed=0, rate=0.0)
    assert not frr_few.passed
    frr_enough = wake_word.FRRResult(total=200, missed=10, rate=0.05)
    assert frr_enough.passed
    frr_too_many_misses = wake_word.FRRResult(total=200, missed=11, rate=0.055)
    assert not frr_too_many_misses.passed


def test_benchmark_result_to_json_dict_round_trips() -> None:
    far = wake_word.FARResult(activations=2, duration_seconds=7200.0, per_hour=1.0)
    frr = wake_word.FRRResult(total=200, missed=2, rate=0.01)
    result = wake_word.BenchmarkResult(
        far=far,
        frr=frr,
        keyword="jarvis",
        sensitivity=0.55,
        sample_rate_hz=16000,
        frame_samples=512,
        porcupine_version="3.0.0",
        synthetic=False,
    )
    payload = result.to_json_dict()
    rendered = json.dumps(payload, sort_keys=True)
    parsed = json.loads(rendered)
    assert parsed["schema_version"] == 1
    assert parsed["far"]["per_hour"] == 1.0
    assert parsed["frr"]["rate"] == 0.01
    assert parsed["thresholds"]["far_per_hour_max"] == 0.5
    assert parsed["thresholds"]["frr_max"] == 0.05
    assert parsed["thresholds"]["minimum_positive_utterances"] == 200
    # Combined pass: per_hour=1.0 > 0.5 → fails FAR.
    assert parsed["passed"] is False


def test_benchmark_result_passed_when_all_within_thresholds() -> None:
    far = wake_word.FARResult(activations=0, duration_seconds=86400.0, per_hour=0.0)
    frr = wake_word.FRRResult(total=200, missed=0, rate=0.0)
    result = wake_word.BenchmarkResult(
        far=far,
        frr=frr,
        keyword="jarvis",
        sensitivity=0.55,
        sample_rate_hz=16000,
        frame_samples=512,
        porcupine_version=None,
    )
    assert result.passed is True


def test_benchmark_result_skipped_is_not_passed() -> None:
    result = wake_word.BenchmarkResult(
        far=None,
        frr=None,
        keyword="jarvis",
        sensitivity=0.55,
        sample_rate_hz=16000,
        frame_samples=512,
        porcupine_version=None,
        skipped=True,
        skipped_reason="no key",
    )
    assert result.passed is False


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def test_synth_negative_pcm_is_aligned_to_sample_size() -> None:
    pcm, duration = wake_word._synth_negative_pcm(0.5, seed=1)
    # 0.5 s * 16000 Hz * 2 bytes/sample == 16000 bytes
    assert len(pcm) == 16000
    assert duration == pytest.approx(0.5, rel=1e-6)
    # All samples are valid 16-bit signed integers.
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert all(-32768 <= s <= 32767 for s in samples)


def test_synth_negative_pcm_is_deterministic_per_seed() -> None:
    a, _ = wake_word._synth_negative_pcm(0.25, seed=42)
    b, _ = wake_word._synth_negative_pcm(0.25, seed=42)
    c, _ = wake_word._synth_negative_pcm(0.25, seed=43)
    assert a == b
    assert a != c


def test_synth_positive_pcm_matches_contract() -> None:
    pcm, duration = wake_word._synth_positive_pcm(seed=0)
    # 1.5 s * 16000 Hz * 2 bytes/sample == 48000 bytes
    assert len(pcm) == 48000
    assert duration == pytest.approx(1.5, rel=1e-6)


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------


def test_run_benchmark_skips_without_access_key() -> None:
    result = wake_word.run_benchmark(access_key=None)
    assert result.skipped is True
    assert result.passed is False
    assert "access key" in (result.skipped_reason or "")


def test_main_skip_path_writes_json_and_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PORCUPINE_ACCESS_KEY", raising=False)
    output_path = tmp_path / "results.json"
    exit_code = wake_word.main(
        ["--quiet", "--output", str(output_path)],
    )
    assert exit_code == 0
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["skipped"] is True
    assert payload["passed"] is False
    # And stdout matches the file contents.
    captured = capsys.readouterr()
    assert json.loads(captured.out)["skipped"] is True


def test_main_corpus_error_returns_exit_code_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "ignored-by-fake-handle")

    # Replace the Porcupine handle with a stub so we never touch the
    # native engine. The stub tracks calls so we can assert the FAR /
    # FRR loops would have fired if we had reached them.
    class _FakeHandle:
        version = "fake"
        keyword_label = "jarvis"

        def __init__(self, **_: object) -> None:  # pragma: no cover - unused
            self.closed = False

        def count_activations(
            self, frames: Iterator[bytes]
        ) -> int:  # pragma: no cover - unreachable
            return 0

        def detected_in_clip(
            self, frames: Iterator[bytes]
        ) -> bool:  # pragma: no cover - unreachable
            return True

        def close(self) -> None:  # pragma: no cover - unreachable
            self.closed = True

    monkeypatch.setattr(wake_word, "_PorcupineHandle", _FakeHandle)

    exit_code = wake_word.main(
        [
            "--negative",
            str(tmp_path / "missing-neg"),
            "--positive",
            str(tmp_path / "missing-pos"),
            "--quiet",
        ]
    )
    assert exit_code == 2
    # No JSON written on the operational-failure path.
    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_benchmark_fake_handle_synthetic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end happy path with a fake Porcupine handle.

    Exercises the FAR / FRR loops, threshold checks, and JSON shape
    without the native engine. The fake reports zero activations on the
    negative pass and detects every positive utterance, which yields a
    passing record.
    """

    class _PerfectHandle:
        version = "fake-perfect"
        keyword_label = "jarvis"

        def __init__(self, **_: object) -> None:
            pass

        def count_activations(self, frames: Iterator[bytes]) -> int:
            # Exhaust the iterator so the harness's frame slicing is
            # actually exercised even on the no-detections branch.
            for _ in frames:
                pass
            return 0

        def detected_in_clip(self, frames: Iterator[bytes]) -> bool:
            for _ in frames:
                pass
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(wake_word, "_PorcupineHandle", _PerfectHandle)

    result = wake_word.run_benchmark(
        access_key="ignored-by-fake",
        synthetic_negative_seconds=2.0,
        synthetic_positive_count=3,
    )
    assert not result.skipped
    assert result.synthetic is True
    assert result.far is not None
    assert result.far.activations == 0
    assert result.far.per_hour == 0.0
    assert result.frr is not None
    assert result.frr.total == 3
    assert result.frr.missed == 0
    assert result.frr.rate == 0.0
    # FRR sample size is below the 200-floor → FRR.passed is False, so
    # the combined result is also not "passed". This is the correct
    # behaviour for synthetic smoke runs.
    assert result.frr.passed is False
    assert result.passed is False


def test_run_benchmark_failing_far_drives_strict_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured FAR above 0.5/hr produces exit code 1 under strict mode."""

    class _NoisyHandle:
        version = "fake-noisy"
        keyword_label = "jarvis"

        def __init__(self, **_: object) -> None:
            pass

        def count_activations(self, frames: Iterator[bytes]) -> int:
            count = 0
            for _ in frames:
                count += 1
            return count

        def detected_in_clip(self, frames: Iterator[bytes]) -> bool:
            for _ in frames:
                pass
            return True

        def close(self) -> None:
            pass

    monkeypatch.setattr(wake_word, "_PorcupineHandle", _NoisyHandle)
    monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "fake")
    output_path = tmp_path / "out.json"

    exit_code = wake_word.main(
        [
            "--quiet",
            "--output",
            str(output_path),
            "--synthetic-negative-seconds",
            "0.5",
            "--synthetic-positive-count",
            "1",
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["far"]["per_hour"] > 0.5
    assert exit_code == 1


def test_main_no_strict_thresholds_always_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-strict-thresholds`` reports the failure but exits 0 anyway."""

    class _NoisyHandle:
        version = "fake-noisy"
        keyword_label = "jarvis"

        def __init__(self, **_: object) -> None:
            pass

        def count_activations(self, frames: Iterator[bytes]) -> int:
            return sum(1 for _ in frames)

        def detected_in_clip(self, frames: Iterator[bytes]) -> bool:
            for _ in frames:
                pass
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(wake_word, "_PorcupineHandle", _NoisyHandle)
    monkeypatch.setenv("PORCUPINE_ACCESS_KEY", "fake")
    output_path = tmp_path / "out.json"

    exit_code = wake_word.main(
        [
            "--quiet",
            "--output",
            str(output_path),
            "--synthetic-negative-seconds",
            "0.5",
            "--synthetic-positive-count",
            "1",
            "--no-strict-thresholds",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
