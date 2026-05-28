"""Wake-word detection benchmark harness (task 23.2).

This module measures the two non-functional acceptance criteria attached
to the Wake_Word_Detector in ``requirements.md`` §Requirement 18:

* **FAR (False Activations per hour)** — Requirement 18.2 caps this at
  ``0.5`` averaged over a 24-hour negative corpus (podcasts, household
  ambient, …). The harness streams the corpus through Porcupine in
  fixed 512-sample / 16 kHz / 16-bit / mono frames (the engine's only
  supported input shape, see :mod:`jarvis.voice.audio_io`), counts every
  positive ``process`` return, and divides by the total negative-audio
  duration in hours.

* **FRR (False Rejection Rate)** — Requirement 18.3 caps this at
  ``0.05`` over >= 200 wake-phrase utterances captured at 1-3 m. Each
  positive utterance is replayed through the same engine and counted as
  *missed* if no detection fires anywhere in the clip. ``FRR =
  missed / total``.

Both metrics are emitted as a single JSON document so CI can track them
over time. The harness is intended to be invoked from a release-cert
workflow rather than per-PR CI (the negative corpus alone is multi-GB),
which is why the ``--negative-dir`` / ``--positive-dir`` arguments are
required for "real" runs and a synthetic fallback exists only for
smoke-testing the harness plumbing.

Design notes
------------

* The harness deliberately reuses :class:`jarvis.voice.wake_word.WakeWordDetector`'s
  audio contract (16 kHz / 16-bit / mono / 512-sample frames) but does
  NOT reuse the detector class itself: the benchmark needs synchronous
  per-frame counting plus the keyword index returned by
  ``Porcupine.process``, neither of which is exposed by the production
  detector. We therefore drive ``pvporcupine.create`` directly. The
  frame-size / sample-rate constants are imported from
  :mod:`jarvis.voice.audio_io` so the harness automatically tracks any
  future ABI change in pvporcupine that the production code adopts.

* :mod:`pvporcupine` requires a Picovoice access key that is *not*
  bundled with the repository. The harness pulls it from (in order):
  ``--access-key`` CLI flag, the ``PORCUPINE_ACCESS_KEY`` environment
  variable, or — when invoked under the JARVIS application — a
  CredentialStore lookup is left to the caller (we do not import the
  store here to keep the benchmark self-contained). When no key is
  available the harness emits a ``skipped`` JSON record and exits ``0``;
  CI can decide whether to treat that as a soft failure.

* Real corpora are read with the standard library's :mod:`wave` module
  (no ``soundfile``/``librosa`` dependency on the benchmark hot path).
  Files must be 16-bit signed-PCM mono at 16 kHz; mismatches raise
  immediately with a precise message rather than silently degrading.

* For offline smoke tests we generate synthetic audio with NumPy
  (Gaussian noise + a sine sweep). Synthetic audio is *not* a substitute
  for the real corpora described in the design; it exists solely so the
  harness itself can be exercised on a developer machine without a
  multi-GB asset download. The emitted JSON includes ``"synthetic":
  true`` so downstream graphs can flag those runs.

Validates: Requirements 18.2, 18.3
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any
import wave

# Allow ``python -m benchmarks.wake_word`` from a repo checkout that has
# *not* been ``pip install -e .``'d. The benchmark only depends on the
# two constants below from :mod:`jarvis.voice.audio_io`, so we add the
# repository's ``src/`` directory to ``sys.path`` if (and only if) the
# ``jarvis`` package is not already importable. This keeps the harness
# usable in both editable-install and bare-checkout configurations
# without changing global tooling.
try:  # pragma: no cover - exercised implicitly by the import below
    from jarvis.voice.audio_io import (
        PORCUPINE_FRAME_SAMPLES,
        PORCUPINE_SAMPLE_RATE_HZ,
    )
except ModuleNotFoundError:  # pragma: no cover - bare-checkout fallback
    _SRC_DIR = Path(__file__).resolve().parent.parent / "src"
    if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from jarvis.voice.audio_io import (
        PORCUPINE_FRAME_SAMPLES,
        PORCUPINE_SAMPLE_RATE_HZ,
    )

logger = logging.getLogger("jarvis.benchmarks.wake_word")

__all__ = [
    "WAKE_WORD_FAR_THRESHOLD",
    "WAKE_WORD_FRR_THRESHOLD",
    "BenchmarkResult",
    "FARResult",
    "FRRResult",
    "main",
    "run_benchmark",
]


# ---------------------------------------------------------------------------
# Thresholds — anchored to Requirement 18.2 / 18.3 verbatim.
# ---------------------------------------------------------------------------

#: Maximum acceptable false activations per hour over the 24-hour negative
#: corpus (Requirement 18.2). Lower is better.
WAKE_WORD_FAR_THRESHOLD: float = 0.5

#: Maximum acceptable false rejection rate over >= 200 positive utterances at
#: 1-3 m (Requirement 18.3). Lower is better.
WAKE_WORD_FRR_THRESHOLD: float = 0.05

# 16-bit signed PCM is the only format Porcupine consumes. ``wave`` reports
# sample width in *bytes*, so the constant is named accordingly.
_REQUIRED_SAMPLE_WIDTH_BYTES = 2
_REQUIRED_CHANNELS = 1
_REQUIRED_SAMPLE_RATE_HZ = PORCUPINE_SAMPLE_RATE_HZ
_FRAME_BYTES = PORCUPINE_FRAME_SAMPLES * _REQUIRED_SAMPLE_WIDTH_BYTES
_FRAME_UNPACK = struct.Struct(f"<{PORCUPINE_FRAME_SAMPLES}h")
_SECONDS_PER_HOUR = 3600.0

# Minimum positive utterance count mandated by Requirement 18.3 — surfaced
# in the JSON output so reviewers can confirm the run met the spec.
_MINIMUM_POSITIVE_UTTERANCES = 200


# ---------------------------------------------------------------------------
# Result types — round-trip safe through ``json.dumps``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FARResult:
    """Outcome of the negative-corpus pass.

    All counts are non-negative integers; ``duration_seconds`` may be a
    float to capture sub-second-precision corpora. ``per_hour`` is
    derived (``activations / duration_hours``) and stored explicitly so
    consumers do not have to recompute it from the JSON record.
    """

    activations: int
    duration_seconds: float
    per_hour: float
    threshold: float = WAKE_WORD_FAR_THRESHOLD
    files: int = 0

    @property
    def passed(self) -> bool:
        """Whether the measured FAR satisfies Requirement 18.2."""
        return self.per_hour <= self.threshold


@dataclass(frozen=True)
class FRRResult:
    """Outcome of the positive-corpus pass.

    ``rate`` is ``missed / total`` and is left as ``0.0`` when the corpus
    is empty (the harness flags an empty positive corpus as a separate
    failure rather than producing a misleading "perfect" FRR).
    """

    total: int
    missed: int
    rate: float
    threshold: float = WAKE_WORD_FRR_THRESHOLD
    minimum_required_total: int = _MINIMUM_POSITIVE_UTTERANCES

    @property
    def passed(self) -> bool:
        """Whether the measured FRR satisfies Requirement 18.3.

        A run that fails to provide the spec-mandated minimum number of
        utterances is treated as *not* passed, even if the measured rate
        is below the threshold — Requirement 18.3 is conditioned on the
        sample size and we refuse to imply confidence we do not have.
        """
        return self.rate <= self.threshold and self.total >= self.minimum_required_total


@dataclass(frozen=True)
class BenchmarkResult:
    """Top-level JSON record emitted by the harness."""

    far: FARResult | None
    frr: FRRResult | None
    keyword: str
    sensitivity: float
    sample_rate_hz: int
    frame_samples: int
    porcupine_version: str | None
    synthetic: bool = False
    skipped: bool = False
    skipped_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        """Combined pass/fail for the run.

        A skipped run is *not* a pass (we cannot prove the spec on a
        skip), but we also do not fail CI on it — the CLI exit code is
        ``0`` for skip and ``1`` only for measured-but-failing runs.
        """
        if self.skipped:
            return False
        far_ok = self.far is None or self.far.passed
        frr_ok = self.frr is None or self.frr.passed
        return far_ok and frr_ok

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise to a plain ``dict`` suitable for ``json.dumps``."""
        return {
            "schema_version": 1,
            "keyword": self.keyword,
            "sensitivity": self.sensitivity,
            "sample_rate_hz": self.sample_rate_hz,
            "frame_samples": self.frame_samples,
            "porcupine_version": self.porcupine_version,
            "synthetic": self.synthetic,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "passed": self.passed,
            "far": asdict(self.far) if self.far is not None else None,
            "frr": asdict(self.frr) if self.frr is not None else None,
            "thresholds": {
                "far_per_hour_max": WAKE_WORD_FAR_THRESHOLD,
                "frr_max": WAKE_WORD_FRR_THRESHOLD,
                "minimum_positive_utterances": _MINIMUM_POSITIVE_UTTERANCES,
            },
        }


# ---------------------------------------------------------------------------
# Audio loading helpers.
# ---------------------------------------------------------------------------


class CorpusError(RuntimeError):
    """Raised when a corpus file cannot be read or fails the audio contract."""


def _iter_wav_files(directory: Path) -> Iterator[Path]:
    """Yield every ``.wav`` file under ``directory`` in sorted order.

    Sorted so the harness output is deterministic across hosts; CI can
    diff JSON records across runs without spurious shuffles. Recursion
    is intentional: real corpora are typically split across speaker /
    distance subdirectories.
    """
    if not directory.is_dir():
        raise CorpusError(f"Corpus directory does not exist: {directory!s}")
    paths = sorted(p for p in directory.rglob("*.wav") if p.is_file())
    yield from paths


def _read_wav_pcm(path: Path) -> tuple[bytes, float]:
    """Read a 16 kHz / 16-bit / mono WAV file and return its raw PCM bytes.

    Returns:
        A ``(pcm_bytes, duration_seconds)`` tuple. ``duration_seconds``
        is computed from the actual frame count rather than any header
        ``ntotalsamples`` claim, so truncated files report their real
        length.

    Raises:
        CorpusError: On any audio-format mismatch. The error message
            names the file so corpus curators can fix the offending
            asset directly.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if channels != _REQUIRED_CHANNELS:
                raise CorpusError(
                    f"{path.name}: expected mono ({_REQUIRED_CHANNELS} channel) "
                    f"audio; got {channels} channels."
                )
            if sample_width != _REQUIRED_SAMPLE_WIDTH_BYTES:
                raise CorpusError(
                    f"{path.name}: expected 16-bit PCM "
                    f"({_REQUIRED_SAMPLE_WIDTH_BYTES} bytes/sample); got "
                    f"{sample_width} bytes/sample."
                )
            if frame_rate != _REQUIRED_SAMPLE_RATE_HZ:
                raise CorpusError(
                    f"{path.name}: expected {_REQUIRED_SAMPLE_RATE_HZ} Hz; "
                    f"got {frame_rate} Hz."
                )
            pcm = wf.readframes(n_frames)
    except CorpusError:
        raise
    except wave.Error as exc:
        raise CorpusError(f"{path!s}: cannot read WAV file ({exc})") from exc
    duration_seconds = len(pcm) / (
        _REQUIRED_SAMPLE_WIDTH_BYTES * _REQUIRED_SAMPLE_RATE_HZ
    )
    return pcm, duration_seconds


def _iter_porcupine_frames(pcm: bytes) -> Iterator[bytes]:
    """Slice a PCM byte string into Porcupine-sized frames.

    Trailing partial frames (i.e. when ``len(pcm) % _FRAME_BYTES != 0``)
    are dropped: Porcupine's ``process`` would reject them, and dropping
    < 32 ms of audio at the tail of a corpus item is statistically
    irrelevant for FAR/FRR.
    """
    nbytes = len(pcm)
    end = nbytes - (nbytes % _FRAME_BYTES)
    for offset in range(0, end, _FRAME_BYTES):
        yield pcm[offset : offset + _FRAME_BYTES]


# ---------------------------------------------------------------------------
# Synthetic fallback corpora — for offline harness smoke tests only.
# ---------------------------------------------------------------------------


def _synth_negative_pcm(
    duration_seconds: float, *, seed: int = 0
) -> tuple[bytes, float]:
    """Generate a deterministic noise + sine-sweep negative clip.

    The clip is *not* representative of a real podcast / ambient corpus
    — Porcupine should treat it as silence-equivalent — but it exercises
    the harness end to end without needing a multi-GB asset.

    NumPy is imported here (rather than at module top) so the harness
    can be imported on environments without NumPy to inspect its CLI
    surface; in practice NumPy is a hard runtime dependency of the
    project so this branch is always reachable in real use.
    """
    import numpy as np  # noqa: PLC0415 — local import; see docstring.

    rng = np.random.default_rng(seed)
    n_samples = round(duration_seconds * _REQUIRED_SAMPLE_RATE_HZ)
    if n_samples <= 0:
        return b"", 0.0
    # Low-amplitude white noise + a slow sine sweep; both are well
    # outside the spectral signature Porcupine learned for "jarvis".
    noise = rng.normal(0.0, 0.02, size=n_samples)
    t = np.arange(n_samples, dtype=np.float64) / _REQUIRED_SAMPLE_RATE_HZ
    sweep_hz = np.linspace(120.0, 320.0, n_samples)
    sweep = 0.05 * np.sin(2.0 * math.pi * sweep_hz * t)
    samples = noise + sweep
    pcm_int16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    pcm_bytes = pcm_int16.tobytes()
    actual_duration = len(pcm_bytes) / (
        _REQUIRED_SAMPLE_WIDTH_BYTES * _REQUIRED_SAMPLE_RATE_HZ
    )
    return pcm_bytes, actual_duration


def _synth_positive_pcm(*, seed: int = 0) -> tuple[bytes, float]:
    """Generate a 1.5-second synthetic "utterance" placeholder.

    There is no acoustic relationship between this clip and a real
    "jarvis" utterance; the harness emits ``synthetic: true`` precisely
    so reviewers do not interpret the resulting FRR figure as a genuine
    measurement of Requirement 18.3.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    duration_seconds = 1.5
    n_samples = round(duration_seconds * _REQUIRED_SAMPLE_RATE_HZ)
    t = np.arange(n_samples, dtype=np.float64) / _REQUIRED_SAMPLE_RATE_HZ
    # Three-tone "word-like" envelope: F0 sweep + two formant-ish
    # harmonics, amplitude-modulated by a Hann-style window so the clip
    # has a clear onset and offset.
    f0 = np.linspace(200.0, 140.0, n_samples)
    harmonics = (
        0.4 * np.sin(2.0 * math.pi * f0 * t)
        + 0.2 * np.sin(2.0 * math.pi * (2.0 * f0) * t)
        + 0.1 * np.sin(2.0 * math.pi * (3.0 * f0) * t)
    )
    window = 0.5 * (1.0 - np.cos(2.0 * math.pi * np.arange(n_samples) / n_samples))
    noise = rng.normal(0.0, 0.005, size=n_samples)
    samples = (harmonics * window) + noise
    pcm_int16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    pcm_bytes = pcm_int16.tobytes()
    actual_duration = len(pcm_bytes) / (
        _REQUIRED_SAMPLE_WIDTH_BYTES * _REQUIRED_SAMPLE_RATE_HZ
    )
    return pcm_bytes, actual_duration


# ---------------------------------------------------------------------------
# Porcupine wrapper.
# ---------------------------------------------------------------------------


class _PorcupineHandle:
    """Thin wrapper that owns a ``Porcupine`` engine for the duration of a run.

    Kept private because the production code paths talk to
    :class:`jarvis.voice.wake_word.WakeWordDetector` instead. The
    benchmark needs synchronous per-frame access plus the keyword index,
    which the production class does not expose.
    """

    def __init__(
        self,
        *,
        access_key: str,
        keyword: str,
        keyword_path: Path | None,
        sensitivity: float,
    ) -> None:
        try:
            import pvporcupine  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "pvporcupine is not installed; cannot run the wake-word "
                "benchmark. Install the 'voice' extras."
            ) from exc

        self._pvporcupine = pvporcupine
        kwargs: dict[str, Any] = {
            "access_key": access_key,
            "sensitivities": [sensitivity],
        }
        if keyword_path is not None:
            kwargs["keyword_paths"] = [str(keyword_path)]
            self.keyword_label = keyword_path.stem
        else:
            kwargs["keywords"] = [keyword]
            self.keyword_label = keyword
        self._engine: Any = pvporcupine.create(**kwargs)

    @property
    def version(self) -> str | None:
        return getattr(self._pvporcupine, "LIBRARY_VERSION", None) or getattr(
            self._pvporcupine, "__version__", None
        )

    def count_activations(self, frames: Iterator[bytes]) -> int:
        """Return the number of positive detections across ``frames``."""
        process = self._engine.process
        unpack = _FRAME_UNPACK.unpack_from
        activations = 0
        for frame in frames:
            pcm = unpack(frame)
            if process(pcm) >= 0:
                activations += 1
        return activations

    def detected_in_clip(self, frames: Iterator[bytes]) -> bool:
        """Return ``True`` iff at least one frame in ``frames`` triggers."""
        process = self._engine.process
        unpack = _FRAME_UNPACK.unpack_from
        for frame in frames:
            pcm = unpack(frame)
            if process(pcm) >= 0:
                return True
        return False

    def close(self) -> None:
        try:
            self._engine.delete()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.exception("Porcupine.delete() raised; ignoring.")


# ---------------------------------------------------------------------------
# Core benchmark loops.
# ---------------------------------------------------------------------------


def _run_far_pass(
    handle: _PorcupineHandle,
    *,
    negative_dir: Path | None,
    synthetic_negative_seconds: float,
) -> FARResult:
    """Stream the negative corpus and count false activations."""
    if negative_dir is not None:
        activations = 0
        total_seconds = 0.0
        files = 0
        for wav_path in _iter_wav_files(negative_dir):
            pcm, duration = _read_wav_pcm(wav_path)
            activations += handle.count_activations(_iter_porcupine_frames(pcm))
            total_seconds += duration
            files += 1
        if files == 0:
            raise CorpusError(
                f"Negative corpus directory is empty: {negative_dir!s}"
            )
    else:
        pcm, total_seconds = _synth_negative_pcm(synthetic_negative_seconds)
        activations = handle.count_activations(_iter_porcupine_frames(pcm))
        files = 1

    duration_hours = max(total_seconds / _SECONDS_PER_HOUR, 1e-9)
    per_hour = activations / duration_hours
    return FARResult(
        activations=activations,
        duration_seconds=total_seconds,
        per_hour=per_hour,
        files=files,
    )


def _run_frr_pass(
    handle: _PorcupineHandle,
    *,
    positive_dir: Path | None,
    synthetic_positive_count: int,
) -> FRRResult:
    """Stream the positive corpus and count missed wake-phrase utterances."""
    if positive_dir is not None:
        total = 0
        missed = 0
        for wav_path in _iter_wav_files(positive_dir):
            pcm, _ = _read_wav_pcm(wav_path)
            total += 1
            if not handle.detected_in_clip(_iter_porcupine_frames(pcm)):
                missed += 1
        if total == 0:
            raise CorpusError(
                f"Positive corpus directory is empty: {positive_dir!s}"
            )
    else:
        total = max(1, synthetic_positive_count)
        missed = 0
        for i in range(total):
            pcm, _ = _synth_positive_pcm(seed=i)
            if not handle.detected_in_clip(_iter_porcupine_frames(pcm)):
                missed += 1

    rate = (missed / total) if total > 0 else 0.0
    return FRRResult(total=total, missed=missed, rate=rate)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    access_key: str | None,
    keyword: str = "jarvis",
    keyword_path: Path | None = None,
    sensitivity: float = 0.55,
    negative_dir: Path | None = None,
    positive_dir: Path | None = None,
    synthetic_negative_seconds: float = 60.0,
    synthetic_positive_count: int = 5,
    skip_far: bool = False,
    skip_frr: bool = False,
) -> BenchmarkResult:
    """Run the wake-word benchmark and return a :class:`BenchmarkResult`.

    A missing ``access_key`` is *not* an error: the function returns a
    ``skipped`` :class:`BenchmarkResult` describing the reason. Callers
    that want hard failure on a missing key should inspect
    :attr:`BenchmarkResult.skipped` themselves.

    The function is the synchronous core of the harness; the CLI wrapper
    in :func:`main` is responsible for argument parsing, JSON output,
    and exit codes.
    """
    started_at = time.time()
    started_monotonic = time.monotonic()
    synthetic = negative_dir is None or positive_dir is None

    if not access_key:
        return BenchmarkResult(
            far=None,
            frr=None,
            keyword=keyword,
            sensitivity=sensitivity,
            sample_rate_hz=_REQUIRED_SAMPLE_RATE_HZ,
            frame_samples=PORCUPINE_FRAME_SAMPLES,
            porcupine_version=None,
            synthetic=synthetic,
            skipped=True,
            skipped_reason=(
                "no Porcupine access key supplied (--access-key / "
                "$PORCUPINE_ACCESS_KEY)"
            ),
            started_at=started_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
        )

    try:
        handle = _PorcupineHandle(
            access_key=access_key,
            keyword=keyword,
            keyword_path=keyword_path,
            sensitivity=sensitivity,
        )
    except RuntimeError as exc:
        return BenchmarkResult(
            far=None,
            frr=None,
            keyword=keyword,
            sensitivity=sensitivity,
            sample_rate_hz=_REQUIRED_SAMPLE_RATE_HZ,
            frame_samples=PORCUPINE_FRAME_SAMPLES,
            porcupine_version=None,
            synthetic=synthetic,
            skipped=True,
            skipped_reason=str(exc),
            started_at=started_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
        )

    try:
        far = (
            None
            if skip_far
            else _run_far_pass(
                handle,
                negative_dir=negative_dir,
                synthetic_negative_seconds=synthetic_negative_seconds,
            )
        )
        frr = (
            None
            if skip_frr
            else _run_frr_pass(
                handle,
                positive_dir=positive_dir,
                synthetic_positive_count=synthetic_positive_count,
            )
        )
    finally:
        handle.close()

    return BenchmarkResult(
        far=far,
        frr=frr,
        keyword=handle.keyword_label,
        sensitivity=sensitivity,
        sample_rate_hz=_REQUIRED_SAMPLE_RATE_HZ,
        frame_samples=PORCUPINE_FRAME_SAMPLES,
        porcupine_version=handle.version,
        synthetic=synthetic,
        skipped=False,
        skipped_reason=None,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started_monotonic,
    )


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks.wake_word",
        description=(
            "Measure Porcupine wake-word FAR/hour and FRR against the "
            "thresholds in Requirement 18.2 / 18.3 and emit a JSON record "
            "for CI tracking."
        ),
    )
    parser.add_argument(
        "--negative",
        "--negative-dir",
        dest="negative_dir",
        type=Path,
        default=None,
        help=(
            "Directory of 16 kHz / 16-bit / mono WAV files representing the "
            "negative corpus (Requirement 18.2 calls for 24 hours). When "
            "omitted a synthetic noise+sweep clip is used; the result is "
            "marked synthetic=true."
        ),
    )
    parser.add_argument(
        "--positive",
        "--positive-dir",
        dest="positive_dir",
        type=Path,
        default=None,
        help=(
            "Directory of 16 kHz / 16-bit / mono WAV files representing the "
            "positive corpus (Requirement 18.3 calls for >= 200 utterances "
            "at 1-3 m). When omitted a synthetic placeholder is used."
        ),
    )
    parser.add_argument(
        "--access-key",
        dest="access_key",
        default=None,
        help=(
            "Picovoice console access key. If omitted, the value of "
            "$PORCUPINE_ACCESS_KEY is used. If neither is present the "
            "harness emits a 'skipped' result and exits 0."
        ),
    )
    parser.add_argument(
        "--keyword",
        dest="keyword",
        default="jarvis",
        help="Built-in Porcupine keyword name (ignored when --keyword-path is given).",
    )
    parser.add_argument(
        "--keyword-path",
        dest="keyword_path",
        type=Path,
        default=None,
        help="Absolute path to a custom .ppn keyword file (overrides --keyword).",
    )
    parser.add_argument(
        "--sensitivity",
        dest="sensitivity",
        type=float,
        default=0.55,
        help="Porcupine sensitivity in [0, 1]; default matches design.md.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        type=Path,
        default=None,
        help="Optional path to write the JSON record to (in addition to stdout).",
    )
    parser.add_argument(
        "--synthetic-negative-seconds",
        dest="synthetic_negative_seconds",
        type=float,
        default=60.0,
        help=(
            "Duration of the synthetic negative clip in seconds when "
            "--negative is not supplied (smoke tests only)."
        ),
    )
    parser.add_argument(
        "--synthetic-positive-count",
        dest="synthetic_positive_count",
        type=int,
        default=5,
        help=(
            "Number of synthetic positive utterances when --positive is not "
            "supplied (smoke tests only; far below the spec-required 200)."
        ),
    )
    parser.add_argument(
        "--skip-far",
        action="store_true",
        help="Skip the negative-corpus pass (FAR will be null in the JSON).",
    )
    parser.add_argument(
        "--skip-frr",
        action="store_true",
        help="Skip the positive-corpus pass (FRR will be null in the JSON).",
    )
    parser.add_argument(
        "--strict-thresholds",
        dest="strict_thresholds",
        action="store_true",
        default=True,
        help=(
            "Exit with status 1 if measured FAR/FRR exceed Requirement "
            "18.2 / 18.3 thresholds. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-strict-thresholds",
        dest="strict_thresholds",
        action="store_false",
        help="Always exit 0 regardless of measured thresholds (CI lab runs).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only emit the JSON record on stdout (suppress info logging).",
    )
    return parser


def _resolve_access_key(cli_value: str | None) -> str | None:
    """Resolve the access key from CLI flag or ``PORCUPINE_ACCESS_KEY``."""
    if cli_value:
        return cli_value
    env_value = os.environ.get("PORCUPINE_ACCESS_KEY")
    if env_value and env_value.strip():
        return env_value.strip()
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit code policy
    ----------------
    * ``0`` — passing run, or skipped run (no access key / no Porcupine).
    * ``1`` — measured run that violates Requirement 18.2 or 18.3 (only
      when ``--strict-thresholds`` is in effect, which is the default).
    * ``2`` — operational failure (missing corpus, bad audio format, …).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    access_key = _resolve_access_key(args.access_key)
    try:
        result = run_benchmark(
            access_key=access_key,
            keyword=args.keyword,
            keyword_path=args.keyword_path,
            sensitivity=args.sensitivity,
            negative_dir=args.negative_dir,
            positive_dir=args.positive_dir,
            synthetic_negative_seconds=args.synthetic_negative_seconds,
            synthetic_positive_count=args.synthetic_positive_count,
            skip_far=args.skip_far,
            skip_frr=args.skip_frr,
        )
    except CorpusError as exc:
        logger.error("Corpus error: %s", exc)
        return 2

    payload = result.to_json_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if result.skipped:
        logger.info("Wake-word benchmark skipped: %s", result.skipped_reason)
        return 0

    if not args.strict_thresholds:
        return 0
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(main())
