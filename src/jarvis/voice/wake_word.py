"""Wake_Word_Detector backed by Picovoice Porcupine.

This module implements task 5.1 from the JARVIS spec: continuously consume
microphone PCM frames and emit a wake event whenever the configured wake
phrase is heard. The implementation wraps the third-party
`pvporcupine <https://picovoice.ai/docs/porcupine/>`_ engine, which the
design selected for its small footprint (<2 MB), <30 ms inference, and
support for custom keyword files (design.md §Wake_Word_Detector).

Design constraints honored here:

* **Frame contract** — Porcupine requires 16 kHz / 16-bit signed-little-endian
  / mono PCM frames of exactly :data:`PORCUPINE_FRAME_SAMPLES` (512) samples.
  The audio capture loop is responsible for delivering frames already
  reframed by :class:`AudioReframer`; this module enforces the contract
  defensively at runtime so a misconfigured upstream surfaces a precise
  error rather than corrupted detections.
* **Latency** — Requirement 1.2 says the Voice_Pipeline must begin capturing
  user speech within 200 ms of wake-word detection. Porcupine processes a
  512-sample (32 ms) frame in well under 10 ms on CPU, and :meth:`run`
  awaits ``on_wake`` synchronously after a positive ``process`` call, so
  the detection-to-callback latency is bounded by ``frame_duration_ms +
  porcupine_inference_ms`` ≈ 40 ms — comfortably inside the 200 ms budget.
* **Built-in vs. custom keywords** — Porcupine ships a curated set of
  built-in keywords (``"jarvis"``, ``"computer"``, ``"alexa"``, …) packaged
  inside :mod:`pvporcupine`. Users may also supply their own ``.ppn`` files
  for custom phrases (Requirement 18.1). The constructor accepts either a
  built-in keyword *name* (str) or an absolute path (:class:`pathlib.Path`)
  to a ``.ppn`` file. Custom files are validated for platform-tag
  compatibility at load time so a misnamed/cross-compiled artefact fails
  fast rather than producing silently-empty detections at runtime.
* **Lazy import** — :mod:`pvporcupine` requires native libraries that are
  not always present on CI runners. Importing this module is therefore
  side-effect free; the actual import happens inside :meth:`start` (or
  :meth:`run`, which calls ``start`` on first use). Type-only imports
  remain at module top level for static analysis.
* **Resource ownership** — every successful :meth:`start` is paired with
  an :meth:`aclose` call (or the async context-manager protocol). The
  detector is single-use: once :meth:`aclose` has fired, callers must
  build a new instance to resume detection.

Validates: Requirements 1.1, 1.2, 18.1
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
import contextlib
import logging
from pathlib import Path
import struct
import sys
from types import TracebackType
from typing import Any, Final

from jarvis.voice.audio_io import (
    PORCUPINE_FRAME_SAMPLES,
    PORCUPINE_SAMPLE_RATE_HZ,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_KEYWORD_JARVIS",
    "Keyword",
    "WakeCallback",
    "WakeWordConfigurationError",
    "WakeWordDetector",
]


# ---------------------------------------------------------------------------
# Public type aliases and constants
# ---------------------------------------------------------------------------

#: Convenience alias for the default JARVIS wake phrase. ``pvporcupine`` ships
#: this as one of its built-in keyword names; passing the string directly to
#: :class:`WakeWordDetector` selects the bundled ``.ppn`` model rather than a
#: user-supplied file.
BUILTIN_KEYWORD_JARVIS: Final[str] = "jarvis"

#: A wake-word entry is either a built-in keyword *name* (e.g. ``"jarvis"``)
#: or an absolute path to a user-supplied ``.ppn`` keyword file. The detector
#: handles both uniformly.
Keyword = str | Path

#: The signature of the user-supplied wake callback. Invoked with no
#: arguments because, as noted in design.md §Wake_Word_Detector, the
#: identity of the matched keyword is not part of the wake event contract —
#: every configured phrase produces the same downstream activation.
WakeCallback = Callable[[], Awaitable[None]]

# Sample width for the 16-bit signed little-endian PCM contract Porcupine
# consumes. Kept private so the public surface is described in terms of
# samples (frame_length) rather than bytes.
_SAMPLE_WIDTH_BYTES: Final[int] = 2
_FRAME_BYTES: Final[int] = PORCUPINE_FRAME_SAMPLES * _SAMPLE_WIDTH_BYTES

# Pre-built struct.Struct for unpacking one Porcupine frame. ``struct`` calls
# are noticeably faster than ``array.array`` round-trips for small fixed-size
# buffers and avoid the overhead of recompiling the format string per frame.
_FRAME_UNPACK = struct.Struct(f"<{PORCUPINE_FRAME_SAMPLES}h")

# Known platform tokens that may appear inside a Porcupine ``.ppn`` filename
# (per Picovoice's distribution convention, e.g. ``Hey-Jarvis_en_windows_v3.ppn``).
# The set is closed: an unrecognised token in the filename is *not* treated as
# a platform mismatch (some legitimate filenames omit the platform tag), only
# a recognised-but-mismatched token is.
_KNOWN_PLATFORM_TAGS: Final[frozenset[str]] = frozenset(
    {
        "windows",
        "mac",
        "linux",
        "raspberry-pi",
        "jetson",
        "beaglebone",
        "android",
        "ios",
        "wasm",
    }
)


def _current_platform_tag() -> str:
    """Return the Porcupine platform tag matching the current interpreter.

    ``sys.platform`` is a runtime value, but mypy treats it as a literal
    derived from the host running the type-checker. We therefore route the
    comparison through a local copy so static analysis does not eliminate
    branches that are valid on other hosts (CI macOS / Linux runners must
    still see meaningful code in this function).
    """
    platform = str(sys.platform)
    if platform.startswith("win"):
        return "windows"
    if platform == "darwin":
        return "mac"
    if platform.startswith("linux"):
        return "linux"
    # Fallback: any unrecognised platform is treated as itself; the validation
    # logic only flags mismatches against *known* platform tokens, so an
    # unmappable host still permits .ppn files that omit the platform tag.
    return platform


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WakeWordConfigurationError(RuntimeError):
    """Raised when the detector cannot be configured or started.

    Wraps the more specific ``pvporcupine`` exception types
    (``PorcupineActivationError``, ``PorcupineInvalidArgumentError``, …) and
    surfaces a single error type to callers. The original exception is
    available via ``__cause__`` for forensic logging in the audit trail.
    """


# ---------------------------------------------------------------------------
# .ppn platform validation
# ---------------------------------------------------------------------------


def _validate_ppn_path(path: Path) -> Path:
    """Validate a user-supplied ``.ppn`` keyword file before Porcupine loads it.

    Two checks happen here:

    1. The path must exist and have a ``.ppn`` extension. Porcupine will
       refuse to load anything else, but failing early produces a clearer
       error than ``PorcupineInvalidArgumentError`` from inside
       ``Porcupine.create``.
    2. The filename's *platform tag* (e.g. ``windows`` in
       ``Hey-Jarvis_en_windows_v3_0_0.ppn``) must match the current host's
       Porcupine platform tag. ``.ppn`` files are platform-specific; loading
       a Linux-tagged file on Windows produces silent zero-detection
       behaviour that is impossible to debug without this check.

    The function is intentionally permissive about *missing* platform tags
    (e.g. ``custom.ppn``) because not every legitimate file carries one;
    only a *recognised-but-mismatched* token raises.

    Returns the (resolved) ``Path`` so callers can pass the canonical form
    to ``pvporcupine.create``.
    """
    resolved = path.expanduser()
    # Use ``resolve(strict=False)`` so symlinks are normalised but a missing
    # file still produces our own friendlier ``WakeWordConfigurationError``
    # below rather than ``FileNotFoundError`` from ``resolve(strict=True)``.
    resolved = resolved.resolve(strict=False)

    if resolved.suffix.lower() != ".ppn":
        raise WakeWordConfigurationError(
            f"Wake-word keyword file must have a .ppn extension; got {resolved!s}"
        )
    if not resolved.is_file():
        raise WakeWordConfigurationError(
            f"Wake-word keyword file does not exist: {resolved!s}"
        )

    # Extract the lowercased filename stem and split on the conventional
    # separators Picovoice uses (underscore is dominant; some older files use
    # dashes). The stem looks like ``Hey-Jarvis_en_windows_v3_0_0``; we want
    # to match the ``windows`` token without false-positives on the
    # ``raspberry-pi`` token, which contains a dash.
    stem_tokens = set(resolved.stem.lower().replace("-", "_").split("_"))
    # ``raspberry-pi`` becomes ``raspberry`` + ``pi`` after the dash split, so
    # restore the canonical form before the comparison.
    if {"raspberry", "pi"} <= stem_tokens:
        stem_tokens.add("raspberry-pi")
    matched_tokens = stem_tokens & _KNOWN_PLATFORM_TAGS
    if not matched_tokens:
        # No recognised platform marker — common for custom-named files.
        # Permit and let Porcupine surface any deeper incompatibility.
        return resolved

    expected = _current_platform_tag()
    if expected in matched_tokens:
        return resolved
    # Any recognised-but-mismatched token is rejected. Sort for a stable
    # error message even when multiple tokens match.
    found = sorted(matched_tokens)
    raise WakeWordConfigurationError(
        f"Wake-word keyword file {resolved.name!r} carries platform tag "
        f"{found!r} but the current platform is {expected!r}. "
        "Re-download the file for this platform from the Picovoice console."
    )


# ---------------------------------------------------------------------------
# WakeWordDetector
# ---------------------------------------------------------------------------


class WakeWordDetector:
    """Wraps Porcupine to emit a wake event when the configured phrase is heard.

    The detector is an async context manager; typical use::

        async with WakeWordDetector(
            access_key=cfg.access_key,
            keyword_paths=[BUILTIN_KEYWORD_JARVIS],
            sensitivity=0.55,
        ) as detector:
            await detector.run(audio_stream, on_wake=trigger_capture)

    The ``run`` coroutine consumes 512-sample / 16 kHz / 16-bit / mono PCM
    frames from ``frames_in`` (typically an :class:`AudioStream` already
    fed through :class:`AudioReframer`) and invokes ``on_wake`` whenever
    Porcupine reports a positive detection. The coroutine completes
    naturally when ``frames_in`` is exhausted (e.g. the audio stream is
    closed) or raises if cancelled.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        access_key: str,
        keyword_paths: Sequence[Keyword],
        sensitivity: float | Sequence[float],
    ) -> None:
        """Configure the detector.

        Args:
            access_key: Picovoice console access key. Mandatory for both
                built-in and custom keywords. The detector treats this
                value as confidential — it is never logged and is held
                only on the instance for the lifetime of the engine.
            keyword_paths: One or more wake-word entries. Each entry is
                either a built-in Porcupine keyword *name* (e.g.
                :data:`BUILTIN_KEYWORD_JARVIS`) or a :class:`pathlib.Path`
                to a user-supplied ``.ppn`` file. Mixing both forms is
                allowed; ordering is preserved so callers reading
                ``Porcupine.process``'s keyword index can correlate
                detections back to the input list (the index is not part
                of the public ``WakeCallback`` contract today, but the
                design leaves room for it).
            sensitivity: Detection sensitivity in ``[0.0, 1.0]``. A
                single float applies the same value to every configured
                keyword; a sequence supplies per-keyword sensitivities
                and must match ``len(keyword_paths)``. Higher values
                trade FAR for FRR (Requirement 18.2 / 18.3).

        Raises:
            WakeWordConfigurationError: If ``access_key`` is empty,
                ``keyword_paths`` is empty, sensitivities are out of
                range, or a supplied ``.ppn`` file fails platform
                validation.
        """
        if not access_key or not access_key.strip():
            raise WakeWordConfigurationError(
                "Porcupine access_key must be a non-empty string. "
                "Configure it in CredentialStore under "
                "'porcupine/access_key'."
            )
        if not keyword_paths:
            raise WakeWordConfigurationError(
                "WakeWordDetector requires at least one keyword "
                "(built-in name or .ppn path)."
            )

        # Materialise the keyword list and split into built-in vs. custom.
        # Order is preserved so the resulting Porcupine keyword index lines
        # up with the caller-supplied ordering.
        normalised: list[tuple[str, str | Path]] = []
        for entry in keyword_paths:
            if isinstance(entry, Path):
                normalised.append(("path", _validate_ppn_path(entry)))
            elif isinstance(entry, str):
                # Empty / whitespace-only strings are an obvious
                # misconfiguration (e.g. an unset config field that fell
                # through default-handling).
                stripped = entry.strip()
                if not stripped:
                    raise WakeWordConfigurationError(
                        "Wake-word keyword name must be non-empty; got "
                        f"{entry!r}."
                    )
                normalised.append(("name", stripped.lower()))
            else:
                raise WakeWordConfigurationError(
                    "Wake-word entries must be `str` (built-in keyword "
                    f"name) or `pathlib.Path` (.ppn file); got {type(entry)!r}."
                )

        # Normalise sensitivities to a per-keyword list so the rest of the
        # class deals with a single shape.
        if isinstance(sensitivity, (int, float)):
            sensitivities = [float(sensitivity)] * len(normalised)
        else:
            sensitivities = [float(s) for s in sensitivity]
            if len(sensitivities) != len(normalised):
                raise WakeWordConfigurationError(
                    "Per-keyword sensitivity list length "
                    f"({len(sensitivities)}) does not match keyword count "
                    f"({len(normalised)})."
                )
        for s in sensitivities:
            if not 0.0 <= s <= 1.0:
                raise WakeWordConfigurationError(
                    f"Sensitivity must lie in [0.0, 1.0]; got {s!r}."
                )

        self._access_key = access_key
        self._keywords: list[tuple[str, str | Path]] = normalised
        self._sensitivities: list[float] = sensitivities
        self._porcupine: Any | None = None
        self._started = False
        self._closed = False
        # Guards against concurrent ``start`` calls racing the lazy import.
        self._start_lock = asyncio.Lock()

    # -------------------------------------------------------- public surface

    @property
    def is_started(self) -> bool:
        """``True`` once :meth:`start` has successfully created the engine."""
        return self._started

    @property
    def frame_length(self) -> int:
        """Required input frame size in samples (512 for Porcupine)."""
        return PORCUPINE_FRAME_SAMPLES

    @property
    def sample_rate_hz(self) -> int:
        """Required input sample rate in Hz (16000 for Porcupine)."""
        return PORCUPINE_SAMPLE_RATE_HZ

    async def __aenter__(self) -> WakeWordDetector:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Lazily construct the underlying ``pvporcupine.Porcupine`` engine.

        Idempotent — calling :meth:`start` repeatedly on a started detector
        is a no-op. ``aclose``-then-``start`` is *not* supported; build a
        fresh detector to resume detection after closure.
        """
        if self._started:
            return
        if self._closed:
            raise RuntimeError(
                "WakeWordDetector has been closed; create a new instance."
            )
        async with self._start_lock:
            # Re-check under the lock so a concurrent ``start`` race does
            # not double-create the engine. mypy narrows ``self._started``
            # to ``False`` after the early-return above and therefore flags
            # the inner ``return`` body as unreachable; in reality another
            # coroutine could have flipped the flag while this one was
            # awaiting the lock, so suppress the warning narrowly on the
            # body line.
            if self._started:
                return  # type: ignore[unreachable]
            # Native engine creation is blocking I/O (it loads the .ppn
            # model file from disk and initialises the underlying C
            # library). Offload to a thread so the event loop stays
            # responsive even on slow disks.
            porcupine = await asyncio.to_thread(self._create_porcupine)
            # Defensive sanity-check: confirm the engine reports the frame
            # size we are about to feed it. Mismatches indicate an ABI
            # change in pvporcupine that the project must adopt explicitly.
            actual_frame = getattr(porcupine, "frame_length", PORCUPINE_FRAME_SAMPLES)
            actual_rate = getattr(porcupine, "sample_rate", PORCUPINE_SAMPLE_RATE_HZ)
            if actual_frame != PORCUPINE_FRAME_SAMPLES:
                # Tear down the freshly-created engine before raising so we
                # do not leak the native handle.
                with contextlib.suppress(Exception):
                    porcupine.delete()
                raise WakeWordConfigurationError(
                    f"Porcupine reported frame_length={actual_frame!r}, "
                    f"expected {PORCUPINE_FRAME_SAMPLES}. The audio_io "
                    "constants are out of sync with the installed pvporcupine."
                )
            if actual_rate != PORCUPINE_SAMPLE_RATE_HZ:
                with contextlib.suppress(Exception):
                    porcupine.delete()
                raise WakeWordConfigurationError(
                    f"Porcupine reported sample_rate={actual_rate!r}, "
                    f"expected {PORCUPINE_SAMPLE_RATE_HZ}."
                )
            self._porcupine = porcupine
            self._started = True
            logger.debug(
                "WakeWordDetector started with %d keyword(s); frame_length=%d, "
                "sample_rate=%d Hz",
                len(self._keywords),
                actual_frame,
                actual_rate,
            )

    async def aclose(self) -> None:
        """Release the native Porcupine handle and mark the detector closed."""
        if self._closed:
            return
        self._closed = True
        self._started = False
        porcupine = self._porcupine
        self._porcupine = None
        if porcupine is None:
            return
        # ``Porcupine.delete`` is a synchronous native-library call; offload
        # so we do not stall the event loop even briefly while the C library
        # frees its resources.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(porcupine.delete)

    async def run(
        self,
        frames_in: AsyncIterable[bytes],
        on_wake: WakeCallback,
    ) -> None:
        """Consume ``frames_in`` and invoke ``on_wake`` upon detection.

        Args:
            frames_in: Async iterator of 16 kHz / 16-bit / mono PCM frames,
                each exactly :pyattr:`PORCUPINE_FRAME_SAMPLES` samples
                (i.e. ``PORCUPINE_FRAME_SAMPLES * 2`` bytes). Frames smaller
                or larger than this contract raise immediately so an
                upstream reframer misconfiguration cannot silently degrade
                detection. Empty (zero-length) frames are skipped, which
                matches the convention used by :class:`AudioReframer.feed`.
            on_wake: Async callable invoked once per positive detection.
                The callable is awaited inline so back-pressure from the
                consumer (e.g. the audio capture loop pivoting from
                "listening for wake" to "capturing utterance") naturally
                propagates back into the wake-word loop. Exceptions raised
                by the callback are NOT swallowed — they propagate out of
                ``run`` so the application's supervisor can decide what to
                do (typically: log and restart the loop).

        The coroutine returns when ``frames_in`` is exhausted. If
        :meth:`start` has not been called yet it is invoked implicitly so
        callers may use the detector in a fire-and-forget fashion.
        """
        await self.start()
        porcupine = self._porcupine
        assert porcupine is not None  # established by ``start``

        # Capture hot-path locals to skip attribute lookups inside the
        # per-frame loop. Detection latency matters for Requirement 1.2.
        process = porcupine.process
        unpack = _FRAME_UNPACK.unpack_from
        expected_bytes = _FRAME_BYTES

        async for frame in frames_in:
            if not frame:
                # Zero-length frames are valid no-ops upstream; do not
                # punish callers for emitting them on edge transitions.
                continue
            if len(frame) != expected_bytes:
                raise ValueError(
                    "WakeWordDetector.run: each frame must be exactly "
                    f"{expected_bytes} bytes ({PORCUPINE_FRAME_SAMPLES} "
                    f"int16 samples); got {len(frame)} bytes. Run frames "
                    "through AudioReframer.for_porcupine() first."
                )
            # ``unpack_from`` accepts any bytes-like object and returns a
            # tuple of Python ints. Porcupine.process accepts any sequence
            # of ints, so the tuple is consumed without an intermediate
            # list allocation.
            pcm = unpack(frame)
            keyword_index = process(pcm)
            if keyword_index >= 0:
                # Detection: surface to the application. ``await`` ensures
                # downstream backpressure (e.g. the capture loop pausing
                # this iterator while it pivots into utterance capture)
                # propagates correctly.
                await on_wake()

    # -------------------------------------------------------- internal helpers

    def _create_porcupine(self) -> Any:
        """Build the Porcupine engine. Runs in a worker thread.

        Lazy-imports :mod:`pvporcupine` so callers can import this module
        on environments without the native engine installed.
        """
        try:
            # pylint: disable=import-outside-toplevel
            import pvporcupine  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise WakeWordConfigurationError(
                "pvporcupine is not installed. Add it to the runtime "
                "dependencies (already declared in pyproject.toml under "
                "the voice-pipeline group) and re-install the project."
            ) from exc

        # Split the keyword list into the two arguments ``Porcupine.create``
        # expects. Built-in names go through ``keywords=``; user-supplied
        # ``.ppn`` files go through ``keyword_paths=``. Sensitivities must
        # be supplied in keyword-list order; we preserved insertion order
        # in ``__init__`` precisely so this works.
        builtin_names: list[str] = []
        builtin_sensitivities: list[float] = []
        custom_paths: list[str] = []
        custom_sensitivities: list[float] = []
        for (kind, value), sens in zip(
            self._keywords, self._sensitivities, strict=True
        ):
            if kind == "name":
                assert isinstance(value, str)
                builtin_names.append(value)
                builtin_sensitivities.append(sens)
            else:
                assert isinstance(value, Path)
                custom_paths.append(str(value))
                custom_sensitivities.append(sens)

        kwargs: dict[str, Any] = {"access_key": self._access_key}
        if builtin_names:
            kwargs["keywords"] = builtin_names
        if custom_paths:
            kwargs["keyword_paths"] = custom_paths
        # Combine sensitivities in the same partitioning order Porcupine
        # expects: builtin first (paired with ``keywords``), then custom
        # (paired with ``keyword_paths``). When a single category is in
        # use, this collapses to the obvious shape.
        kwargs["sensitivities"] = builtin_sensitivities + custom_sensitivities

        try:
            return pvporcupine.create(**kwargs)
        except Exception as exc:  # pragma: no cover - environment-specific
            # Wrap every Porcupine-side failure so callers only have to
            # handle one error type. ``__cause__`` preserves the original
            # exception (PorcupineActivationError, PorcupineActivationLimitError,
            # PorcupineInvalidArgumentError, …) for diagnostic logging.
            raise WakeWordConfigurationError(
                f"Failed to initialise Porcupine: {exc}"
            ) from exc
