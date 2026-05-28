"""Unit tests for ``jarvis.voice.audio_io.AudioReframer``.

The reframer is the pure-Python adapter that stitches arbitrary PortAudio
callback chunk sizes into the fixed-size frames demanded by Porcupine
(512-sample / 16 kHz / 16-bit mono) and silero-vad (480-sample). Two
correctness invariants govern its behaviour:

* **Byte preservation** — concatenating every frame yielded by ``feed`` plus
  any trailing partial flush reproduces the input byte-for-byte. The reframer
  is a *partitioner*, never a transformer.
* **Frame size invariant** — every frame returned by ``feed`` is exactly
  ``frame_bytes`` long; ``flush(pad=False)`` returns ``None`` for partial
  tails so this invariant is never violated by an exposed frame.

These tests cover the documented happy paths (single-byte feeds, large-chunk
feeds, exact-multiple feeds, empty feeds), the ``flush`` / ``reset`` /
``for_*`` classmethod surface, and a Hypothesis-driven property suite that
verifies both invariants across arbitrary chunk sequences and frame sizes.

Validates: Requirements 1.2
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from jarvis.voice.audio_io import (
    PORCUPINE_FRAME_SAMPLES,
    VAD_FRAME_SAMPLES,
    AudioFormat,
    AudioReframer,
)

# ---------------------------------------------------------------------------
# Construction and convenience constructors
# ---------------------------------------------------------------------------


def test_default_state_has_empty_buffer() -> None:
    r = AudioReframer(frame_bytes=8)
    assert r.frame_bytes == 8
    assert r.buffered_bytes == 0


@pytest.mark.parametrize("bad", [0, -1, -1024])
def test_constructor_rejects_non_positive_frame_bytes(bad: int) -> None:
    with pytest.raises(ValueError):
        AudioReframer(frame_bytes=bad)


def test_for_porcupine_uses_512_sample_16bit_mono() -> None:
    r = AudioReframer.for_porcupine()
    assert r.frame_bytes == PORCUPINE_FRAME_SAMPLES * 2  # int16 mono


def test_for_vad_uses_30ms_16bit_mono() -> None:
    r = AudioReframer.for_vad()
    assert r.frame_bytes == VAD_FRAME_SAMPLES * 2


def test_for_format_matches_format_frame_bytes() -> None:
    fmt = AudioFormat(sample_rate_hz=16000, frame_samples=480)
    r = AudioReframer.for_format(fmt)
    assert r.frame_bytes == fmt.frame_bytes


# ---------------------------------------------------------------------------
# feed: empty / partial / exact / large
# ---------------------------------------------------------------------------


def test_feed_empty_chunk_returns_empty_list_and_no_buffering() -> None:
    r = AudioReframer(frame_bytes=4)
    assert r.feed(b"") == []
    assert r.buffered_bytes == 0


def test_feed_partial_buffers_without_emitting() -> None:
    r = AudioReframer(frame_bytes=4)
    assert r.feed(b"\x01\x02") == []
    assert r.buffered_bytes == 2
    # Still nothing on a second sub-frame chunk.
    assert r.feed(b"\x03") == []
    assert r.buffered_bytes == 3


def test_feed_exact_single_frame_returns_one_frame_and_clears_buffer() -> None:
    r = AudioReframer(frame_bytes=4)
    out = r.feed(b"\x01\x02\x03\x04")
    assert out == [b"\x01\x02\x03\x04"]
    assert r.buffered_bytes == 0


def test_feed_exact_multiple_returns_all_frames_in_order() -> None:
    r = AudioReframer(frame_bytes=2)
    out = r.feed(b"\x01\x02\x03\x04\x05\x06")
    assert out == [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
    assert r.buffered_bytes == 0


def test_feed_large_chunk_emits_frames_and_keeps_remainder() -> None:
    r = AudioReframer(frame_bytes=3)
    out = r.feed(b"abcdefghij")  # 10 bytes -> 3 frames + 1 byte tail
    assert out == [b"abc", b"def", b"ghi"]
    assert r.buffered_bytes == 1
    # The retained tail byte combines with the next chunk to form a frame.
    out2 = r.feed(b"kl")
    assert out2 == [b"jkl"]
    assert r.buffered_bytes == 0


def test_feed_completes_partial_frame_across_calls() -> None:
    r = AudioReframer(frame_bytes=4)
    assert r.feed(b"\x10\x11") == []
    assert r.feed(b"\x12") == []
    out = r.feed(b"\x13\x14\x15\x16\x17\x18")
    # 2 + 1 + 6 = 9 bytes total -> 2 full frames + 1-byte tail.
    # Frame 1 = carry-over (\x10\x11\x12) + first new byte (\x13).
    # Frame 2 = next 4 new bytes (\x14..\x17). Tail = \x18.
    assert out == [b"\x10\x11\x12\x13", b"\x14\x15\x16\x17"]
    assert r.buffered_bytes == 1
    # Feeding 3 more bytes completes the next frame and leaves nothing.
    assert r.feed(b"\x19\x1a\x1b") == [b"\x18\x19\x1a\x1b"]
    assert r.buffered_bytes == 0


# ---------------------------------------------------------------------------
# feed: input type handling
# ---------------------------------------------------------------------------


def test_feed_accepts_bytearray_and_memoryview() -> None:
    r = AudioReframer(frame_bytes=2)
    assert r.feed(bytearray(b"\x01\x02")) == [b"\x01\x02"]
    assert r.feed(memoryview(b"\x03\x04")) == [b"\x03\x04"]


@pytest.mark.parametrize("bad", [123, "abc", 1.0, None, [b"\x00"]])
def test_feed_rejects_non_bytes_like(bad: object) -> None:
    r = AudioReframer(frame_bytes=2)
    with pytest.raises(TypeError):
        r.feed(bad)  # type: ignore[arg-type]


def test_feed_returns_independent_bytes_objects() -> None:
    """Mutating the input bytearray after ``feed`` must not affect emitted frames."""
    r = AudioReframer(frame_bytes=2)
    src = bytearray(b"\x01\x02\x03\x04")
    out = r.feed(src)
    assert out == [b"\x01\x02", b"\x03\x04"]
    src[0] = 0xFF
    # Frames should be untouched by the post-feed mutation.
    assert out[0] == b"\x01\x02"


# ---------------------------------------------------------------------------
# Single-byte streaming
# ---------------------------------------------------------------------------


def test_single_byte_feeds_emit_frame_only_at_boundary() -> None:
    r = AudioReframer(frame_bytes=4)
    payload = b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8"
    emitted: list[bytes] = []
    for byte in payload:
        emitted.extend(r.feed(bytes([byte])))
    assert emitted == [b"\xa0\xa1\xa2\xa3", b"\xa4\xa5\xa6\xa7"]
    assert r.buffered_bytes == 1
    assert r.flush(pad=False) is None
    assert r.buffered_bytes == 0


# ---------------------------------------------------------------------------
# flush behaviour
# ---------------------------------------------------------------------------


def test_flush_on_empty_buffer_returns_none_regardless_of_pad() -> None:
    r = AudioReframer(frame_bytes=4)
    assert r.flush(pad=False) is None
    assert r.flush(pad=True) is None


def test_flush_partial_pad_false_returns_none_and_drops_buffer() -> None:
    r = AudioReframer(frame_bytes=4)
    r.feed(b"\x01\x02\x03")  # 3-byte tail
    assert r.buffered_bytes == 3
    assert r.flush(pad=False) is None
    assert r.buffered_bytes == 0


def test_flush_partial_pad_true_returns_padded_full_frame() -> None:
    r = AudioReframer(frame_bytes=5)
    r.feed(b"AB")
    out = r.flush(pad=True)
    assert out == b"AB\x00\x00\x00"
    assert r.buffered_bytes == 0


def test_flush_partial_pad_true_with_custom_pad_value() -> None:
    r = AudioReframer(frame_bytes=4)
    r.feed(b"\xfe")
    out = r.flush(pad=True, pad_value=0xAB)
    assert out == b"\xfe\xab\xab\xab"


@pytest.mark.parametrize("bad", [-1, 256, 1024])
def test_flush_rejects_pad_value_out_of_range(bad: int) -> None:
    r = AudioReframer(frame_bytes=4)
    r.feed(b"\x01")
    with pytest.raises(ValueError):
        r.flush(pad=True, pad_value=bad)
    # On error the buffer is left untouched so the caller can retry.
    assert r.buffered_bytes == 1


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_drops_buffered_bytes_silently() -> None:
    r = AudioReframer(frame_bytes=4)
    r.feed(b"\x01\x02\x03")
    assert r.buffered_bytes == 3
    r.reset()
    assert r.buffered_bytes == 0
    # After reset, the next feed must not see any of the discarded bytes.
    assert r.feed(b"\xff\xff\xff\xff") == [b"\xff\xff\xff\xff"]


def test_reset_on_empty_buffer_is_idempotent() -> None:
    r = AudioReframer(frame_bytes=4)
    r.reset()
    r.reset()
    assert r.buffered_bytes == 0


# ---------------------------------------------------------------------------
# Byte-preservation across mixed chunk sizes
# ---------------------------------------------------------------------------


def _drive(reframer: AudioReframer, chunks: list[bytes]) -> tuple[list[bytes], bytes]:
    frames: list[bytes] = []
    for c in chunks:
        frames.extend(reframer.feed(c))
    leftover_len = reframer.buffered_bytes
    if leftover_len == 0:
        return frames, b""
    # Capture the still-buffered tail by padding to a full frame and trimming.
    padded = reframer.flush(pad=True)
    assert padded is not None
    return frames, padded[:leftover_len]


def test_byte_preservation_under_mixed_chunk_sizes() -> None:
    r = AudioReframer(frame_bytes=4)
    chunks = [b"\x00\x01", b"\x02", b"\x03\x04\x05\x06\x07", b"", b"\x08\x09"]
    frames, tail = _drive(r, chunks)
    assert b"".join(frames) + tail == b"".join(chunks)
    for f in frames:
        assert len(f) == 4


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


# Bound the explored frame size to the realistic range used by the voice
# pipeline (Porcupine = 1024 bytes, VAD = 960 bytes) plus some headroom for
# alternative configurations. Bound chunk sizes to PortAudio-realistic values.
_MAX_FRAME_BYTES = 4096
_MAX_CHUNK_LEN = 2048
_MAX_CHUNK_COUNT = 24


_chunks_strategy = st.lists(
    st.binary(min_size=0, max_size=_MAX_CHUNK_LEN),
    min_size=0,
    max_size=_MAX_CHUNK_COUNT,
)


@given(
    frame_bytes=st.integers(min_value=1, max_value=_MAX_FRAME_BYTES),
    chunks=_chunks_strategy,
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_byte_preservation_and_frame_size_invariant(
    frame_bytes: int, chunks: list[bytes]
) -> None:
    """Concatenating all emitted frames + leftover reproduces input exactly.

    Every frame returned by ``feed`` is exactly ``frame_bytes`` bytes long.
    """
    r = AudioReframer(frame_bytes=frame_bytes)
    emitted: list[bytes] = []
    for c in chunks:
        out = r.feed(c)
        for frame in out:
            assert len(frame) == frame_bytes
        emitted.extend(out)

    # All complete frames combined with the still-buffered tail must equal the
    # full input, byte for byte. This catches any insertion, drop, or reorder.
    expected = b"".join(chunks)
    leftover_len = r.buffered_bytes
    assert leftover_len < frame_bytes  # tail is always shorter than a frame
    assert len(emitted) * frame_bytes + leftover_len == len(expected)

    # Recover the leftover bytes via padded flush so we can compare end-to-end.
    if leftover_len == 0:
        tail = b""
        assert r.flush(pad=False) is None
    else:
        padded = r.flush(pad=True)
        assert padded is not None
        assert len(padded) == frame_bytes
        tail = padded[:leftover_len]

    assert b"".join(emitted) + tail == expected
    # Buffer is always empty after flush.
    assert r.buffered_bytes == 0


@given(
    frame_bytes=st.integers(min_value=1, max_value=_MAX_FRAME_BYTES),
    chunks=_chunks_strategy,
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_flush_pad_false_returns_none_iff_partial_tail(
    frame_bytes: int, chunks: list[bytes]
) -> None:
    """``flush(pad=False)`` returns ``None`` exactly when the tail is short."""
    r = AudioReframer(frame_bytes=frame_bytes)
    for c in chunks:
        r.feed(c)
    leftover_len = r.buffered_bytes
    # Invariant: leftover is always strictly less than a full frame because
    # feed eagerly emits whenever it has enough bytes.
    assert leftover_len < frame_bytes
    out = r.flush(pad=False)
    if leftover_len == 0:
        assert out is None
    else:
        # Partial tail is dropped without padding.
        assert out is None
    assert r.buffered_bytes == 0


@given(
    frame_bytes=st.integers(min_value=1, max_value=_MAX_FRAME_BYTES),
    chunks=_chunks_strategy,
    pad_value=st.integers(min_value=0, max_value=255),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_flush_pad_true_yields_full_frame_with_correct_pad(
    frame_bytes: int, chunks: list[bytes], pad_value: int
) -> None:
    """``flush(pad=True)`` always returns a full-length frame when bytes remain."""
    r = AudioReframer(frame_bytes=frame_bytes)
    for c in chunks:
        r.feed(c)
    leftover_len = r.buffered_bytes
    expected_tail = b"".join(chunks)[len(b"".join(chunks)) - leftover_len :]
    out = r.flush(pad=True, pad_value=pad_value)
    if leftover_len == 0:
        assert out is None
    else:
        assert out is not None
        assert len(out) == frame_bytes
        assert out[:leftover_len] == expected_tail
        assert out[leftover_len:] == bytes([pad_value]) * (frame_bytes - leftover_len)
    assert r.buffered_bytes == 0


@given(
    frame_bytes=st.integers(min_value=1, max_value=_MAX_FRAME_BYTES),
    chunks=_chunks_strategy,
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_reset_clears_buffer(
    frame_bytes: int, chunks: list[bytes]
) -> None:
    r = AudioReframer(frame_bytes=frame_bytes)
    for c in chunks:
        r.feed(c)
    r.reset()
    assert r.buffered_bytes == 0
    # After reset, feeding fresh data behaves identically to a brand-new instance.
    follow = b"\xaa" * frame_bytes
    assert r.feed(follow) == [follow]
