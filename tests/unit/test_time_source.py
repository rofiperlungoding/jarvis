"""Unit tests for ``jarvis.utils.time_source``."""

from __future__ import annotations

from datetime import UTC, datetime
import time

import pytest

from jarvis.utils.time_source import FakeTimeSource, SystemTimeSource, TimeSource

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_system_time_source_satisfies_protocol() -> None:
    src: TimeSource = SystemTimeSource()
    assert isinstance(src, TimeSource)


def test_fake_time_source_satisfies_protocol() -> None:
    src: TimeSource = FakeTimeSource()
    assert isinstance(src, TimeSource)


# ---------------------------------------------------------------------------
# SystemTimeSource
# ---------------------------------------------------------------------------


def test_system_now_is_timezone_aware_utc() -> None:
    src = SystemTimeSource()
    value = src.now()
    assert value.tzinfo is not None
    assert value.utcoffset() == UTC.utcoffset(value)


def test_system_monotonic_is_non_decreasing() -> None:
    src = SystemTimeSource()
    a = src.monotonic()
    # Sleep a tiny amount to give the clock a chance to tick.
    time.sleep(0.001)
    b = src.monotonic()
    assert b >= a


# ---------------------------------------------------------------------------
# FakeTimeSource construction
# ---------------------------------------------------------------------------


def test_fake_default_start_is_aware() -> None:
    src = FakeTimeSource()
    assert src.now().tzinfo is not None
    assert src.monotonic() == 0.0


def test_fake_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        FakeTimeSource(now=datetime(2024, 1, 1, 0, 0, 0))


def test_fake_rejects_negative_monotonic() -> None:
    with pytest.raises(ValueError):
        FakeTimeSource(monotonic=-0.1)


def test_fake_accepts_custom_aware_start() -> None:
    start = datetime(2030, 6, 1, 12, 0, 0, tzinfo=UTC)
    src = FakeTimeSource(now=start, monotonic=42.5)
    assert src.now() == start
    assert src.monotonic() == 42.5


# ---------------------------------------------------------------------------
# FakeTimeSource.advance
# ---------------------------------------------------------------------------


def test_advance_moves_both_clocks_forward() -> None:
    src = FakeTimeSource()
    before_now = src.now()
    before_mono = src.monotonic()

    src.advance(2.5)

    assert src.monotonic() == pytest.approx(before_mono + 2.5)
    assert (src.now() - before_now).total_seconds() == pytest.approx(2.5)


def test_advance_zero_is_noop() -> None:
    src = FakeTimeSource()
    a = src.now()
    src.advance(0)
    assert src.now() == a


def test_advance_rejects_negative() -> None:
    src = FakeTimeSource()
    with pytest.raises(ValueError):
        src.advance(-1.0)


def test_advance_preserves_timezone() -> None:
    src = FakeTimeSource()
    src.advance(60)
    assert src.now().tzinfo is UTC


# ---------------------------------------------------------------------------
# FakeTimeSource setters
# ---------------------------------------------------------------------------


def test_set_now_does_not_disturb_monotonic() -> None:
    src = FakeTimeSource(monotonic=10.0)
    new_now = datetime(2099, 1, 1, tzinfo=UTC)
    src.set_now(new_now)
    assert src.now() == new_now
    assert src.monotonic() == 10.0


def test_set_now_rejects_naive() -> None:
    src = FakeTimeSource()
    with pytest.raises(ValueError):
        src.set_now(datetime(2030, 1, 1))


def test_set_monotonic_forward_only() -> None:
    src = FakeTimeSource(monotonic=5.0)
    src.set_monotonic(7.0)
    assert src.monotonic() == 7.0


def test_set_monotonic_rejects_backwards() -> None:
    src = FakeTimeSource(monotonic=5.0)
    with pytest.raises(ValueError):
        src.set_monotonic(4.0)


# ---------------------------------------------------------------------------
# Integration-style: simulate a reminder firing window (Requirements 6.2/6.4)
# ---------------------------------------------------------------------------


def test_fake_supports_scheduling_simulation() -> None:
    src = FakeTimeSource()
    trigger_at = src.now()
    # Advance time and verify ordering against a stored absolute trigger.
    src.advance(1.0)
    assert src.now() > trigger_at
    # Monotonic measures elapsed wall-time on a stable clock for timeout
    # scheduling (Requirement 17.3).
    start_mono = src.monotonic()
    src.advance(0.5)
    assert src.monotonic() - start_mono == pytest.approx(0.5)
