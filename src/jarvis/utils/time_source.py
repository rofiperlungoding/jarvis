"""Injectable time source abstractions.

A single point of indirection for *both* wall-clock time (`now`) and the
monotonic clock (`monotonic`). Production code receives a `TimeSource` and
calls these methods rather than reaching for :mod:`datetime` or :mod:`time`
directly. Tests substitute :class:`FakeTimeSource` to drive deterministic
behaviour in components such as the reminder service (Requirements 6.2,
6.4) and the voice-pipeline timeout/error paths (Requirement 17.3).

The `now()` contract returns a *timezone-aware* :class:`datetime`. Naive
datetimes are explicitly rejected when feeding the fake source so the rest
of the codebase can rely on aware semantics for serialization and
comparison.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Protocol, runtime_checkable

__all__ = ["FakeTimeSource", "SystemTimeSource", "TimeSource"]


@runtime_checkable
class TimeSource(Protocol):
    """Abstract clock used throughout the application.

    Implementations MUST return timezone-aware datetimes from :meth:`now`
    and a monotonically non-decreasing float (in seconds) from
    :meth:`monotonic`. The two clocks are independent: ``now`` may jump
    when the wall clock is corrected, while ``monotonic`` never jumps
    backwards. Code that schedules timeouts (Requirement 17.3) or measures
    elapsed durations should prefer :meth:`monotonic`; code that persists
    an absolute trigger time (Requirements 6.2, 6.4) should use
    :meth:`now`.
    """

    def now(self) -> datetime:
        """Return the current wall-clock time as an aware datetime."""
        ...

    def monotonic(self) -> float:
        """Return a monotonic clock reading in seconds."""
        ...


class SystemTimeSource:
    """Default :class:`TimeSource` backed by the operating system clocks.

    ``now`` returns :func:`datetime.datetime.now` in UTC so persisted
    timestamps are stable across process restarts and time-zone changes.
    Callers that need a local representation can convert via
    :meth:`datetime.astimezone` at the rendering layer.
    """

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class FakeTimeSource:
    """Manually-advanced :class:`TimeSource` for deterministic tests.

    Both clocks start from caller-supplied values and only move forward
    when :meth:`advance` (or the explicit setters) is called. Advancing
    progresses *both* clocks by the same delta by default, which mirrors
    real-world behaviour and keeps the two readings in sync for the common
    case. The wall-clock can also be overridden independently via
    :meth:`set_now` to exercise scenarios where the system clock is
    adjusted (e.g., NTP correction) without disturbing monotonic
    measurements.
    """

    __slots__ = ("_monotonic", "_now")

    _DEFAULT_NOW = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    def __init__(
        self,
        *,
        now: datetime | None = None,
        monotonic: float = 0.0,
    ) -> None:
        start = now if now is not None else self._DEFAULT_NOW
        self._validate_aware(start)
        if monotonic < 0:
            raise ValueError("monotonic start value must be non-negative")
        self._now: datetime = start
        self._monotonic: float = float(monotonic)

    # -- TimeSource protocol --------------------------------------------------

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    # -- Test helpers ---------------------------------------------------------

    def advance(self, seconds: float) -> None:
        """Move both clocks forward by ``seconds``.

        ``seconds`` must be non-negative; advancing the monotonic clock
        backwards would violate the protocol contract and is rejected.
        """
        if seconds < 0:
            raise ValueError("cannot advance time backwards")
        delta = float(seconds)
        # Update via timestamp arithmetic to avoid timedelta(microseconds=...)
        # rounding for sub-microsecond resolution (datetime is microsecond
        # precision, so we still floor at that granularity, but this keeps
        # the behaviour predictable for callers using small floats).
        self._now = self._now.fromtimestamp(
            self._now.timestamp() + delta, tz=self._now.tzinfo
        )
        self._monotonic += delta

    def set_now(self, value: datetime) -> None:
        """Override the wall clock without touching the monotonic clock."""
        self._validate_aware(value)
        self._now = value

    def set_monotonic(self, value: float) -> None:
        """Override the monotonic clock; refuses to move backwards."""
        if value < self._monotonic:
            raise ValueError("monotonic clock must be non-decreasing")
        self._monotonic = float(value)

    # -- Internal -------------------------------------------------------------

    @staticmethod
    def _validate_aware(value: datetime) -> None:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("FakeTimeSource requires a timezone-aware datetime")
