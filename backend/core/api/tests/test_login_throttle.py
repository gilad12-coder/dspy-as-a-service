"""Tests for the per-email failed-login throttle.

The clock is injected so the rolling window can be exercised deterministically
without sleeping.
"""

from __future__ import annotations

import pytest

from ..errors import DomainError
from ..login_throttle import LoginThrottle


class _Clock:
    """A hand-cranked monotonic clock for the throttle under test."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds: How far to advance, in seconds.
        """
        self.now += seconds


def test_allows_attempts_below_threshold() -> None:
    """Fewer failures than the threshold never lock the key out."""
    throttle = LoginThrottle(max_failures=3, window_seconds=100, clock=_Clock())
    throttle.record_failure("a@x.com")
    throttle.record_failure("a@x.com")
    throttle.check("a@x.com")  # 2 < 3, so no raise


def test_locks_out_at_threshold() -> None:
    """Reaching the failure threshold locks the key with a 429 and retry hint."""
    clock = _Clock()
    throttle = LoginThrottle(max_failures=3, window_seconds=100, clock=clock)
    for _ in range(3):
        throttle.record_failure("a@x.com")

    clock.advance(10)
    with pytest.raises(DomainError) as exc:
        throttle.check("a@x.com")
    assert exc.value.status_code == 429
    assert exc.value.code == "accounts.too_many_attempts"
    assert exc.value.params["retry_after"] == 91  # 100 - 10 + 1


def test_check_does_not_extend_the_lockout() -> None:
    """Blocked checks record nothing, so the window still expires on schedule."""
    clock = _Clock()
    throttle = LoginThrottle(max_failures=2, window_seconds=100, clock=clock)
    throttle.record_failure("a@x.com")
    throttle.record_failure("a@x.com")
    for _ in range(5):
        clock.advance(10)
        with pytest.raises(DomainError):
            throttle.check("a@x.com")

    clock.advance(51)  # now 101s past the oldest failure
    throttle.check("a@x.com")  # window elapsed, so no raise


def test_window_slides_off_old_failures() -> None:
    """Failures aging past the window free up room under the threshold."""
    clock = _Clock()
    throttle = LoginThrottle(max_failures=3, window_seconds=100, clock=clock)
    throttle.record_failure("a@x.com")  # t=0
    clock.advance(50)
    throttle.record_failure("a@x.com")  # t=50
    clock.advance(40)
    throttle.record_failure("a@x.com")  # t=90 -> 3 in window, locked

    with pytest.raises(DomainError):
        throttle.check("a@x.com")

    clock.advance(11)  # t=101, the t=0 failure ages out -> 2 in window
    throttle.check("a@x.com")  # no raise


def test_reset_clears_failures() -> None:
    """A reset (successful sign-in) wipes the failure history."""
    throttle = LoginThrottle(max_failures=2, window_seconds=100, clock=_Clock())
    throttle.record_failure("a@x.com")
    throttle.record_failure("a@x.com")
    throttle.reset("a@x.com")
    throttle.check("a@x.com")  # counter cleared, so no raise


def test_keys_are_independent() -> None:
    """One locked-out email does not affect a different email."""
    throttle = LoginThrottle(max_failures=2, window_seconds=100, clock=_Clock())
    throttle.record_failure("a@x.com")
    throttle.record_failure("a@x.com")
    with pytest.raises(DomainError):
        throttle.check("a@x.com")
    throttle.check("b@x.com")  # untouched key is free
