"""In-memory, per-email throttle for failed sign-in attempts.

Skynet's ``/auth/login`` sits behind the shared ``BACKEND_AUTH_SECRET``, so every
request reaches it from the one trusted frontend server. An IP-based limiter
would see a single source for the whole world and a per-IP counter would be
meaningless; the brute-force surface that actually remains is online guessing of
a known account's password (or its 2FA code). This limiter therefore keys on the
email and locks it out once failures pile up inside a rolling window.

State is per-process, which is exactly right while the backend runs as a single
uvicorn process (see ``backend/main.py``). A future multi-instance deployment
would need to move this state to a shared store (Redis/Postgres) so the counter
is not trivially reset by load-balancing across replicas.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from .errors import DomainError

_DEFAULT_MAX_FAILURES = 5
_DEFAULT_WINDOW_SECONDS = 900.0
# Cap distinct tracked emails so an attacker spraying many addresses cannot grow
# the map without bound; least-recently-touched keys are evicted past this.
_MAX_TRACKED_KEYS = 4096


class LoginThrottle:
    """Locks an email out of sign-in after too many recent failures.

    Thread-safe: FastAPI runs synchronous routes in a threadpool, so several
    requests may touch the same email concurrently. A single lock guards the map;
    contention is negligible because every operation is O(1) amortised.
    """

    def __init__(
        self,
        *,
        max_failures: int = _DEFAULT_MAX_FAILURES,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the failure threshold and rolling window.

        Args:
            max_failures: Number of failures within the window that trip the lock.
            window_seconds: Length of the rolling window, in seconds.
            clock: Monotonic time source in seconds; injectable so tests drive it.
        """
        self._max_failures = max_failures
        self._window = window_seconds
        self._clock = clock
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Reject the attempt when ``key`` is currently locked out.

        Call this before verifying a credential. It never records anything, so a
        locked-out caller cannot extend their own lockout by hammering.

        Args:
            key: The account email being signed in to.

        Raises:
            DomainError: 429 ``accounts.too_many_attempts`` while locked, carrying
                a ``retry_after`` param (whole seconds until the window reopens).
        """
        with self._lock:
            recent = self._prune(key)
            if len(recent) < self._max_failures:
                return
            retry_after = max(int(self._window - (self._clock() - recent[0])) + 1, 1)
            raise DomainError(
                "accounts.too_many_attempts",
                status=429,
                params={"retry_after": retry_after},
            )

    def record_failure(self, key: str) -> None:
        """Record one failed attempt for ``key`` at the current time.

        Args:
            key: The account email the failed attempt targeted.
        """
        with self._lock:
            recent = self._prune(key)
            recent.append(self._clock())
            self._failures[key] = recent
            self._failures.move_to_end(key)
            self._evict()

    def reset(self, key: str) -> None:
        """Clear a key's failure history after a successful sign-in.

        Args:
            key: The account email that just authenticated.
        """
        with self._lock:
            self._failures.pop(key, None)

    def _prune(self, key: str) -> deque[float]:
        """Return ``key``'s in-window failures, dropping any that have aged out.

        Args:
            key: The account email to read.

        Returns:
            The deque of failure timestamps still inside the rolling window —
            empty (and detached from the map) when the key is unknown or has
            fully aged out.
        """
        recent = self._failures.get(key)
        if recent is None:
            return deque()
        cutoff = self._clock() - self._window
        while recent and recent[0] <= cutoff:
            recent.popleft()
        if not recent:
            self._failures.pop(key, None)
        return recent

    def _evict(self) -> None:
        """Drop least-recently-touched keys once the map exceeds its cap."""
        while len(self._failures) > _MAX_TRACKED_KEYS:
            self._failures.popitem(last=False)
