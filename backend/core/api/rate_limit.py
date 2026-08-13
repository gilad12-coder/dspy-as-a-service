"""Cross-replica request-rate limiting and login lockout, backed by Redis.

Skynet's backend never sees a real client IP: every form request arrives from
the one trusted frontend server, and the public dev API authenticates by
``skyd_`` key rather than address (see :mod:`core.api.login_throttle` for the
same reasoning applied to sign-in). A per-IP limiter would therefore see a
single source for the whole world. The meaningful key is the *account* —
username for the cost-driving ``/run`` and ``/grid-search`` submissions, email
for the account-action endpoints — which is what the helpers here throttle.

State lives in Redis so the counters are shared across replicas: the in-memory
:class:`core.api.login_throttle.LoginThrottle` is trivially reset by
load-balancing across processes, whereas a Redis counter is not. Every limiter
**fails open**: when ``REDIS_URL`` is unset or Redis is unreachable the request
is allowed rather than rejected, so a limiter outage degrades to today's
unlimited behaviour instead of a hard outage. That is the right trade for a
cost/abuse guardrail — a brief window of unthrottled traffic is cheaper than
refusing every legitimate user because the limiter's backing store blipped.

The window primitive is a fixed-window counter (``INCR`` + ``EXPIRE NX``), which
needs Redis 7.0+ for the ``NX`` flag — the version Railway and every current
managed provider ship.
"""

from __future__ import annotations

import logging

from redis import Redis
from redis.exceptions import RedisError

from ..config import settings
from .errors import DomainError
from .login_throttle import _DEFAULT_MAX_FAILURES, _DEFAULT_WINDOW_SECONDS

logger = logging.getLogger(__name__)

# Namespaces the limiter's keys so they never collide with other Redis users
# sharing the instance (e.g. LiteLLM) and are trivially greppable/flushable.
_KEY_PREFIX = "skynet:rl:"

_shared_client_cache: Redis | None = None
_shared_client_resolved = False


def build_redis_client(url: str | None) -> Redis | None:
    """Build a Redis client from a connection URL, or ``None`` when unset.

    Short socket timeouts keep a stalled or unreachable Redis from blocking a
    request: the limiter would rather fail open in a quarter-second than hang.
    Construction does not connect — redis-py dials lazily on the first command —
    so this is safe to call at import time even with Redis down.

    Args:
        url: Redis connection URL (``redis://…``), or a falsy value to disable.

    Returns:
        A configured client, or ``None`` when ``url`` is falsy.
    """
    if not url:
        return None
    return Redis.from_url(
        url,
        socket_connect_timeout=0.25,
        socket_timeout=0.25,
        decode_responses=True,
    )


def shared_redis_client() -> Redis | None:
    """Return the process-wide limiter client, built once from settings.

    Cached across calls so every limiter shares one connection pool. Returns
    ``None`` when ``REDIS_URL`` is unset, which is how the helpers below select
    their fail-open / in-memory behaviour.

    Returns:
        The shared client, or ``None`` when no ``REDIS_URL`` is configured.
    """
    global _shared_client_cache, _shared_client_resolved
    if not _shared_client_resolved:
        _shared_client_cache = build_redis_client(settings.redis_url)
        _shared_client_resolved = True
    return _shared_client_cache


def _hit_fixed_window(client: Redis, key: str, window_seconds: int) -> tuple[int, int]:
    """Count one hit in ``key``'s fixed window and report the count and TTL.

    ``INCR`` creates the key at 1; ``EXPIRE NX`` stamps the window only when no
    expiry exists yet, so the window is anchored to the first hit and does not
    slide under sustained traffic (which would stop it ever resetting).

    Args:
        client: Redis client to run the pipeline against.
        key: Fully-namespaced counter key.
        window_seconds: Length of the fixed window, in whole seconds.

    Returns:
        A ``(count, ttl)`` pair: the post-increment hit count and the key's
        remaining time-to-live in seconds.
    """
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_seconds, nx=True)
    pipe.ttl(key)
    count, _, ttl = pipe.execute()
    return int(count), int(ttl)


def _retry_after(ttl: int, window_seconds: int) -> int:
    """Pick a client-facing ``retry_after`` from a key's TTL, with a floor.

    A non-positive TTL (``-1`` no-expiry, ``-2`` key gone) means the window
    boundary is unknown, so fall back to the full window length.

    Args:
        ttl: Remaining time-to-live reported by Redis, in seconds.
        window_seconds: The configured window length, used as the fallback.

    Returns:
        Whole seconds the caller should wait before retrying (at least 1).
    """
    return max(ttl if ttl > 0 else window_seconds, 1)


class RateLimiter:
    """Fixed-window request-rate limiter over a Redis client (fail-open).

    Wraps a single Redis client; ``None`` (no ``REDIS_URL``) makes every check a
    no-op so single-instance and test deployments keep today's behaviour.
    """

    def __init__(self, client: Redis | None) -> None:
        """Bind the limiter to a Redis client.

        Args:
            client: Redis client, or ``None`` to disable (always allow).
        """
        self._client = client

    def enforce(self, key: str, *, limit: int, window_seconds: int) -> None:
        """Admit one request against ``key``'s window, or reject it as 429.

        No-op when the limiter is disabled (``limit <= 0`` or no client) or when
        Redis errors — the fail-open contract. Only a definite over-limit count
        from a healthy Redis raises.

        Args:
            key: Caller-scoped sub-key (e.g. ``"submit:alice"``); namespaced
                internally.
            limit: Maximum hits allowed within the window; ``<= 0`` disables.
            window_seconds: Length of the fixed window, in whole seconds.

        Raises:
            DomainError: 429 ``rate_limit.exceeded`` with a ``retry_after`` param
                when the windowed count exceeds ``limit``.
        """
        if limit <= 0 or self._client is None:
            return
        try:
            count, ttl = _hit_fixed_window(self._client, _KEY_PREFIX + key, window_seconds)
        except RedisError:
            logger.warning("rate limiter unavailable; failing open for %s", key, exc_info=True)
            return
        if count > limit:
            raise DomainError(
                "rate_limit.exceeded",
                status=429,
                params={"retry_after": _retry_after(ttl, window_seconds)},
            )


class RedisLoginThrottle:
    """Redis-backed drop-in for :class:`core.api.login_throttle.LoginThrottle`.

    Same ``check`` / ``record_failure`` / ``reset`` contract and the same
    ``accounts.too_many_attempts`` rejection, but the failure counter lives in
    Redis so a lockout holds across replicas rather than resetting whenever the
    load balancer picks a different process. Fails open (no lockout) on any Redis
    error, matching :class:`RateLimiter`.

    Uses the same fixed-window counter as the request limiter: ``check`` peeks
    without incrementing (so a locked-out caller cannot extend their own lockout
    by hammering) and ``record_failure`` increments. The trade versus the
    in-memory rolling window is that a boundary crossing can allow up to twice
    the threshold across two adjacent windows — still ample brute-force cover.
    """

    def __init__(
        self,
        client: Redis,
        *,
        max_failures: int = _DEFAULT_MAX_FAILURES,
        window_seconds: int = int(_DEFAULT_WINDOW_SECONDS),
    ) -> None:
        """Configure the failure threshold and lockout window.

        Args:
            client: Redis client backing the shared counter.
            max_failures: Failures within the window that trip the lock.
            window_seconds: Length of the lockout window, in whole seconds.
        """
        self._client = client
        self._max_failures = max_failures
        self._window = window_seconds

    def check(self, key: str) -> None:
        """Reject the sign-in attempt when ``key`` is currently locked out.

        Peeks the counter without incrementing, so calling this never extends a
        lockout. Fails open (returns) on any Redis error.

        Args:
            key: The account email being signed in to.

        Raises:
            DomainError: 429 ``accounts.too_many_attempts`` while locked, carrying
                a ``retry_after`` param (whole seconds until the window reopens).
        """
        redis_key = _KEY_PREFIX + "login:" + key
        try:
            pipe = self._client.pipeline()
            pipe.get(redis_key)
            pipe.ttl(redis_key)
            value, ttl = pipe.execute()
        except RedisError:
            logger.warning("login throttle unavailable; failing open for %s", key, exc_info=True)
            return
        if int(value or 0) < self._max_failures:
            return
        raise DomainError(
            "accounts.too_many_attempts",
            status=429,
            params={"retry_after": _retry_after(int(ttl), self._window)},
        )

    def record_failure(self, key: str) -> None:
        """Record one failed attempt for ``key`` at the current time.

        Fails open (no-op) on any Redis error.

        Args:
            key: The account email the failed attempt targeted.
        """
        try:
            _hit_fixed_window(self._client, _KEY_PREFIX + "login:" + key, self._window)
        except RedisError:
            logger.warning("login throttle unavailable; failure not recorded for %s", key, exc_info=True)

    def reset(self, key: str) -> None:
        """Clear a key's failure count after a successful sign-in.

        Fails open (no-op) on any Redis error.

        Args:
            key: The account email that just authenticated.
        """
        try:
            self._client.delete(_KEY_PREFIX + "login:" + key)
        except RedisError:
            logger.warning("login throttle unavailable; reset skipped for %s", key, exc_info=True)


def enforce_submission_rate(username: str) -> None:
    """Throttle a user's run/grid-search submissions to the configured rate.

    A single shared key for both submission endpoints, since it is total run
    volume — not any one route — that drives cost. No-op when the cap is unset
    or Redis is unavailable.

    Args:
        username: Account submitting the run.

    Raises:
        DomainError: 429 ``rate_limit.exceeded`` when over the per-minute cap.
    """
    RateLimiter(shared_redis_client()).enforce(
        f"submit:{username}",
        limit=settings.rate_limit_submissions_per_minute,
        window_seconds=60,
    )


def enforce_account_rate(email: str, action: str) -> None:
    """Throttle per-email account-action requests to the configured rate.

    Args:
        email: Target account email.
        action: Short action tag (e.g. ``"register"``) keeping the register,
            password-reset, and email-verify counters independent.

    Raises:
        DomainError: 429 ``rate_limit.exceeded`` when over the per-hour cap.
    """
    RateLimiter(shared_redis_client()).enforce(
        f"acct:{action}:{email}",
        limit=settings.rate_limit_account_requests_per_hour,
        window_seconds=3600,
    )


def build_login_throttle() -> RedisLoginThrottle | None:
    """Return a Redis-backed login throttle, or ``None`` to fall back in-memory.

    Returns:
        A :class:`RedisLoginThrottle` when ``REDIS_URL`` is configured, else
        ``None`` so the accounts router keeps its per-process
        :class:`core.api.login_throttle.LoginThrottle`.
    """
    client = shared_redis_client()
    return RedisLoginThrottle(client) if client is not None else None
