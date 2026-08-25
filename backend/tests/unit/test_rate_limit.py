"""Unit tests for the Redis-backed rate limiter and login lockout.

Exercises the fixed-window request limiter (:class:`RateLimiter`), the
cross-replica login throttle (:class:`RedisLoginThrottle`), and the module
helpers that wire them to settings — all against an in-memory ``fakeredis``
double so no real Redis is required. The fail-open contract (allow on a missing
or erroring client) is covered explicitly, since it is the behaviour a
production outage depends on.
"""

from __future__ import annotations

import fakeredis
import pytest
from redis import Redis
from redis.exceptions import RedisError

from core.api import rate_limit
from core.api.errors import DomainError
from core.api.rate_limit import (
    RateLimiter,
    RedisLoginThrottle,
    build_login_throttle,
    build_redis_client,
    enforce_account_rate,
    enforce_submission_rate,
)
from core.config import settings


@pytest.fixture
def fake_redis() -> fakeredis.FakeStrictRedis:
    """Return a fresh in-memory Redis double with string decoding on."""
    return fakeredis.FakeStrictRedis(decode_responses=True)


class _RaisingPipe:
    """Pipeline stub whose buffered commands are no-ops and ``execute`` fails."""

    def incr(self, *args: object, **kwargs: object) -> _RaisingPipe:
        """Ignore a buffered INCR and allow chaining."""
        return self

    def expire(self, *args: object, **kwargs: object) -> _RaisingPipe:
        """Ignore a buffered EXPIRE and allow chaining."""
        return self

    def ttl(self, *args: object, **kwargs: object) -> _RaisingPipe:
        """Ignore a buffered TTL and allow chaining."""
        return self

    def get(self, *args: object, **kwargs: object) -> _RaisingPipe:
        """Ignore a buffered GET and allow chaining."""
        return self

    def execute(self) -> list[object]:
        """Raise as though Redis were unreachable when the pipeline runs."""
        raise RedisError("boom")


class _RaisingRedis:
    """Client stub that raises on every operation, to drive the fail-open path."""

    def pipeline(self) -> _RaisingPipe:
        """Return a pipeline whose execution raises."""
        return _RaisingPipe()

    def delete(self, *args: object, **kwargs: object) -> None:
        """Raise as though the DEL could not reach Redis."""
        raise RedisError("boom")


def test_enforce_allows_up_to_limit_then_rejects(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """The first ``limit`` hits pass; the next raises a 429 with a retry hint."""
    limiter = RateLimiter(fake_redis)

    for _ in range(3):
        limiter.enforce("submit:alice", limit=3, window_seconds=60)

    with pytest.raises(DomainError) as exc:
        limiter.enforce("submit:alice", limit=3, window_seconds=60)

    assert exc.value.code == "rate_limit.exceeded"
    assert exc.value.status_code == 429
    assert exc.value.params["retry_after"] >= 1


def test_enforce_keys_are_independent(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """A limit reached for one key does not spill over onto another."""
    limiter = RateLimiter(fake_redis)

    for _ in range(2):
        limiter.enforce("submit:alice", limit=2, window_seconds=60)
    with pytest.raises(DomainError):
        limiter.enforce("submit:alice", limit=2, window_seconds=60)

    limiter.enforce("submit:bob", limit=2, window_seconds=60)


def test_enforce_disabled_when_limit_zero(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """A non-positive limit disables the check regardless of hit volume."""
    limiter = RateLimiter(fake_redis)

    for _ in range(100):
        limiter.enforce("submit:alice", limit=0, window_seconds=60)


def test_enforce_no_client_is_noop() -> None:
    """With no Redis client every request is admitted (fail-open when unset)."""
    limiter = RateLimiter(None)

    for _ in range(100):
        limiter.enforce("submit:alice", limit=1, window_seconds=60)


def test_enforce_fails_open_on_redis_error() -> None:
    """A Redis error is swallowed and the request admitted, never a 500."""
    limiter = RateLimiter(_RaisingRedis())

    limiter.enforce("submit:alice", limit=1, window_seconds=60)


def test_login_throttle_trips_after_threshold(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """After ``max_failures`` recorded failures ``check`` locks the email out."""
    throttle = RedisLoginThrottle(fake_redis, max_failures=3, window_seconds=900)

    throttle.check("user@example.com")
    for _ in range(3):
        throttle.record_failure("user@example.com")

    with pytest.raises(DomainError) as exc:
        throttle.check("user@example.com")

    assert exc.value.code == "accounts.too_many_attempts"
    assert exc.value.status_code == 429
    assert exc.value.params["retry_after"] >= 1


def test_login_throttle_check_does_not_increment(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """Repeated ``check`` calls never lock an account on their own."""
    throttle = RedisLoginThrottle(fake_redis, max_failures=2, window_seconds=900)

    for _ in range(50):
        throttle.check("user@example.com")


def test_login_throttle_reset_clears_failures(fake_redis: fakeredis.FakeStrictRedis) -> None:
    """A reset after a successful sign-in reopens the account immediately."""
    throttle = RedisLoginThrottle(fake_redis, max_failures=2, window_seconds=900)

    for _ in range(2):
        throttle.record_failure("user@example.com")
    with pytest.raises(DomainError):
        throttle.check("user@example.com")

    throttle.reset("user@example.com")

    throttle.check("user@example.com")


def test_login_throttle_fails_open_on_redis_error() -> None:
    """All three throttle operations swallow Redis errors and never lock out."""
    throttle = RedisLoginThrottle(_RaisingRedis(), max_failures=1, window_seconds=900)

    throttle.record_failure("user@example.com")
    throttle.reset("user@example.com")
    throttle.check("user@example.com")


def test_build_redis_client_returns_none_when_unset() -> None:
    """A falsy URL disables the limiter by yielding no client."""
    assert build_redis_client(None) is None
    assert build_redis_client("") is None


def test_build_redis_client_builds_without_connecting() -> None:
    """A URL yields a client; construction must not require a live server."""
    client = build_redis_client("redis://localhost:6379/0")

    assert isinstance(client, Redis)


def test_enforce_submission_rate_uses_settings(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.FakeStrictRedis
) -> None:
    """The submission helper reads its cap from settings and rejects over it."""
    monkeypatch.setattr(rate_limit, "shared_redis_client", lambda: fake_redis)
    monkeypatch.setattr(settings, "rate_limit_submissions_per_minute", 2)

    for _ in range(2):
        enforce_submission_rate("alice")

    with pytest.raises(DomainError) as exc:
        enforce_submission_rate("alice")
    assert exc.value.code == "rate_limit.exceeded"


def test_enforce_account_rate_separates_actions(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.FakeStrictRedis
) -> None:
    """Distinct action tags keep per-email counters independent."""
    monkeypatch.setattr(rate_limit, "shared_redis_client", lambda: fake_redis)
    monkeypatch.setattr(settings, "rate_limit_account_requests_per_hour", 1)

    enforce_account_rate("user@example.com", "register")
    with pytest.raises(DomainError):
        enforce_account_rate("user@example.com", "register")

    enforce_account_rate("user@example.com", "pwreset")


def test_build_login_throttle_follows_client_availability(
    monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.FakeStrictRedis
) -> None:
    """The factory returns a Redis throttle only when a client is configured."""
    monkeypatch.setattr(rate_limit, "shared_redis_client", lambda: None)
    assert build_login_throttle() is None

    monkeypatch.setattr(rate_limit, "shared_redis_client", lambda: fake_redis)
    assert isinstance(build_login_throttle(), RedisLoginThrottle)
