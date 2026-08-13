"""Tests for monthly-active-user admission control."""

from __future__ import annotations

from datetime import date

import pytest

from ...config import settings
from .. import monthly_active_users
from ..errors import DomainError

_MONTH_START = date(2026, 8, 1)


class _Store:
    """Record admission calls and return a configured decision."""

    def __init__(self, admitted: bool = True) -> None:
        """Initialize the decision and empty call list.

        Args:
            admitted: Decision returned by the fake store.
        """
        self.admitted = admitted
        self.calls: list[tuple[str, date, int]] = []

    def admit_monthly_active_user(self, username: str, month_start: date, limit: int) -> bool:
        """Record and return one admission decision.

        Args:
            username: Normalized identity.
            month_start: UTC month being admitted.
            limit: Configured monthly limit.

        Returns:
            The configured decision.
        """
        self.calls.append((username, month_start, limit))
        return self.admitted


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module cache and freeze the month for deterministic tests."""
    monthly_active_users._ADMITTED_IDENTITIES.clear()
    monkeypatch.setattr(monthly_active_users, "_CACHE_MONTH", None)
    monkeypatch.setattr(monthly_active_users, "_utc_month_start", lambda: _MONTH_START)


def test_first_request_admits_and_later_request_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One identity reaches the store once per process and month."""
    monkeypatch.setattr(settings, "max_monthly_active_users", 10_000)
    store = _Store()

    monthly_active_users.enforce_monthly_active_user_limit(store, " Alice@Example.com ")
    monthly_active_users.enforce_monthly_active_user_limit(store, "alice@example.com")

    assert store.calls == [("alice@example.com", _MONTH_START, 10_000)]


def test_new_identity_is_rejected_when_capacity_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store refusal becomes the stable monthly-capacity 503 error."""
    monkeypatch.setattr(settings, "max_monthly_active_users", 10_000)

    with pytest.raises(DomainError) as exc:
        monthly_active_users.enforce_monthly_active_user_limit(_Store(admitted=False), "new@example.com")

    assert exc.value.code == "accounts.monthly_capacity_reached"
    assert exc.value.status_code == 503


def test_admin_and_disabled_limit_skip_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operational admins and disabled deployments never consume capacity."""
    store = _Store(admitted=False)
    monkeypatch.setattr(settings, "max_monthly_active_users", 10_000)
    monthly_active_users.enforce_monthly_active_user_limit(store, "admin@example.com", exempt=True)
    monkeypatch.setattr(settings, "max_monthly_active_users", 0)
    monthly_active_users.enforce_monthly_active_user_limit(store, "user@example.com")

    assert store.calls == []
