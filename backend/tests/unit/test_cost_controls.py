"""Unit tests for the P0 cost-control guardrails.

Covers the three admission guards that bound platform cost without new infra:
the sign-up cap (:func:`accounts._enforce_signup_cap`), the per-user concurrency
cap and global kill-switches (:func:`submissions._enforce_submission_admission`
and :func:`submissions._enforce_global_daily_spend_ceiling`), and the ledger
query that backs the daily spend switch (``StripeBillingService.credits_spent_since``).

The guard logic is exercised with duck-typed fake stores and a patched billing
service so no database is required; ``credits_spent_since`` runs against an
in-memory SQLite engine holding only the ``credit_ledger`` table.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from core.api.errors import DomainError
from core.api.routers import accounts, submissions
from core.billing.service import StripeBillingService
from core.config import settings
from core.storage.models import CreditLedgerModel


class _FakeStore:
    """Minimal duck-typed job store exposing only the counters the guards read."""

    def __init__(
        self,
        *,
        users: int = 0,
        monthly_active: int = 0,
        active: int = 0,
        engine: object | None = object(),
    ) -> None:
        """Record the fixed counts and engine the guards will observe.

        Args:
            users: Value returned by :meth:`count_users`.
            monthly_active: Value returned by :meth:`count_monthly_active_users`.
            active: Number of active runs reported by :meth:`count_jobs_by_status`.
            engine: Stand-in ORM engine; ``None`` mimics a legacy/in-memory store.
        """
        self._users = users
        self._monthly_active = monthly_active
        self._active = active
        self.engine = engine

    def count_users(self) -> int:
        """Return the configured total-account count."""
        return self._users

    def count_monthly_active_users(self, month_start: date) -> int:
        """Return the configured monthly-active count.

        Args:
            month_start: First UTC date of the month, accepted for parity with
                the production store.

        Returns:
            The configured count.
        """
        return self._monthly_active

    def count_jobs_by_status(self, *, username: str) -> dict[str, int]:
        """Return the configured active-run count under a single status bucket."""
        return {"running": self._active}


class _NoCounterStore:
    """A store lacking the count methods, to exercise the getattr fall-throughs."""

    def __init__(self) -> None:
        """Expose a null engine so no billing lookup is attempted."""
        self.engine = None


def _patch_billing_spend(monkeypatch: pytest.MonkeyPatch, spent: int) -> None:
    """Replace the submission path's billing service with one reporting fixed spend.

    Args:
        monkeypatch: Pytest patcher.
        spent: Value every ``credits_spent_since`` call should return.
    """

    class _FakeBilling:
        def __init__(self, *, engine: object) -> None:
            self._engine = engine

        def credits_spent_since(self, since: datetime) -> int:
            return spent

    monkeypatch.setattr(submissions, "StripeBillingService", _FakeBilling)


def test_signup_cap_blocks_at_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """At or above ``max_total_users`` registration is refused with a 503 code."""
    monkeypatch.setattr(settings, "max_total_users", 10)
    monkeypatch.setattr(settings, "max_monthly_active_users", 0)

    with pytest.raises(DomainError) as exc:
        accounts._enforce_signup_cap(_FakeStore(users=10))

    assert exc.value.code == "accounts.signups_closed"
    assert exc.value.status_code == 503


def test_signup_cap_allows_below_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the cap, registration proceeds without raising."""
    monkeypatch.setattr(settings, "max_total_users", 10)
    monkeypatch.setattr(settings, "max_monthly_active_users", 0)

    accounts._enforce_signup_cap(_FakeStore(users=9))


def test_signup_cap_disabled_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap of 0 disables the guard even when many accounts exist."""
    monkeypatch.setattr(settings, "max_total_users", 0)
    monkeypatch.setattr(settings, "max_monthly_active_users", 0)

    accounts._enforce_signup_cap(_FakeStore(users=1_000_000))


def test_signup_cap_skips_store_without_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store without ``count_users`` is skipped rather than erroring."""
    monkeypatch.setattr(settings, "max_total_users", 1)
    monkeypatch.setattr(settings, "max_monthly_active_users", 0)

    accounts._enforce_signup_cap(_NoCounterStore())


def test_signup_cap_blocks_when_monthly_capacity_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full monthly-active population pauses further registrations."""
    monkeypatch.setattr(settings, "max_total_users", 0)
    monkeypatch.setattr(settings, "max_monthly_active_users", 10)

    with pytest.raises(DomainError) as exc:
        accounts._enforce_signup_cap(_FakeStore(monthly_active=10))

    assert exc.value.code == "accounts.signups_closed"
    assert exc.value.status_code == 503


def test_admission_paused_switch_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manual pause switch refuses submissions with a generic 503."""
    monkeypatch.setattr(settings, "submissions_paused", True)
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)
    monkeypatch.setattr(settings, "max_concurrent_jobs_per_user", 0)

    with pytest.raises(DomainError) as exc:
        submissions._enforce_submission_admission(_FakeStore(), "alice")

    assert exc.value.code == "submission.capacity_reached"
    assert exc.value.status_code == 503


def test_admission_concurrency_cap_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the per-user active-run cap a submission is refused (429, limit param)."""
    monkeypatch.setattr(settings, "submissions_paused", False)
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)
    monkeypatch.setattr(settings, "max_concurrent_jobs_per_user", 2)

    with pytest.raises(DomainError) as exc:
        submissions._enforce_submission_admission(_FakeStore(active=2), "alice")

    assert exc.value.code == "quota.concurrent_reached"
    assert exc.value.status_code == 429
    assert exc.value.params["limit"] == 2


def test_admission_concurrency_cap_allows_below(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the active-run cap the submission is admitted."""
    monkeypatch.setattr(settings, "submissions_paused", False)
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)
    monkeypatch.setattr(settings, "max_concurrent_jobs_per_user", 2)

    submissions._enforce_submission_admission(_FakeStore(active=1), "alice")


def test_admission_concurrency_disabled_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A concurrency cap of 0 disables the per-user guard."""
    monkeypatch.setattr(settings, "submissions_paused", False)
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)
    monkeypatch.setattr(settings, "max_concurrent_jobs_per_user", 0)

    submissions._enforce_submission_admission(_FakeStore(active=1_000), "alice")


def test_admission_skips_store_without_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store lacking ``count_jobs_by_status`` skips the concurrency check."""
    monkeypatch.setattr(settings, "submissions_paused", False)
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)
    monkeypatch.setattr(settings, "max_concurrent_jobs_per_user", 1)

    submissions._enforce_submission_admission(_NoCounterStore(), "alice")


def test_global_ceiling_blocks_at_or_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trailing-window spend at the ceiling refuses new submissions (503)."""
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 100)
    _patch_billing_spend(monkeypatch, 100)

    with pytest.raises(DomainError) as exc:
        submissions._enforce_global_daily_spend_ceiling(_FakeStore())

    assert exc.value.code == "submission.capacity_reached"
    assert exc.value.status_code == 503


def test_global_ceiling_allows_below(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spend below the ceiling admits the submission."""
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 100)
    _patch_billing_spend(monkeypatch, 99)

    submissions._enforce_global_daily_spend_ceiling(_FakeStore())


def test_global_ceiling_disabled_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling of 0 disables the kill-switch without touching billing."""
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 0)

    submissions._enforce_global_daily_spend_ceiling(_FakeStore())


def test_global_ceiling_skips_engineless_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store with no SQL engine skips the spend lookup."""
    monkeypatch.setattr(settings, "global_daily_spend_ceiling_credits", 100)

    submissions._enforce_global_daily_spend_ceiling(_FakeStore(engine=None))


def test_credits_spent_since_sums_only_recent_debits() -> None:
    """Only negative ledger deltas at/after the window count, as positive magnitude."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    CreditLedgerModel.__table__.create(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        session.add_all(
            [
                CreditLedgerModel(
                    username="a", delta_credits=-30, kind="run", created_at=now - timedelta(hours=1)
                ),
                CreditLedgerModel(
                    username="b", delta_credits=-12, kind="run", created_at=now - timedelta(hours=2)
                ),
                CreditLedgerModel(
                    username="a", delta_credits=500, kind="topup", created_at=now - timedelta(hours=1)
                ),
                CreditLedgerModel(
                    username="c", delta_credits=-99, kind="run", created_at=now - timedelta(hours=48)
                ),
            ]
        )
        session.commit()

    service = StripeBillingService(engine=engine)

    assert service.credits_spent_since(now - timedelta(hours=24)) == 42


def test_credits_spent_since_empty_window_is_zero() -> None:
    """With no debits in the window the spend total is zero, not an error."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    CreditLedgerModel.__table__.create(engine)

    service = StripeBillingService(engine=engine)

    assert service.credits_spent_since(datetime.now(UTC) - timedelta(hours=24)) == 0
