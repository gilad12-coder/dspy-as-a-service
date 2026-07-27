"""Tests for ``StripeBillingService``: metered overage, the credit ledger, and gate.

Covers ``report_run_usage`` (whether/with-what-units a meter event is pushed),
``create_subscription_checkout`` (whether the metered price rides on the
subscription), and the Phase-0 credit-ledger backbone — the one-time free grant
and the renewing Premium allotment, run debiting (grant before paid balance), and
the ``spendable_credits`` figure the submit gate reads. Each test
stands up an in-memory SQLite engine with the billing tables and patches the
``stripe`` module so no network call is made.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import stripe
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.api.errors import DomainError
from core.billing.pricing import ModelUsage, credits_for_usage
from core.billing.service import (
    FOUNDERS_LOCK_DAYS,
    FREE_GRANT_CREDITS,
    GRANT_WINDOW_DAYS,
    PREMIUM_GRANT_CREDITS,
    StripeBillingService,
    cost_ceiling_budget,
    platform_fee_credits_for_usage,
)
from core.config import settings
from core.constants import TOKEN_SOURCE_BYOK, TOKEN_SOURCE_MANAGED
from core.storage.models import (
    Base,
    BillingCustomerModel,
    BillingWebhookEventModel,
    CreditLedgerModel,
)


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


def _usages(input_tokens: int, output_tokens: int = 0) -> list[ModelUsage]:
    """Single-model usage on an unpriced model so default per-token costs apply.

    Pricing is then deterministic (the module default 1e-6 input / 3e-6 output ×
    MARKUP) regardless of LiteLLM's live table, so a debit's credit cost is stable
    across versions. ``_usages(100_000)`` ≈ 15 credits; ``_usages(200_000)`` ≈ 30.
    """
    return [ModelUsage(model="test/unpriced", input_tokens=input_tokens, output_tokens=output_tokens)]


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake Stripe secret key so mutations are not short-circuited."""
    monkeypatch.setattr(settings, "stripe_secret_key", SecretStr("sk_test_dummy"))


def _seed_customer(engine: object, username: str) -> None:
    """Insert a billing customer row so the account is metered-eligible.

    Args:
        engine: The SQLite engine to write to.
        username: Account identity to create a Stripe-customer link for.
    """
    with Session(engine) as session:
        session.add(BillingCustomerModel(username=username, stripe_customer_id=f"cus_{username}", credit_balance=0))
        session.commit()


def test_report_run_usage_noop_when_stripe_unconfigured(engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """No meter event is pushed when no Stripe secret key is configured."""
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("u@x.com", 5000)
    create.assert_not_called()


def test_report_run_usage_noop_without_billing_customer(engine: object, configured: None) -> None:
    """Usage for an account that never touched billing is not metered (no customer sprawl)."""
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("nobody@x.com", 5000)
    create.assert_not_called()
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "nobody@x.com") is None


def test_report_run_usage_noop_for_zero_credits(engine: object, configured: None) -> None:
    """A run that cost nothing reports no meter event."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("u@x.com", 0)
    create.assert_not_called()


def test_report_run_usage_meters_credits_for_customer(engine: object, configured: None) -> None:
    """Credits are metered one-to-one (one unit per credit) for the right customer."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("u@x.com", 7)
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["event_name"] == settings.stripe_meter_event_name
    assert kwargs["payload"]["stripe_customer_id"] == "cus_u@x.com"
    assert kwargs["payload"]["value"] == "7"


def test_subscription_checkout_includes_metered_item(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured metered price rides on the subscription as a quantity-less item."""
    monkeypatch.setattr(settings, "stripe_price_premium", "price_premium")
    monkeypatch.setattr(settings, "stripe_price_metered", "price_metered")
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.checkout.Session.create") as create:
        create.return_value = SimpleNamespace(url="https://checkout.test/abc")
        url = service.create_subscription_checkout("u@x.com")
    assert url == "https://checkout.test/abc"
    assert create.call_args.kwargs["line_items"] == [
        {"price": "price_premium", "quantity": 1},
        {"price": "price_metered"},
    ]


def test_subscription_checkout_omits_metered_when_unconfigured(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no metered price configured, only the flat Premium item is sent."""
    monkeypatch.setattr(settings, "stripe_price_premium", "price_premium")
    monkeypatch.setattr(settings, "stripe_price_metered", "")
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.checkout.Session.create") as create:
        create.return_value = SimpleNamespace(url="https://checkout.test/abc")
        service.create_subscription_checkout("u@x.com")
    assert create.call_args.kwargs["line_items"] == [{"price": "price_premium", "quantity": 1}]


def test_founders_rate_open_before_close_date(engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before the close date the offer is open and reports a 12-month lock window."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2099-12-31T23:59:59Z")
    service = StripeBillingService(engine=engine)
    now = datetime(2026, 6, 28, tzinfo=UTC)
    status = service.founders_rate_status(now=now)
    assert status.open is True
    assert status.price_locked_until == (now + timedelta(days=FOUNDERS_LOCK_DAYS)).isoformat()


def test_founders_rate_closed_after_close_date(engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the configured deadline the offer reports closed."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2026-07-31T23:59:59Z")
    service = StripeBillingService(engine=engine)
    status = service.founders_rate_status(now=datetime(2026, 8, 1, tzinfo=UTC))
    assert status.open is False


def test_founders_checkout_stamps_price_lock_metadata(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Founder's Rate checkout carries the lock metadata onto the subscription."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2099-12-31T23:59:59Z")
    monkeypatch.setattr(settings, "stripe_price_founders", "price_founders")
    monkeypatch.setattr(settings, "stripe_price_metered", "")
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.checkout.Session.create") as create:
        create.return_value = SimpleNamespace(url="https://checkout.test/f")
        url = service.create_founders_checkout("u@x.com")
    assert url == "https://checkout.test/f"
    kwargs = create.call_args.kwargs
    assert kwargs["mode"] == "subscription"
    assert kwargs["line_items"] == [{"price": "price_founders", "quantity": 1}]
    assert kwargs["metadata"]["founders_rate"] == "true"
    assert "price_locked_until" in kwargs["metadata"]
    assert kwargs["subscription_data"]["metadata"]["founders_rate"] == "true"
    # No explicit payment_method_types — Stripe picks from the Dashboard config.
    assert "payment_method_types" not in kwargs


def test_founders_checkout_falls_back_to_premium_price(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no dedicated Founder's price, the Premium price backs the subscription."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2099-12-31T23:59:59Z")
    monkeypatch.setattr(settings, "stripe_price_founders", "")
    monkeypatch.setattr(settings, "stripe_price_premium", "price_premium")
    monkeypatch.setattr(settings, "stripe_price_metered", "")
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.checkout.Session.create") as create:
        create.return_value = SimpleNamespace(url="https://checkout.test/f")
        service.create_founders_checkout("u@x.com")
    assert create.call_args.kwargs["line_items"] == [{"price": "price_premium", "quantity": 1}]


def test_founders_checkout_rejected_after_deadline(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the deadline a checkout attempt is gated with a 410, no Stripe call."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2020-01-01T00:00:00Z")
    monkeypatch.setattr(settings, "stripe_price_founders", "price_founders")
    service = StripeBillingService(engine=engine)
    with patch("stripe.checkout.Session.create") as create, pytest.raises(DomainError) as exc:
        service.create_founders_checkout("u@x.com")
    assert exc.value.status_code == 410
    create.assert_not_called()


def _as_utc(value: datetime) -> datetime:
    """Coerce a possibly-naive timestamp to UTC (SQLite drops tzinfo on read).

    Args:
        value: A datetime read back from the SQLite billing tables.

    Returns:
        The same instant tagged UTC when it was naive.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _grant_remaining(engine: object, username: str) -> int | None:
    """Read the persisted grant_remaining for an account.

    Args:
        engine: The SQLite engine to read from.
        username: Account whose grant column to read.

    Returns:
        The stored ``grant_remaining`` (``None`` when the row or column is unseeded).
    """
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, username)
        return None if customer is None else customer.grant_remaining


def test_wallet_reports_full_grant_for_new_account(engine: object) -> None:
    """A brand-new account reads a full grant without a row being created."""
    snapshot = StripeBillingService(engine=engine).get_wallet("new@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "new@x.com") is None


def test_wallet_seeds_one_time_free_grant_on_first_read(engine: object) -> None:
    """Reading a free row with no grant seeds the one-time grant and no reset anchor."""
    _seed_customer(engine, "u@x.com")
    snapshot = StripeBillingService(engine=engine).get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    assert snapshot.free_grant_resets_at is None
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "u@x.com")
    assert customer.grant_remaining == FREE_GRANT_CREDITS
    assert customer.grant_reset_at is None


def test_debit_run_draws_from_grant_first(engine: object) -> None:
    """A run debit decrements the free grant before touching the paid balance."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    usages = _usages(100_000)
    expected = credits_for_usage(usages)
    cost = service.debit_run("u@x.com", usages, model="openai/gpt-5.5-mini", description="run-a")
    assert cost == expected
    assert 0 < expected < FREE_GRANT_CREDITS
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS - expected
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        rows = session.query(CreditLedgerModel).filter_by(username="u@x.com").all()
    assert len(rows) == 1
    assert rows[0].delta_credits == -expected
    assert rows[0].kind == "run"
    assert rows[0].model == "openai/gpt-5.5-mini"


def test_debit_run_overflows_grant_into_paid_balance(engine: object) -> None:
    """A debit larger than the remaining grant drains it, then the paid balance."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=100,
                grant_remaining=10,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    expected = credits_for_usage(usages)
    cost = service.debit_run("u@x.com", usages, model=None, description="big")
    assert cost == expected
    assert expected > 10  # must exceed the remaining grant to overflow into paid
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 0
    assert snapshot.paid_balance_credits == 100 - (expected - 10)


def test_debit_run_creates_local_row_for_customerless_account(engine: object) -> None:
    """A run for an account that never touched Stripe seeds a local billing row."""
    service = StripeBillingService(engine=engine)
    usages = _usages(20_000)
    service.debit_run("free@x.com", usages, model=None, description="r")
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "free@x.com")
    assert customer is not None
    assert customer.stripe_customer_id.startswith("local:")
    assert customer.grant_remaining == FREE_GRANT_CREDITS - credits_for_usage(usages)


def test_debit_run_zero_cost_writes_nothing(engine: object) -> None:
    """A run under one credit's worth of tokens writes no ledger row or debit."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    assert service.debit_run("u@x.com", [], model=None, description="r") == 0
    with Session(engine) as session:
        assert session.query(CreditLedgerModel).filter_by(username="u@x.com").count() == 0
        assert session.get(BillingCustomerModel, "u@x.com").grant_remaining in (None, FREE_GRANT_CREDITS)


def test_free_grant_is_one_time_and_never_resets(engine: object) -> None:
    """A free grant never tops up — even past a stale anchor the leftover stands."""
    past = datetime.now(UTC) - timedelta(days=1)
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=0,
                grant_remaining=40,
                grant_reset_at=past,
            )
        )
        session.commit()
    snapshot = StripeBillingService(engine=engine).get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 40
    assert snapshot.free_grant_resets_at is None
    assert _grant_remaining(engine, "u@x.com") == 40


def test_grant_does_not_reset_before_window_elapses(engine: object) -> None:
    """Inside the window the partially-spent grant is preserved (no early top-up)."""
    future = datetime.now(UTC) + timedelta(days=10)
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=0,
                grant_remaining=40,
                grant_reset_at=future,
            )
        )
        session.commit()
    snapshot = StripeBillingService(engine=engine).get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 40
    assert _grant_remaining(engine, "u@x.com") == 40


def test_spendable_credits_sums_grant_and_paid_balance(engine: object) -> None:
    """The submit-gate figure is grant remaining plus purchased balance."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=70,
                grant_remaining=30,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()
    assert StripeBillingService(engine=engine).spendable_credits("u@x.com") == 100


def test_spendable_credits_zero_when_grant_and_balance_exhausted(engine: object) -> None:
    """A drained grant and zero paid balance reads as no spendable credits (gate trips)."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=0,
                grant_remaining=0,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()
    assert StripeBillingService(engine=engine).spendable_credits("u@x.com") == 0


def test_spendable_credits_full_for_new_account(engine: object) -> None:
    """A brand-new account has a full grant of spendable credits (gate passes)."""
    assert StripeBillingService(engine=engine).spendable_credits("new@x.com") == FREE_GRANT_CREDITS


def _balance(engine: object, username: str) -> tuple[int, int]:
    """Read (grant_remaining, paid_balance) for an account."""
    with Session(engine) as session:
        c = session.get(BillingCustomerModel, username)
        return (0, 0) if c is None else (int(c.grant_remaining or 0), int(c.credit_balance))


def test_debit_run_byok_charges_only_platform_fee(engine: object) -> None:
    """A BYOK run debits only Skynet's platform fee — zero at-cost pricing."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    cost = service.debit_run(
        "u@x.com",
        usages,
        model="m1",
        description="byok-run",
        token_source=TOKEN_SOURCE_BYOK,
    )
    assert cost == platform_fee_credits_for_usage(usages)
    # At-cost pricing: the provider tokens ran on the user's own key and the
    # platform takes no cut, so the run is free and the balance untouched.
    assert cost == 0
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS


def test_debit_run_managed_still_charges_full_cost(engine: object) -> None:
    """A managed run is unaffected — it still pays the full per-token credit cost."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    cost = service.debit_run("u@x.com", usages, model="m1", description="managed-run")
    assert cost == credits_for_usage(usages)


def test_cost_ceiling_budget_managed_is_the_balance(engine: object) -> None:
    """A managed run's ceiling budget is exactly the spendable balance."""
    assert cost_ceiling_budget(0, TOKEN_SOURCE_MANAGED) == 0
    assert cost_ceiling_budget(100, TOKEN_SOURCE_MANAGED) == 100


def test_cost_ceiling_budget_byok_is_fee_aware_and_larger(engine: object) -> None:
    """A fee-less BYOK run gets an effectively unlimited ceiling from any balance."""
    assert cost_ceiling_budget(0, TOKEN_SOURCE_BYOK) == 0
    budget = cost_ceiling_budget(100, TOKEN_SOURCE_BYOK)
    # At-cost pricing: the run can never touch the balance, so the ceiling is
    # far beyond any real run while a drained account still gets nothing.
    assert budget > 100
    assert budget >= 10**9
    assert platform_fee_credits_for_usage(_usages(10_000_000)) == 0


def test_wallet_reports_premium_grant_total_for_subscriber(engine: object) -> None:
    """An active Premium account's grant total is the larger Premium allotment."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="pro@x.com",
                stripe_customer_id="cus_pro",
                credit_balance=0,
                grant_remaining=2000,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
                subscription_status="active",
            )
        )
        session.commit()
    snapshot = StripeBillingService(engine=engine).get_wallet("pro@x.com")
    assert snapshot.free_grant_total == PREMIUM_GRANT_CREDITS
    assert snapshot.free_grant_remaining == 2000
    assert snapshot.premium_active is True


def test_premium_grant_resets_to_premium_allotment(engine: object) -> None:
    """Past the window, a Premium account tops up to the Premium allotment, not 500."""
    past = datetime.now(UTC) - timedelta(days=1)
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="pro@x.com",
                stripe_customer_id="cus_pro",
                credit_balance=0,
                grant_remaining=10,
                grant_reset_at=past,
                subscription_status="active",
            )
        )
        session.commit()
    snapshot = StripeBillingService(engine=engine).get_wallet("pro@x.com")
    assert snapshot.free_grant_remaining == PREMIUM_GRANT_CREDITS


def test_subscription_activation_grants_premium_allotment_immediately(engine: object) -> None:
    """A fresh activation tops the grant up to the Premium allotment right away."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="up@x.com",
                stripe_customer_id="cus_up",
                credit_balance=0,
                grant_remaining=300,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
            )
        )
        session.commit()
    period_end = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp())
    sub = {
        "customer": "cus_up",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_premium"}, "current_period_end": period_end}]},
    }
    service = StripeBillingService(engine=engine)
    with Session(engine) as session:
        service._on_subscription_change(session, sub)
        session.commit()
    grant, _paid = _balance(engine, "up@x.com")
    assert grant == PREMIUM_GRANT_CREDITS
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "up@x.com")
    assert _as_utc(customer.grant_reset_at) == datetime(2026, 8, 1, tzinfo=UTC)


def test_subscription_update_while_active_does_not_refill_grant(engine: object) -> None:
    """An update event on an already-active sub must not top the grant back up."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="act@x.com",
                stripe_customer_id="cus_act",
                credit_balance=0,
                grant_remaining=100,
                grant_reset_at=datetime.now(UTC) + timedelta(days=GRANT_WINDOW_DAYS),
                subscription_status="active",
            )
        )
        session.commit()
    period_end = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
    sub = {
        "customer": "cus_act",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_premium"}, "current_period_end": period_end}]},
    }
    service = StripeBillingService(engine=engine)
    with Session(engine) as session:
        service._on_subscription_change(session, sub)
        session.commit()
    assert _balance(engine, "act@x.com")[0] == 100


def _add_ledger(
    engine: object,
    username: str,
    *,
    delta: int,
    kind: str,
    model: str | None,
    when: datetime,
    description: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Insert one credit-ledger row at an explicit instant.

    Args:
        engine: SQLite engine to write to.
        username: Account the row belongs to.
        delta: Signed credit delta (negative for a spend).
        kind: Ledger kind ('run', 'topup', 'grant').
        model: Model id, or None for non-run rows.
        when: ``created_at`` instant the row is stamped with.
        description: Row label; defaults to the kind.
        input_tokens: Measured input tokens, or None for legacy/non-run rows.
        output_tokens: Measured output tokens, or None for legacy/non-run rows.
    """
    with Session(engine) as session:
        session.add(
            CreditLedgerModel(
                username=username,
                delta_credits=delta,
                kind=kind,
                description=description or kind,
                model=model,
                created_at=when,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        session.commit()


def test_get_usage_aggregates_runs_by_day_and_model(engine: object) -> None:
    """get_usage sums billed spend, counts runs, and rolls up by day and model."""
    user = "u@x.com"
    now = datetime.now(UTC)
    day1 = now - timedelta(days=1)
    day2 = now - timedelta(days=2)
    _add_ledger(engine, user, delta=-300, kind="run", model="openai/gpt-5.5", when=day1)
    _add_ledger(engine, user, delta=-50, kind="run", model="anthropic/claude", when=day1)
    # A positive run row (legacy correction) is ignored by the rollups.
    _add_ledger(engine, user, delta=120, kind="run", model="openai/gpt-5.5", when=day1)
    _add_ledger(engine, user, delta=-100, kind="run", model="openai/gpt-5.5", when=day2)
    _add_ledger(engine, user, delta=2200, kind="topup", model=None, when=day1)

    service = StripeBillingService(engine=engine)
    snapshot = service.get_usage(user, now - timedelta(days=3), now)

    assert snapshot.billed_credits == 450
    assert snapshot.runs == 3
    # Top-ups and positive run rows are excluded from the spend rollups but
    # still ride along in entries.
    assert len(snapshot.entries) == 5
    assert [m.model for m in snapshot.by_model] == ["openai/gpt-5.5", "anthropic/claude"]
    assert snapshot.by_model[0].credits == 400
    assert snapshot.by_model[0].runs == 2
    assert [d.date for d in snapshot.by_day] == [day2.date().isoformat(), day1.date().isoformat()]
    day1_row = next(d for d in snapshot.by_day if d.date == day1.date().isoformat())
    assert day1_row.billed_credits == 350


def test_debit_run_stamps_token_counts(engine: object) -> None:
    """The ledger row records the measured input/output tokens behind the charge."""
    service = StripeBillingService(engine=engine)
    service.debit_run("u@x.com", _usages(100_000, 40_000), model="m", description="Run")
    with Session(engine) as session:
        row = session.query(CreditLedgerModel).one()
    assert row.input_tokens == 100_000
    assert row.output_tokens == 40_000


def test_get_usage_rolls_up_token_counts_per_model(engine: object) -> None:
    """Per-model token sums cover stamped rows; legacy rows contribute zero."""
    user = "u@x.com"
    now = datetime.now(UTC)
    when = now - timedelta(days=1)
    _add_ledger(engine, user, delta=-30, kind="run", model="m", when=when, input_tokens=1000, output_tokens=200)
    _add_ledger(engine, user, delta=-20, kind="run", model="m", when=when, input_tokens=500, output_tokens=100)
    _add_ledger(engine, user, delta=-10, kind="run", model="m", when=when)

    service = StripeBillingService(engine=engine)
    snapshot = service.get_usage(user, now - timedelta(days=3), now)

    (model_row,) = snapshot.by_model
    assert model_row.input_tokens == 1500
    assert model_row.output_tokens == 300
    assert model_row.runs == 3


def test_get_usage_excludes_rows_outside_window(engine: object) -> None:
    """Rows older than the window's start are not counted in the rollup."""
    user = "u@x.com"
    now = datetime.now(UTC)
    _add_ledger(engine, user, delta=-40, kind="run", model="m", when=now - timedelta(days=1))
    _add_ledger(engine, user, delta=-999, kind="run", model="m", when=now - timedelta(days=40))

    service = StripeBillingService(engine=engine)
    snapshot = service.get_usage(user, now - timedelta(days=7), now)

    assert snapshot.billed_credits == 40
    assert snapshot.runs == 1
    assert len(snapshot.entries) == 1


def _checkout_event(event_id: str, username: str, credits: int, pack_id: str = "pack_small") -> dict:
    """Build a minimal ``checkout.session.completed`` event for a paid pack purchase.

    Args:
        event_id: Stripe event id (the idempotency key the handler records).
        username: Buyer the credits land on.
        credits: Credit quantity the pack grants.
        pack_id: Pack identifier carried in the session metadata.

    Returns:
        An event dict shaped like the fields ``handle_webhook`` reads.
    """
    return {
        "id": event_id,
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "mode": "payment",
                "payment_status": "paid",
                "customer": f"cus_{username}",
                "metadata": {"username": username, "credits": str(credits), "pack_id": pack_id},
            }
        },
    }


@pytest.fixture
def webhook_ready(configured: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure both the Stripe secret and the webhook signing secret."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", SecretStr("whsec_test"))


def test_webhook_raises_503_when_signing_secret_unconfigured(
    engine: object, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfigured webhook secret is a 503, never a silent accept."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", None)
    service = StripeBillingService(engine=engine)
    with pytest.raises(DomainError) as exc:
        service.handle_webhook(b"{}", "sig")
    assert exc.value.status_code == 503


def test_webhook_rejects_bad_signature_with_400(engine: object, webhook_ready: None) -> None:
    """A signature that fails verification is rejected as a 400, nothing applied."""
    service = StripeBillingService(engine=engine)
    err = stripe.SignatureVerificationError("invalid signature", "t=1,v1=bad")
    with (
        patch("stripe.Webhook.construct_event", side_effect=err),
        pytest.raises(DomainError) as exc,
    ):
        service.handle_webhook(b"{}", "t=1,v1=bad")
    assert exc.value.status_code == 400
    assert exc.value.code == "billing.webhook_invalid"


def test_webhook_credits_pack_topup(engine: object, webhook_ready: None) -> None:
    """A verified paid checkout credits the buyer and writes one topup ledger row."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    event = _checkout_event("evt_1", "u@x.com", 500)
    with patch("stripe.Webhook.construct_event", return_value=event):
        service.handle_webhook(b"{}", "sig")
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "u@x.com").credit_balance == 500
        rows = session.query(CreditLedgerModel).filter_by(username="u@x.com", kind="topup").all()
        assert len(rows) == 1
        assert rows[0].delta_credits == 500
        assert rows[0].stripe_event_id == "evt_1"
        assert session.get(BillingWebhookEventModel, "evt_1") is not None


def test_webhook_is_idempotent_on_redelivery(engine: object, webhook_ready: None) -> None:
    """A redelivered event (same id) credits exactly once — no double top-up."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    event = _checkout_event("evt_dup", "u@x.com", 500)
    with patch("stripe.Webhook.construct_event", return_value=event):
        service.handle_webhook(b"{}", "sig")
        service.handle_webhook(b"{}", "sig")
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "u@x.com").credit_balance == 500
        rows = session.query(CreditLedgerModel).filter_by(username="u@x.com", kind="topup").all()
        assert len(rows) == 1
