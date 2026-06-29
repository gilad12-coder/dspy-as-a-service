"""Tests for ``StripeBillingService``: metered overage, the credit ledger, and gate.

Covers ``report_run_usage`` (whether/with-what-units a meter event is pushed),
``create_subscription_checkout`` (whether the metered price rides on the
subscription), and the Phase-0 credit-ledger backbone — ``credits_for_tokens``,
the rolling/non-cumulative free-grant reset, run debiting (grant before paid
balance), and the ``spendable_credits`` figure the submit gate reads. Each test
stands up an in-memory SQLite engine with the billing tables and patches the
``stripe`` module so no network call is made.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.api.errors import DomainError
from core.billing.service import (
    FOUNDERS_LOCK_DAYS,
    FREE_GRANT_CREDITS,
    GRANT_WINDOW_DAYS,
    METER_UNIT_TOKENS,
    TOKENS_PER_CREDIT,
    StripeBillingService,
    credits_for_tokens,
    platform_fee_credits,
)
from core.config import settings
from core.constants import GUARANTEE_BASIS_TEST, GUARANTEE_BASIS_VAL, TOKEN_SOURCE_BYOK, TOKEN_SOURCE_MANAGED
from core.storage.models import Base, BillingCustomerModel, CreditLedgerModel, GuaranteeRunModel


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


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
        session.add(
            BillingCustomerModel(
                username=username, stripe_customer_id=f"cus_{username}", credit_balance=0
            )
        )
        session.commit()


def test_report_run_usage_noop_when_stripe_unconfigured(
    engine: object, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_report_run_usage_noop_below_one_unit(engine: object, configured: None) -> None:
    """A run smaller than one meter unit reports nothing rather than rounding up."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("u@x.com", METER_UNIT_TOKENS - 1)
    create.assert_not_called()


def test_report_run_usage_meters_whole_units_for_customer(engine: object, configured: None) -> None:
    """Tokens are floored to whole meter units and pushed for the right customer."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    with patch("stripe.billing.MeterEvent.create") as create:
        service.report_run_usage("u@x.com", 2 * METER_UNIT_TOKENS + 500)
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["event_name"] == settings.stripe_meter_event_name
    assert kwargs["payload"]["stripe_customer_id"] == "cus_u@x.com"
    assert kwargs["payload"]["value"] == "2"


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


def test_founders_rate_open_before_close_date(
    engine: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the close date the offer is open and reports a 12-month lock window."""
    monkeypatch.setattr(settings, "founders_rate_closes_at", "2099-12-31T23:59:59Z")
    service = StripeBillingService(engine=engine)
    now = datetime(2026, 6, 28, tzinfo=UTC)
    status = service.founders_rate_status(now=now)
    assert status.open is True
    assert status.price_locked_until == (now + timedelta(days=FOUNDERS_LOCK_DAYS)).isoformat()


def test_founders_rate_closed_after_close_date(
    engine: object, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_credits_for_tokens_rounds_up_and_floors_at_zero() -> None:
    """A partial credit's tokens cost a whole credit; non-positive costs nothing."""
    assert credits_for_tokens(0) == 0
    assert credits_for_tokens(-10) == 0
    assert credits_for_tokens(1) == 1
    assert credits_for_tokens(TOKENS_PER_CREDIT) == 1
    assert credits_for_tokens(TOKENS_PER_CREDIT + 1) == 2
    assert credits_for_tokens(3 * TOKENS_PER_CREDIT) == 3


def test_wallet_reports_full_grant_for_new_account(engine: object) -> None:
    """A brand-new account reads a full grant without a row being created."""
    snapshot = StripeBillingService(engine=engine).get_wallet("new@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "new@x.com") is None


def test_wallet_seeds_grant_window_on_first_read(engine: object) -> None:
    """Reading an existing row with no window seeds a full grant and a +30d anchor."""
    _seed_customer(engine, "u@x.com")
    before = datetime.now(UTC)
    snapshot = StripeBillingService(engine=engine).get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "u@x.com")
    assert customer.grant_remaining == FREE_GRANT_CREDITS
    assert customer.grant_reset_at is not None
    expected = before + timedelta(days=GRANT_WINDOW_DAYS)
    assert abs((_as_utc(customer.grant_reset_at) - expected).total_seconds()) < 5


def test_debit_run_draws_from_grant_first(engine: object) -> None:
    """A run debit decrements the free grant before touching the paid balance."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    cost = service.debit_run(
        "u@x.com", 5 * TOKENS_PER_CREDIT, model="openai/gpt-5.5-mini", description="run-a"
    )
    assert cost == 5
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS - 5
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        rows = session.query(CreditLedgerModel).filter_by(username="u@x.com").all()
    assert len(rows) == 1
    assert rows[0].delta_credits == -5
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
    cost = service.debit_run("u@x.com", 30 * TOKENS_PER_CREDIT, model=None, description="big")
    assert cost == 30
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 0
    assert snapshot.paid_balance_credits == 80


def test_debit_run_creates_local_row_for_customerless_account(engine: object) -> None:
    """A run for an account that never touched Stripe seeds a local billing row."""
    service = StripeBillingService(engine=engine)
    service.debit_run("free@x.com", 3 * TOKENS_PER_CREDIT, model=None, description="r")
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "free@x.com")
    assert customer is not None
    assert customer.stripe_customer_id.startswith("local:")
    assert customer.grant_remaining == FREE_GRANT_CREDITS - 3


def test_debit_run_zero_cost_writes_nothing(engine: object) -> None:
    """A run under one credit's worth of tokens writes no ledger row or debit."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    assert service.debit_run("u@x.com", 0, model=None, description="r") == 0
    with Session(engine) as session:
        assert session.query(CreditLedgerModel).filter_by(username="u@x.com").count() == 0
        assert session.get(BillingCustomerModel, "u@x.com").grant_remaining in (None, FREE_GRANT_CREDITS)


def test_grant_reset_is_rolling_and_non_cumulative(engine: object) -> None:
    """Past the window the grant tops up to a flat 500; leftover does not bank."""
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
    service = StripeBillingService(engine=engine)
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "u@x.com")
    assert _as_utc(customer.grant_reset_at) > datetime.now(UTC)
    assert customer.grant_remaining == FREE_GRANT_CREDITS


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


def _lift(baseline: float, optimized: float, *, basis: str = GUARANTEE_BASIS_TEST) -> dict:
    """Build a guarantee block with the given basis and scores."""
    return {"basis": basis, "baseline": baseline, "optimized": optimized}


def _balance(engine: object, username: str) -> tuple[int, int]:
    """Read (grant_remaining, paid_balance) for an account."""
    with Session(engine) as session:
        c = session.get(BillingCustomerModel, username)
        return (0, 0) if c is None else (int(c.grant_remaining or 0), int(c.credit_balance))


def test_platform_fee_is_a_floor_and_fraction_of_cost() -> None:
    """The platform fee is at least one credit and a fraction of the full cost."""
    assert platform_fee_credits(0) == 0
    assert platform_fee_credits(TOKENS_PER_CREDIT) == 1  # 20% of 1 credit, floored up to 1
    assert platform_fee_credits(100 * TOKENS_PER_CREDIT) == 20  # 20% of 100


def test_guarantee_refunds_full_run_on_no_lift_managed(engine: object) -> None:
    """A managed first run with no lift is fully refunded and the slot is flagged."""
    service = StripeBillingService(engine=engine)
    # Debit 50 credits worth first (mirrors the worker's debit before adjudication).
    service.debit_run("u@x.com", 50 * TOKENS_PER_CREDIT, model="m1", description="r")
    assert _balance(engine, "u@x.com")[0] == FREE_GRANT_CREDITS - 50
    refunded = service.adjudicate_guarantee(
        "u@x.com",
        "task-abc",
        "opt-1",
        _lift(0.7, 0.7),
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=50 * TOKENS_PER_CREDIT,
        model="m1",
        description="No lift — refunded",
    )
    assert refunded == 50
    assert _balance(engine, "u@x.com")[0] == FREE_GRANT_CREDITS  # restored
    with Session(engine) as session:
        claim = session.get(GuaranteeRunModel, ("u@x.com", "task-abc"))
        assert claim is not None
        assert claim.refunded is True
        rows = session.query(CreditLedgerModel).filter_by(username="u@x.com").all()
    assert sorted(r.delta_credits for r in rows) == [-50, 50]


def test_guarantee_refunds_only_platform_fee_on_byok_no_lift(engine: object) -> None:
    """A BYOK no-lift run refunds only Skynet's platform fee, not provider tokens."""
    service = StripeBillingService(engine=engine)
    refunded = service.adjudicate_guarantee(
        "u@x.com",
        "task-byok",
        "opt-1",
        _lift(0.5, 0.5),
        token_source=TOKEN_SOURCE_BYOK,
        total_tokens=100 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert refunded == platform_fee_credits(100 * TOKENS_PER_CREDIT)
    assert refunded == 20


def test_guarantee_no_refund_when_run_has_lift(engine: object) -> None:
    """Any improvement on the basis counts as lift — the run stays billed."""
    service = StripeBillingService(engine=engine)
    refunded = service.adjudicate_guarantee(
        "u@x.com",
        "task-lift",
        "opt-1",
        _lift(0.6, 0.61),
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=10 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert refunded == 0
    with Session(engine) as session:
        claim = session.get(GuaranteeRunModel, ("u@x.com", "task-lift"))
        assert claim is not None
        assert claim.refunded is False
        assert session.query(CreditLedgerModel).filter_by(username="u@x.com").count() == 0


def test_guarantee_only_covers_first_run_per_task(engine: object) -> None:
    """A re-run on the same task bills regardless — only the first run is covered."""
    service = StripeBillingService(engine=engine)
    first = service.adjudicate_guarantee(
        "u@x.com",
        "task-dup",
        "opt-1",
        _lift(0.7, 0.7),
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=20 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert first == 20
    second = service.adjudicate_guarantee(
        "u@x.com",
        "task-dup",
        "opt-2",
        _lift(0.7, 0.7),
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=20 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert second == 0
    with Session(engine) as session:
        # Only the first run's slot exists, still pointing at opt-1.
        claim = session.get(GuaranteeRunModel, ("u@x.com", "task-dup"))
        assert claim.optimization_id == "opt-1"


def test_guarantee_valset_basis_judges_lift(engine: object) -> None:
    """The valset fallback basis is honored when no test split was reserved."""
    service = StripeBillingService(engine=engine)
    refunded = service.adjudicate_guarantee(
        "u@x.com",
        "task-val",
        "opt-1",
        _lift(0.8, 0.8, basis=GUARANTEE_BASIS_VAL),
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=5 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert refunded == 5


def test_guarantee_billed_when_no_comparable_scores(engine: object) -> None:
    """A run with no guarantee block still spends its slot but is not refunded."""
    service = StripeBillingService(engine=engine)
    refunded = service.adjudicate_guarantee(
        "u@x.com",
        "task-none",
        "opt-1",
        None,
        token_source=TOKEN_SOURCE_MANAGED,
        total_tokens=5 * TOKENS_PER_CREDIT,
        model=None,
        description="No lift — refunded",
    )
    assert refunded == 0
    with Session(engine) as session:
        assert session.get(GuaranteeRunModel, ("u@x.com", "task-none")) is not None
