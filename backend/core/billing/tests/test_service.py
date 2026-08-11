"""Tests for ``StripeBillingService``: the credit ledger and the submit gate.

Covers the credit-ledger backbone — the at-cost pricing policy (no free
allowance, packs at par), run debiting (legacy grant before paid balance), the
``spendable_credits`` figure the submit gate reads, the usage rollup, and
webhook idempotency. Each test stands up an
in-memory SQLite engine with the billing tables and patches the ``stripe``
module so no network call is made.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import stripe
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.api.errors import DomainError
from core.billing.pricing import ModelUsage, credits_for_usage
from core.billing.service import (
    CUSTOM_CREDITS_MAX,
    CUSTOM_CREDITS_MIN,
    FREE_GRANT_CREDITS,
    PACK_CREDITS,
    PLATFORM_FEE_FRACTION,
    StripeBillingService,
    committed_spend_credits,
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
    """Insert a billing customer row with a real-looking Stripe link.

    Args:
        engine: The SQLite engine to write to.
        username: Account identity to create a Stripe-customer link for.
    """
    with Session(engine) as session:
        session.add(BillingCustomerModel(username=username, stripe_customer_id=f"cus_{username}", credit_balance=0))
        session.commit()


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


def test_pricing_policy_no_subsidy_no_profit() -> None:
    """The at-cost policy: no free allowance, and packs grant exactly their price in cents."""
    assert FREE_GRANT_CREDITS == 0
    # One credit is one cent, so at-par packs must grant exactly the Stripe
    # unit_amount provisioned in scripts/provision_stripe.py ($5 / $20 / $50).
    assert PACK_CREDITS == {"starter": 500, "plus": 2000, "pro": 5000}


def test_wallet_reports_empty_grant_for_new_account(engine: object) -> None:
    """A brand-new account reads a zero grant without a row being created."""
    snapshot = StripeBillingService(engine=engine).get_wallet("new@x.com")
    assert snapshot.free_grant_remaining == 0
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "new@x.com") is None


def test_wallet_seeds_grant_column_on_first_read(engine: object) -> None:
    """Reading a row with a NULL grant seeds it to the (zero) allowance and persists it."""
    _seed_customer(engine, "u@x.com")
    snapshot = StripeBillingService(engine=engine).get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == FREE_GRANT_CREDITS
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "u@x.com")
    assert customer.grant_remaining == FREE_GRANT_CREDITS


def test_debit_run_draws_from_legacy_grant_first(engine: object) -> None:
    """A run debit decrements a remaining legacy grant before touching the paid balance."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=0,
                grant_remaining=50,
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(100_000)
    expected = credits_for_usage(usages)
    cost = service.debit_run("u@x.com", usages, model="openai/gpt-5.5-mini", description="run-a")
    assert cost == expected
    assert 0 < expected < 50
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 50 - expected
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
    """A run for an account that never touched Stripe seeds a local billing row.

    Credits are prepaid, so with no free allowance and no purchased balance
    there is nothing to draw from: the charge clamps to zero (the shortfall is
    absorbed, never lent) and the balance stays at exactly zero.
    """
    service = StripeBillingService(engine=engine)
    usages = _usages(20_000)
    charged = service.debit_run("free@x.com", usages, model=None, description="r")
    assert charged == 0
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "free@x.com")
        ledger_rows = session.query(CreditLedgerModel).filter_by(username="free@x.com").count()
    assert customer is not None
    assert customer.stripe_customer_id.startswith("local:")
    assert customer.grant_remaining == 0
    assert customer.credit_balance == 0
    assert ledger_rows == 0


def test_debit_run_clamps_charge_to_available_balance(engine: object) -> None:
    """A run costing more than grant + paid drains both to zero, never below.

    The ledger row records the clamped amount actually charged, so the audit
    trail still sums to the stored balance.
    """
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=3,
                grant_remaining=5,
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    assert credits_for_usage(usages) > 8
    charged = service.debit_run("u@x.com", usages, model=None, description="big")
    assert charged == 8
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.free_grant_remaining == 0
    assert snapshot.paid_balance_credits == 0
    with Session(engine) as session:
        (row,) = session.query(CreditLedgerModel).filter_by(username="u@x.com").all()
    assert row.delta_credits == -8


def test_debit_run_repeated_overdraw_floors_at_zero(engine: object) -> None:
    """A second overdrawing run charges nothing — the balance can never go negative."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=2,
                grant_remaining=0,
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    assert service.debit_run("u@x.com", usages, model=None, description="first") == 2
    assert service.debit_run("u@x.com", usages, model=None, description="second") == 0
    assert service.spendable_credits("u@x.com") == 0
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "u@x.com")
        ledger_rows = session.query(CreditLedgerModel).filter_by(username="u@x.com").all()
    assert customer.credit_balance == 0
    assert [row.delta_credits for row in ledger_rows] == [-2]


def test_debit_run_zero_cost_writes_nothing(engine: object) -> None:
    """A run under one credit's worth of tokens writes no ledger row or debit."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    assert service.debit_run("u@x.com", [], model=None, description="r") == 0
    with Session(engine) as session:
        assert session.query(CreditLedgerModel).filter_by(username="u@x.com").count() == 0
        assert session.get(BillingCustomerModel, "u@x.com").grant_remaining in (None, FREE_GRANT_CREDITS)


def test_free_grant_is_one_time_and_never_resets(engine: object) -> None:
    """A legacy account's partially-spent grant is honored as-is — never topped up."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=0,
                grant_remaining=40,
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
            )
        )
        session.commit()
    assert StripeBillingService(engine=engine).spendable_credits("u@x.com") == 0


def test_spendable_credits_zero_for_new_account(engine: object) -> None:
    """A brand-new account has no spendable credits — the submit gate trips until it buys."""
    assert StripeBillingService(engine=engine).spendable_credits("new@x.com") == 0


def test_debit_run_byok_charges_only_platform_fee(engine: object) -> None:
    """A BYOK run debits only the infra platform fee — a fraction of the full cost."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=100,
                grant_remaining=0,
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    cost = service.debit_run(
        "u@x.com",
        usages,
        model="m1",
        description="byok-run",
        token_source=TOKEN_SOURCE_BYOK,
    )
    fee = platform_fee_credits_for_usage(usages)
    assert cost == fee
    # The provider tokens ran on the user's own key, so only the compute/storage
    # share is charged — more than zero, well under the managed full cost.
    assert 0 < fee < credits_for_usage(usages)
    snapshot = service.get_wallet("u@x.com")
    assert snapshot.paid_balance_credits == 100 - fee


def test_debit_run_managed_still_charges_full_cost(engine: object) -> None:
    """A managed run is unaffected — it still pays the full per-token credit cost."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=1000,
                grant_remaining=0,
            )
        )
        session.commit()
    service = StripeBillingService(engine=engine)
    usages = _usages(200_000)
    cost = service.debit_run("u@x.com", usages, model="m1", description="managed-run")
    assert cost == credits_for_usage(usages)


def test_cost_ceiling_budget_managed_is_the_balance(engine: object) -> None:
    """A managed run's ceiling budget is exactly the spendable balance."""
    assert cost_ceiling_budget(0, TOKEN_SOURCE_MANAGED) == 0
    assert cost_ceiling_budget(100, TOKEN_SOURCE_MANAGED) == 100


def test_cost_ceiling_budget_byok_is_fee_aware_and_larger(engine: object) -> None:
    """A BYOK ceiling is the largest full-cost budget whose platform fee fits the balance."""
    assert cost_ceiling_budget(0, TOKEN_SOURCE_BYOK) == 0
    budget = cost_ceiling_budget(100, TOKEN_SOURCE_BYOK)
    # The run only spends the fee fraction, so the same balance backs a
    # proportionally larger — but finite — ceiling, maximal within the balance.
    assert budget > 100
    assert math.ceil(budget * PLATFORM_FEE_FRACTION) <= 100
    assert math.ceil((budget + 1) * PLATFORM_FEE_FRACTION) > 100


def test_committed_spend_credits_managed_is_the_full_budget() -> None:
    """A managed run's committed spend equals its full cost ceiling."""
    assert committed_spend_credits(200, TOKEN_SOURCE_MANAGED) == 200
    assert committed_spend_credits(0, TOKEN_SOURCE_MANAGED) == 0
    assert committed_spend_credits(-5, TOKEN_SOURCE_MANAGED) == 0


def test_committed_spend_credits_byok_is_fee_sized() -> None:
    """A BYOK run commits only the platform fee of its ceiling, at least one credit."""
    assert committed_spend_credits(1000, TOKEN_SOURCE_BYOK) == math.ceil(1000 * PLATFORM_FEE_FRACTION)
    assert committed_spend_credits(1, TOKEN_SOURCE_BYOK) == 1
    assert committed_spend_credits(0, TOKEN_SOURCE_BYOK) == 0


def test_committed_spend_credits_inverts_cost_ceiling_budget() -> None:
    """The ceiling granted for a balance never commits more than that balance."""
    for balance in (1, 12, 100, 500):
        for source in (TOKEN_SOURCE_MANAGED, TOKEN_SOURCE_BYOK):
            budget = cost_ceiling_budget(balance, source)
            assert committed_spend_credits(budget, source) <= balance


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
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="u@x.com",
                stripe_customer_id="cus_u",
                credit_balance=1000,
                grant_remaining=0,
            )
        )
        session.commit()
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


def _checkout_event(
    event_id: str,
    username: str,
    credits: int,
    pack_id: str = "pack_small",
    payment_intent: str | None = None,
) -> dict:
    """Build a minimal ``checkout.session.completed`` event for a paid pack purchase.

    Args:
        event_id: Stripe event id (the idempotency key the handler records).
        username: Buyer the credits land on.
        credits: Credit quantity the pack grants.
        pack_id: Pack identifier carried in the session metadata.
        payment_intent: PaymentIntent id stamped on the top-up row, so a later
            refund/dispute can resolve back to it.

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
                "payment_intent": payment_intent,
                "metadata": {"username": username, "credits": str(credits), "pack_id": pack_id},
            }
        },
    }


def _refund_event(event_id: str, payment_intent: str, amount_refunded: int) -> dict:
    """Build a ``charge.refunded`` event whose charge cumulatively refunded ``amount_refunded``.

    Args:
        event_id: Stripe event id (the idempotency key the handler records).
        payment_intent: The PaymentIntent behind the refunded charge.
        amount_refunded: Cumulative refunded cents on the charge (one credit per cent).

    Returns:
        An event dict shaped like the fields the refund handler reads.
    """
    return {
        "id": event_id,
        "type": "charge.refunded",
        "data": {"object": {"payment_intent": payment_intent, "amount_refunded": amount_refunded}},
    }


def _dispute_event(event_id: str, payment_intent: str, amount: int) -> dict:
    """Build a ``charge.dispute.created`` event disputing ``amount`` cents of a charge.

    Args:
        event_id: Stripe event id (the idempotency key the handler records).
        payment_intent: The PaymentIntent behind the disputed charge.
        amount: Disputed cents (one credit per cent).

    Returns:
        An event dict shaped like the fields the dispute handler reads.
    """
    return {
        "id": event_id,
        "type": "charge.dispute.created",
        "data": {"object": {"payment_intent": payment_intent, "amount": amount}},
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


def test_custom_checkout_rejects_out_of_bounds_amount(engine: object) -> None:
    """Amounts outside the custom bounds are a 400 before any Stripe call."""
    service = StripeBillingService(engine=engine)
    for credits in (CUSTOM_CREDITS_MIN - 1, 0, -5, CUSTOM_CREDITS_MAX + 1):
        with pytest.raises(DomainError) as exc:
            service.create_custom_checkout("u@x.com", credits)
        assert exc.value.status_code == 400
        assert exc.value.code == "billing.invalid_amount"


def test_custom_checkout_builds_ad_hoc_price_and_metadata(
    engine: object, configured: None
) -> None:
    """A custom top-up charges credits-as-cents and stamps webhook metadata."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    captured: dict[str, object] = {}

    def _create(**kwargs: object) -> object:
        captured.update(kwargs)
        return type("Obj", (), {"url": "https://stripe.test/c/cs_123"})()

    with patch("stripe.checkout.Session.create", side_effect=_create):
        url = service.create_custom_checkout("u@x.com", 1234)
    assert url == "https://stripe.test/c/cs_123"
    line_items = captured["line_items"]
    assert isinstance(line_items, list)
    price_data = line_items[0]["price_data"]
    assert price_data["unit_amount"] == 1234
    assert price_data["currency"] == "usd"
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["credits"] == "1234"
    assert metadata["pack_id"] == "custom"
    assert metadata["username"] == "u@x.com"


def _deliver(service: StripeBillingService, event: dict) -> None:
    """Deliver a prebuilt event dict through the webhook with signature verification stubbed."""
    with patch("stripe.Webhook.construct_event", return_value=event):
        service.handle_webhook(b"{}", "sig")


def _balance(engine: object, username: str) -> int:
    """Read the persisted purchased credit balance for an account (0 when it has no row)."""
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, username)
        return 0 if customer is None else int(customer.credit_balance)


def _ledger_rows(engine: object, username: str, kind: str) -> list[CreditLedgerModel]:
    """Return an account's ledger rows of a given kind, oldest first.

    Args:
        engine: The SQLite engine to read from.
        username: Account whose ledger to read.
        kind: The ``kind`` column value to filter on (e.g. ``"refund"``).

    Returns:
        The matching rows ordered by insertion.
    """
    with Session(engine) as session:
        return (
            session.query(CreditLedgerModel)
            .filter_by(username=username, kind=kind)
            .order_by(CreditLedgerModel.id)
            .all()
        )


def test_webhook_refund_claws_back_credits(engine: object, webhook_ready: None) -> None:
    """A full refund removes the topped-up credits and writes one refund ledger row."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_1"))
    assert _balance(engine, "u@x.com") == 500

    _deliver(service, _refund_event("evt_ref", "pi_1", 500))

    assert _balance(engine, "u@x.com") == 0
    rows = _ledger_rows(engine, "u@x.com", "refund")
    assert len(rows) == 1
    assert rows[0].delta_credits == -500
    assert rows[0].stripe_payment_intent_id == "pi_1"
    assert rows[0].stripe_event_id == "evt_ref"


def test_webhook_partial_refunds_are_incremental(engine: object, webhook_ready: None) -> None:
    """Two partial refunds each claw back only the new slice — the cumulative total is never removed twice."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_1"))

    _deliver(service, _refund_event("evt_ref1", "pi_1", 200))
    assert _balance(engine, "u@x.com") == 300
    # The second event carries the charge's cumulative refunded total, not just the new slice.
    _deliver(service, _refund_event("evt_ref2", "pi_1", 500))
    assert _balance(engine, "u@x.com") == 0

    rows = _ledger_rows(engine, "u@x.com", "refund")
    assert [r.delta_credits for r in rows] == [-200, -300]


def test_webhook_refund_clamps_to_spent_balance(engine: object, webhook_ready: None) -> None:
    """A refund of credits already spent claws back only what remains, never below zero."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_1"))
    # The buyer spent 300 of the 500 before the refund lands.
    with Session(engine) as session:
        session.get(BillingCustomerModel, "u@x.com").credit_balance = 200
        session.commit()

    _deliver(service, _refund_event("evt_ref", "pi_1", 500))

    assert _balance(engine, "u@x.com") == 0
    rows = _ledger_rows(engine, "u@x.com", "refund")
    assert len(rows) == 1
    assert rows[0].delta_credits == -200


def test_webhook_refund_is_idempotent_on_redelivery(engine: object, webhook_ready: None) -> None:
    """A redelivered refund event claws back exactly once."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_1"))

    event = _refund_event("evt_ref", "pi_1", 500)
    _deliver(service, event)
    _deliver(service, event)

    assert _balance(engine, "u@x.com") == 0
    assert len(_ledger_rows(engine, "u@x.com", "refund")) == 1


def test_webhook_refund_for_unknown_payment_intent_is_noop(engine: object, webhook_ready: None) -> None:
    """A refund for a charge Skynet never credited touches no balance and writes no row."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)

    _deliver(service, _refund_event("evt_ref", "pi_unknown", 500))

    assert _balance(engine, "u@x.com") == 0
    assert _ledger_rows(engine, "u@x.com", "refund") == []


def test_webhook_dispute_claws_back_credits(engine: object, webhook_ready: None) -> None:
    """A chargeback removes the disputed credits and writes one dispute ledger row."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_2"))

    _deliver(service, _dispute_event("evt_dis", "pi_2", 500))

    assert _balance(engine, "u@x.com") == 0
    rows = _ledger_rows(engine, "u@x.com", "dispute")
    assert len(rows) == 1
    assert rows[0].delta_credits == -500
    assert rows[0].description == "Chargeback"
    assert rows[0].stripe_payment_intent_id == "pi_2"


def test_webhook_dispute_after_partial_refund_nets(engine: object, webhook_ready: None) -> None:
    """A dispute after a partial refund claws back only the still-unreversed remainder."""
    _seed_customer(engine, "u@x.com")
    service = StripeBillingService(engine=engine)
    _deliver(service, _checkout_event("evt_pay", "u@x.com", 500, payment_intent="pi_1"))
    _deliver(service, _refund_event("evt_ref", "pi_1", 200))
    assert _balance(engine, "u@x.com") == 300

    _deliver(service, _dispute_event("evt_dis", "pi_1", 500))

    assert _balance(engine, "u@x.com") == 0
    dispute_rows = _ledger_rows(engine, "u@x.com", "dispute")
    assert len(dispute_rows) == 1
    assert dispute_rows[0].delta_credits == -300
