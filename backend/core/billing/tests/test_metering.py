"""Tests for ``core.billing.metering.meter_llm_run`` — the interactive-run seam.

Each test stands up an in-memory SQLite engine with the billing tables and
drives the helper with fake LMs shaped like ``dspy.LM`` history carriers, the
same contract ``usage_by_model_from_history`` reads in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.billing.metering import estimate_run_credits, meter_llm_run
from core.storage.models import Base, BillingCustomerModel, CreditLedgerModel


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


class _FakeLm:
    """History-carrying LM double, optionally with a served-model stash."""

    def __init__(
        self,
        history: list[dict[str, Any]],
        model: str = "openrouter/test/unpriced",
        served: str | None = None,
    ) -> None:
        """Store the canned history, model id, and optional served-model stash.

        Args:
            history: Entries shaped like ``dspy.LM.history`` rows.
            model: Model id the LM reports.
            served: Concrete model to stash as ``last_response_model``.
        """
        self.history = history
        self.model = model
        if served is not None:
            self.last_request_model = model
            self.last_response_model = served


def _ledger_rows(engine: object) -> list[CreditLedgerModel]:
    """Return every credit-ledger row, oldest first."""
    with Session(engine) as session:
        return session.query(CreditLedgerModel).order_by(CreditLedgerModel.id).all()


def _fund(engine: object, username: str, credits: int = 100_000) -> None:
    """Seed a billing row with a paid balance so a debit has something to draw.

    The clamped debit charges at most what the account holds, so a test that
    asserts a real charge landed must start from a funded balance.

    Args:
        engine: The SQLite engine to write to.
        username: Account to fund.
        credits: Paid balance to seed.
    """
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username=username,
                stripe_customer_id=f"cus_{username}",
                credit_balance=credits,
                grant_remaining=0,
            )
        )
        session.commit()


def test_meter_llm_run_debits_and_stamps_tokens(engine: object) -> None:
    """A tracked run writes one run row with the measured token counts."""
    _fund(engine, "alice@x.io")
    lm = _FakeLm([{"usage": {"prompt_tokens": 100_000, "completion_tokens": 40_000}}])
    credits = meter_llm_run(engine, "alice@x.io", [lm], description="Agent chat")
    assert credits > 0
    rows = _ledger_rows(engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == "run"
    assert row.description == "Agent chat"
    assert row.model == "openrouter/test/unpriced"
    assert row.delta_credits == -credits
    assert row.input_tokens == 100_000
    assert row.output_tokens == 40_000
    with Session(engine) as session:
        customer = session.get(BillingCustomerModel, "alice@x.io")
        assert customer is not None
        assert int(customer.credit_balance) == 100_000 - credits


def test_meter_llm_run_rekeys_auto_route_to_served_model(engine: object) -> None:
    """An auto-routed run's charge lands on the concrete model the router served."""
    _fund(engine, "alice@x.io")
    lm = _FakeLm(
        [{"usage": {"prompt_tokens": 50_000, "completion_tokens": 10_000}}],
        model="litellm_proxy/openrouter/auto-beta",
        served="google/gemini-2.5-flash-lite",
    )
    assert meter_llm_run(engine, "alice@x.io", [lm], description="Agent chat") > 0
    (row,) = _ledger_rows(engine)
    assert row.model == "google/gemini-2.5-flash-lite"


def test_meter_llm_run_strips_proxy_prefix(engine: object) -> None:
    """An explicit pick behind the managed proxy books under its catalog id."""
    _fund(engine, "alice@x.io")
    lm = _FakeLm(
        [{"usage": {"prompt_tokens": 50_000, "completion_tokens": 10_000}}],
        model="litellm_proxy/openrouter/test/unpriced",
    )
    assert meter_llm_run(engine, "alice@x.io", [lm], description="Code interview") > 0
    (row,) = _ledger_rows(engine)
    assert row.model == "openrouter/test/unpriced"


def test_meter_llm_run_skips_untracked_usage(engine: object) -> None:
    """No usage info anywhere bills nothing and writes no row."""
    assert meter_llm_run(engine, "alice@x.io", [_FakeLm([{"response": "hi"}])], description="x") == 0
    assert meter_llm_run(engine, "alice@x.io", [], description="x") == 0
    assert meter_llm_run(engine, "alice@x.io", [None], description="x") == 0
    assert _ledger_rows(engine) == []


def test_meter_llm_run_skips_without_engine_or_username(engine: object) -> None:
    """A store with no engine, or an anonymous caller, meters nothing."""
    lm = _FakeLm([{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}])
    assert meter_llm_run(None, "alice@x.io", [lm], description="x") == 0
    assert meter_llm_run(engine, "", [lm], description="x") == 0
    assert _ledger_rows(engine) == []


def test_estimate_run_credits_prices_without_debiting(engine: object) -> None:
    """The in-flight estimate matches what a debit would charge and writes nothing."""
    _fund(engine, "alice@x.io")
    history = [{"usage": {"prompt_tokens": 100_000, "completion_tokens": 40_000}}]
    estimate = estimate_run_credits([_FakeLm(history)])
    assert estimate > 0
    assert _ledger_rows(engine) == []
    assert estimate == meter_llm_run(engine, "alice@x.io", [_FakeLm(history)], description="x")


def test_estimate_run_credits_applies_byok_platform_fee() -> None:
    """The live credit watch prices BYOK usage at the same reduced source rate."""
    history = [{"usage": {"prompt_tokens": 100_000, "completion_tokens": 40_000}}]
    managed = estimate_run_credits([_FakeLm(history)], "managed")
    byok = estimate_run_credits([_FakeLm(history)], "byok")
    assert 0 < byok < managed


def test_estimate_run_credits_handles_empty_and_untracked_lms() -> None:
    """No LMs or no tracked usage estimates to zero instead of raising."""
    assert estimate_run_credits([]) == 0
    assert estimate_run_credits([None]) == 0
    assert estimate_run_credits([_FakeLm([{"response": "hi"}])]) == 0


def test_meter_llm_run_never_raises_on_billing_failure(engine: object) -> None:
    """A broken billing backend logs and returns 0 — the user's turn already succeeded."""
    lm = _FakeLm([{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}])
    assert meter_llm_run(object(), "alice@x.io", [lm], description="x") == 0
