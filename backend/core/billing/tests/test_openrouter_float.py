"""Tests for the OpenRouter float monitor and its Stripe-webhook trigger.

Covers the balance read (dollars → credits, fail-open on every HTTP or shape
error), the floor check (disabled when the floor is non-positive or the key is
unset, warns only below the floor), the outstanding-liability aggregate, and the
post-commit webhook hook (fires only on a freshly-applied checkout, never
double-runs on redelivery, and can never break the money path). Each test uses
an in-memory SQLite engine and patches ``httpx``/``stripe`` so no network call is
made.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.billing.openrouter_float import (
    FloatStatus,
    check_float,
    read_account_balance_credits,
)
from core.billing.service import StripeBillingService
from core.config import settings
from core.storage.models import Base, BillingCustomerModel

_MONITOR_LOGGER = "skynet.billing.openrouter_float"


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def monitor_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the monitor: a master key plus a $10 (1000-credit) floor."""
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("sk-or-master"))
    monkeypatch.setattr(settings, "openrouter_balance_floor_credits", 1000)


@pytest.fixture
def stripe_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the Stripe secret and webhook signing secret for webhook tests."""
    monkeypatch.setattr(settings, "stripe_secret_key", SecretStr("sk_test_dummy"))
    monkeypatch.setattr(settings, "stripe_webhook_secret", SecretStr("whsec_test"))


def _credits_response(total_credits: float, total_usage: float, status_code: int = 200) -> object:
    """Build a stand-in httpx response for ``GET /api/v1/credits``.

    Args:
        total_credits: Dollars ever added to the master account.
        total_usage: Dollars ever spent from it.
        status_code: HTTP status the fake response reports.

    Returns:
        An object exposing the ``status_code`` and ``json`` attributes the reader
        touches.
    """
    body = {"data": {"total_credits": total_credits, "total_usage": total_usage}}
    return SimpleNamespace(status_code=status_code, json=lambda: body)


def _checkout_event(event_id: str, username: str, credits: int) -> dict:
    """Build a minimal verified ``checkout.session.completed`` event.

    Args:
        event_id: Stripe event id (the idempotency key the handler records).
        username: Buyer the credits land on.
        credits: Credit quantity the pack grants.

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
                "metadata": {"username": username, "credits": str(credits), "pack_id": "starter"},
            }
        },
    }


def test_float_status_covered_boundary() -> None:
    """``covered`` is true at or above the floor, false one credit below it."""
    assert FloatStatus(balance_credits=1000, floor_credits=1000, liability_credits=0).covered
    assert FloatStatus(balance_credits=1500, floor_credits=1000, liability_credits=0).covered
    assert not FloatStatus(balance_credits=999, floor_credits=1000, liability_credits=0).covered


def test_read_balance_converts_dollars_to_credits(monitor_on: None) -> None:
    """A healthy read returns ``(total_credits - total_usage)`` in credits."""
    with patch("httpx.get", return_value=_credits_response(20.0, 5.5)):
        assert read_account_balance_credits() == 1450


def test_read_balance_none_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no master key configured the reader returns None and makes no request."""
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    get = Mock()
    with patch("httpx.get", get):
        assert read_account_balance_credits() is None
    get.assert_not_called()


def test_read_balance_none_on_network_error(monitor_on: None) -> None:
    """A transport error fails open to None rather than raising."""
    with patch("httpx.get", side_effect=httpx.ConnectError("boom")):
        assert read_account_balance_credits() is None


def test_read_balance_none_on_non_2xx(monitor_on: None) -> None:
    """A non-2xx status fails open to None."""
    with patch("httpx.get", return_value=_credits_response(20.0, 5.0, status_code=500)):
        assert read_account_balance_credits() is None


def test_read_balance_none_on_unexpected_shape(monitor_on: None) -> None:
    """A 200 whose body lacks the expected keys fails open to None."""
    response = SimpleNamespace(status_code=200, json=lambda: {"data": {}})
    with patch("httpx.get", return_value=response):
        assert read_account_balance_credits() is None


def test_read_balance_none_when_body_not_object(monitor_on: None) -> None:
    """A 200 whose JSON is not an object (no ``.get``) fails open to None."""
    response = SimpleNamespace(status_code=200, json=lambda: ["unexpected"])
    with patch("httpx.get", return_value=response):
        assert read_account_balance_credits() is None


def test_check_float_disabled_when_floor_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A floor of 0 disables the monitor: no balance read, no status."""
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("sk-or-master"))
    monkeypatch.setattr(settings, "openrouter_balance_floor_credits", 0)
    get = Mock()
    with patch("httpx.get", get):
        assert check_float(5000) is None
    get.assert_not_called()


def test_check_float_none_when_balance_unreadable(monitor_on: None) -> None:
    """A configured floor but an unreadable balance yields None (nothing to compare)."""
    with patch("httpx.get", return_value=_credits_response(0.0, 0.0, status_code=503)):
        assert check_float(5000) is None


def test_check_float_warns_below_floor(monitor_on: None, caplog: pytest.LogCaptureFixture) -> None:
    """A balance under the floor logs a WARNING and returns an uncovered status."""
    with (
        caplog.at_level(logging.WARNING, logger=_MONITOR_LOGGER),
        patch("httpx.get", return_value=_credits_response(5.0, 0.0)),
    ):
        status = check_float(outstanding_credits=3000)
    assert status is not None
    assert not status.covered
    assert status.balance_credits == 500
    assert status.liability_credits == 3000
    assert "OpenRouter float low" in caplog.text


def test_check_float_quiet_when_covered(monitor_on: None, caplog: pytest.LogCaptureFixture) -> None:
    """A balance at or above the floor returns a covered status and logs no warning."""
    with (
        caplog.at_level(logging.WARNING, logger=_MONITOR_LOGGER),
        patch("httpx.get", return_value=_credits_response(15.0, 0.0)),
    ):
        status = check_float(outstanding_credits=3000)
    assert status is not None
    assert status.covered
    assert status.balance_credits == 1500
    assert "OpenRouter float low" not in caplog.text


def test_total_outstanding_credits_zero_without_customers(engine: object) -> None:
    """No customers means no liability."""
    assert StripeBillingService(engine=engine).total_outstanding_credits() == 0


def test_total_outstanding_credits_sums_balances_and_grants(engine: object) -> None:
    """Liability is every account's paid balance plus remaining grant, NULL grants coalesced."""
    with Session(engine) as session:
        session.add_all(
            [
                BillingCustomerModel(username="a@x.com", stripe_customer_id="cus_a", credit_balance=500, grant_remaining=100),
                BillingCustomerModel(username="b@x.com", stripe_customer_id="cus_b", credit_balance=2000, grant_remaining=None),
            ]
        )
        session.commit()
    assert StripeBillingService(engine=engine).total_outstanding_credits() == 2600


def test_webhook_triggers_monitor_and_still_credits(
    engine: object, stripe_ready: None, monitor_on: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A paid checkout credits the buyer and runs the float check against the new liability."""
    service = StripeBillingService(engine=engine)
    event = _checkout_event("evt_float", "u@x.com", 500)
    with (
        caplog.at_level(logging.WARNING, logger=_MONITOR_LOGGER),
        patch("stripe.Webhook.construct_event", return_value=event),
        patch("httpx.get", return_value=_credits_response(2.0, 0.0)),
    ):
        service.handle_webhook(b"{}", "sig")
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "u@x.com").credit_balance == 500
    assert "OpenRouter float low" in caplog.text


def test_webhook_survives_monitor_failure(engine: object, stripe_ready: None) -> None:
    """A monitor that raises never breaks the webhook — the credit is already committed."""
    service = StripeBillingService(engine=engine)
    event = _checkout_event("evt_boom", "u@x.com", 500)
    with (
        patch("stripe.Webhook.construct_event", return_value=event),
        patch("core.billing.service.check_float", side_effect=RuntimeError("monitor down")),
    ):
        service.handle_webhook(b"{}", "sig")
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "u@x.com").credit_balance == 500


def test_webhook_skips_monitor_for_non_checkout_event(engine: object, stripe_ready: None) -> None:
    """A non-checkout event records normally but never runs the float monitor."""
    service = StripeBillingService(engine=engine)
    event = {"id": "evt_sub", "type": "customer.subscription.updated", "data": {"object": {}}}
    monitor = Mock()
    with (
        patch("stripe.Webhook.construct_event", return_value=event),
        patch("core.billing.service.check_float", monitor),
    ):
        service.handle_webhook(b"{}", "sig")
    monitor.assert_not_called()


def test_webhook_monitor_runs_once_on_redelivery(engine: object, stripe_ready: None) -> None:
    """A redelivered checkout credits once and runs the monitor once (fresh-apply path only)."""
    service = StripeBillingService(engine=engine)
    event = _checkout_event("evt_dup", "u@x.com", 500)
    monitor = Mock()
    with (
        patch("stripe.Webhook.construct_event", return_value=event),
        patch("core.billing.service.check_float", monitor),
    ):
        service.handle_webhook(b"{}", "sig")
        service.handle_webhook(b"{}", "sig")
    assert monitor.call_count == 1
    with Session(engine) as session:
        assert session.get(BillingCustomerModel, "u@x.com").credit_balance == 500
