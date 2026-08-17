"""Tests for the OpenRouter float monitor and its Stripe-webhook trigger.

Covers the balance read (dollars → credits, fail-open on every HTTP or shape
error), the floor check (disabled when the floor is non-positive or the key is
unset, warns only below the floor), the low-float notification fan-out (webhook
+ email behind a cooldown that prefers Redis), the periodic sweeper (disabled
unless fully configured, swallows failures), the outstanding-liability
aggregate, and the post-commit webhook hook (fires only on a freshly-applied
checkout, never double-runs on redelivery, and can never break the money path).
Each test uses an in-memory SQLite engine and patches ``httpx``/``stripe``/SMTP
so no network call is made.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.billing import openrouter_float
from core.billing.openrouter_float import (
    FloatStatus,
    OpenRouterFloatSweeper,
    check_float,
    notify_low_float,
    read_account_balance_credits,
    start_openrouter_float_sweeper,
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
def notify_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-local cooldown and configure both notification channels.

    Redis is forced off so the cooldown exercises the per-process fallback;
    individual tests override ``shared_redis_client`` when they need the shared
    path.
    """
    monkeypatch.setattr(openrouter_float, "_local_cooldown_until", 0.0)
    monkeypatch.setattr(openrouter_float, "shared_redis_client", lambda: None)
    monkeypatch.setattr(settings, "openrouter_float_alert_cooldown_seconds", 3600.0)
    monkeypatch.setattr(settings, "openrouter_float_alert_email", "ops@example.com")
    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.example.com/x")
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")


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


def test_check_float_notifies_below_floor(monitor_on: None) -> None:
    """A breach hands the uncovered status to the notifier; a covered read does not."""
    with (
        patch.object(openrouter_float, "notify_low_float") as notify,
        patch("httpx.get", return_value=_credits_response(5.0, 0.0)),
    ):
        status = check_float(outstanding_credits=3000)
    notify.assert_called_once_with(status)
    with (
        patch.object(openrouter_float, "notify_low_float") as notify,
        patch("httpx.get", return_value=_credits_response(15.0, 0.0)),
    ):
        check_float(outstanding_credits=3000)
    notify.assert_not_called()


def _wait_for(mock: Mock, timeout: float = 2.0) -> None:
    """Block until a mock dispatched on a daemon thread has been called.

    Args:
        mock: The patched callable the background thread invokes.
        timeout: Seconds to wait before giving up.
    """
    deadline = time.monotonic() + timeout
    while not mock.called and time.monotonic() < deadline:
        time.sleep(0.01)


def test_notify_low_float_fans_out_to_webhook_and_email(notify_ready: None) -> None:
    """A breach posts to the alert webhook and emails the operator with the figures."""
    status = FloatStatus(balance_credits=500, floor_credits=1500, liability_credits=3000)
    with (
        patch.object(openrouter_float, "send_alert") as alert,
        patch.object(openrouter_float, "send_email") as email,
    ):
        assert notify_low_float(status) is True
        _wait_for(email)
    alert.assert_called_once()
    assert alert.call_args.kwargs["level"] == "WARNING"
    assert "$5.00" in alert.call_args.args[0]
    assert "$15.00" in alert.call_args.args[0]
    email.assert_called_once()
    to, subject, body = email.call_args.args
    assert to == "ops@example.com"
    assert "OpenRouter float low" in subject
    assert "$30.00" in body
    assert "openrouter.ai/settings/credits" in body


def test_notify_low_float_skips_email_when_unconfigured(notify_ready: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """No recipient (or no SMTP host) means no email thread; the webhook still fires."""
    status = FloatStatus(balance_credits=500, floor_credits=1500, liability_credits=0)
    monkeypatch.setattr(settings, "openrouter_float_alert_email", "")
    with (
        patch.object(openrouter_float, "send_alert") as alert,
        patch.object(openrouter_float, "send_email") as email,
    ):
        assert notify_low_float(status) is True
        time.sleep(0.05)
    alert.assert_called_once()
    email.assert_not_called()


def test_notify_low_float_honours_process_cooldown(notify_ready: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Redis a second breach inside the cooldown is dropped; 0 disables the gate."""
    status = FloatStatus(balance_credits=500, floor_credits=1500, liability_credits=0)
    with patch.object(openrouter_float, "send_alert") as alert:
        assert notify_low_float(status, now=100.0) is True
        assert notify_low_float(status, now=200.0) is False
        assert notify_low_float(status, now=100.0 + 3600.0) is True
    assert alert.call_count == 2
    monkeypatch.setattr(settings, "openrouter_float_alert_cooldown_seconds", 0.0)
    with patch.object(openrouter_float, "send_alert") as alert:
        assert notify_low_float(status, now=1.0) is True
        assert notify_low_float(status, now=1.0) is True
    assert alert.call_count == 2


def test_notify_low_float_prefers_redis_cooldown(notify_ready: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With Redis available the shared ``SET NX EX`` decides; a held key suppresses the send."""
    status = FloatStatus(balance_credits=500, floor_credits=1500, liability_credits=0)
    client = Mock()
    client.set.return_value = False
    monkeypatch.setattr(openrouter_float, "shared_redis_client", lambda: client)
    with patch.object(openrouter_float, "send_alert") as alert:
        assert notify_low_float(status) is False
    alert.assert_not_called()
    client.set.assert_called_once_with(openrouter_float._ALERT_COOLDOWN_REDIS_KEY, "1", nx=True, ex=3600)
    client.set.return_value = True
    with patch.object(openrouter_float, "send_alert") as alert:
        assert notify_low_float(status) is True
    alert.assert_called_once()


def test_notify_low_float_never_raises(notify_ready: None) -> None:
    """A failing webhook send is swallowed and reported as not-notified."""
    status = FloatStatus(balance_credits=500, floor_credits=1500, liability_credits=0)
    with patch.object(openrouter_float, "send_alert", side_effect=RuntimeError("boom")):
        assert notify_low_float(status) is False


def test_start_sweeper_requires_full_configuration(monitor_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The periodic sweeper only starts with a key, a positive floor and a positive interval."""
    monkeypatch.setattr(settings, "openrouter_float_check_interval_seconds", 0.0)
    assert start_openrouter_float_sweeper(None, lambda: 0) is None
    monkeypatch.setattr(settings, "openrouter_float_check_interval_seconds", 900.0)
    monkeypatch.setattr(settings, "openrouter_balance_floor_credits", 0)
    assert start_openrouter_float_sweeper(None, lambda: 0) is None
    monkeypatch.setattr(settings, "openrouter_balance_floor_credits", 1000)
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    assert start_openrouter_float_sweeper(None, lambda: 0) is None
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("sk-or-master"))
    with patch.object(OpenRouterFloatSweeper, "start") as start:
        sweeper = start_openrouter_float_sweeper(None, lambda: 0)
    assert isinstance(sweeper, OpenRouterFloatSweeper)
    start.assert_called_once()


def test_sweeper_sweep_once_runs_check(monitor_on: None, engine: object) -> None:
    """On a non-Postgres engine one sweep reads the balance against the injected liability."""
    sweeper = OpenRouterFloatSweeper(engine, lambda: 3000, interval_seconds=1.0)
    with (
        patch.object(openrouter_float, "notify_low_float"),
        patch("httpx.get", return_value=_credits_response(5.0, 0.0)),
    ):
        status = sweeper.sweep_once()
    assert status is not None
    assert status.balance_credits == 500
    assert status.liability_credits == 3000
    assert sweeper._interval_seconds == 60.0


def test_sweeper_sweep_once_swallows_failures(monitor_on: None, engine: object) -> None:
    """A liability read that raises yields None instead of killing the loop."""

    def _boom() -> int:
        raise RuntimeError("db down")

    sweeper = OpenRouterFloatSweeper(engine, _boom)
    assert sweeper.sweep_once() is None


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
