"""Tests for optional privacy-safe Sentry initialization."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from core import error_reporting
from core.config import settings


@pytest.fixture(autouse=True)
def _reset_error_reporting() -> None:
    """Reset module configuration state before each test."""
    error_reporting._configured = False


def test_configure_is_disabled_without_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset DSN leaves the SDK inactive."""
    monkeypatch.setattr(settings, "sentry_dsn", None)

    assert error_reporting.configure_error_reporting("backend") is False


def test_configure_uses_private_low_volume_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sentry initializes without request bodies, locals, or default PII."""
    monkeypatch.setattr(settings, "sentry_dsn", SecretStr("https://key@example.invalid/1"))
    monkeypatch.setattr(settings, "sentry_environment", "production")
    monkeypatch.setattr(settings, "sentry_traces_sample_rate", 0.05)
    captured: dict[str, object] = {}

    def fake_init(**kwargs: object) -> None:
        """Capture Sentry initialization options."""
        captured.update(kwargs)

    monkeypatch.setattr(error_reporting.sentry_sdk, "init", fake_init)
    monkeypatch.setattr(error_reporting.sentry_sdk, "set_tag", lambda *args: None)

    assert error_reporting.configure_error_reporting("worker") is True
    assert captured["dsn"] == "https://key@example.invalid/1"
    assert captured["send_default_pii"] is False
    assert captured["include_local_variables"] is False
    assert captured["max_request_body_size"] == "never"
    assert captured["traces_sample_rate"] == 0.05
    assert captured["before_send"] is error_reporting._scrub_event


def test_scrub_event_removes_sensitive_request_fields() -> None:
    """The final transport filter strips identity and request secrets."""
    event = {
        "user": {"email": "alice@example.com"},
        "request": {
            "url": "https://skynetml.com/account",
            "headers": {"authorization": "secret"},
            "cookies": {"session": "secret"},
            "data": {"prompt": "private"},
            "query_string": "token=secret",
        },
    }

    scrubbed = error_reporting._scrub_event(event, {})

    assert "user" not in scrubbed
    assert scrubbed["request"] == {"url": "https://skynetml.com/account"}


def test_capture_is_noop_until_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Captures are skipped when an operator has not supplied a DSN."""
    called = False

    def fake_capture(exc: BaseException) -> None:
        """Record an unexpected SDK call."""
        nonlocal called
        called = True

    monkeypatch.setattr(error_reporting.sentry_sdk, "capture_exception", fake_capture)

    error_reporting.capture_exception(RuntimeError("boom"))

    assert called is False
