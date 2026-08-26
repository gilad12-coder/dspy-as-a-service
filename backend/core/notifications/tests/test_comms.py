"""Tests for the ``core.notifications.comms`` SMTP transport."""

from __future__ import annotations

import pytest

import core.notifications.comms as comms_module
from core.notifications.comms import resolve_email, send_mail


def test_resolve_email_returns_normalized_registered_address() -> None:
    """Normalize a registered email identity for notification delivery."""
    assert resolve_email(" Alice@Example.COM ") == "alice@example.com"


@pytest.mark.parametrize("identity", ["alice", "@example.com", "alice@", "alice@@example.com", ""])
def test_resolve_email_rejects_non_email_identity(identity: str) -> None:
    """Skip a legacy non-email identity without attempting delivery."""
    assert resolve_email(identity) is None


def test_send_mail_passes_html_to_shared_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the rendered notification through the shared SMTP transport."""
    calls: list[tuple[str, str, str, str | None]] = []

    def _send(to: str, subject: str, body: str, *, html_body: str | None = None) -> None:
        """Record the shared SMTP sender invocation."""
        calls.append((to, subject, body, html_body))

    monkeypatch.setattr(comms_module, "send_email", _send)

    assert send_mail("alice@example.com", "hello", "<p>hi</p>") is True
    assert calls == [("alice@example.com", "hello", "hello", "<p>hi</p>")]


def test_send_mail_returns_false_on_smtp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swallow an SMTP failure so notifications cannot break app work."""

    def _fail(to: str, subject: str, body: str, *, html_body: str | None = None) -> None:
        """Raise a representative transport failure."""
        raise OSError("SMTP boom")

    monkeypatch.setattr(comms_module, "send_email", _fail)

    assert send_mail("alice@example.com", "hello", "<p>hi</p>") is False


def test_send_mail_logs_warning_on_smtp_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log the SMTP failure for production alerting."""

    def _fail(to: str, subject: str, body: str, *, html_body: str | None = None) -> None:
        """Raise a representative transport failure."""
        raise OSError("SMTP boom")

    monkeypatch.setattr(comms_module, "send_email", _fail)

    with caplog.at_level("WARNING", logger="core.notifications.comms"):
        send_mail("alice@example.com", "hello", "<p>hi</p>")

    assert "Failed to send notification email" in caplog.text
    assert "SMTP boom" in caplog.text
