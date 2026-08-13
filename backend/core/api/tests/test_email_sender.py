"""Tests for the shared SMTP email transport."""

from __future__ import annotations

from email.message import EmailMessage
from types import TracebackType

import pytest
from pydantic import SecretStr

import core.api.email_sender as email_sender_module
from core.api.email_sender import send_email
from core.config import settings


class FakeSmtp:
    """Capture SMTP session behavior without opening a network connection."""

    instances: list[FakeSmtp] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        """Record connection parameters for assertions.

        Args:
            host: SMTP relay hostname.
            port: SMTP relay port.
            timeout: Connection timeout in seconds.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_called = False
        self.login_args: tuple[str, str] | None = None
        self.message: EmailMessage | None = None
        self.instances.append(self)

    def __enter__(self) -> FakeSmtp:
        """Enter the fake SMTP context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the fake SMTP context."""

    def starttls(self) -> None:
        """Record the STARTTLS upgrade."""
        self.starttls_called = True

    def login(self, username: str, password: str) -> None:
        """Record SMTP authentication credentials.

        Args:
            username: SMTP username.
            password: SMTP password.
        """
        self.login_args = (username, password)

    def send_message(self, message: EmailMessage) -> None:
        """Capture the composed message.

        Args:
            message: Email message submitted to the relay.
        """
        self.message = message


def test_send_email_composes_plain_and_html_alternatives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose a rich notification while preserving a plain-text fallback."""
    FakeSmtp.instances.clear()
    monkeypatch.setattr(email_sender_module.smtplib, "SMTP", FakeSmtp)
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_username", "mailer@example.com")
    monkeypatch.setattr(settings, "smtp_password", SecretStr("secret"))
    monkeypatch.setattr(settings, "smtp_from", "Skynet <no-reply@example.com>")
    monkeypatch.setattr(settings, "smtp_starttls", True)

    send_email("user@example.com", "Run complete", "Run complete", html_body="<p>Run complete</p>")

    smtp = FakeSmtp.instances[-1]
    assert (smtp.host, smtp.port, smtp.timeout) == ("smtp.example.com", 587, 15)
    assert smtp.starttls_called is True
    assert smtp.login_args == ("mailer@example.com", "secret")
    assert smtp.message is not None
    assert smtp.message["To"] == "user@example.com"
    assert smtp.message["From"] == "Skynet <no-reply@example.com>"
    assert smtp.message["Subject"] == "Run complete"
    assert smtp.message.get_body(preferencelist=("plain",)).get_content().strip() == "Run complete"
    assert smtp.message.get_body(preferencelist=("html",)).get_content().strip() == "<p>Run complete</p>"
