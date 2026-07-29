"""Outbound SMTP mail for account-security flows.

Transport-only: callers gate on :func:`email_configured` and translate
delivery failures into typed :class:`~core.api.errors.DomainError`\\ s. Kept
separate from :mod:`core.notifications.comms` (the Windows-only Outlook COM
path) because sign-in codes must deliver from the hosted Linux deployment.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from ..config import settings


def email_configured() -> bool:
    """Return whether an SMTP relay is configured for outbound mail."""
    return bool(settings.smtp_host)


def send_email(to: str, subject: str, body: str) -> None:
    """Send a plain-text email through the configured SMTP relay.

    Args:
        to: Recipient address.
        subject: Message subject line.
        body: Plain-text message body.

    Raises:
        RuntimeError: When no SMTP host is configured.
        OSError, smtplib.SMTPException: On connection or delivery failure.
    """
    if not settings.smtp_host:
        raise RuntimeError("SMTP_HOST is not configured")
    message = EmailMessage()
    message["From"] = settings.smtp_from or settings.smtp_username or "no-reply@skynet.local"
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_username and settings.smtp_password:
            smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
        smtp.send_message(message)
