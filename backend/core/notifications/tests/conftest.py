"""Shared pytest fixtures for the ``core.notifications`` test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ..preferences import configure_notification_preferences


class FakeMail:
    """Capture ``send_mail`` calls in-memory for behaviour assertions."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list[dict] = []

    def send_mail(self, to: str, subject: str, html_body: str) -> bool:
        """Record a ``send_mail`` invocation and pretend it succeeded.

        Args:
            to: Resolved recipient address.
            subject: Mail subject line.
            html_body: Rendered HTML body.

        Returns:
            Always ``True`` to mimic successful SMTP delivery.
        """
        self.calls.append({"to": to, "subject": subject, "html": html_body})
        return True

    @property
    def call_count(self) -> int:
        """Return how many times ``send_mail`` has been invoked."""
        return len(self.calls)

    def last(self) -> dict:
        """Return the most recent recorded call payload."""
        assert self.calls, "no send_mail calls recorded"
        return self.calls[-1]


@pytest.fixture
def fake_mail() -> FakeMail:
    """Return a fresh ``FakeMail`` instance for each test.

    Returns:
        A ``FakeMail`` ready to record ``send_mail`` invocations.
    """
    return FakeMail()


@pytest.fixture(autouse=True)
def _reset_notification_preference_engine() -> Iterator[None]:
    """Keep notifier tests independent from application-factory side effects."""
    configure_notification_preferences(None)
    yield
    configure_notification_preferences(None)
