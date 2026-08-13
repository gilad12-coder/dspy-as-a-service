"""Tests for the database-backed notification preference resolver."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import NotificationPreferenceModel
from ..preferences import (
    configure_notification_preferences,
    notification_category_enabled,
)


def _engine():
    """Create an in-memory engine containing only the preference table."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    NotificationPreferenceModel.__table__.create(engine)
    return engine


def test_missing_preference_row_defaults_both_categories_on() -> None:
    """An identity without overrides keeps historical delivery behavior."""
    engine = _engine()
    configure_notification_preferences(engine)

    assert notification_category_enabled("new@example.com", "job_updates") is True
    assert notification_category_enabled("new@example.com", "sharing_updates") is True


def test_stored_preference_disables_only_selected_category() -> None:
    """Category checks honor independent job and sharing switches."""
    engine = _engine()
    with Session(engine) as session:
        session.add(
            NotificationPreferenceModel(
                username="alice@example.com",
                job_updates_enabled=False,
                sharing_updates_enabled=True,
            )
        )
        session.commit()
    configure_notification_preferences(engine)

    assert notification_category_enabled("alice@example.com", "job_updates") is False
    assert notification_category_enabled("alice@example.com", "sharing_updates") is True
