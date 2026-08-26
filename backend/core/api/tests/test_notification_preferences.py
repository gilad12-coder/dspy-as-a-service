"""Tests for caller-scoped email-notification preference endpoints."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import Base, NotificationPreferenceModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..routers.notification_preferences import create_notification_preferences_router

_USER = "oauth-user@example.com"


def _harness() -> tuple[TestClient, object]:
    """Build an authenticated preference client over an in-memory database.

    Returns:
        Client and shared SQLAlchemy engine.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.include_router(create_notification_preferences_router(job_store=SimpleNamespace(engine=engine)))
    app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
        username=_USER,
        role="user",
        groups=(),
    )
    return TestClient(app), engine


def test_missing_row_returns_enabled_defaults_without_creating_data() -> None:
    """A new identity starts enabled without requiring a local account row."""
    client, engine = _harness()

    response = client.get("/account/notification-preferences")

    assert response.status_code == 200
    assert response.json() == {
        "job_updates_enabled": True,
        "sharing_updates_enabled": True,
    }
    with Session(engine) as session:
        assert session.get(NotificationPreferenceModel, _USER) is None


def test_patch_persists_one_category_and_preserves_the_other() -> None:
    """A partial update stores the changed switch and keeps defaults intact."""
    client, engine = _harness()

    response = client.patch(
        "/account/notification-preferences",
        json={"job_updates_enabled": False},
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_updates_enabled": False,
        "sharing_updates_enabled": True,
    }
    assert client.get("/account/notification-preferences").json() == response.json()
    with Session(engine) as session:
        row = session.get(NotificationPreferenceModel, _USER)
        assert row is not None
        assert row.job_updates_enabled is False
        assert row.sharing_updates_enabled is True
