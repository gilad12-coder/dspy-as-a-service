"""Tests for passwordless local login and ADFS account provisioning."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import UserModel
from ..routers import accounts as accounts_module
from ..routers.accounts import create_accounts_router


@pytest.fixture
def accounts_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, object]]:
    """Build an isolated account router backed by SQLite."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    UserModel.__table__.create(engine)
    monkeypatch.setattr(accounts_module.settings, "backend_auth_secret", SecretStr("internal-test"))
    monkeypatch.setattr(accounts_module.settings, "admin_usernames", "bootstrap")
    store = SimpleNamespace(engine=engine)
    app = FastAPI()
    app.include_router(create_accounts_router(job_store=store))
    with TestClient(app) as client:
        yield client, engine
    engine.dispose()


def _headers() -> dict[str, str]:
    """Return the trusted frontend header."""
    return {"X-Internal-Auth": "internal-test"}


def test_unknown_local_username_is_rejected(accounts_client: tuple[TestClient, object]) -> None:
    """Reject a local username that an administrator has not approved."""
    client, _engine = accounts_client

    response = client.post("/auth/local/login", headers=_headers(), json={"username": "unknown"})

    assert response.status_code == 401


def test_environment_admin_is_created_on_first_login(accounts_client: tuple[TestClient, object]) -> None:
    """Create the environment bootstrap administrator lazily."""
    client, engine = accounts_client

    response = client.post("/auth/local/login", headers=_headers(), json={"username": "Bootstrap"})

    assert response.status_code == 200
    assert response.json() == {
        "username": "bootstrap",
        "display_name": "bootstrap",
        "role": "admin",
    }
    with Session(engine) as session:
        row = session.get(UserModel, "bootstrap")
        assert row is not None
        assert row.local_enabled is True
        assert row.is_admin is True


def test_adfs_auto_provisions_normalized_identity(accounts_client: tuple[TestClient, object]) -> None:
    """Provision valid ADFS identities without a prior local allowlist entry."""
    client, engine = accounts_client

    response = client.post(
        "/auth/sso/provision",
        headers=_headers(),
        json={"username": " Alice@Example.COM ", "display_name": "Alice", "external_admin": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "username": "alice@example.com",
        "display_name": "Alice",
        "role": "admin",
    }
    with Session(engine) as session:
        row = session.get(UserModel, "alice@example.com")
        assert row is not None
        assert row.local_enabled is False
        assert row.is_admin is False
        assert row.adfs_seen_at is not None


def test_authentication_routes_require_internal_secret(
    accounts_client: tuple[TestClient, object],
) -> None:
    """Reject calls that do not originate from the trusted frontend server."""
    client, _engine = accounts_client

    local = client.post("/auth/local/login", json={"username": "bootstrap"})
    sso = client.post("/auth/sso/provision", json={"username": "alice"})

    assert local.status_code == 403
    assert sso.status_code == 403
