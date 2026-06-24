"""Tests for email/password account registration, sign-in, and hashing.

The endpoint tests run against a shared in-memory SQLite engine (StaticPool so
the table and rows persist across sessions and the TestClient's worker thread)
and set a known ``BACKEND_AUTH_SECRET`` so the internal-auth gate is exercised
rather than bypassed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ...config import settings
from ...storage.models import UserModel
from ..passwords import hash_password, verify_password
from ..routers.accounts import create_accounts_router

_SECRET = "test-internal-secret"
_AUTH_HEADER = {"X-Internal-Auth": _SECRET}


def test_hash_password_round_trips() -> None:
    """A hashed password verifies, a wrong one does not, and salt randomizes."""
    encoded = hash_password("correct horse battery")
    assert verify_password("correct horse battery", encoded) is True
    assert verify_password("wrong password", encoded) is False
    assert encoded != hash_password("correct horse battery")
    assert encoded.startswith("scrypt$")


def test_verify_password_tolerates_malformed_hash() -> None:
    """A garbage stored value verifies as False instead of raising."""
    assert verify_password("anything", "not-a-real-hash") is False
    assert verify_password("anything", "") is False


class _Store:
    """Minimal store exposing only the SQLAlchemy engine the routes need."""

    def __init__(self, engine: Any) -> None:
        """Hold the engine the accounts router opens sessions on.

        Args:
            engine: A SQLAlchemy engine with the ``users`` table created.
        """
        self.engine = engine


@pytest.fixture
def accounts_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a client for the accounts router over in-memory SQLite.

    Args:
        monkeypatch: Pytest fixture used to set the shared internal secret.

    Returns:
        A ``TestClient`` whose app serves register + login against a shared
        in-memory store holding the ``users`` table.
    """
    monkeypatch.setattr(settings, "backend_auth_secret", SecretStr(_SECRET))
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    UserModel.__table__.create(engine)
    store = _Store(engine)
    app = FastAPI()
    app.state.job_store = store
    app.include_router(create_accounts_router(job_store=store))
    return TestClient(app)


def test_register_then_login_succeeds(accounts_client: TestClient) -> None:
    """A registered account can sign in and identity is the lowercased email."""
    created = accounts_client.post(
        "/auth/register",
        json={"email": "Alice@Example.com", "password": "hunter2hunter", "name": "Alice"},
        headers=_AUTH_HEADER,
    )
    assert created.status_code == 201
    assert created.json() == {"email": "alice@example.com", "name": "Alice", "role": "user"}

    ok = accounts_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "hunter2hunter"},
        headers=_AUTH_HEADER,
    )
    assert ok.status_code == 200
    assert ok.json()["email"] == "alice@example.com"


def test_register_rejects_duplicate_email(accounts_client: TestClient) -> None:
    """Re-registering an existing email is a 409, regardless of casing."""
    body = {"email": "bob@example.com", "password": "longenough1", "name": "Bob"}
    assert accounts_client.post("/auth/register", json=body, headers=_AUTH_HEADER).status_code == 201
    dupe = accounts_client.post(
        "/auth/register",
        json={"email": "BOB@example.com", "password": "longenough1"},
        headers=_AUTH_HEADER,
    )
    assert dupe.status_code == 409


def test_register_validates_email_and_password(accounts_client: TestClient) -> None:
    """A malformed email or a short password is rejected with 422."""
    bad_email = accounts_client.post(
        "/auth/register",
        json={"email": "not-an-email", "password": "longenough1"},
        headers=_AUTH_HEADER,
    )
    assert bad_email.status_code == 422
    short_pw = accounts_client.post(
        "/auth/register",
        json={"email": "c@example.com", "password": "short"},
        headers=_AUTH_HEADER,
    )
    assert short_pw.status_code == 422


def test_login_rejects_wrong_password_and_unknown_email(accounts_client: TestClient) -> None:
    """Bad password and unknown email both fail with 401."""
    accounts_client.post(
        "/auth/register",
        json={"email": "dora@example.com", "password": "correctpass1"},
        headers=_AUTH_HEADER,
    )
    wrong = accounts_client.post(
        "/auth/login",
        json={"email": "dora@example.com", "password": "wrongpass1"},
        headers=_AUTH_HEADER,
    )
    assert wrong.status_code == 401
    unknown = accounts_client.post(
        "/auth/login",
        json={"email": "ghost@example.com", "password": "whatever12"},
        headers=_AUTH_HEADER,
    )
    assert unknown.status_code == 401


def test_endpoints_require_internal_secret(accounts_client: TestClient) -> None:
    """Calls without the shared internal secret are forbidden."""
    no_secret = accounts_client.post(
        "/auth/register",
        json={"email": "eve@example.com", "password": "longenough1"},
    )
    assert no_secret.status_code == 403
    wrong_secret = accounts_client.post(
        "/auth/login",
        json={"email": "eve@example.com", "password": "longenough1"},
        headers={"X-Internal-Auth": "nope"},
    )
    assert wrong_secret.status_code == 403
