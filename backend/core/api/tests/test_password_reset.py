"""Tests for the forgot-password flow: emailed reset codes and confirmation.

The endpoint tests run against a shared in-memory SQLite engine (StaticPool so
tables and rows persist across sessions and the TestClient worker thread), set a
known ``BACKEND_AUTH_SECRET`` so the internal-auth gate is exercised, point
``SMTP_HOST`` at a dummy relay so ``email_configured()`` is true, and stub the
transport so the 6-digit code is captured from the message body instead of sent.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...config import settings
from ...i18n_en import t_en
from ...storage.models import EmailVerificationCodeModel, PasswordResetCodeModel, UserModel
from .. import password_reset
from ..password_reset import (
    RESET_CODE_MAX_ATTEMPTS,
    issue_reset_code,
    reset_code_on_cooldown,
    verify_reset_code,
)
from ..passwords import hash_password, verify_password
from ..routers import accounts as accounts_module
from ..routers.accounts import create_accounts_router

_SECRET = "test-internal-secret"
_AUTH_HEADER = {"X-Internal-Auth": _SECRET}
_CODE_RE = re.compile(r"code is (\d{6})")


class _Store:
    """Minimal store exposing only the SQLAlchemy engine the routes need."""

    def __init__(self, engine: Any) -> None:
        """Hold the engine the accounts router opens sessions on.

        Args:
            engine: A SQLAlchemy engine with the account tables created.
        """
        self.engine = engine


@pytest.fixture
def reset_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, list[dict[str, str]]]:
    """Build an accounts client and capture outbound reset emails.

    Args:
        monkeypatch: Pytest fixture used to set the internal secret, enable
            email delivery, and replace the SMTP transport with a recorder.

    Returns:
        A ``(client, sent)`` pair, where ``sent`` accumulates one dict per
        captured message with ``to``, ``subject``, and ``body`` keys.
    """
    monkeypatch.setattr(settings, "backend_auth_secret", SecretStr(_SECRET))
    monkeypatch.setattr(settings, "smtp_host", "smtp.test.local")
    sent: list[dict[str, str]] = []

    def _capture(to: str, subject: str, body: str) -> None:
        sent.append({"to": to, "subject": subject, "body": body})

    monkeypatch.setattr(accounts_module, "send_email", _capture)
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    UserModel.__table__.create(engine)
    PasswordResetCodeModel.__table__.create(engine)
    EmailVerificationCodeModel.__table__.create(engine)
    store = _Store(engine)
    app = FastAPI()
    app.state.job_store = store
    app.include_router(create_accounts_router(job_store=store))
    return TestClient(app), sent


def _register(
    client: TestClient, sent: list[dict[str, str]], email: str, password: str
) -> None:
    """Register an account, mark it verified, and clear the email recorder.

    Under a configured SMTP relay ``register`` creates the account unverified
    and emails a confirmation code. The reset flows all assume an account that
    has already confirmed its email, so this flips ``email_verified`` directly
    and drops the captured confirmation message, restoring the recorder to empty
    for the reset-email assertions that follow.

    Args:
        client: The accounts test client.
        sent: The fixture's outbound-email recorder, cleared after registration.
        email: Account email.
        password: Initial password.
    """
    created = client.post(
        "/auth/register", json={"email": email, "password": password}, headers=_AUTH_HEADER
    )
    assert created.status_code == 201
    with Session(client.app.state.job_store.engine) as session:
        session.get(UserModel, email).email_verified = True
        session.commit()
    sent.clear()


def _last_code(sent: list[dict[str, str]]) -> str:
    """Extract the 6-digit code from the most recently captured message.

    Args:
        sent: The recorder populated by the fixture.

    Returns:
        The 6-digit code parsed from the message body.
    """
    match = _CODE_RE.search(sent[-1]["body"])
    assert match is not None
    return match.group(1)


def test_request_then_confirm_resets_password(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A reset code lets the user set a new password; old one stops working."""
    client, sent = reset_client
    _register(client, sent, "amy@example.com", "originalpass1")

    requested = client.post(
        "/auth/password-reset/request", json={"email": "amy@example.com"}, headers=_AUTH_HEADER
    )
    assert requested.status_code == 200
    assert len(sent) == 1
    code = _last_code(sent)

    confirmed = client.post(
        "/auth/password-reset/confirm",
        json={"email": "amy@example.com", "code": code, "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    assert confirmed.status_code == 200

    assert (
        client.post(
            "/auth/login",
            json={"email": "amy@example.com", "password": "brandnewpass9"},
            headers=_AUTH_HEADER,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/auth/login",
            json={"email": "amy@example.com", "password": "originalpass1"},
            headers=_AUTH_HEADER,
        ).status_code
        == 401
    )


def test_request_unknown_email_is_ok_and_silent(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """An unknown address gets the same 200 but no email is sent."""
    client, sent = reset_client
    res = client.post(
        "/auth/password-reset/request", json={"email": "ghost@example.com"}, headers=_AUTH_HEADER
    )
    assert res.status_code == 200
    assert sent == []


def test_request_without_smtp_is_422(
    reset_client: tuple[TestClient, list[dict[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no SMTP relay the request is refused for every address alike."""
    client, _ = reset_client
    monkeypatch.setattr(settings, "smtp_host", None)
    res = client.post(
        "/auth/password-reset/request", json={"email": "amy@example.com"}, headers=_AUTH_HEADER
    )
    assert res.status_code == 422


def test_request_cooldown_suppresses_second_send(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A second request inside the cooldown returns 200 without re-sending."""
    client, sent = reset_client
    _register(client, sent, "cara@example.com", "originalpass1")
    first = client.post(
        "/auth/password-reset/request", json={"email": "cara@example.com"}, headers=_AUTH_HEADER
    )
    second = client.post(
        "/auth/password-reset/request", json={"email": "cara@example.com"}, headers=_AUTH_HEADER
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(sent) == 1


def test_confirm_wrong_code_is_422(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A wrong code is rejected and does not change the password."""
    client, sent = reset_client
    _register(client, sent, "dan@example.com", "originalpass1")
    client.post(
        "/auth/password-reset/request", json={"email": "dan@example.com"}, headers=_AUTH_HEADER
    )
    res = client.post(
        "/auth/password-reset/confirm",
        json={"email": "dan@example.com", "code": "000000", "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422
    assert (
        client.post(
            "/auth/login",
            json={"email": "dan@example.com", "password": "originalpass1"},
            headers=_AUTH_HEADER,
        ).status_code
        == 200
    )


def test_confirm_unknown_email_is_422(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Confirming for an address with no account fails like a bad code."""
    client, _ = reset_client
    res = client.post(
        "/auth/password-reset/confirm",
        json={"email": "ghost@example.com", "code": "123456", "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422
    assert res.json()["detail"] == t_en("accounts.invalid_reset_code")


def test_confirm_weak_password_is_422_and_keeps_code(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A weak new password is rejected before the code is spent, so retry works."""
    client, sent = reset_client
    _register(client, sent, "eli@example.com", "originalpass1")
    client.post(
        "/auth/password-reset/request", json={"email": "eli@example.com"}, headers=_AUTH_HEADER
    )
    code = _last_code(sent)
    weak = client.post(
        "/auth/password-reset/confirm",
        json={"email": "eli@example.com", "code": code, "new_password": "short"},
        headers=_AUTH_HEADER,
    )
    assert weak.status_code == 422
    retry = client.post(
        "/auth/password-reset/confirm",
        json={"email": "eli@example.com", "code": code, "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    assert retry.status_code == 200


def test_confirm_expired_code_is_422(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """An expired code is rejected even when the digits are correct."""
    client, sent = reset_client
    _register(client, sent, "fay@example.com", "originalpass1")
    client.post(
        "/auth/password-reset/request", json={"email": "fay@example.com"}, headers=_AUTH_HEADER
    )
    code = _last_code(sent)
    engine = client.app.state.job_store.engine
    with Session(engine) as session:
        row = session.get(PasswordResetCodeModel, "fay@example.com")
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session.commit()
    res = client.post(
        "/auth/password-reset/confirm",
        json={"email": "fay@example.com", "code": code, "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422


def test_confirm_leaves_second_factor_intact(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Resetting the password does not disable the account's email 2FA."""
    client, sent = reset_client
    _register(client, sent, "gus@example.com", "originalpass1")
    engine = client.app.state.job_store.engine
    with Session(engine) as session:
        session.get(UserModel, "gus@example.com").email_2fa_enabled = True
        session.commit()
    client.post(
        "/auth/password-reset/request", json={"email": "gus@example.com"}, headers=_AUTH_HEADER
    )
    code = _last_code(sent)
    client.post(
        "/auth/password-reset/confirm",
        json={"email": "gus@example.com", "code": code, "new_password": "brandnewpass9"},
        headers=_AUTH_HEADER,
    )
    with Session(engine) as session:
        assert session.get(UserModel, "gus@example.com").email_2fa_enabled is True


def test_endpoints_require_internal_secret(
    reset_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Both reset routes reject callers without the shared internal secret."""
    client, _ = reset_client
    assert (
        client.post(
            "/auth/password-reset/request", json={"email": "amy@example.com"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/auth/password-reset/confirm",
            json={"email": "amy@example.com", "code": "123456", "new_password": "brandnewpass9"},
        ).status_code
        == 403
    )


@pytest.fixture
def code_engine() -> Any:
    """Provide an in-memory engine with the reset-code table for helper tests.

    Returns:
        A SQLAlchemy engine holding an empty ``password_reset_codes`` table.
    """
    engine = create_engine("sqlite://", poolclass=StaticPool)
    PasswordResetCodeModel.__table__.create(engine)
    return engine


def test_verify_reset_code_consumes_on_success(code_engine: Any) -> None:
    """A correct code verifies once and is then gone."""
    with Session(code_engine) as session:
        code = issue_reset_code(session, "hal@example.com")
        session.commit()
    with Session(code_engine) as session:
        assert verify_reset_code(session, "hal@example.com", code) is True
        session.commit()
    with Session(code_engine) as session:
        assert verify_reset_code(session, "hal@example.com", code) is False


def test_verify_reset_code_caps_attempts(code_engine: Any) -> None:
    """After the attempt cap the code is burned even if later guessed right."""
    with Session(code_engine) as session:
        code = issue_reset_code(session, "ida@example.com")
        session.commit()
    with Session(code_engine) as session:
        for _ in range(RESET_CODE_MAX_ATTEMPTS):
            assert verify_reset_code(session, "ida@example.com", "000000") is False
        session.commit()
    with Session(code_engine) as session:
        assert verify_reset_code(session, "ida@example.com", code) is False


def test_reset_code_cooldown_reflects_recent_send(code_engine: Any) -> None:
    """Cooldown is false with no code, true right after a send."""
    with Session(code_engine) as session:
        assert reset_code_on_cooldown(session, "jan@example.com") is False
        issue_reset_code(session, "jan@example.com")
        session.commit()
        assert reset_code_on_cooldown(session, "jan@example.com") is True


def test_issue_reset_code_hashes_and_rotates(code_engine: Any) -> None:
    """The stored value is a scrypt hash, and re-issuing replaces it."""
    with Session(code_engine) as session:
        first = issue_reset_code(session, "kim@example.com")
        session.commit()
        row = session.get(PasswordResetCodeModel, "kim@example.com")
        assert row.code_hash.startswith("scrypt$")
        assert verify_password(first, row.code_hash) is True
        second = issue_reset_code(session, "kim@example.com")
        session.commit()
        assert first != second
        assert verify_password(first, str(row.code_hash)) is False


def test_password_reset_module_exports_helpers() -> None:
    """The leaf module surfaces the three functions the router imports."""
    assert hasattr(password_reset, "issue_reset_code")
    assert hasattr(password_reset, "verify_reset_code")
    assert hasattr(password_reset, "reset_code_on_cooldown")
    assert hash_password("x").startswith("scrypt$")
