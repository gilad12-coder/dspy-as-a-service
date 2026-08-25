"""Tests for registration email verification: emailed confirmation codes.

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
from ...storage.models import EmailVerificationCodeModel, UserModel
from .. import email_verification
from ..email_verification import (
    VERIFY_CODE_MAX_ATTEMPTS,
    issue_verification_code,
    verification_code_on_cooldown,
    verify_verification_code,
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
def verify_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, list[dict[str, str]]]:
    """Build an accounts client and capture outbound confirmation emails.

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
    EmailVerificationCodeModel.__table__.create(engine)
    store = _Store(engine)
    app = FastAPI()
    app.state.job_store = store
    app.include_router(create_accounts_router(job_store=store))
    return TestClient(app), sent


def _register(client: TestClient, email: str, password: str) -> None:
    """Register an account, asserting the 201 the route returns.

    Args:
        client: The accounts test client.
        email: Account email.
        password: Initial password.
    """
    created = client.post(
        "/auth/register", json={"email": email, "password": password}, headers=_AUTH_HEADER
    )
    assert created.status_code == 201


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


def _login(client: TestClient, email: str, password: str) -> Any:
    """POST /auth/login and return the raw response.

    Args:
        client: The accounts test client.
        email: Account email.
        password: Account password.

    Returns:
        The raw login response.
    """
    return client.post(
        "/auth/login", json={"email": email, "password": password}, headers=_AUTH_HEADER
    )


def test_register_emails_code_and_blocks_login_until_confirmed(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Registration sends a code; login is refused until the code confirms it."""
    client, sent = verify_client
    _register(client, "amy@example.com", "originalpass1")
    assert len(sent) == 1
    code = _last_code(sent)

    blocked = _login(client, "amy@example.com", "originalpass1")
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == t_en("accounts.email_not_verified")

    confirmed = client.post(
        "/auth/email-verify/confirm",
        json={"email": "amy@example.com", "code": code},
        headers=_AUTH_HEADER,
    )
    assert confirmed.status_code == 200
    assert _login(client, "amy@example.com", "originalpass1").status_code == 200


def test_register_without_smtp_auto_verifies(
    verify_client: tuple[TestClient, list[dict[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no relay there is nothing to confirm, so the account signs in at once."""
    client, sent = verify_client
    monkeypatch.setattr(settings, "smtp_host", None)
    _register(client, "bea@example.com", "originalpass1")
    assert sent == []
    assert _login(client, "bea@example.com", "originalpass1").status_code == 200


def test_confirm_wrong_code_is_422_and_keeps_account_unverified(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A wrong code is rejected and the account stays locked out of login."""
    client, _ = verify_client
    _register(client, "dan@example.com", "originalpass1")
    res = client.post(
        "/auth/email-verify/confirm",
        json={"email": "dan@example.com", "code": "000000"},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422
    assert res.json()["detail"] == t_en("accounts.invalid_verification_code")
    assert _login(client, "dan@example.com", "originalpass1").status_code == 403


def test_confirm_unknown_email_is_422(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Confirming for an address with no account fails like a bad code."""
    client, _ = verify_client
    res = client.post(
        "/auth/email-verify/confirm",
        json={"email": "ghost@example.com", "code": "123456"},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422
    assert res.json()["detail"] == t_en("accounts.invalid_verification_code")


def test_confirm_expired_code_is_422(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """An expired code is rejected even when the digits are correct."""
    client, sent = verify_client
    _register(client, "fay@example.com", "originalpass1")
    code = _last_code(sent)
    engine = client.app.state.job_store.engine
    with Session(engine) as session:
        row = session.get(EmailVerificationCodeModel, "fay@example.com")
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session.commit()
    res = client.post(
        "/auth/email-verify/confirm",
        json={"email": "fay@example.com", "code": code},
        headers=_AUTH_HEADER,
    )
    assert res.status_code == 422


def test_resend_request_sends_fresh_code(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """The resend route emails a new code an unverified account can confirm with."""
    client, sent = verify_client
    _register(client, "gus@example.com", "originalpass1")
    engine = client.app.state.job_store.engine
    # Clear the registration cooldown so the resend actually re-sends.
    with Session(engine) as session:
        row = session.get(EmailVerificationCodeModel, "gus@example.com")
        row.sent_at = datetime.now(UTC) - timedelta(minutes=5)
        session.commit()
    res = client.post(
        "/auth/email-verify/request", json={"email": "gus@example.com"}, headers=_AUTH_HEADER
    )
    assert res.status_code == 200
    assert len(sent) == 2
    code = _last_code(sent)
    confirmed = client.post(
        "/auth/email-verify/confirm",
        json={"email": "gus@example.com", "code": code},
        headers=_AUTH_HEADER,
    )
    assert confirmed.status_code == 200


def test_resend_is_silent_for_unknown_and_verified(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """An unknown or already-verified address gets the same 200 with no send."""
    client, sent = verify_client
    unknown = client.post(
        "/auth/email-verify/request", json={"email": "ghost@example.com"}, headers=_AUTH_HEADER
    )
    assert unknown.status_code == 200
    assert sent == []

    _register(client, "hal@example.com", "originalpass1")
    code = _last_code(sent)
    client.post(
        "/auth/email-verify/confirm",
        json={"email": "hal@example.com", "code": code},
        headers=_AUTH_HEADER,
    )
    before = len(sent)
    verified = client.post(
        "/auth/email-verify/request", json={"email": "hal@example.com"}, headers=_AUTH_HEADER
    )
    assert verified.status_code == 200
    assert len(sent) == before


def test_resend_cooldown_suppresses_second_send(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """A resend inside the cooldown returns 200 without re-sending."""
    client, sent = verify_client
    _register(client, "ida@example.com", "originalpass1")
    assert len(sent) == 1
    again = client.post(
        "/auth/email-verify/request", json={"email": "ida@example.com"}, headers=_AUTH_HEADER
    )
    assert again.status_code == 200
    assert len(sent) == 1


def test_resend_without_smtp_is_422(
    verify_client: tuple[TestClient, list[dict[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no SMTP relay the resend is refused for every address alike."""
    client, _ = verify_client
    monkeypatch.setattr(settings, "smtp_host", None)
    res = client.post(
        "/auth/email-verify/request", json={"email": "amy@example.com"}, headers=_AUTH_HEADER
    )
    assert res.status_code == 422


def test_endpoints_require_internal_secret(
    verify_client: tuple[TestClient, list[dict[str, str]]],
) -> None:
    """Both verification routes reject callers without the shared internal secret."""
    client, _ = verify_client
    assert (
        client.post("/auth/email-verify/request", json={"email": "amy@example.com"}).status_code
        == 403
    )
    assert (
        client.post(
            "/auth/email-verify/confirm", json={"email": "amy@example.com", "code": "123456"}
        ).status_code
        == 403
    )


@pytest.fixture
def code_engine() -> Any:
    """Provide an in-memory engine with the confirmation-code table.

    Returns:
        A SQLAlchemy engine holding an empty ``email_verification_codes`` table.
    """
    engine = create_engine("sqlite://", poolclass=StaticPool)
    EmailVerificationCodeModel.__table__.create(engine)
    return engine


def test_verify_code_consumes_on_success(code_engine: Any) -> None:
    """A correct code verifies once and is then gone."""
    with Session(code_engine) as session:
        code = issue_verification_code(session, "jan@example.com")
        session.commit()
    with Session(code_engine) as session:
        assert verify_verification_code(session, "jan@example.com", code) is True
        session.commit()
    with Session(code_engine) as session:
        assert verify_verification_code(session, "jan@example.com", code) is False


def test_verify_code_caps_attempts(code_engine: Any) -> None:
    """After the attempt cap the code is burned even if later guessed right."""
    with Session(code_engine) as session:
        code = issue_verification_code(session, "kim@example.com")
        session.commit()
    with Session(code_engine) as session:
        for _ in range(VERIFY_CODE_MAX_ATTEMPTS):
            assert verify_verification_code(session, "kim@example.com", "000000") is False
        session.commit()
    with Session(code_engine) as session:
        assert verify_verification_code(session, "kim@example.com", code) is False


def test_verification_code_cooldown_reflects_recent_send(code_engine: Any) -> None:
    """Cooldown is false with no code, true right after a send."""
    with Session(code_engine) as session:
        assert verification_code_on_cooldown(session, "leo@example.com") is False
        issue_verification_code(session, "leo@example.com")
        session.commit()
        assert verification_code_on_cooldown(session, "leo@example.com") is True


def test_issue_verification_code_hashes_and_rotates(code_engine: Any) -> None:
    """The stored value is a scrypt hash, and re-issuing replaces it."""
    with Session(code_engine) as session:
        first = issue_verification_code(session, "mia@example.com")
        session.commit()
        row = session.get(EmailVerificationCodeModel, "mia@example.com")
        assert row.code_hash.startswith("scrypt$")
        assert verify_password(first, row.code_hash) is True
        second = issue_verification_code(session, "mia@example.com")
        session.commit()
        assert first != second
        assert verify_password(first, str(row.code_hash)) is False


def test_email_verification_module_exports_helpers() -> None:
    """The leaf module surfaces the three functions the router imports."""
    assert hasattr(email_verification, "issue_verification_code")
    assert hasattr(email_verification, "verify_verification_code")
    assert hasattr(email_verification, "verification_code_on_cooldown")
    assert hash_password("x").startswith("scrypt$")
