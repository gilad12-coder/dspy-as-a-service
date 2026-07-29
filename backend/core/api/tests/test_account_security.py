"""Tests for two-factor auth and passkey endpoints.

Runs the accounts + account-security routers against a shared in-memory
SQLite engine. TOTP and recovery flows are exercised end-to-end with real
``pyotp`` codes; the WebAuthn signature checks are monkeypatched (attestation
requires a hardware authenticator) while the challenge lifecycle, storage,
and ownership checks around them run for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pyotp
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from webauthn.helpers import bytes_to_base64url

from ...config import settings
from ...storage.models import (
    TwoFactorEmailCodeModel,
    UserModel,
    WebAuthnChallengeModel,
    WebAuthnCredentialModel,
)
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers import account_security as security_module
from ..routers.account_security import create_account_security_router
from ..routers.accounts import create_accounts_router

_SECRET = "test-internal-secret"
_AUTH_HEADER = {"X-Internal-Auth": _SECRET}
_EMAIL = "user@example.com"
_PASSWORD = "correct horse battery"


class _Store:
    """Minimal store exposing only the SQLAlchemy engine the routes need."""

    def __init__(self, engine: Any) -> None:
        """Hold the engine the routers open sessions on.

        Args:
            engine: A SQLAlchemy engine with the auth tables created.
        """
        self.engine = engine


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a client serving both auth routers over in-memory SQLite.

    The bearer dependency is overridden to the registered test user, so the
    settings-facing routes act on a fixed identity without minting JWTs.

    Args:
        monkeypatch: Pytest fixture used to set the shared internal secret.

    Returns:
        A ``TestClient`` with one registered email/password account.
    """
    monkeypatch.setattr(settings, "backend_auth_secret", SecretStr(_SECRET))
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    for model in (
        UserModel,
        WebAuthnCredentialModel,
        WebAuthnChallengeModel,
        TwoFactorEmailCodeModel,
    ):
        model.__table__.create(engine)
    store = _Store(engine)
    app = FastAPI()
    app.state.job_store = store

    # The real app's problem-details handler attaches ``code`` + ``params``;
    # mirror just those fields so the assertions see the same envelope.
    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    app.include_router(create_accounts_router(job_store=store))
    app.include_router(create_account_security_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: AuthenticatedUser(
        username=_EMAIL, role="user", groups=()
    )
    test_client = TestClient(app)
    resp = test_client.post(
        "/auth/register",
        json={"email": _EMAIL, "password": _PASSWORD},
        headers=_AUTH_HEADER,
    )
    assert resp.status_code == 201
    return test_client


def _login(client: TestClient, **codes: str):
    """POST /auth/login for the test account with optional 2FA codes.

    Args:
        client: The fixture client.
        **codes: Optional ``totp_code`` / ``email_code`` / ``recovery_code``.

    Returns:
        The raw response.
    """
    return client.post(
        "/auth/login",
        json={"email": _EMAIL, "password": _PASSWORD, **codes},
        headers=_AUTH_HEADER,
    )


def _enable_totp(client: TestClient) -> tuple[str, list[str]]:
    """Run the full TOTP enrollment and return the secret + recovery codes.

    Args:
        client: The fixture client.

    Returns:
        The base32 secret and the one-time recovery codes.
    """
    setup = client.post("/auth/security/totp/setup").json()
    code = pyotp.TOTP(setup["secret"]).now()
    enable = client.post("/auth/security/totp/enable", json={"code": code})
    assert enable.status_code == 200
    return setup["secret"], enable.json()["recovery_codes"]


def test_security_status_defaults(client: TestClient) -> None:
    """A fresh local account has a password, no 2FA, and no passkeys."""
    status = client.get("/auth/security").json()
    assert status == {
        "has_password": True,
        "totp_enabled": False,
        "email_2fa_enabled": False,
        "email_2fa_available": False,
        "passkeys": [],
    }


def test_totp_enable_requires_setup_and_valid_code(client: TestClient) -> None:
    """Enable without setup 422s; a wrong first code 401s and stays pending."""
    resp = client.post("/auth/security/totp/enable", json={"code": "000000"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "accounts.totp_setup_required"
    client.post("/auth/security/totp/setup")
    resp = client.post("/auth/security/totp/enable", json={"code": "000000"})
    assert resp.status_code == 401
    assert client.get("/auth/security").json()["totp_enabled"] is False


def test_totp_gates_login_and_recovery_codes_burn(client: TestClient) -> None:
    """TOTP-enabled login demands a code; recovery codes work exactly once."""
    secret, recovery = _enable_totp(client)
    assert len(recovery) == 8

    bare = _login(client)
    assert bare.status_code == 401
    body = bare.json()
    assert body["code"] == "accounts.two_factor_required"
    assert body["params"]["methods"] == "totp"

    assert _login(client, totp_code="000000").status_code == 401
    assert _login(client, totp_code=pyotp.TOTP(secret).now()).status_code == 200

    assert _login(client, recovery_code=recovery[0]).status_code == 200
    replay = _login(client, recovery_code=recovery[0])
    assert replay.status_code == 401
    assert replay.json()["code"] == "accounts.invalid_second_factor"


def test_totp_disable_restores_plain_login(client: TestClient) -> None:
    """Disabling with a live code clears the factor and the login gate."""
    secret, _ = _enable_totp(client)
    resp = client.post(
        "/auth/security/totp/disable", json={"code": pyotp.TOTP(secret).now()}
    )
    assert resp.status_code == 200
    assert client.get("/auth/security").json()["totp_enabled"] is False
    assert _login(client).status_code == 200


def test_email_code_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Email 2FA sends a code (password re-proved) that then unlocks login."""
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(settings, "smtp_host", "smtp.test")
    monkeypatch.setattr(
        security_module, "send_email", lambda to, subject, body: sent.append((to, subject, body))
    )
    assert client.put("/auth/security/email-codes", json={"enabled": True}).status_code == 200

    wrong = client.post(
        "/auth/2fa/email/send",
        json={"email": _EMAIL, "password": "not-the-password"},
        headers=_AUTH_HEADER,
    )
    assert wrong.status_code == 401
    assert sent == []

    resp = client.post(
        "/auth/2fa/email/send",
        json={"email": _EMAIL, "password": _PASSWORD},
        headers=_AUTH_HEADER,
    )
    assert resp.status_code == 200
    assert len(sent) == 1
    assert sent[0][0] == _EMAIL
    code = next(word for word in sent[0][2].split() if word.rstrip(".").isdigit()).rstrip(".")

    bare = _login(client)
    assert bare.status_code == 401
    assert bare.json()["params"]["methods"] == "email"
    assert _login(client, email_code="999999").status_code == 401
    assert _login(client, email_code=code).status_code == 200
    replay = _login(client, email_code=code)
    assert replay.status_code == 401


def test_email_code_toggle_requires_smtp(client: TestClient) -> None:
    """Enabling email codes on a mail-less deployment is a typed 422."""
    resp = client.put("/auth/security/email-codes", json={"enabled": True})
    assert resp.status_code == 422
    assert resp.json()["code"] == "accounts.email_delivery_unavailable"


def _fake_credential(challenge_b64: str, raw_id: bytes = b"cred-1") -> dict[str, Any]:
    """Build the minimal browser-credential JSON the endpoints inspect.

    Args:
        challenge_b64: The base64url challenge echoed in clientDataJSON.
        raw_id: Raw credential id bytes.

    Returns:
        A credential dict whose signature checks are monkeypatched in tests.
    """
    client_data = json.dumps({"type": "webauthn.create", "challenge": challenge_b64}).encode()
    return {
        "id": bytes_to_base64url(raw_id),
        "rawId": bytes_to_base64url(raw_id),
        "response": {"clientDataJSON": bytes_to_base64url(client_data), "transports": ["internal"]},
        "type": "public-key",
    }


def test_passkey_register_list_delete(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration verifies against the stored challenge, then lists/deletes."""
    options = client.post("/auth/security/passkeys/options").json()
    assert options["rp"]["name"] == "Skynet"
    challenge = options["challenge"]

    monkeypatch.setattr(
        security_module,
        "verify_registration_response",
        lambda **kwargs: SimpleNamespace(
            credential_id=b"cred-1", credential_public_key=b"pubkey", sign_count=0
        ),
    )
    created = client.post(
        "/auth/security/passkeys",
        json={"credential": _fake_credential(challenge), "nickname": "MacBook Touch ID"},
    )
    assert created.status_code == 201
    assert created.json()["nickname"] == "MacBook Touch ID"
    credential_id = created.json()["credential_id"]

    replay = client.post(
        "/auth/security/passkeys",
        json={"credential": _fake_credential(challenge)},
    )
    assert replay.status_code == 401
    assert replay.json()["code"] == "webauthn.challenge_expired"

    status = client.get("/auth/security").json()
    assert [p["credential_id"] for p in status["passkeys"]] == [credential_id]

    assert client.delete(f"/auth/security/passkeys/{credential_id}").status_code == 200
    assert client.get("/auth/security").json()["passkeys"] == []
    assert client.delete(f"/auth/security/passkeys/{credential_id}").status_code == 404


def test_passkey_signin_resolves_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified assertion signs the credential's owner in and bumps counters."""
    reg_challenge = client.post("/auth/security/passkeys/options").json()["challenge"]
    monkeypatch.setattr(
        security_module,
        "verify_registration_response",
        lambda **kwargs: SimpleNamespace(
            credential_id=b"cred-1", credential_public_key=b"pubkey", sign_count=0
        ),
    )
    assert (
        client.post(
            "/auth/security/passkeys", json={"credential": _fake_credential(reg_challenge)}
        ).status_code
        == 201
    )

    options = client.post("/auth/webauthn/options", headers=_AUTH_HEADER).json()
    monkeypatch.setattr(
        security_module,
        "verify_authentication_response",
        lambda **kwargs: SimpleNamespace(new_sign_count=7),
    )
    resp = client.post(
        "/auth/webauthn/verify",
        json={"credential": _fake_credential(options["challenge"])},
        headers=_AUTH_HEADER,
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == _EMAIL

    store: _Store = client.app.state.job_store
    with Session(store.engine) as session:
        row = session.query(WebAuthnCredentialModel).one()
        assert int(row.sign_count) == 7
        assert row.last_used_at is not None


def test_passkey_signin_rejects_unknown_credential(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An assertion for an unregistered credential fails closed."""
    options = client.post("/auth/webauthn/options", headers=_AUTH_HEADER).json()
    resp = client.post(
        "/auth/webauthn/verify",
        json={"credential": _fake_credential(options["challenge"], raw_id=b"nobody")},
        headers=_AUTH_HEADER,
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "webauthn.invalid_credential"


def test_internal_routes_reject_bad_secret(client: TestClient) -> None:
    """The pre-session routes stay closed without the shared secret."""
    assert client.post("/auth/webauthn/options").status_code == 403
    assert (
        client.post(
            "/auth/2fa/email/send", json={"email": _EMAIL, "password": _PASSWORD}
        ).status_code
        == 403
    )
