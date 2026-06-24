"""Email/password account registration and sign-in. [INTERNAL]

Backs the "create an account in Skynet" path on the login screen. These are the
only unauthenticated routes in the API — they bootstrap a session before any
token exists. To keep them from being a public abuse vector they are gated by
the shared ``BACKEND_AUTH_SECRET``: only the Skynet frontend, which holds that
secret, may call them (server-side, from its NextAuth credentials provider).
OAuth (Google/GitHub) sign-ins never touch this router — they resolve identity
at the provider and mint the session JWT directly.
"""

from __future__ import annotations

import hmac
import re
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import settings
from ...storage.models import UserModel
from ..errors import DomainError
from ..passwords import hash_password, verify_password

_MIN_PASSWORD_LENGTH = 8
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# Credentials supplied when creating a Skynet-native account.
class RegisterRequest(BaseModel):
    email: str = Field(description="Account email; also the cross-app identity.")
    password: str = Field(description="Plaintext password; stored only as a scrypt hash.")
    name: str = Field(default="", description="Display name shown in the app header.")


# Credentials supplied when signing in to a Skynet-native account.
class LoginRequest(BaseModel):
    email: str = Field(description="Account email.")
    password: str = Field(description="Plaintext password to verify.")


# The resolved account the frontend turns into a session — never carries a secret.
class AccountInfo(BaseModel):
    email: str = Field(description="Lowercased account email, which is the identity.")
    name: str = Field(description="Display name.")
    role: str = Field(description="Authorization role: 'admin' or 'user'.")


def _normalise_email(raw: str) -> str:
    """Lowercase and trim an email for use as the stable identity.

    Args:
        raw: Email as supplied by the client.

    Returns:
        The normalized email.
    """
    return raw.strip().lower()


def _role_for(email: str) -> str:
    """Resolve the authorization role for an account email.

    Args:
        email: Normalized account email.

    Returns:
        ``"admin"`` when the email is in the admin allowlist, else ``"user"``.
    """
    return "admin" if email in settings.admin_usernames_set else "user"


def _require_internal_auth(header_value: str | None) -> None:
    """Authorize a call from the trusted frontend via the shared secret.

    Args:
        header_value: Value of the ``X-Internal-Auth`` request header.

    Raises:
        DomainError: 500 when no shared secret is configured (the deployment is
            half-wired); 403 when the header is missing or does not match.
    """
    secret = settings.backend_auth_secret
    if secret is None:
        raise DomainError("auth.not_configured", status=500)
    if not header_value or not hmac.compare_digest(header_value, secret.get_secret_value()):
        raise DomainError("auth.missing_token", status=403)


def create_accounts_router(*, job_store) -> APIRouter:
    """Build the email/password account router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.

    Returns:
        A FastAPI ``APIRouter`` exposing register + login for local accounts.
    """
    router = APIRouter()

    @router.post(
        "/auth/register",
        response_model=AccountInfo,
        status_code=201,
        summary="Create a Skynet-native email/password account",
    )
    def register(
        body: RegisterRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> AccountInfo:
        """Create a new local account and return its identity.

        Args:
            body: Email, password, and optional display name.
            x_internal_auth: Shared-secret header proving the caller is the
                trusted frontend.

        Returns:
            The created account (email, display name, role).

        Raises:
            DomainError: 403 on a bad internal secret; 422 on an invalid email
                or too-short password; 409 when the email is already registered.
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        if not _EMAIL_RE.match(email):
            raise DomainError("accounts.invalid_email", status=422)
        if len(body.password) < _MIN_PASSWORD_LENGTH:
            raise DomainError("accounts.weak_password", status=422)
        name = body.name.strip() or email
        with Session(job_store.engine) as session:
            if session.get(UserModel, email) is not None:
                raise DomainError("accounts.email_taken", status=409)
            session.add(
                UserModel(
                    email=email,
                    name=name,
                    password_hash=hash_password(body.password),
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()
        return AccountInfo(email=email, name=name, role=_role_for(email))

    @router.post(
        "/auth/login",
        response_model=AccountInfo,
        summary="Verify email/password credentials",
    )
    def login(
        body: LoginRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> AccountInfo:
        """Verify credentials and return the account identity.

        Args:
            body: Email and password to verify.
            x_internal_auth: Shared-secret header proving the caller is the
                trusted frontend.

        Returns:
            The authenticated account (email, display name, role).

        Raises:
            DomainError: 403 on a bad internal secret; 401 when the email is
                unknown or the password does not match.
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            if row is None or not verify_password(body.password, str(row.password_hash)):
                raise DomainError("accounts.invalid_credentials", status=401)
            row.last_login_at = datetime.now(UTC)
            name = str(row.name)
            session.commit()
        return AccountInfo(email=email, name=name, role=_role_for(email))

    return router
