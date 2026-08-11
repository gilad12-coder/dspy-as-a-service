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
from ..login_throttle import LoginThrottle
from ..password_policy import validate_password
from ..passwords import hash_password, verify_password
from ..two_factor import enforce_second_factor

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_USE_CASES = frozenset({"classification", "extraction", "rag_agents", "generation", "other"})
_EXPERIENCE_LEVELS = frozenset({"new", "familiar", "expert"})
_JOB_ROLES = frozenset({"engineer", "researcher", "pm", "other"})


# Credentials supplied when creating a Skynet-native account.
class RegisterRequest(BaseModel):
    email: str = Field(description="Account email; also the cross-app identity.")
    password: str = Field(description="Plaintext password; stored only as a scrypt hash.")
    name: str = Field(default="", description="Display name shown in the app header.")
    use_case: str = Field(default="", description="Primary optimization use-case from sign-up.")
    experience_level: str = Field(default="", description="Self-reported optimization experience.")
    job_role: str = Field(default="", description="Optional job function from sign-up.")


# Credentials supplied when signing in to a Skynet-native account. At most one
# of the three second-factor codes is honored (TOTP > email > recovery).
class LoginRequest(BaseModel):
    email: str = Field(description="Account email.")
    password: str = Field(description="Plaintext password to verify.")
    totp_code: str = Field(default="", description="Authenticator-app code when TOTP 2FA is enabled.")
    email_code: str = Field(default="", description="Emailed one-time code when email 2FA is enabled.")
    recovery_code: str = Field(default="", description="Single-use recovery code standing in for a lost factor.")


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


def _coerce_choice(value: str, allowed: frozenset[str]) -> str | None:
    """Return the value if it is a recognized choice, else ``None``.

    Args:
        value: Raw choice string from the client.
        allowed: The permitted values for this field.

    Returns:
        The trimmed value when it is in ``allowed``; ``None`` otherwise. An
        unknown or blank choice is stored as "not provided" rather than
        rejected, since these profile fields are non-critical metadata and the
        sign-up form already constrains input to the known options.
    """
    cleaned = value.strip()
    return cleaned if cleaned in allowed else None


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


def create_accounts_router(*, job_store, login_throttle: LoginThrottle | None = None) -> APIRouter:
    """Build the email/password account router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.
        login_throttle: Per-email failed-login limiter guarding ``/auth/login``.
            Defaults to a fresh in-memory instance scoped to this router (and so
            to the process); tests inject one with a controlled clock or lower
            threshold.

    Returns:
        A FastAPI ``APIRouter`` exposing register + login for local accounts.
    """
    router = APIRouter()
    throttle = login_throttle or LoginThrottle()

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
                or a password the acceptance policy rejects; 409 when the email
                is already registered.
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        if not _EMAIL_RE.match(email):
            raise DomainError("accounts.invalid_email", status=422)
        validate_password(body.password, email)
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
                    use_case=_coerce_choice(body.use_case, _USE_CASES),
                    experience_level=_coerce_choice(body.experience_level, _EXPERIENCE_LEVELS),
                    job_role=_coerce_choice(body.job_role, _JOB_ROLES),
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
            DomainError: 403 on a bad internal secret; 429
                (``accounts.too_many_attempts``) when the email is locked out
                after repeated failures; 401 when the email is unknown, the
                password does not match, or the account's second factor is
                missing (``accounts.two_factor_required``, carrying the usable
                methods in ``params``) or wrong (``accounts.invalid_second_factor``).
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        throttle.check(email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            if row is None or not verify_password(body.password, str(row.password_hash)):
                throttle.record_failure(email)
                raise DomainError("accounts.invalid_credentials", status=401)
            try:
                enforce_second_factor(
                    session,
                    row,
                    totp_code=body.totp_code,
                    email_code=body.email_code,
                    recovery_code=body.recovery_code,
                )
            except DomainError as exc:
                # A missing factor is the benign first leg of a legit 2FA login;
                # only a *wrong* code counts as a guess against the throttle.
                if exc.code == "accounts.invalid_second_factor":
                    throttle.record_failure(email)
                raise
            row.last_login_at = datetime.now(UTC)
            name = str(row.name)
            session.commit()
        throttle.reset(email)
        return AccountInfo(email=email, name=name, role=_role_for(email))

    return router
