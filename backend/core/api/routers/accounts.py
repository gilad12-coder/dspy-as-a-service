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
import logging
import re
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import settings
from ...storage.models import UserModel
from ..email_sender import email_configured, send_email
from ..email_verification import (
    issue_verification_code,
    verification_code_on_cooldown,
    verify_verification_code,
)
from ..errors import DomainError
from ..login_throttle import LoginThrottle
from ..password_policy import validate_password
from ..password_reset import issue_reset_code, reset_code_on_cooldown, verify_reset_code
from ..passwords import hash_password, verify_password
from ..two_factor import enforce_second_factor

logger = logging.getLogger(__name__)

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


# Simple acknowledgement for state-changing auth calls. Lives here (not in the
# account-security router) because that router imports from this leaf module, so
# a shared response model can only flow in this direction without a cycle.
class OkResponse(BaseModel):
    ok: bool = Field(default=True, description="Always true on success.")


# Account email to email a password-reset code to (forgot-password step one).
class PasswordResetRequest(BaseModel):
    email: str = Field(description="Account email to send a one-time reset code to.")


# Reset code plus the new password (forgot-password step two).
class PasswordResetConfirm(BaseModel):
    email: str = Field(description="Account email being reset.")
    code: str = Field(description="One-time code delivered by email.")
    new_password: str = Field(description="New plaintext password; stored only as a scrypt hash.")


# Account email to (re-)send an email-confirmation code to.
class EmailVerifyRequest(BaseModel):
    email: str = Field(description="Account email to send a one-time confirmation code to.")


# Emailed confirmation code proving control of the sign-up address.
class EmailVerifyConfirm(BaseModel):
    email: str = Field(description="Account email being confirmed.")
    code: str = Field(description="One-time code delivered by email.")


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


def _enforce_signup_cap(job_store) -> None:
    """Refuse new registrations once the platform-wide account cap is reached.

    A cost guardrail that bounds total accounts (and thus the ceiling on
    concurrent demand) at :data:`core.config.settings.max_total_users`. No-op
    when the cap is disabled (``0``) or the store cannot count users. The cap is
    soft: the count runs outside the insert transaction, so simultaneous sign-ups
    at the boundary may overshoot by a few — acceptable for a cost bound.

    Args:
        job_store: Job-store instance; a store without ``count_users`` is skipped.

    Raises:
        DomainError: 503 ``accounts.signups_closed`` when the account count is at
            or above the cap.
    """
    cap = settings.max_total_users
    if cap <= 0:
        return
    counter = getattr(job_store, "count_users", None)
    if not callable(counter):
        return
    if counter() >= cap:
        raise DomainError("accounts.signups_closed", status=503)


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

        When email delivery is configured the account is created unverified and
        a one-time confirmation code is emailed; sign-in is refused until it is
        confirmed. On a deployment with no SMTP relay there is nothing to verify
        against, so the account is created already-verified. A failure to send
        the code is swallowed: the account exists and the user can trigger a
        resend from the verification step.

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
        verified = not email_configured()
        _enforce_signup_cap(job_store)
        with Session(job_store.engine) as session:
            if session.get(UserModel, email) is not None:
                raise DomainError("accounts.email_taken", status=409)
            session.add(
                UserModel(
                    email=email,
                    name=name,
                    password_hash=hash_password(body.password),
                    created_at=datetime.now(UTC),
                    email_verified=verified,
                    use_case=_coerce_choice(body.use_case, _USE_CASES),
                    experience_level=_coerce_choice(body.experience_level, _EXPERIENCE_LEVELS),
                    job_role=_coerce_choice(body.job_role, _JOB_ROLES),
                )
            )
            code = None if verified else issue_verification_code(session, email)
            session.commit()
        if code is not None:
            # Best-effort: the account already exists, so a send failure must
            # not fail registration — the user can trigger a resend. Logged so a
            # broken relay surfaces (the alert sink forwards warnings) instead of
            # silently stranding new sign-ups on the verification step.
            try:
                send_email(
                    email,
                    "Confirm your Skynet email",
                    f"Your Skynet confirmation code is {code}. It expires in 30 minutes.\n\n"
                    "If you didn't create a Skynet account, you can ignore this email.",
                )
            except (OSError, RuntimeError):
                logger.warning("Failed to send verification email at registration", exc_info=True)
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
            DomainError: 403 on a bad internal secret, or when the account's
                email is not yet confirmed (``accounts.email_not_verified``);
                429 (``accounts.too_many_attempts``) when the email is locked
                out after repeated failures; 401 when the email is unknown, the
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
            # Only meaningful once the password is proven, so an unconfirmed
            # account never becomes an oracle for a wrong password. Gated on
            # ``email_configured`` so a mail-less deployment (where accounts are
            # created already-verified) can still sign in.
            if email_configured() and not bool(row.email_verified):
                raise DomainError("accounts.email_not_verified", status=403)
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

    @router.post(
        "/auth/password-reset/request",
        response_model=OkResponse,
        summary="Email a one-time password-reset code (internal)",
    )
    def password_reset_request(
        body: PasswordResetRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> OkResponse:
        """Email a reset code to an account that has forgotten its password.

        Returns the same acknowledgement whether or not the address has an
        account, so this route can't be used to enumerate registered emails. A
        per-account cooldown bounds how often a code is sent, keeping it from
        being a mail-bomb oracle since no password is required here.

        Args:
            body: The account email to send a code to.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            Acknowledgement (identical for known and unknown emails).

        Raises:
            DomainError: 403 on a bad internal secret; 422 when the deployment
                has no SMTP relay (a global, account-independent condition);
                502 when the relay rejects the send.
        """
        _require_internal_auth(x_internal_auth)
        if not email_configured():
            raise DomainError("accounts.email_delivery_unavailable", status=422)
        email = _normalise_email(body.email)
        with Session(job_store.engine) as session:
            if session.get(UserModel, email) is None or reset_code_on_cooldown(session, email):
                return OkResponse()
            code = issue_reset_code(session, email)
            session.commit()
        try:
            send_email(
                email,
                "Your Skynet password-reset code",
                f"Your Skynet password-reset code is {code}. It expires in 30 minutes.\n\n"
                "If you didn't ask to reset your password, you can ignore this email.",
            )
        except (OSError, RuntimeError) as exc:
            raise DomainError("accounts.email_send_failed", status=502) from exc
        return OkResponse()

    @router.post(
        "/auth/password-reset/confirm",
        response_model=OkResponse,
        summary="Set a new password with a reset code (internal)",
    )
    def password_reset_confirm(
        body: PasswordResetConfirm,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> OkResponse:
        """Verify a reset code and set the account's new password.

        The account's second factor is left untouched, so a reset restores only
        password knowledge and a later sign-in still passes through 2FA.

        Args:
            body: Account email, the emailed code, and the new password.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 403 on a bad internal secret; 422 when the new password
                fails the acceptance policy, or ``accounts.invalid_reset_code``
                when the code is unknown, wrong, expired, or spent (also raised
                for an unknown email, so neither leaks account existence).
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        validate_password(body.new_password, email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            if row is None or not verify_reset_code(session, email, body.code):
                session.commit()
                raise DomainError("accounts.invalid_reset_code", status=422)
            row.password_hash = hash_password(body.new_password)
            session.commit()
        return OkResponse()

    @router.post(
        "/auth/email-verify/request",
        response_model=OkResponse,
        summary="Email a one-time account-confirmation code (internal)",
    )
    def email_verify_request(
        body: EmailVerifyRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> OkResponse:
        """Email a confirmation code to an account that has not verified yet.

        Returns the same acknowledgement whether or not the address has an
        account and whether or not it is already verified, so this route can't
        be used to enumerate accounts or their verification state. A per-account
        cooldown bounds how often a code is sent, keeping it from being a
        mail-bomb oracle since no password is required here.

        Args:
            body: The account email to send a code to.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            Acknowledgement (identical for known, unknown, and already-verified
            emails).

        Raises:
            DomainError: 403 on a bad internal secret; 422 when the deployment
                has no SMTP relay; 502 when the relay rejects the send.
        """
        _require_internal_auth(x_internal_auth)
        if not email_configured():
            raise DomainError("accounts.email_delivery_unavailable", status=422)
        email = _normalise_email(body.email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            nothing_to_send = (
                row is None
                or bool(row.email_verified)
                or verification_code_on_cooldown(session, email)
            )
            if nothing_to_send:
                return OkResponse()
            code = issue_verification_code(session, email)
            session.commit()
        try:
            send_email(
                email,
                "Confirm your Skynet email",
                f"Your Skynet confirmation code is {code}. It expires in 30 minutes.\n\n"
                "If you didn't create a Skynet account, you can ignore this email.",
            )
        except (OSError, RuntimeError) as exc:
            raise DomainError("accounts.email_send_failed", status=502) from exc
        return OkResponse()

    @router.post(
        "/auth/email-verify/confirm",
        response_model=OkResponse,
        summary="Confirm an account email with a one-time code (internal)",
    )
    def email_verify_confirm(
        body: EmailVerifyConfirm,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> OkResponse:
        """Verify an emailed code and mark the account's email confirmed.

        Confirming grants no session — it only flips the verified flag, so a
        later sign-in still needs the password (and any second factor). The same
        error is returned for an unknown email and a bad code, so neither leaks
        account existence.

        Args:
            body: Account email and the emailed confirmation code.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 403 on a bad internal secret;
                ``accounts.invalid_verification_code`` (422) when the code is
                unknown, wrong, expired, or spent.
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            if row is None or not verify_verification_code(session, email, body.code):
                session.commit()
                raise DomainError("accounts.invalid_verification_code", status=422)
            row.email_verified = True
            session.commit()
        return OkResponse()

    return router
