"""Second-factor verification shared by sign-in and security settings.

Leaf module (imports no routers) so both :mod:`core.api.routers.accounts`
(the login gate) and :mod:`core.api.routers.account_security` (enrollment)
can use it without a cycle. Covers the three accepted factors for password
sign-ins: TOTP codes from an authenticator app, emailed one-time codes, and
single-use recovery codes.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta

import pyotp
from sqlalchemy.orm import Session

from ..storage.models import TwoFactorEmailCodeModel, UserModel
from .email_sender import email_configured
from .errors import DomainError
from .passwords import hash_password, verify_password

TOTP_ISSUER = "Skynet"
EMAIL_CODE_TTL = timedelta(minutes=10)
EMAIL_CODE_MAX_ATTEMPTS = 5
_RECOVERY_CODE_COUNT = 8
# Unambiguous Crockford-style alphabet (no 0/O/1/I) so codes survive being
# read aloud or retyped from paper.
_RECOVERY_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_totp_secret() -> str:
    """Return a fresh base32 TOTP secret."""
    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, email: str) -> str:
    """Build the ``otpauth://`` URI an authenticator app enrolls from.

    Args:
        secret: Base32 TOTP secret.
        email: Account email shown as the account label in the app.

    Returns:
        The provisioning URI (rendered as a QR code by the frontend).
    """
    return pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name=TOTP_ISSUER)


def verify_totp(secret: str, code: str) -> bool:
    """Check a 6-digit TOTP code against a secret.

    Args:
        secret: Base32 TOTP secret.
        code: User-supplied code; spaces are tolerated.

    Returns:
        True when the code matches the current or adjacent time step.
    """
    cleaned = code.strip().replace(" ", "")
    if not cleaned:
        return False
    return pyotp.TOTP(secret).verify(cleaned, valid_window=1)


def generate_recovery_codes() -> list[str]:
    """Return a fresh set of plaintext recovery codes (shown to the user once)."""
    return [
        "-".join(
            "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(4)) for _ in range(2)
        )
        for _ in range(_RECOVERY_CODE_COUNT)
    ]


def hash_recovery_codes(codes: list[str]) -> str:
    """Hash plaintext recovery codes into the JSON blob stored on the user row.

    Args:
        codes: Plaintext codes from :func:`generate_recovery_codes`.

    Returns:
        A JSON array of scrypt hashes.
    """
    return json.dumps([hash_password(_normalise_recovery_code(code)) for code in codes])


def _normalise_recovery_code(code: str) -> str:
    """Uppercase a recovery code and strip separators so retyping is forgiving."""
    return code.strip().upper().replace("-", "").replace(" ", "")


def consume_recovery_code(row: UserModel, code: str) -> bool:
    """Verify a recovery code and burn it on success.

    Args:
        row: The user's ORM row (mutated in place; caller commits).
        code: User-supplied recovery code.

    Returns:
        True when the code matched an unused entry.
    """
    cleaned = _normalise_recovery_code(code)
    if not cleaned:
        return False
    try:
        hashes: list[str] = json.loads(str(row.recovery_codes or "[]"))
    except ValueError:
        return False
    for stored in hashes:
        if verify_password(cleaned, stored):
            hashes.remove(stored)
            row.recovery_codes = json.dumps(hashes)
            return True
    return False


def second_factor_methods(row: UserModel) -> list[str]:
    """Return the second-factor methods currently usable for this account.

    Args:
        row: The user's ORM row.

    Returns:
        A subset of ``["totp", "email"]``; empty when 2FA is off. Email-code
        2FA silently drops out when the deployment has no SMTP relay, so a
        config regression can't lock every email-2FA account out.
    """
    methods: list[str] = []
    if row.totp_secret:
        methods.append("totp")
    if bool(row.email_2fa_enabled) and email_configured():
        methods.append("email")
    return methods


def issue_email_code(session: Session, email: str) -> str:
    """Create (or replace) the active emailed sign-in code for an account.

    Args:
        session: Open ORM session (caller commits).
        email: Normalized account email.

    Returns:
        The plaintext 6-digit code to deliver.
    """
    code = f"{secrets.randbelow(1_000_000):06d}"
    row = session.get(TwoFactorEmailCodeModel, email)
    if row is None:
        row = TwoFactorEmailCodeModel(email=email)
        session.add(row)
    row.code_hash = hash_password(code)
    row.expires_at = datetime.now(UTC) + EMAIL_CODE_TTL
    row.attempts = 0
    return code


def verify_email_code(session: Session, email: str, code: str) -> bool:
    """Check an emailed code, counting attempts and consuming it on success.

    Args:
        session: Open ORM session (caller commits — including after a failed
            attempt, so the attempt counter always persists).
        email: Normalized account email.
        code: User-supplied 6-digit code.

    Returns:
        True when the code is valid, unexpired, and under the attempt cap.
    """
    row = session.get(TwoFactorEmailCodeModel, email)
    if row is None:
        return False
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < datetime.now(UTC) or int(row.attempts) >= EMAIL_CODE_MAX_ATTEMPTS:
        session.delete(row)
        return False
    if not verify_password(code.strip(), str(row.code_hash)):
        row.attempts = int(row.attempts) + 1
        return False
    session.delete(row)
    return True


def enforce_second_factor(
    session: Session,
    row: UserModel,
    *,
    totp_code: str,
    email_code: str,
    recovery_code: str,
) -> None:
    """Gate a password sign-in on the account's enabled second factor.

    No-op when the account has no usable second factor. Exactly one supplied
    code is checked, preferring TOTP > email > recovery when several arrive.

    Args:
        session: Open ORM session (caller commits; failed email attempts and
            consumed recovery codes mutate rows).
        row: The user's ORM row, already password-verified.
        totp_code: Authenticator code from the sign-in form ("" when absent).
        email_code: Emailed code from the sign-in form ("" when absent).
        recovery_code: Recovery code from the sign-in form ("" when absent).

    Raises:
        DomainError: 401 ``accounts.two_factor_required`` (with a ``methods``
            param) when no code was supplied; 401
            ``accounts.invalid_second_factor`` when the supplied code fails.
    """
    methods = second_factor_methods(row)
    if not methods:
        return
    if totp_code.strip():
        if "totp" in methods and verify_totp(str(row.totp_secret), totp_code):
            return
    elif email_code.strip():
        if "email" in methods and verify_email_code(session, str(row.email), email_code):
            return
    elif recovery_code.strip():
        if consume_recovery_code(row, recovery_code):
            return
    else:
        raise DomainError(
            "accounts.two_factor_required", status=401, methods=" ".join(methods)
        )
    # Persist the failed-attempt counter (and any consumed state) before the
    # error unwinds past the caller's commit.
    session.commit()
    raise DomainError("accounts.invalid_second_factor", status=401)
