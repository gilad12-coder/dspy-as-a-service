"""Email-confirmation code lifecycle for Skynet-native accounts.

Leaf module (imports no routers) used by :mod:`core.api.routers.accounts` to
confirm that a new account controls the address it signed up with: issue a
one-time emailed code at registration, then verify it before the account is
allowed to sign in. Mirrors the emailed-code machinery in
:mod:`core.api.password_reset` but keeps a separate table so a pending
confirmation never collides with a pending sign-in or reset code.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from ..storage.models import EmailVerificationCodeModel
from .passwords import hash_password, verify_password

VERIFY_CODE_TTL = timedelta(minutes=30)
VERIFY_CODE_MAX_ATTEMPTS = 5
# Ceiling on how often a fresh code is emailed to one address. The resend route
# takes no password, so this cooldown is the mail-bomb guard.
VERIFY_CODE_RESEND_COOLDOWN = timedelta(seconds=60)


def _aware(value: datetime) -> datetime:
    """Return a timezone-aware copy of a datetime, assuming UTC when naive.

    SQLite round-trips ``DateTime(timezone=True)`` columns as naive, so a value
    read back from the test store needs its UTC tzinfo restored before it can be
    compared against :func:`datetime.now`.

    Args:
        value: A datetime that may be naive (from SQLite) or aware (Postgres).

    Returns:
        The same instant with UTC tzinfo attached when it was missing.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def verification_code_on_cooldown(session: Session, email: str) -> bool:
    """Return whether a confirmation code was emailed to this account too recently.

    Args:
        session: Open ORM session.
        email: Normalized account email.

    Returns:
        True when an unexpired-cooldown code exists, so the caller should skip
        re-sending; False when no code exists or the cooldown has elapsed.
    """
    row = session.get(EmailVerificationCodeModel, email)
    if row is None:
        return False
    return _aware(row.sent_at) + VERIFY_CODE_RESEND_COOLDOWN > datetime.now(UTC)


def issue_verification_code(session: Session, email: str) -> str:
    """Create (or replace) the active email-confirmation code for an account.

    Args:
        session: Open ORM session (caller commits).
        email: Normalized account email.

    Returns:
        The plaintext 6-digit code to deliver.
    """
    code = f"{secrets.randbelow(1_000_000):06d}"
    now = datetime.now(UTC)
    row = session.get(EmailVerificationCodeModel, email)
    if row is None:
        row = EmailVerificationCodeModel(email=email)
        session.add(row)
    row.code_hash = hash_password(code)
    row.expires_at = now + VERIFY_CODE_TTL
    row.sent_at = now
    row.attempts = 0
    return code


def verify_verification_code(session: Session, email: str, code: str) -> bool:
    """Check a confirmation code, counting attempts and consuming it on success.

    Args:
        session: Open ORM session (caller commits — including after a failed
            attempt, so the attempt counter always persists).
        email: Normalized account email.
        code: User-supplied 6-digit code.

    Returns:
        True when the code is valid, unexpired, and under the attempt cap. The
        row is consumed on success and dropped once expired or exhausted.
    """
    row = session.get(EmailVerificationCodeModel, email)
    if row is None:
        return False
    if _aware(row.expires_at) < datetime.now(UTC) or int(row.attempts) >= VERIFY_CODE_MAX_ATTEMPTS:
        session.delete(row)
        return False
    if not verify_password(code.strip(), str(row.code_hash)):
        row.attempts = int(row.attempts) + 1
        return False
    session.delete(row)
    return True
