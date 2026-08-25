"""add two-factor columns and passkey tables

Revision ID: 0f2a4c6e8b1d
Revises: b1c2d3e4f5a6
Create Date: 2026-07-29 12:00:00.000000

Account security: TOTP + emailed-code two-factor for password sign-ins and
WebAuthn passkeys for every identity. Adds the ``users`` 2FA columns
(``totp_secret``, ``totp_pending_secret``, ``email_2fa_enabled``,
``recovery_codes``), the ``webauthn_credentials`` table (registered passkeys,
keyed by base64url credential id, owner indexed), the single-use
``webauthn_challenges`` table backing in-flight ceremonies across replicas,
and ``two_factor_email_codes`` (one active emailed code per account).
SQLite test schemas come from ``create_all``; this migration is
Postgres-only, matching the storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0f2a4c6e8b1d"
down_revision: str | None = "b1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the 2FA columns and create the passkey/challenge/code tables."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret VARCHAR(64)")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_pending_secret VARCHAR(64)")
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_2fa_enabled BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS recovery_codes TEXT")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS webauthn_credentials (
            credential_id VARCHAR(1024) PRIMARY KEY,
            user_email VARCHAR(255) NOT NULL,
            public_key TEXT NOT NULL,
            sign_count INTEGER NOT NULL DEFAULT 0,
            transports VARCHAR(255),
            nickname VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            last_used_at TIMESTAMP WITH TIME ZONE
        )
        """
    )
    op.create_index(
        "ix_webauthn_credentials_user_email",
        "webauthn_credentials",
        ["user_email"],
        unique=False,
        if_not_exists=True,
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS webauthn_challenges (
            challenge VARCHAR(255) PRIMARY KEY,
            purpose VARCHAR(16) NOT NULL,
            user_email VARCHAR(255),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS two_factor_email_codes (
            email VARCHAR(255) PRIMARY KEY,
            code_hash VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def downgrade() -> None:
    """Drop the passkey/challenge/code tables and the ``users`` 2FA columns."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS two_factor_email_codes")
    op.execute("DROP TABLE IF EXISTS webauthn_challenges")
    op.execute("DROP TABLE IF EXISTS webauthn_credentials")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS recovery_codes")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS email_2fa_enabled")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS totp_pending_secret")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS totp_secret")
