"""add email verification (users.email_verified + email_verification_codes)

Revision ID: b8d2fa41c96e
Revises: a7c1e93f24b8
Create Date: 2026-08-11 13:00:00.000000

Registration email confirmation: a ``users.email_verified`` flag plus one active
emailed confirmation code per account (scrypt hash, ``expires_at``, attempt
counter, and a ``sent_at`` backing the resend cooldown). Existing rows are
backfilled verified via the column default, so this migration never locks out
accounts created before it. SQLite test schemas come from ``create_all``; this
migration is Postgres-only, matching the storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b8d2fa41c96e"
down_revision: str | None = "a7c1e93f24b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``users.email_verified`` and create ``email_verification_codes``."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT TRUE"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS email_verification_codes (
            email VARCHAR(255) PRIMARY KEY,
            code_hash VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def downgrade() -> None:
    """Drop the confirmation-code table and the ``email_verified`` column."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS email_verification_codes")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS email_verified")
