"""add password_reset_codes table

Revision ID: a7c1e93f24b8
Revises: 5c6d7e8f9a0b
Create Date: 2026-08-11 12:00:00.000000

Forgot-password flow: one active emailed reset code per account, holding only
the scrypt hash of the code, an ``expires_at``, an attempt counter, and a
``sent_at`` backing the resend cooldown. SQLite test schemas come from
``create_all``; this migration is Postgres-only, matching the storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a7c1e93f24b8"
down_revision: str | None = "5c6d7e8f9a0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``password_reset_codes`` table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_codes (
            email VARCHAR(255) PRIMARY KEY,
            code_hash VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def downgrade() -> None:
    """Drop the ``password_reset_codes`` table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS password_reset_codes")
