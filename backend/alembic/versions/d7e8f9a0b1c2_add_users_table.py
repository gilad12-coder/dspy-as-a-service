"""add users table for email/password accounts

Revision ID: d7e8f9a0b1c2
Revises: c1d2e3f4a5b6
Create Date: 2026-06-24 12:00:00.000000

Backs the Skynet-native email/password sign-in shipped with hosted auth. OAuth
users never get a row here; this table holds only local accounts, keyed by the
lowercased email that doubles as the cross-app identity (``username``).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "d7e8f9a0b1c2"
down_revision: str | Sequence[str] | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``users`` table for local accounts.

    ``IF NOT EXISTS`` mirrors the api_tokens / search_query_log migrations: the
    app runs ``Base.metadata.create_all`` on boot, so the table may already
    exist when migrations run — the create must be idempotent.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            last_login_at TIMESTAMP WITH TIME ZONE
        )
        """
    )


def downgrade() -> None:
    """Drop the ``users`` table."""
    op.execute("DROP TABLE IF EXISTS users")
