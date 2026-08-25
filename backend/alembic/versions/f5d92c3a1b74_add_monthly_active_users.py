"""add monthly active user admission table

Revision ID: f5d92c3a1b74
Revises: f4c81a9b7e30
Create Date: 2026-08-12 22:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f5d92c3a1b74"
down_revision: str | Sequence[str] | None = "f4c81a9b7e30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the UTC monthly identity-admission table idempotently."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_active_users (
            month_start DATE NOT NULL,
            username VARCHAR(255) NOT NULL,
            first_seen_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (month_start, username)
        )
        """
    )


def downgrade() -> None:
    """Drop the monthly identity-admission table."""
    op.execute("DROP TABLE IF EXISTS monthly_active_users")
