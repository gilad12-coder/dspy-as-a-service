"""Add optional email-notification preferences.

Revision ID: fa6b7c8d9e0f
Revises: f5d92c3a1b74
Create Date: 2026-08-13 00:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "fa6b7c8d9e0f"
down_revision: str | Sequence[str] | None = "f5d92c3a1b74"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the durable per-identity preference table idempotently."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_preferences (
            username VARCHAR(255) PRIMARY KEY,
            job_updates_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            sharing_updates_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    """Drop the notification preference table."""
    op.execute("DROP TABLE IF EXISTS notification_preferences")
