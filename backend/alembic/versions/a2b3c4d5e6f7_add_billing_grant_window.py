"""add rolling free-grant window columns to billing_customers

Revision ID: a2b3c4d5e6f7
Revises: f0a1b2c3d4e5
Create Date: 2026-06-28 12:00:00.000000

Adds the two columns that back the rolling, non-cumulative free grant:
``grant_remaining`` (credits left in the current 200-credit window) and
``grant_reset_at`` (when it tops back up to a flat 200). Both are nullable so
existing rows are treated as a full grant on their next wallet read, which seeds
the window lazily. Postgres-only with ``IF NOT EXISTS`` to stay idempotent
against the boot-time ``create_all`` — the SQLite test schema comes from the ORM
models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "f0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add grant_remaining and grant_reset_at to billing_customers."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS grant_remaining BIGINT")
    op.execute(
        "ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS "
        "grant_reset_at TIMESTAMP WITH TIME ZONE"
    )


def downgrade() -> None:
    """Drop the rolling free-grant window columns."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS grant_reset_at")
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS grant_remaining")
