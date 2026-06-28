"""add guarantee_runs table for first-run-per-task tracking

Revision ID: b3c4d5e6f7a9
Revises: a2b3c4d5e6f7
Create Date: 2026-06-28 13:00:00.000000

Backs the "No lift, no charge" guarantee's anti-gaming rule: one row per
``(username, task_fingerprint)`` records that an account has spent its single
guaranteed run on that task, so re-runs bill normally. ``refunded`` flips true
when that first run had no lift and was auto-refunded. Postgres-only with
``IF NOT EXISTS`` to stay idempotent against the boot-time ``create_all`` — the
SQLite test schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b3c4d5e6f7a9"
down_revision: str | Sequence[str] | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the guarantee_runs table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "CREATE TABLE IF NOT EXISTS guarantee_runs ("
        "username VARCHAR(255) NOT NULL, "
        "task_fingerprint VARCHAR(64) NOT NULL, "
        "optimization_id VARCHAR(36) NOT NULL, "
        "refunded BOOLEAN NOT NULL DEFAULT false, "
        "created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
        "PRIMARY KEY (username, task_fingerprint))"
    )


def downgrade() -> None:
    """Drop the guarantee_runs table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS guarantee_runs")
