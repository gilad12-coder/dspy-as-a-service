"""drop guarantee_runs — the no-lift refund feature is removed

Revision ID: f2a3b4c5d6e7
Revises: 7a40cd36ecc8
Create Date: 2026-07-27 12:00:00.000000

The "No lift, no charge" guarantee no longer exists: every run bills normally,
so the first-run-per-task claim table has nothing to back. Postgres-only with
``IF EXISTS`` to stay idempotent — the SQLite test schema comes from the ORM
models directly (which no longer declare the table).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "7a40cd36ecc8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the guarantee_runs table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS guarantee_runs")


def downgrade() -> None:
    """Recreate the guarantee_runs table as the add_guarantee_runs revision defined it."""
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
