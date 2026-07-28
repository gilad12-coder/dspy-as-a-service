"""add jobs.parent_optimization_id + pair_index — distributed grid pairs

Revision ID: b1c2d3e4f5a6
Revises: a9b8c7d6e5f4
Create Date: 2026-07-28 12:00:00.000000

A grid search fans each (generation, reflection) pair out as its own
claimable job row so pairs distribute across worker pods instead of
sharing one child process. Child rows point at their grid parent and
carry the global pair index; the self-referential cascade removes them
with the parent. Postgres-only with ``IF NOT EXISTS`` to stay
idempotent — the SQLite test schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: str | Sequence[str] | None = "a9b8c7d6e5f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the grid-pair child columns and their parent index."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        ALTER TABLE jobs
            ADD COLUMN IF NOT EXISTS parent_optimization_id VARCHAR(36)
                REFERENCES jobs (optimization_id) ON DELETE CASCADE,
            ADD COLUMN IF NOT EXISTS pair_index INTEGER
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_jobs_parent_optimization_id ON jobs (parent_optimization_id)"
    )


def downgrade() -> None:
    """Drop the grid-pair child columns."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_jobs_parent_optimization_id")
    op.execute(
        "ALTER TABLE jobs DROP COLUMN IF EXISTS pair_index, DROP COLUMN IF EXISTS parent_optimization_id"
    )
