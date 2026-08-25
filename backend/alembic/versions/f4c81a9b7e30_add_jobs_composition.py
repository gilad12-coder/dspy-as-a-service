"""add jobs.composition classifier

Revision ID: f4c81a9b7e30
Revises: c212bc389087
Create Date: 2026-08-12 09:00:00.000000

Adds the ``jobs.composition`` column — "single" for a run over one atomic DSPy
module, "workflow" for a run over a workflow (a DAG of module nodes). It is a
first-class indexed classifier, orthogonal to ``optimization_type``, hoisted
from the payload overview at submit so a run's shape is queryable without
probing ``module_name`` inside the JSON overview. The write path maintains it
going forward; this migration backfills existing rows once from the persisted
overview. Postgres-only: SQLite test schemas come from ``create_all`` with the
ORM column already declared, so the guard early-returns on other dialects.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f4c81a9b7e30"
down_revision: str | None = "c212bc389087"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``jobs.composition`` and backfill it from the persisted overview."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS composition VARCHAR(32)")
    op.execute(
        """
        UPDATE jobs
        SET composition = CASE
            WHEN lower(payload_overview ->> 'module_name') = 'workflow' THEN 'workflow'
            ELSE 'single'
        END
        WHERE composition IS NULL
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_jobs_composition ON jobs (composition)")


def downgrade() -> None:
    """Drop the ``jobs.composition`` column and its index."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_jobs_composition")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS composition")
