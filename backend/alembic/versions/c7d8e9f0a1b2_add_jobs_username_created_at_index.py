"""add composite index on jobs(username, created_at)

Revision ID: c7d8e9f0a1b2
Revises: b3c4d5e6f7a8
Create Date: 2026-07-04 09:00:00.000000

The dominant list/sidebar query is ``WHERE username = ? ORDER BY created_at
DESC LIMIT n`` (``list_jobs`` / ``list_jobs_sidebar``), which previously
filtered through the single-column ``username`` index and then sorted. This
composite index turns it into a single backward index range scan.

Built as a plain (transactional) CREATE INDEX, not CONCURRENTLY:
``sync_migration_head`` applies pending migrations on the schema-bootstrap
lock connection inside its transaction, where Alembic's ``autocommit_block``
cannot commit — a concurrent build here fails the boot-time upgrade
(exercised by ``test_adopted_database_applies_pending_migrations``). The
SHARE lock this takes on ``jobs`` lasts only as long as the build; the table
holds one row per optimization, so that is brief.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "c7d8e9f0a1b2"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the composite (username, created_at) index on jobs."""
    op.create_index(
        "ix_jobs_username_created_at",
        "jobs",
        ["username", "created_at"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    """Drop the composite (username, created_at) index."""
    op.drop_index(
        "ix_jobs_username_created_at",
        table_name="jobs",
        if_exists=True,
    )
