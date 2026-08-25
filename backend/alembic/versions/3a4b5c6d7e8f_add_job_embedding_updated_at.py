"""Track the last successful write to each job embedding row.

The Explore cache needs to distinguish a refreshed embedding from an older
row with the same optimization id. ``created_at`` identifies when a row was
first inserted and therefore cannot invalidate the cache after a resumed job
updates its metrics and summary.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from core.config import embeddings_schema_enabled

revision: str = "3a4b5c6d7e8f"
down_revision: str | Sequence[str] | None = "29d9f24ddac9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add and backfill the embedding freshness timestamp."""
    if not embeddings_schema_enabled():
        return
    op.add_column(
        "job_embeddings",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE job_embeddings SET updated_at = created_at WHERE updated_at IS NULL")
    op.alter_column("job_embeddings", "updated_at", nullable=False)
    op.create_index(
        op.f("ix_job_embeddings_updated_at"),
        "job_embeddings",
        ["updated_at"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    """Remove the embedding freshness timestamp."""
    if not embeddings_schema_enabled():
        return
    op.drop_index(op.f("ix_job_embeddings_updated_at"), table_name="job_embeddings", if_exists=True)
    op.drop_column("job_embeddings", "updated_at")
