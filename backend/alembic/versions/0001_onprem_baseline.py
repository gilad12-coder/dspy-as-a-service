"""Create the fresh PostgreSQL schema for an on-premises deployment."""

from __future__ import annotations

from alembic import op

from core.storage.models import Base, ConversationEmbeddingModel, JobEmbeddingModel

revision = "0001_onprem"
down_revision = None
branch_labels = None
depends_on = None


def _baseline_tables() -> list:
    """Return tables that work on ordinary PostgreSQL without extensions."""
    optional_embeddings = {
        JobEmbeddingModel.__table__,
        ConversationEmbeddingModel.__table__,
    }
    return [table for table in Base.metadata.sorted_tables if table not in optional_embeddings]


def upgrade() -> None:
    """Create the complete extension-free on-premises baseline."""
    Base.metadata.create_all(bind=op.get_bind(), tables=_baseline_tables())


def downgrade() -> None:
    """Drop the on-premises baseline schema."""
    Base.metadata.drop_all(bind=op.get_bind(), tables=_baseline_tables())
