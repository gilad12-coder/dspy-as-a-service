"""drop job_templates table

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-27 14:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the ``job_templates`` table, retiring the saved-config feature."""
    op.execute("DROP TABLE IF EXISTS job_templates")


def downgrade() -> None:
    """Recreate the baseline ``job_templates`` shape so a full downgrade chain stays consistent."""
    op.create_table(
        "job_templates",
        sa.Column("template_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("username", sa.String(length=255), nullable=False),
        sa.Column(
            "config",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("template_id"),
        if_not_exists=True,
    )
