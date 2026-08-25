"""add tagging_sessions table (merges byok + telemetry heads)

Revision ID: f2b3c4d5e6a7
Revises: c5d6e7f8a9b0, d8e9f0a1b2c3
Create Date: 2026-06-30 12:00:00.000000

Adds persistence for text-labeling (tagger) sessions so users can resume
annotating across refreshes and devices, mirroring how optimizations persist.
``tagging_sessions`` holds the full session payload (config, columns, data,
annotations) alongside denormalized sidebar columns (name, phase, row_count,
tagged_count, pinned). This revision also merges the two open heads — byok
provider connections and telemetry events — back into a single linear head.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f2b3c4d5e6a7"
down_revision: str | Sequence[str] | None = ("c5d6e7f8a9b0", "d8e9f0a1b2c3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create tagging_sessions with its owner/recency/pinned indexes."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tagging_sessions (
            id VARCHAR(36) PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            name VARCHAR(200) NOT NULL DEFAULT '',
            phase VARCHAR(20) NOT NULL DEFAULT 'annotating',
            config JSONB NOT NULL DEFAULT '{}'::jsonb,
            columns JSONB NOT NULL DEFAULT '[]'::jsonb,
            data JSONB NOT NULL DEFAULT '[]'::jsonb,
            annotations JSONB NOT NULL DEFAULT '{}'::jsonb,
            current_index INTEGER NOT NULL DEFAULT 0,
            row_count INTEGER NOT NULL DEFAULT 0,
            tagged_count INTEGER NOT NULL DEFAULT 0,
            pinned BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )
    op.create_index(
        "ix_tagging_sessions_username",
        "tagging_sessions",
        ["username"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_tagging_sessions_user_updated",
        "tagging_sessions",
        ["username", "updated_at"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_tagging_sessions_user_pinned",
        "tagging_sessions",
        ["username", "pinned"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    """Drop the tagging_sessions table and its indexes."""
    op.drop_index("ix_tagging_sessions_user_pinned", table_name="tagging_sessions", if_exists=True)
    op.drop_index("ix_tagging_sessions_user_updated", table_name="tagging_sessions", if_exists=True)
    op.drop_index("ix_tagging_sessions_username", table_name="tagging_sessions", if_exists=True)
    op.execute("DROP TABLE IF EXISTS tagging_sessions")
