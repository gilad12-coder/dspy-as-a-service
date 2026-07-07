"""add tagging_sessions.assist column for AI co-tagging state

Revision ID: b9c0d1e2f3a4
Revises: c7d8e9f0a1b2
Create Date: 2026-07-07 12:00:00.000000

Adds a nullable ``assist`` JSONB column to ``tagging_sessions`` carrying the
AI co-tagging state (assist mode, interview transcript, labeling rubric,
calibration/review progress, per-row predictions with provenance and
confidence, and bulk auto-tag job bookkeeping). ``NULL`` means a plain manual
session — pre-existing rows need no backfill and the manual tagger never
writes it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b9c0d1e2f3a4"
down_revision: str | Sequence[str] | None = "c7d8e9f0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable assist JSONB column."""
    op.execute("ALTER TABLE tagging_sessions ADD COLUMN IF NOT EXISTS assist JSONB")


def downgrade() -> None:
    """Drop the assist column."""
    op.execute("ALTER TABLE tagging_sessions DROP COLUMN IF EXISTS assist")
