"""add agent memory settings table

Revision ID: 29d9f24ddac9
Revises: 113ea57a3b82
Create Date: 2026-07-29 18:00:00.000000

Per-user overrides for the agent-memory size knobs (the OptMem ``config``
surface): wake budget, entry length, and recall output cap. A NULL column —
or no row — means the knob follows the tool default. SQLite test schemas
come from ``create_all``; this migration is Postgres-only, matching the
storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "29d9f24ddac9"
down_revision: str | None = "113ea57a3b82"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the agent memory settings table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_settings (
            username VARCHAR(255) PRIMARY KEY,
            wake_lines INTEGER,
            entry_chars INTEGER,
            recall_chars INTEGER,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    """Drop the agent memory settings table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS agent_memory_settings")
