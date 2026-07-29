"""add agent memory log and summary tables

Revision ID: 113ea57a3b82
Revises: 0f2a4c6e8b1d
Create Date: 2026-07-29 12:00:00.000000

Permanent per-user memory for the generalist agent (an OptMem port):
``agent_memories`` is the append-only log — one ≤280-char line per memory,
densely numbered per user — and ``agent_memory_summaries`` caches the
agent-authored compressions of aligned power-of-two blocks that form the
binary merge tree the wake context is rendered from. SQLite test schemas
come from ``create_all``; this migration is Postgres-only, matching the
storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "113ea57a3b82"
down_revision: str | None = "0f2a4c6e8b1d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the agent memory log and summary-tree tables."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memories (
            username VARCHAR(255) NOT NULL,
            seq INTEGER NOT NULL,
            content VARCHAR(280) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (username, seq)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_summaries (
            username VARCHAR(255) NOT NULL,
            block_size INTEGER NOT NULL,
            block_index INTEGER NOT NULL,
            content VARCHAR(280) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (username, block_size, block_index)
        )
        """
    )


def downgrade() -> None:
    """Drop the agent memory tables."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS agent_memory_summaries")
    op.execute("DROP TABLE IF EXISTS agent_memories")
