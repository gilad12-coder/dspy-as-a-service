"""add agent_approvals — cross-replica tool-approval handoff

Revision ID: a9b8c7d6e5f4
Revises: e2f3a4b5c6d7
Create Date: 2026-07-27 12:00:00.000000

The generalist agent's approval registry is per-process: the SSE stream
awaiting a decision and the confirm POST resolving it can land on different
replicas, in which case the confirm writes its decision here and the owning
replica's poll loop consumes it. Postgres-only with ``IF NOT EXISTS`` to stay
idempotent — the SQLite test schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a9b8c7d6e5f4"
down_revision: str | Sequence[str] | None = "e2f3a4b5c6d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the agent_approvals decision-handoff table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_approvals (
            call_id VARCHAR(64) PRIMARY KEY,
            approved BOOLEAN NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL
        )
        """
    )


def downgrade() -> None:
    """Drop the agent_approvals table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS agent_approvals")
