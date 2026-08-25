"""add credit_ledger token-count columns

Revision ID: 7a40cd36ecc8
Revises: a3b4c5d6e7f8
Create Date: 2026-07-23 00:00:00.000000

Adds nullable input/output token counts to the credit ledger so run rows
record the measured usage behind their credit charge — the basis for the
Usage tab's per-model token breakdown. Nullable: pre-migration rows and
non-run rows (top-ups, grants) simply carry no counts.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "7a40cd36ecc8"
down_revision: str | None = "a3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable token-count columns to credit_ledger."""
    # SQLite (tests) builds the full schema via create_all and lacks
    # ``ADD COLUMN IF NOT EXISTS``; only Postgres runs migrations.
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        ALTER TABLE credit_ledger
            ADD COLUMN IF NOT EXISTS input_tokens BIGINT,
            ADD COLUMN IF NOT EXISTS output_tokens BIGINT
        """
    )


def downgrade() -> None:
    """Drop the token-count columns."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        ALTER TABLE credit_ledger
            DROP COLUMN IF EXISTS output_tokens,
            DROP COLUMN IF EXISTS input_tokens
        """
    )
