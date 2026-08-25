"""add user profile fields captured at sign-up

Revision ID: e9f0a1b2c3d4
Revises: d7e8f9a0b1c2
Create Date: 2026-06-25 12:00:00.000000

Adds nullable ``use_case``, ``experience_level``, and ``job_role`` columns to
``users``. They hold the optional profile collected on the rich sign-up form;
nullable because OAuth accounts (which never get a ``users`` row) and rows
created before this migration never set them, and ``job_role`` is optional even
on the form. Postgres-only, matching the rest of the suite — SQLite test schemas
come from ``create_all`` and already include the columns from the ORM model.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "e9f0a1b2c3d4"
down_revision: str | Sequence[str] | None = "d7e8f9a0b1c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the three nullable profile columns to ``users``."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS use_case VARCHAR(32)")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS experience_level VARCHAR(16)")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS job_role VARCHAR(16)")


def downgrade() -> None:
    """Drop the sign-up profile columns from ``users``."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS job_role")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS experience_level")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS use_case")
