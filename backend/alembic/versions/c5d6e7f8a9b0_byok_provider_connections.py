"""generalize byok provider keys into connections (id PK, api_base, params, label)

Revision ID: c5d6e7f8a9b0
Revises: c4d5e6f7a8b0
Create Date: 2026-06-29 12:00:00.000000

Generalizes ``billing_provider_keys`` from one-key-per-provider into flexible
LiteLLM-style *connections*: a surrogate ``id`` becomes the primary key (so an
account may hold several connections for one provider — e.g. two
OpenAI-compatible endpoints), and ``api_base`` / ``params`` (JSONB) / ``label``
columns let a connection target any custom host with extra kwargs. Existing
rows are preserved: each keeps its ``(username, provider)`` data and is assigned
a deterministic ``id`` so no key is lost. ``(username, provider)`` stays indexed
for the run-path and settings lookups. Postgres-only with ``IF [NOT] EXISTS``
guards to stay idempotent against the boot-time ``create_all`` — the SQLite test
schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "c5d6e7f8a9b0"
down_revision: str | Sequence[str] | None = "c4d5e6f7a8b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the connection columns and move the primary key onto a surrogate id."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE billing_provider_keys ADD COLUMN IF NOT EXISTS id VARCHAR(32)")
    op.execute("ALTER TABLE billing_provider_keys ADD COLUMN IF NOT EXISTS label VARCHAR(120)")
    op.execute("ALTER TABLE billing_provider_keys ADD COLUMN IF NOT EXISTS api_base VARCHAR(255)")
    op.execute(
        "ALTER TABLE billing_provider_keys "
        "ADD COLUMN IF NOT EXISTS params JSONB NOT NULL DEFAULT '{}'::jsonb"
    )
    # Deterministic id from the old composite key — unique because (username,
    # provider) was the prior primary key, so every legacy row is preserved.
    op.execute(
        "UPDATE billing_provider_keys SET id = md5(username || ':' || provider) WHERE id IS NULL"
    )
    op.execute("ALTER TABLE billing_provider_keys ALTER COLUMN id SET NOT NULL")
    op.execute("ALTER TABLE billing_provider_keys DROP CONSTRAINT IF EXISTS billing_provider_keys_pkey")
    op.execute("ALTER TABLE billing_provider_keys ADD PRIMARY KEY (id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_billing_provider_keys_username "
        "ON billing_provider_keys (username)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_billing_provider_keys_username_provider "
        "ON billing_provider_keys (username, provider)"
    )


def downgrade() -> None:
    """Restore the ``(username, provider)`` primary key and drop the new columns.

    Best-effort: re-adding the composite primary key fails if an account holds
    more than one connection for a provider (only possible once the generalized
    UI is in use), which is the intended guard against silent data loss.
    """
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_billing_provider_keys_username_provider")
    op.execute("DROP INDEX IF EXISTS ix_billing_provider_keys_username")
    op.execute("ALTER TABLE billing_provider_keys DROP CONSTRAINT IF EXISTS billing_provider_keys_pkey")
    op.execute("ALTER TABLE billing_provider_keys ADD PRIMARY KEY (username, provider)")
    op.execute("ALTER TABLE billing_provider_keys DROP COLUMN IF EXISTS params")
    op.execute("ALTER TABLE billing_provider_keys DROP COLUMN IF EXISTS api_base")
    op.execute("ALTER TABLE billing_provider_keys DROP COLUMN IF EXISTS label")
    op.execute("ALTER TABLE billing_provider_keys DROP COLUMN IF EXISTS id")
