"""drop subscription columns — Premium and the Founder's Rate are removed

Revision ID: e2f3a4b5c6d7
Revises: f2a3b4c5d6e7
Create Date: 2026-07-27 12:00:00.000000

Billing is pay-as-you-go prepaid credits only: there is no subscription to
mirror from Stripe and no renewing grant, so the ``subscription_*`` cache
columns and the ``grant_reset_at`` renewal anchor have nothing to back.
Postgres-only with ``IF EXISTS`` to stay idempotent — the SQLite test schema
comes from the ORM models directly (which no longer declare the columns).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: str | Sequence[str] | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the subscription-mirror and grant-renewal columns from billing_customers."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS subscription_status")
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS subscription_price_id")
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS subscription_current_period_end")
    op.execute("ALTER TABLE billing_customers DROP COLUMN IF EXISTS grant_reset_at")


def downgrade() -> None:
    """Re-add the columns as the add_billing_tables / grant_window revisions defined them."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(32)")
    op.execute("ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS subscription_price_id VARCHAR(64)")
    op.execute(
        "ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS subscription_current_period_end TIMESTAMP WITH TIME ZONE"
    )
    op.execute("ALTER TABLE billing_customers ADD COLUMN IF NOT EXISTS grant_reset_at TIMESTAMP WITH TIME ZONE")
