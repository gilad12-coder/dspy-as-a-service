"""add billing tables (Stripe customers, credit ledger, webhook idempotency)

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
Create Date: 2026-06-26 12:00:00.000000

Adds the three tables backing managed-credit billing: ``billing_customers``
(per-user Stripe customer link + denormalized credit balance + cached Premium
subscription state), ``credit_ledger`` (append-only signed credit movements),
and ``billing_webhook_events`` (the at-least-once webhook idempotency ledger).
All keyed on ``username`` (the lowercased email used as cross-app identity) so
SSO accounts without a ``users`` row are billed too. Postgres-only with
``IF NOT EXISTS`` to stay idempotent against the boot-time ``create_all`` — the
SQLite test schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f0a1b2c3d4e5"
down_revision: str | Sequence[str] | None = "e9f0a1b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the billing_customers, credit_ledger and webhook-event tables."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_customers (
            username VARCHAR(255) PRIMARY KEY,
            stripe_customer_id VARCHAR(64) NOT NULL UNIQUE,
            credit_balance BIGINT NOT NULL DEFAULT 0,
            subscription_status VARCHAR(32),
            subscription_price_id VARCHAR(64),
            subscription_current_period_end TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_billing_customers_stripe_customer_id "
        "ON billing_customers (stripe_customer_id)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS credit_ledger (
            id BIGSERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            delta_credits BIGINT NOT NULL,
            kind VARCHAR(16) NOT NULL,
            description VARCHAR(255) NOT NULL DEFAULT '',
            model VARCHAR(128),
            stripe_event_id VARCHAR(64),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_credit_ledger_username ON credit_ledger (username)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_credit_ledger_stripe_event_id "
        "ON credit_ledger (stripe_event_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_credit_ledger_created_at ON credit_ledger (created_at)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_webhook_events (
            event_id VARCHAR(64) PRIMARY KEY,
            event_type VARCHAR(64) NOT NULL,
            received_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    """Drop the billing tables."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS billing_webhook_events")
    op.execute("DROP TABLE IF EXISTS credit_ledger")
    op.execute("DROP TABLE IF EXISTS billing_customers")
