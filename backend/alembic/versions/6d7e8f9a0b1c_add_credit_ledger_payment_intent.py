"""add credit_ledger.stripe_payment_intent_id for refund/dispute clawbacks

Revision ID: 6d7e8f9a0b1c
Revises: 5c6d7e8f9a0b
Create Date: 2026-08-11 12:00:00.000000

Adds the ``credit_ledger.stripe_payment_intent_id`` column that ties a top-up
row to the Stripe PaymentIntent behind it. It is the join key the
``charge.refunded`` and ``charge.dispute.created`` webhook handlers use to find
the account and the credits to claw back — the one id present on the checkout,
charge, and dispute objects alike (a dispute event carries no ``customer``). The
same column is stamped on the negative refund/dispute ledger rows so a later
clawback nets against what was already reversed. Top-ups predating this column
leave it NULL and cannot be auto-reversed. Postgres-only with
``IF NOT EXISTS`` to stay idempotent against the boot-time ``create_all`` — the
SQLite test schema takes the column from the ORM model directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "6d7e8f9a0b1c"
down_revision: str | Sequence[str] | None = "5c6d7e8f9a0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable payment-intent column and its lookup index."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE credit_ledger ADD COLUMN IF NOT EXISTS stripe_payment_intent_id VARCHAR(64)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_credit_ledger_stripe_payment_intent_id "
        "ON credit_ledger (stripe_payment_intent_id)"
    )


def downgrade() -> None:
    """Drop the payment-intent column and its index."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_credit_ledger_stripe_payment_intent_id")
    op.execute("ALTER TABLE credit_ledger DROP COLUMN IF EXISTS stripe_payment_intent_id")
