"""repair negative credit balances and forbid them at the DB layer

Revision ID: 4b5c6d7e8f9a
Revises: 3a4b5c6d7e8f
Create Date: 2026-08-10 12:00:00.000000

Historically ``debit_run`` subtracted a run's full cost unconditionally, so an
uncapped run could drive ``billing_customers.credit_balance`` negative (the
demo account sat at -409). Credits are prepaid — a negative balance is platform
money already spent, not user debt to collect — so this migration writes the
deficit off and then makes the state unrepresentable:

- Every negative ``credit_balance`` / ``grant_remaining`` is zeroed, with a
  matching positive ``adjustment`` ledger row so the audit trail still sums to
  the stored balance.
- ``CHECK`` constraints then forbid either column from going below zero, backing
  up the clamped debit in ``StripeBillingService.debit_run``.

Postgres-only; the SQLite test schema takes the same constraints from the ORM
models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "4b5c6d7e8f9a"
down_revision: str | Sequence[str] | None = "3a4b5c6d7e8f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Zero negative balances (ledgered as adjustments) and add the CHECKs."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "INSERT INTO credit_ledger (username, delta_credits, kind, description, created_at) "
        "SELECT username, -credit_balance, 'adjustment', 'Negative balance write-off', now() "
        "FROM billing_customers WHERE credit_balance < 0"
    )
    op.execute(
        "UPDATE billing_customers SET credit_balance = 0, updated_at = now() "
        "WHERE credit_balance < 0"
    )
    op.execute(
        "UPDATE billing_customers SET grant_remaining = 0, updated_at = now() "
        "WHERE grant_remaining < 0"
    )
    op.create_check_constraint(
        "ck_billing_customers_credit_balance_non_negative",
        "billing_customers",
        "credit_balance >= 0",
    )
    op.create_check_constraint(
        "ck_billing_customers_grant_remaining_non_negative",
        "billing_customers",
        "grant_remaining IS NULL OR grant_remaining >= 0",
    )


def downgrade() -> None:
    """Drop the CHECKs; the written-off deficits are not recoverable."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.drop_constraint(
        "ck_billing_customers_grant_remaining_non_negative",
        "billing_customers",
        type_="check",
    )
    op.drop_constraint(
        "ck_billing_customers_credit_balance_non_negative",
        "billing_customers",
        type_="check",
    )
