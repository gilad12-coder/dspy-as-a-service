"""grant existing free accounts the one-time 500-credit lifetime grant

Revision ID: b3c4d5e6f7a8
Revises: f2b3c4d5e6a7
Create Date: 2026-07-02 11:30:00.000000

The free grant moved from a renewing 200-credit/30-day window to a one-time
500-credit lifetime grant (see ``core.billing.service.FREE_GRANT_CREDITS``). This
data migration brings existing accounts onto the new deal in one shot:

- Every non-Premium account is topped up to at least 500 credits (``GREATEST`` so a
  former subscriber's larger leftover is never clawed back) and its reset anchor is
  cleared to NULL — the grant no longer renews.
- Active/trialing/past_due (Premium) accounts are left untouched: their monthly
  allotment still renews, anchored to the Stripe billing period.

New accounts that never materialized a billing row are unaffected — they are seeded
to the one-time 500 lazily on their first wallet read or run. Postgres-only; the
SQLite test schema comes from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: str | Sequence[str] | None = "f2b3c4d5e6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Top non-Premium accounts up to the one-time 500 grant and clear their reset."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "UPDATE billing_customers "
        "SET grant_remaining = GREATEST(COALESCE(grant_remaining, 0), 500), "
        "    grant_reset_at = NULL, "
        "    updated_at = now() "
        "WHERE subscription_status IS NULL "
        "   OR subscription_status NOT IN ('active', 'trialing', 'past_due')"
    )


def downgrade() -> None:
    """No-op: the prior per-account grant/anchor values are not recoverable."""
    return
