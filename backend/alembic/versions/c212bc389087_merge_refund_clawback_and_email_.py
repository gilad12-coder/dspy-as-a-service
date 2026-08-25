"""merge refund-clawback and email-verification heads

Revision ID: c212bc389087
Revises: 6d7e8f9a0b1c, b8d2fa41c96e
Create Date: 2026-08-11 16:29:05.809451

"""
from __future__ import annotations

from collections.abc import Sequence

revision: str = 'c212bc389087'
down_revision: str | None = ('6d7e8f9a0b1c', 'b8d2fa41c96e')
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
