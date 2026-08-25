"""add billing_provider_keys table for the encrypt-at-rest BYOK vault

Revision ID: c4d5e6f7a8b0
Revises: b3c4d5e6f7a9
Create Date: 2026-06-28 14:00:00.000000

Backs BYOK mode's key vault: one row per ``(username, provider)`` holding the
Fernet-encrypted secret (``secret_ciphertext`` — never plaintext), the masked
``last4`` tail for display, and a ``status`` (``unverified`` / ``verified`` /
``invalid``) set by the verify probe. Saving a key for a provider replaces the
account's previous one for it. Postgres-only with ``IF NOT EXISTS`` to stay
idempotent against the boot-time ``create_all`` — the SQLite test schema comes
from the ORM models directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "c4d5e6f7a8b0"
down_revision: str | Sequence[str] | None = "b3c4d5e6f7a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the billing_provider_keys table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "CREATE TABLE IF NOT EXISTS billing_provider_keys ("
        "username VARCHAR(255) NOT NULL, "
        "provider VARCHAR(32) NOT NULL, "
        "secret_ciphertext BYTEA NOT NULL, "
        "last4 VARCHAR(8) NOT NULL, "
        "status VARCHAR(16) NOT NULL DEFAULT 'unverified', "
        "created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
        "updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
        "PRIMARY KEY (username, provider))"
    )


def downgrade() -> None:
    """Drop the billing_provider_keys table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS billing_provider_keys")
