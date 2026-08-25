"""add billing_openrouter_keys table for per-user provisioned runtime keys

Revision ID: 5c6d7e8f9a0b
Revises: 4b5c6d7e8f9a
Create Date: 2026-08-10 12:00:00.000000

Backs provider-side spend capping for managed runs: one row per account holding
the Fernet-encrypted OpenRouter runtime key minted through the key-management
API (``secret_ciphertext`` — never plaintext) and the ``key_hash`` addressing it
there. The worker syncs each key's spend limit to the account's credit balance
before dispatch, so OpenRouter itself refuses requests past the prepaid
balance. Postgres-only with ``IF NOT EXISTS`` to stay idempotent against the
boot-time ``create_all`` — the SQLite test schema comes from the ORM models
directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "5c6d7e8f9a0b"
down_revision: str | Sequence[str] | None = "4b5c6d7e8f9a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the billing_openrouter_keys table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "CREATE TABLE IF NOT EXISTS billing_openrouter_keys ("
        "username VARCHAR(255) NOT NULL, "
        "key_hash VARCHAR(128) NOT NULL, "
        "secret_ciphertext BYTEA NOT NULL, "
        "created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
        "updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
        "PRIMARY KEY (username))"
    )


def downgrade() -> None:
    """Drop the billing_openrouter_keys table."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS billing_openrouter_keys")
