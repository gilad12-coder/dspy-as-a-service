"""add tagging_session_share_links and tagging_session_share_grants tables

Revision ID: a3b4c5d6e7f8
Revises: d9e0f1a2b3c4
Create Date: 2026-07-22 10:00:00.000000

Mirrors the dataset sharing ACL for saved tagger sessions. Creates
``tagging_session_share_links`` (the active ``revoked_at IS NULL`` row per
session is the sharing config — ``general_access`` ``'restricted'`` /
``'anyone'`` with a signed-in ``general_role``) and
``tagging_session_share_grants`` for per-user member grants (``viewer`` /
``editor`` / ``owner``). Both ``session_id`` foreign keys cascade so deleting a
session clears its sharing rows. ``grantee_username`` is indexed for the
shared-with-me listing.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a3b4c5d6e7f8"
down_revision: str | None = "d9e0f1a2b3c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tagging-session share-links and share-grants tables."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tagging_session_share_links (
            token VARCHAR(48) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL REFERENCES tagging_sessions(id) ON DELETE CASCADE,
            created_by VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            revoked_at TIMESTAMP WITH TIME ZONE,
            general_access VARCHAR(16) NOT NULL DEFAULT 'restricted',
            general_role VARCHAR(16) NOT NULL DEFAULT 'viewer'
        )
        """
    )
    op.create_index(
        "ix_tagging_session_share_links_session_id",
        "tagging_session_share_links",
        ["session_id"],
        if_not_exists=True,
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tagging_session_share_grants (
            session_id VARCHAR(36) NOT NULL REFERENCES tagging_sessions(id) ON DELETE CASCADE,
            grantee_username VARCHAR(255) NOT NULL,
            role VARCHAR(16) NOT NULL,
            created_by VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, grantee_username)
        )
        """
    )
    op.create_index(
        "ix_tagging_session_share_grants_grantee",
        "tagging_session_share_grants",
        ["grantee_username"],
        if_not_exists=True,
    )


def downgrade() -> None:
    """Drop the tagging-session share-grants and share-links tables and their indexes."""
    op.drop_index(
        "ix_tagging_session_share_grants_grantee",
        table_name="tagging_session_share_grants",
        if_exists=True,
    )
    op.execute("DROP TABLE IF EXISTS tagging_session_share_grants")
    op.drop_index(
        "ix_tagging_session_share_links_session_id",
        table_name="tagging_session_share_links",
        if_exists=True,
    )
    op.execute("DROP TABLE IF EXISTS tagging_session_share_links")
