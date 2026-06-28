"""add telemetry_events for first-party product analytics

Revision ID: d8e9f0a1b2c3
Revises: e9f0a1b2c3d4
Create Date: 2026-06-28 10:00:00.000000

Creates the append-only ``telemetry_events`` table the browser SDK batches
interaction events into (page views, labelled clicks, named flow events) plus
the two composite recency indexes its read endpoints scan: by event name and by
username, each ordered on ``received_at``. A standalone ``received_at`` index
backs the unfiltered global time-series, and ``anonymous_id`` is indexed for
logged-out funnel counts.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "d8e9f0a1b2c3"
down_revision: str | None = "e9f0a1b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the telemetry_events table and its read indexes (idempotently).

    ``IF NOT EXISTS`` mirrors the search_query_log / api_tokens migrations: the
    app runs ``Base.metadata.create_all`` on boot, so the table may already
    exist when migrations run — every create here must be idempotent.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id BIGSERIAL PRIMARY KEY,
            event_name VARCHAR(80) NOT NULL,
            occurred_at TIMESTAMP WITH TIME ZONE,
            received_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            username VARCHAR(255),
            anonymous_id VARCHAR(64),
            session_id VARCHAR(64),
            path VARCHAR(512),
            locale VARCHAR(16),
            app_version VARCHAR(40),
            properties JSONB,
            context JSONB
        )
        """
    )
    op.create_index(
        "ix_telemetry_events_received_at",
        "telemetry_events",
        ["received_at"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_telemetry_events_anonymous_id",
        "telemetry_events",
        ["anonymous_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_telemetry_events_name_received",
        "telemetry_events",
        ["event_name", "received_at"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_telemetry_events_user_received",
        "telemetry_events",
        ["username", "received_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    """Drop the telemetry_events table and its indexes."""
    for name in (
        "ix_telemetry_events_user_received",
        "ix_telemetry_events_name_received",
        "ix_telemetry_events_anonymous_id",
        "ix_telemetry_events_received_at",
    ):
        op.drop_index(name, table_name="telemetry_events", if_exists=True)
    op.execute("DROP TABLE IF EXISTS telemetry_events")
