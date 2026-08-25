"""Record product-telemetry milestones from server code paths.

The browser SDK can only report what the browser sees. Two funnel milestones
happen out of its sight — a Stripe checkout completing (a webhook, often after
the tab closed) and an optimisation run reaching a terminal state (the worker,
minutes to hours later). Those are recorded here so the ``telemetry_events``
table and the PostHog export hold the whole funnel, attributed by the same
server-trusted ``username`` and (best-effort) the user's last-known browser
``anonymous_id`` so the PostHog distinct id lines up with the browser events.

Everything in this module is best-effort: a telemetry failure must never turn
a credited purchase or a finished run into an error.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ..api.posthog import export_telemetry_events
from ..config import settings
from ..storage.models import TelemetryEventModel

logger = logging.getLogger("skynet.telemetry")

_SERVER_CONTEXT: dict[str, Any] = {"source": "server"}


def _latest_anonymous_id(session: Session, username: str) -> str | None:
    """Return the most recent browser id seen for ``username``, if any.

    Args:
        session: Open session on the telemetry engine.
        username: Server-trusted account identity.

    Returns:
        The latest non-null ``anonymous_id`` from the user's browser events, or
        ``None`` when the user has never sent browser telemetry.
    """
    row = (
        session.query(TelemetryEventModel.anonymous_id)
        .filter(
            TelemetryEventModel.username == username,
            TelemetryEventModel.anonymous_id.isnot(None),
        )
        .order_by(TelemetryEventModel.received_at.desc(), TelemetryEventModel.id.desc())
        .first()
    )
    return row[0] if row else None


def record_server_event(
    engine: Engine | None,
    *,
    username: str | None,
    name: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Persist a server-side milestone and forward it to PostHog.

    Args:
        engine: SQLAlchemy engine backing ``telemetry_events``; ``None`` (e.g. a
            store without an engine) makes the call a no-op.
        username: Server-trusted account identity the milestone belongs to.
        name: Event name; must be on the PostHog allowlist to be exported.
        properties: Structural, PII-free descriptors (pack id, optimizer, ...).

    Never raises: failures are logged at warning level and swallowed.
    """
    if engine is None or not settings.telemetry_enabled:
        return
    props = dict(properties or {})
    now = datetime.now(UTC)
    try:
        with Session(engine) as session:
            anonymous_id = _latest_anonymous_id(session, username) if username else None
            session.add(
                TelemetryEventModel(
                    event_name=name,
                    occurred_at=now,
                    received_at=now,
                    username=username,
                    anonymous_id=anonymous_id,
                    session_id=None,
                    path=None,
                    locale=None,
                    app_version=None,
                    properties=props,
                    context=dict(_SERVER_CONTEXT),
                )
            )
            session.commit()
    except Exception:
        logger.warning("telemetry: failed to record server event %s", name, exc_info=True)
        return
    if settings.posthog_project_api_key is None:
        return
    event = {
        "name": name,
        "timestamp": now.isoformat(),
        "path": None,
        "locale": None,
        "app_version": None,
        "properties": props,
        "context": dict(_SERVER_CONTEXT),
    }
    # The export is a network call with its own timeout; a daemon thread keeps
    # it off the webhook/worker critical path without needing an event loop.
    threading.Thread(
        target=export_telemetry_events,
        kwargs={
            "username": username,
            "anonymous_id": anonymous_id,
            "session_id": None,
            "events": [event],
        },
        name=f"posthog-export-{name}",
        daemon=True,
    ).start()
