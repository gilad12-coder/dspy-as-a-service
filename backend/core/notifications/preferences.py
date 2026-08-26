"""Durable email-notification preference lookup shared by API and workers."""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..storage.models import NotificationPreferenceModel

logger = logging.getLogger(__name__)

NotificationCategory = Literal["job_updates", "sharing_updates"]
_engine: Engine | None = None


def configure_notification_preferences(engine: Engine | None) -> None:
    """Set the database engine used for notification preference reads.

    Args:
        engine: Shared application engine, or ``None`` to restore the default
            enabled behavior in isolated tests.
    """
    global _engine
    _engine = engine


def notification_category_enabled(username: str, category: NotificationCategory) -> bool:
    """Return whether ``username`` allows the optional email category.

    Missing rows and unavailable storage fail open so a preference lookup can
    never fail the job or sharing action that triggered it. Account-security
    mail does not pass through this function and remains unaffected.

    Args:
        username: Recipient identity.
        category: Optional email category being considered.

    Returns:
        ``False`` only when the stored category preference is explicitly off.
    """
    if _engine is None:
        return True
    try:
        with Session(_engine) as session:
            row = session.get(NotificationPreferenceModel, username)
    except Exception:
        logger.warning("Notification preference lookup failed; using enabled default", exc_info=True)
        return True
    if row is None:
        return True
    if category == "job_updates":
        return bool(row.job_updates_enabled)
    return bool(row.sharing_updates_enabled)
