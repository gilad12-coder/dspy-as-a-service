"""Cross-replica monthly-active-user admission control."""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime
from typing import Any

from ..config import settings
from .errors import DomainError

_CACHE_LOCK = threading.Lock()
_CACHE_MONTH: date | None = None
_ADMITTED_IDENTITIES: set[str] = set()


def _utc_month_start() -> date:
    """Return the first date of the current UTC calendar month."""
    now = datetime.now(UTC)
    return date(now.year, now.month, 1)


def enforce_monthly_active_user_limit(
    job_store: Any,
    username: str,
    *,
    exempt: bool = False,
) -> None:
    """Admit an identity for this month or reject it at the configured cap.

    The process cache removes a database round trip from every request after an
    identity's first request on a replica. The store remains authoritative and
    serializes first admission across replicas.

    Args:
        job_store: Store exposing ``admit_monthly_active_user``.
        username: Authenticated identity to admit.
        exempt: Whether the identity is an administrator that must retain
            operational access at capacity.

    Raises:
        DomainError: 500 when the configured store cannot enforce the cap; 503
            when no capacity remains for a new identity this month.
    """
    cap = settings.max_monthly_active_users
    if cap <= 0 or exempt:
        return
    normalized = username.strip().lower()
    month_start = _utc_month_start()
    global _CACHE_MONTH
    with _CACHE_LOCK:
        if month_start != _CACHE_MONTH:
            _ADMITTED_IDENTITIES.clear()
            _CACHE_MONTH = month_start
        if normalized in _ADMITTED_IDENTITIES:
            return
        admit = getattr(job_store, "admit_monthly_active_user", None)
        if not callable(admit):
            raise DomainError("auth.not_configured", status=500)
        if not admit(normalized, month_start, cap):
            raise DomainError("accounts.monthly_capacity_reached", status=503)
        _ADMITTED_IDENTITIES.add(normalized)
