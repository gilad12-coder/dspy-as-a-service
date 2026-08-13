"""Notification public surface.

Re-exports the helpers used by the worker and API layers for job lifecycle and
sharing events; the shared SMTP transport is encapsulated in
``core.notifications.comms``.
"""

from .notifier import (
    notify_job_completed,
    notify_job_started,
    notify_ownership_transfer,
    notify_role_change,
    notify_share_invite,
)

__all__ = [
    "notify_job_completed",
    "notify_job_started",
    "notify_ownership_transfer",
    "notify_role_change",
    "notify_share_invite",
]
