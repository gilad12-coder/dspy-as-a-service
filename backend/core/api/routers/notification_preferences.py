"""Authenticated email-notification preference endpoints. [INTERNAL]"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...storage.models import NotificationPreferenceModel
from ..auth import AuthenticatedUser, get_authenticated_user

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


class NotificationPreferencesResponse(BaseModel):
    job_updates_enabled: bool = Field(description="Email when the caller's optimization starts or finishes.")
    sharing_updates_enabled: bool = Field(description="Email when an optimization is shared or access changes.")


class NotificationPreferencesUpdate(BaseModel):
    job_updates_enabled: bool | None = Field(default=None)
    sharing_updates_enabled: bool | None = Field(default=None)


def _response(row: NotificationPreferenceModel | None) -> NotificationPreferencesResponse:
    """Render a stored row or the enabled-by-default preference set.

    Args:
        row: Stored preferences, or ``None`` for an identity without overrides.

    Returns:
        API response with both effective category values.
    """
    if row is None:
        return NotificationPreferencesResponse(
            job_updates_enabled=True,
            sharing_updates_enabled=True,
        )
    return NotificationPreferencesResponse(
        job_updates_enabled=bool(row.job_updates_enabled),
        sharing_updates_enabled=bool(row.sharing_updates_enabled),
    )


def create_notification_preferences_router(*, job_store) -> APIRouter:
    """Build the caller-scoped notification preference router.

    Args:
        job_store: Job-store instance whose ORM engine persists preferences.

    Returns:
        A router exposing get and patch operations for the authenticated caller.
    """
    router = APIRouter()

    @router.get(
        "/account/notification-preferences",
        response_model=NotificationPreferencesResponse,
        summary="Read the caller's optional email-notification preferences",
    )
    def get_notification_preferences(
        user: AuthenticatedUserDep,
    ) -> NotificationPreferencesResponse:
        """Return the caller's stored preferences or enabled defaults.

        Args:
            user: Authenticated caller whose preferences are requested.

        Returns:
            Effective optional email preferences.
        """
        with Session(job_store.engine) as session:
            return _response(session.get(NotificationPreferenceModel, user.username))

    @router.patch(
        "/account/notification-preferences",
        response_model=NotificationPreferencesResponse,
        summary="Update the caller's optional email-notification preferences",
    )
    def update_notification_preferences(
        body: NotificationPreferencesUpdate,
        user: AuthenticatedUserDep,
    ) -> NotificationPreferencesResponse:
        """Persist the supplied category switches for the caller.

        Args:
            body: Partial set of preference changes.
            user: Authenticated caller whose preferences are updated.

        Returns:
            Effective preferences after the update.
        """
        with Session(job_store.engine) as session:
            row = session.get(NotificationPreferenceModel, user.username)
            if row is None:
                row = NotificationPreferenceModel(username=user.username)
                session.add(row)
            updates = body.model_dump(exclude_none=True)
            for field, value in updates.items():
                setattr(row, field, value)
            row.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            return _response(row)

    return router
