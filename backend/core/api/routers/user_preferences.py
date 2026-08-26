"""Agent-facing browser preference updates. [INTERNAL]

The generalist agent can change the small set of local UI preferences exposed
in the settings modal. The browser applies the returned patch to its own
``localStorage``; the API deliberately does not persist these device-scoped
preferences on the server.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import AuthenticatedUser, get_authenticated_user

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


class UserPreferencesUpdate(BaseModel):
    """Partial set of browser preferences the generalist may change."""

    advanced_mode: bool | None = Field(
        default=None,
        description="Show expert optimization controls in the app.",
    )
    expand_advanced: bool | None = Field(
        default=None,
        description="Start advanced sections expanded in forms.",
    )
    lite_mode: bool | None = Field(
        default=None,
        description="Use the lower-motion, lighter-weight interface.",
    )
    wizard_code_assist: Literal["auto", "manual"] | None = Field(
        default=None,
        description="Whether the optimization wizard should draft code automatically.",
    )
    wizard_split_mode: Literal["auto", "manual"] | None = Field(
        default=None,
        description="Whether dataset splits should be selected automatically.",
    )
    tagger_assist: bool | None = Field(
        default=None,
        description="Enable AI-assisted tagging in new labeling sessions.",
    )
    dictation_enabled: bool | None = Field(
        default=None,
        description="Show voice dictation in the shared composer.",
    )


class UserPreferencesUpdateResponse(BaseModel):
    """Validated browser-preference patch for the frontend to apply."""

    updates: dict[str, bool | str]
    changed: list[str]


def create_user_preferences_router() -> APIRouter:
    """Build the agent-facing browser-preferences router.

    Returns:
        A FastAPI router exposing the validated preference patch tool.
    """
    router = APIRouter()

    @router.post(
        "/settings/user-preferences",
        response_model=UserPreferencesUpdateResponse,
        operation_id="update_user_preferences",
        summary="Apply local UI preference changes in the current browser",
        tags=["agent"],
    )
    def update_user_preferences(
        req: UserPreferencesUpdate,
        _user: AuthenticatedUserDep,
    ) -> UserPreferencesUpdateResponse:
        """Validate and return the requested browser-preference patch.

        Args:
            req: The subset of supported preferences to change.
            _user: Authenticated caller; the browser applies the response only
                to the caller's local settings.

        Returns:
            The normalized wire-format patch and its changed field names.
        """
        updates = req.model_dump(exclude_unset=True, exclude_none=True)
        return UserPreferencesUpdateResponse(updates=updates, changed=list(updates))

    return router
