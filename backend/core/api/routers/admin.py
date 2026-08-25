"""Expose administrator-only storage quota and directory search routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...config import settings
from ..auth import AuthenticatedUser, get_authenticated_user, require_admin_user
from ..directory_client import DirectoryClient, NullDirectoryClient
from ..errors import DomainError

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


# Per-user storage ceiling and current use returned to the administrator UI.
class StorageQuotaOverrideResponse(BaseModel):
    username: str
    quota_bytes: int | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    effective_bytes: int = 0
    used_bytes: int = 0


# Administrator request for setting a user's storage ceiling.
class StorageQuotaOverrideRequest(BaseModel):
    username: str = Field(min_length=1)
    quota_bytes: int = Field(ge=1)


# Envelope for all explicit per-user storage overrides.
class StorageQuotaOverrideListResponse(BaseModel):
    default_bytes: int
    overrides: list[StorageQuotaOverrideResponse]


# One autocomplete match from the database or configured directory.
class DirectoryUserMatch(BaseModel):
    username: str
    display_name: str | None = None
    email: str | None = None
    source: str


# Envelope for administrator user-search results.
class DirectoryUserSearchResponse(BaseModel):
    matches: list[DirectoryUserMatch]


def _build_storage_quota_response(
    row: dict[str, Any], *, job_store: Any
) -> StorageQuotaOverrideResponse:
    """Build a storage quota response with live use.

    Args:
        row: Raw storage override mapping from the job store.
        job_store: Store used to resolve effective quota and storage use.

    Returns:
        Storage quota row for the administrator UI.
    """
    username = str(row["username"])
    return StorageQuotaOverrideResponse(
        username=username,
        quota_bytes=row.get("quota_bytes"),
        updated_at=row.get("updated_at"),
        updated_by=row.get("updated_by"),
        effective_bytes=job_store.get_effective_user_storage_quota(username),
        used_bytes=job_store.compute_user_storage(username).total,
    )


def _require_admin_dependency(user: AuthenticatedUserDep) -> AuthenticatedUser:
    """Require an authenticated administrator.

    Args:
        user: Authenticated bearer-token identity.

    Returns:
        Authorized administrator identity.
    """
    return require_admin_user(user)


AdminUserDep = Annotated[AuthenticatedUser, Depends(_require_admin_dependency)]


def create_admin_router(
    *,
    job_store: Any,
    directory_client: DirectoryClient | None = None,
) -> APIRouter:
    """Build administrator routes for storage quotas and user discovery.

    Args:
        job_store: Store used for quota and username operations.
        directory_client: Optional ADFS-compatible directory search provider.

    Returns:
        Configured administrator router.
    """
    router = APIRouter(prefix="/admin")
    resolved_directory_client = directory_client or NullDirectoryClient()

    @router.get("/storage-quotas", response_model=StorageQuotaOverrideListResponse)
    def list_user_storage_quota_overrides(
        admin_user: AdminUserDep,
    ) -> StorageQuotaOverrideListResponse:
        """List explicit storage overrides and live use.

        Args:
            admin_user: Authenticated administrator.

        Returns:
            Default quota and all explicit per-user overrides.
        """
        del admin_user
        rows = job_store.list_user_storage_quota_overrides()
        return StorageQuotaOverrideListResponse(
            default_bytes=settings.user_storage_quota_bytes,
            overrides=[_build_storage_quota_response(row, job_store=job_store) for row in rows],
        )

    @router.put("/storage-quotas", response_model=StorageQuotaOverrideResponse)
    def set_user_storage_quota_override(
        payload: StorageQuotaOverrideRequest,
        admin_user: AdminUserDep,
    ) -> StorageQuotaOverrideResponse:
        """Set a per-user storage ceiling.

        Args:
            payload: Username and replacement byte ceiling.
            admin_user: Authenticated administrator making the change.

        Returns:
            Saved override with effective quota and live use.

        Raises:
            DomainError: When the normalized username is empty.
        """
        normalized_username = payload.username.strip().casefold()
        if not normalized_username:
            raise DomainError("admin.invalid_username", status=400)
        job_store.set_user_storage_quota_override(
            normalized_username,
            payload.quota_bytes,
            updated_by=admin_user.username,
        )
        return _build_storage_quota_response(
            {
                "username": normalized_username,
                "quota_bytes": payload.quota_bytes,
                "updated_at": None,
                "updated_by": admin_user.username,
            },
            job_store=job_store,
        )

    @router.delete("/storage-quotas/{username}", response_model=StorageQuotaOverrideResponse)
    def delete_user_storage_quota_override(
        username: str,
        admin_user: AdminUserDep,
    ) -> StorageQuotaOverrideResponse:
        """Restore the default storage ceiling for one user.

        Args:
            username: User whose explicit override should be removed.
            admin_user: Authenticated administrator.

        Returns:
            Effective default quota and current storage use.

        Raises:
            DomainError: When the normalized username is empty.
        """
        del admin_user
        normalized_username = username.strip().casefold()
        if not normalized_username:
            raise DomainError("admin.invalid_username", status=400)
        job_store.delete_user_storage_quota_override(normalized_username)
        return _build_storage_quota_response(
            {
                "username": normalized_username,
                "quota_bytes": None,
                "updated_at": None,
                "updated_by": None,
            },
            job_store=job_store,
        )

    @router.get("/users/search", response_model=DirectoryUserSearchResponse)
    def search_users(
        admin_user: AdminUserDep,
        q: Annotated[str, Query(description="Free-text fragment to match.")] = "",
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> DirectoryUserSearchResponse:
        """Search known and directory identities for administrator workflows.

        Args:
            admin_user: Authenticated administrator.
            q: Username, display-name, or email fragment.
            limit: Maximum distinct matches.

        Returns:
            Merged case-normalized matches.
        """
        del admin_user
        query = q.strip()
        if not query:
            return DirectoryUserSearchResponse(matches=[])

        matches: list[DirectoryUserMatch] = []
        seen: set[str] = set()
        for username in job_store.search_usernames(query, limit=limit):
            normalized = (username or "").strip().casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            matches.append(DirectoryUserMatch(username=normalized, source="database"))

        for entry in resolved_directory_client.search_users(query, limit=limit):
            normalized = entry.username.strip().casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            matches.append(
                DirectoryUserMatch(
                    username=normalized,
                    display_name=entry.display_name,
                    email=entry.email,
                    source="directory",
                )
            )

        return DirectoryUserSearchResponse(matches=matches[:limit])

    return router
