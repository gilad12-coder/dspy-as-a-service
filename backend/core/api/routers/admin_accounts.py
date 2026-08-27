"""Administer local usernames and persistent account roles."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import settings
from ...storage.models import UserModel
from ..account_data_service import delete_account
from ..errors import DomainError
from .accounts import normalize_username
from .admin import AdminUserDep


# Administrator request for adding a passwordless local username.
class CreateLocalAccountRequest(BaseModel):
    username: str = Field(description="Username to approve for local fallback login.")
    is_admin: bool = Field(default=False, description="Whether the account is an administrator.")


# Administrator request for changing a persisted account role.
class UpdateAccountRoleRequest(BaseModel):
    is_admin: bool = Field(description="Whether the account is an administrator.")


# Account-management row shown in the administrator UI, including the
# account's storage budget and live use so one table can manage both.
class ManagedAccountResponse(BaseModel):
    username: str
    local_enabled: bool
    adfs_seen: bool
    is_admin: bool
    created_at: str | None = None
    last_login_at: str | None = None
    quota_bytes: int | None = None
    effective_bytes: int = 0
    used_bytes: int = 0
    quota_updated_by: str | None = None


# Envelope returned by the account-management list endpoint.
class ManagedAccountListResponse(BaseModel):
    accounts: list[ManagedAccountResponse]
    default_bytes: int


# Outcome returned after irreversible account deletion.
class ManagedAccountDeletionResponse(BaseModel):
    username: str
    deleted_rows: int
    anonymized_rows: int


def _iso(value: datetime | None) -> str | None:
    """Render an optional timestamp for the API response.

    Args:
        value: Timestamp to serialize.

    Returns:
        ISO-8601 text or ``None``.
    """
    return value.isoformat() if value is not None else None


def _quota_overrides_by_username(job_store: Any) -> dict[str, dict[str, Any]]:
    """Map usernames to their explicit storage-quota override rows.

    Args:
        job_store: Store holding the saved storage overrides.

    Returns:
        Override rows keyed by username.
    """
    return {str(row["username"]): row for row in job_store.list_user_storage_quota_overrides()}


def _response(
    row: UserModel,
    *,
    job_store: Any,
    quota_overrides: dict[str, dict[str, Any]],
) -> ManagedAccountResponse:
    """Build one administrator account row with storage state.

    Args:
        row: Stored identity.
        job_store: Store used to resolve effective quota and storage use.
        quota_overrides: Explicit storage-quota override rows keyed by username.

    Returns:
        Account-management response.
    """
    override = quota_overrides.get(row.username, {})
    return ManagedAccountResponse(
        username=row.username,
        local_enabled=row.local_enabled,
        adfs_seen=row.adfs_seen_at is not None,
        is_admin=row.is_admin,
        created_at=_iso(row.created_at),
        last_login_at=_iso(row.last_login_at),
        quota_bytes=override.get("quota_bytes"),
        effective_bytes=job_store.get_effective_user_storage_quota(row.username),
        used_bytes=job_store.compute_user_storage(row.username).total,
        quota_updated_by=override.get("updated_by"),
    )


def create_admin_accounts_router(*, job_store) -> APIRouter:
    """Build administrator account-management routes.

    Args:
        job_store: Job-store instance whose engine stores identities and owned data.

    Returns:
        Router for listing, creating, promoting and deleting accounts.
    """
    router = APIRouter(prefix="/admin/accounts")

    @router.get("", response_model=ManagedAccountListResponse)
    def list_accounts(admin_user: AdminUserDep) -> ManagedAccountListResponse:
        """List every known ADFS or local account with storage budget and use.

        Args:
            admin_user: Authenticated administrator.

        Returns:
            Accounts sorted by canonical username, plus the default quota.
        """
        del admin_user
        quota_overrides = _quota_overrides_by_username(job_store)
        with Session(job_store.engine) as session:
            rows = session.scalars(select(UserModel).order_by(UserModel.username)).all()
            return ManagedAccountListResponse(
                accounts=[
                    _response(row, job_store=job_store, quota_overrides=quota_overrides)
                    for row in rows
                ],
                default_bytes=settings.user_storage_quota_bytes,
            )

    @router.post("", response_model=ManagedAccountResponse, status_code=201)
    def create_local_account(
        body: CreateLocalAccountRequest,
        admin_user: AdminUserDep,
    ) -> ManagedAccountResponse:
        """Approve a username for passwordless local login.

        Existing ADFS identities are upgraded in place so both login methods
        continue to resolve to the same account.

        Args:
            body: Username and initial role.
            admin_user: Authenticated administrator creating the account.

        Returns:
            Created or upgraded account.

        Raises:
            DomainError: When the username is already locally enabled.
        """
        username = normalize_username(body.username)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, username)
            if row is not None and row.local_enabled:
                raise DomainError("accounts.username_taken", status=409)
            if row is None:
                row = UserModel(
                    username=username,
                    display_name=username,
                    local_enabled=True,
                    is_admin=body.is_admin,
                    created_by=admin_user.username,
                )
                session.add(row)
            else:
                row.local_enabled = True
                row.is_admin = body.is_admin
            session.commit()
            session.refresh(row)
            return _response(
                row,
                job_store=job_store,
                quota_overrides=_quota_overrides_by_username(job_store),
            )

    @router.put("/{username}/role", response_model=ManagedAccountResponse)
    def update_account_role(
        username: str,
        body: UpdateAccountRoleRequest,
        admin_user: AdminUserDep,
    ) -> ManagedAccountResponse:
        """Promote or demote a persisted account.

        Args:
            username: Account whose stored role should change.
            body: New administrator state.
            admin_user: Authenticated administrator making the change.

        Returns:
            Updated account.

        Raises:
            DomainError: When the account does not exist.
        """
        del admin_user
        normalized = normalize_username(username)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, normalized)
            if row is None:
                raise DomainError("accounts.not_found", status=404)
            row.is_admin = body.is_admin
            session.commit()
            session.refresh(row)
            return _response(
                row,
                job_store=job_store,
                quota_overrides=_quota_overrides_by_username(job_store),
            )

    @router.delete("/{username}", response_model=ManagedAccountDeletionResponse)
    def delete_managed_account(
        username: str,
        admin_user: AdminUserDep,
    ) -> ManagedAccountDeletionResponse:
        """Permanently delete an account and all owned data.

        Args:
            username: Account to delete.
            admin_user: Authenticated administrator confirming the deletion.

        Returns:
            Counts of removed and anonymized rows.

        Raises:
            DomainError: When the account does not exist.
        """
        del admin_user
        normalized = normalize_username(username)
        with Session(job_store.engine) as session:
            if session.get(UserModel, normalized) is None:
                raise DomainError("accounts.not_found", status=404)
            summary = delete_account(session, normalized)
            session.commit()
        return ManagedAccountDeletionResponse(
            username=normalized,
            deleted_rows=summary.deleted_rows,
            anonymized_rows=summary.anonymized_rows,
        )

    return router
