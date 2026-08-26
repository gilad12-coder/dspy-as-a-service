"""Provision ADFS identities and validate passwordless local usernames."""

from __future__ import annotations

import hmac
import re
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import settings
from ...storage.models import UserModel
from ..errors import DomainError

_USERNAME_RE = re.compile(r"^[^\s\x00-\x1f\x7f]{1,255}$")


# Username submitted through the local fallback form.
class LocalLoginRequest(BaseModel):
    username: str = Field(description="Administrator-approved local username.")


# Trusted identity attributes resolved from an ADFS profile.
class SsoProvisionRequest(BaseModel):
    username: str = Field(description="Stable username selected from the ADFS claims.")
    display_name: str = Field(default="", description="Human-readable ADFS display name.")
    external_admin: bool = Field(
        default=False,
        description="Whether configured ADFS groups or the environment grant admin access.",
    )


# Resolved account returned to the frontend authentication service.
class AccountInfo(BaseModel):
    username: str = Field(description="Canonical account identity.")
    display_name: str = Field(description="Human-readable account name.")
    role: str = Field(description="Authorization role: admin or user.")


def normalize_username(raw: str) -> str:
    """Normalize and validate a cross-provider username.

    Args:
        raw: Username supplied by ADFS or the local fallback form.

    Returns:
        Canonical case-folded username.

    Raises:
        DomainError: When the username is empty, contains whitespace or control
            characters, or exceeds the storage limit.
    """
    username = raw.strip().casefold()
    if not _USERNAME_RE.fullmatch(username):
        raise DomainError("accounts.invalid_username", status=422)
    return username


def _require_internal_auth(header_value: str | None) -> None:
    """Authorize a trusted frontend authentication call.

    Args:
        header_value: Shared secret supplied by the server-side frontend.

    Raises:
        DomainError: When the deployment secret is missing or does not match.
    """
    secret = settings.backend_auth_secret
    if secret is None:
        raise DomainError("auth.not_configured", status=500)
    if not header_value or not hmac.compare_digest(header_value, secret.get_secret_value()):
        raise DomainError("auth.missing_token", status=403)


def _resolved_role(row: UserModel, *, external_admin: bool = False) -> str:
    """Resolve a stored identity's effective role.

    Args:
        row: Persisted identity row.
        external_admin: Whether ADFS groups grant administrator access for the
            current session.

    Returns:
        ``admin`` when any configured source grants it, otherwise ``user``.
    """
    if external_admin or row.is_admin or row.username in settings.admin_usernames_set:
        return "admin"
    return "user"


def _account_info(row: UserModel, *, external_admin: bool = False) -> AccountInfo:
    """Build the authentication response for a stored identity.

    Args:
        row: Persisted identity row.
        external_admin: Whether the current ADFS session grants admin access.

    Returns:
        Resolved account information.
    """
    return AccountInfo(
        username=row.username,
        display_name=row.display_name,
        role=_resolved_role(row, external_admin=external_admin),
    )


def create_accounts_router(*, job_store) -> APIRouter:
    """Build the local-fallback and ADFS-provisioning routes.

    Args:
        job_store: Job-store instance whose PostgreSQL engine stores identities.

    Returns:
        Router exposing trusted server-to-server authentication endpoints.
    """
    router = APIRouter()

    @router.post(
        "/auth/local/login",
        response_model=AccountInfo,
        summary="Validate an administrator-approved local username",
    )
    def local_login(
        body: LocalLoginRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> AccountInfo:
        """Validate a passwordless local username.

        Environment bootstrap administrators are created lazily so a fresh
        database can be administered before any UI-managed account exists.

        Args:
            body: Local username submitted through the fallback login form.
            x_internal_auth: Shared frontend-to-backend secret.

        Returns:
            The approved local account.

        Raises:
            DomainError: When the username is not locally approved.
        """
        _require_internal_auth(x_internal_auth)
        username = normalize_username(body.username)
        now = datetime.now(UTC)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, username)
            if row is None and username in settings.admin_usernames_set:
                row = UserModel(
                    username=username,
                    display_name=username,
                    local_enabled=True,
                    is_admin=True,
                    created_by="environment",
                    created_at=now,
                )
                session.add(row)
            elif row is not None and username in settings.admin_usernames_set:
                row.local_enabled = True
            if row is None or not row.local_enabled:
                raise DomainError("accounts.invalid_credentials", status=401)
            row.last_login_at = now
            response = _account_info(row)
            session.commit()
        return response

    @router.post(
        "/auth/sso/provision",
        response_model=AccountInfo,
        summary="Provision or refresh an ADFS identity",
    )
    def provision_sso(
        body: SsoProvisionRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> AccountInfo:
        """Create or refresh a normalized identity after valid ADFS login.

        Args:
            body: Trusted attributes resolved by the frontend OIDC provider.
            x_internal_auth: Shared frontend-to-backend secret.

        Returns:
            The unified ADFS/local account and effective role.
        """
        _require_internal_auth(x_internal_auth)
        username = normalize_username(body.username)
        display_name = body.display_name.strip() or username
        now = datetime.now(UTC)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, username)
            if row is None:
                row = UserModel(
                    username=username,
                    display_name=display_name,
                    local_enabled=False,
                    is_admin=False,
                    created_by="adfs",
                    created_at=now,
                )
                session.add(row)
            else:
                row.display_name = display_name
            row.adfs_seen_at = now
            row.last_login_at = now
            response = _account_info(row, external_admin=body.external_admin)
            session.commit()
        return response

    return router
