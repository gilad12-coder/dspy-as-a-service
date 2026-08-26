"""Named, authenticated sharing for personal-library datasets.

The dataset twin of :mod:`core.api.routers.share`, minus the optimization-only
concerns (no inference, no model-secret scrubbing, no Explorer publication).
Owner-gated management endpoints plus an access-gated named-member read:

* ``GET    /datasets/library/{id}/sharing`` — current sharing config (owner).
* ``PUT    /datasets/library/{id}/sharing`` — mint or update a restricted link.
* ``POST   /datasets/library/{id}/sharing/members`` — add/replace a member grant.
* ``PATCH  /datasets/library/{id}/sharing/members/{username}`` — change a role.
* ``DELETE /datasets/library/{id}/sharing/members/{username}`` — remove a grant.
* ``POST   /datasets/library/{id}/sharing/transfer`` — reassign ownership to an
  existing member (the previous owner is demoted to an editor).
* ``GET    /datasets/share/{token}`` — composite read for a named member.
* ``POST   /datasets/share/{token}/claim`` — resolve the restricted link.

The token alone grants nothing; the caller must own the dataset, be an
administrator, or hold a named member grant. The invite people-picker reuses
the shared ``GET /users/search`` autocomplete.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...storage.dataset_library import DatasetLibraryStore, PostgresDatasetBlobStore
from ...storage.models import DatasetShareGrantModel, DatasetShareLinkModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..dataset_access import (
    dataset_owner,
    get_active_link,
    get_grant,
    get_link_by_token,
    list_grants,
    require_role,
    resolve_share_access,
)
from ..errors import DomainError
from ..sharing_access import (
    GENERAL_ACCESS_RESTRICTED,
    LINK_GRANT_MARKER,
    LINK_ROLES,
    MEMBER_ROLES,
    ShareRole,
)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

_GENERAL_ACCESS_VALUES = (GENERAL_ACCESS_RESTRICTED,)
_LINK_ROLE_VALUES = tuple(sorted(LINK_ROLES))


class DatasetSharingMember(BaseModel):
    """One invited member of a dataset (username + tier role)."""

    username: str
    role: str


class DatasetSharingState(BaseModel):
    """Owner-facing sharing config for one library dataset."""

    general_access: str
    general_role: str = "viewer"
    token: str | None = None
    share_path: str | None = None
    owner: str | None = None
    members: list[DatasetSharingMember] = Field(default_factory=list)


class PutDatasetSharingRequest(BaseModel):
    """Request body for ``PUT /datasets/library/{id}/sharing``."""

    general_access: str
    general_role: str | None = None


class AddDatasetMemberRequest(BaseModel):
    """Request body for ``POST /datasets/library/{id}/sharing/members``."""

    username: str
    role: str


class UpdateDatasetMemberRequest(BaseModel):
    """Request body for ``PATCH /datasets/library/{id}/sharing/members/{username}``."""

    role: str


class TransferDatasetOwnershipRequest(BaseModel):
    """Request body for ``POST /datasets/library/{id}/sharing/transfer``."""

    username: str


class ClaimDatasetResponse(BaseModel):
    """Envelope for ``POST /datasets/share/{token}/claim`` — the target dataset."""

    dataset_id: str
    role: str


def create_dataset_share_router(*, job_store) -> APIRouter:
    """Build the Google-Drive-style dataset sharing router.

    Args:
        job_store: Storage backend whose ``engine`` carries the ``datasets`` and
            dataset-share tables; used for the access resolver and the public
            composite read.

    Returns:
        A FastAPI ``APIRouter`` with the owner-gated management routes and the
        access-gated public surface.
    """
    store = DatasetLibraryStore(job_store.engine, PostgresDatasetBlobStore(job_store.engine))
    router = APIRouter()

    def _require_manage(session: Session, dataset_id: str, user: AuthenticatedUser) -> str | None:
        """Ensure ``user`` may manage sharing for ``dataset_id``, returning its owner.

        Management is owner-only: only the dataset owner or an admin may invite
        people, change roles, transfer ownership, and set general access.

        Args:
            session: Open DB session.
            dataset_id: Dataset being managed.
            user: Authenticated caller.

        Returns:
            The dataset owner's lowercased username.

        Raises:
            DomainError: 404 when the caller has no access to the dataset; 403
                when the caller can reach it but is not the owner/admin.
        """
        require_role(session, dataset_id, user, ShareRole.owner)
        return dataset_owner(session, dataset_id)

    def _sharing_state(session: Session, dataset_id: str, owner: str | None) -> DatasetSharingState:
        """Assemble the current :class:`DatasetSharingState` for a dataset.

        Args:
            session: Open DB session.
            dataset_id: Dataset to describe.
            owner: The dataset owner username (shown to the manager).

        Returns:
            The populated :class:`DatasetSharingState`.
        """
        link = get_active_link(session, dataset_id)
        general_access = link.general_access if link is not None else GENERAL_ACCESS_RESTRICTED
        general_role = link.general_role if link is not None else str(ShareRole.viewer)
        token = link.token if link is not None else None
        members = [
            DatasetSharingMember(username=g.grantee_username, role=g.role)
            for g in list_grants(session, dataset_id)
            if g.created_by != LINK_GRANT_MARKER
        ]
        return DatasetSharingState(
            general_access=general_access,
            general_role=general_role,
            token=token,
            share_path=f"/datasets/share/{token}" if token else None,
            owner=owner,
            members=members,
        )

    def _ensure_link(session: Session, dataset_id: str, created_by: str) -> DatasetShareLinkModel:
        """Return the active link, minting one if none exists.

        Args:
            session: Open DB session (caller commits).
            dataset_id: Dataset the link belongs to.
            created_by: Username recorded as the link creator.

        Returns:
            The active :class:`DatasetShareLinkModel`.
        """
        link = get_active_link(session, dataset_id)
        if link is None:
            link = DatasetShareLinkModel(
                token=secrets.token_urlsafe(24),
                dataset_id=dataset_id,
                created_by=created_by,
                created_at=datetime.now(UTC),
                general_access=GENERAL_ACCESS_RESTRICTED,
                general_role=str(ShareRole.viewer),
            )
            session.add(link)
        return link

    def _delete_link_memberships(session: Session, dataset_id: str) -> None:
        """Delete legacy link-derived memberships without touching named grants.

        Args:
            session: Open DB session.
            dataset_id: Dataset whose link memberships are reconciled.
        """
        markers = session.scalars(
            select(DatasetShareGrantModel).where(
                DatasetShareGrantModel.dataset_id == dataset_id,
                DatasetShareGrantModel.created_by == LINK_GRANT_MARKER,
            )
        )
        for grant in markers:
            session.delete(grant)

    @router.get(
        "/datasets/library/{dataset_id}/sharing",
        response_model=DatasetSharingState,
        summary="Get the sharing config (general access + members) for a dataset",
    )
    def get_sharing(dataset_id: str, current_user: AuthenticatedUserDep) -> DatasetSharingState:
        """Return the dataset's sharing config for its owner.

        Args:
            dataset_id: Dataset to inspect.
            current_user: Authenticated owner/admin.

        Returns:
            The current :class:`DatasetSharingState`.

        Raises:
            DomainError: 404 when unknown/inaccessible; 403 when the caller may
                not manage sharing.
        """
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            return _sharing_state(session, dataset_id, owner)

    @router.put(
        "/datasets/library/{dataset_id}/sharing",
        response_model=DatasetSharingState,
        summary="Mint or update a restricted named-share link",
    )
    def put_sharing(
        dataset_id: str, req: PutDatasetSharingRequest, current_user: AuthenticatedUserDep
    ) -> DatasetSharingState:
        """Mint a restricted link or update its compatibility metadata.

        Only ``restricted`` is accepted. ``general_role`` never grants access
        without a matching named member row.

        Args:
            dataset_id: Dataset to update.
            req: Body carrying the new ``general_access`` policy and optional
                ``general_role`` tier.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`DatasetSharingState`.

        Raises:
            DomainError: 404/403 on access; 400 when ``general_access`` or
                ``general_role`` is invalid.
        """
        if req.general_access not in _GENERAL_ACCESS_VALUES:
            raise DomainError(
                "share.invalid_general_access", status=400, allowed=", ".join(_GENERAL_ACCESS_VALUES)
            )
        if req.general_role is not None and req.general_role not in LINK_ROLES:
            raise DomainError(
                "share.invalid_role", status=400, role=req.general_role, allowed=", ".join(_LINK_ROLE_VALUES)
            )
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            link = _ensure_link(session, dataset_id, current_user.username)
            link.general_access = req.general_access
            if req.general_role is not None:
                link.general_role = req.general_role
            _delete_link_memberships(session, dataset_id)
            session.commit()
            return _sharing_state(session, dataset_id, owner)

    @router.post(
        "/datasets/library/{dataset_id}/sharing/members",
        response_model=DatasetSharingState,
        summary="Invite a user (add or replace a member grant)",
    )
    def add_member(
        dataset_id: str, req: AddDatasetMemberRequest, current_user: AuthenticatedUserDep
    ) -> DatasetSharingState:
        """Add or replace a member grant on the dataset.

        Args:
            dataset_id: Dataset to share.
            req: Body carrying the grantee ``username`` and tier ``role``.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`DatasetSharingState`.

        Raises:
            DomainError: 404/403 on access; 400 when the role is invalid or the
                caller tries to grant the owner themselves.
        """
        if req.role not in MEMBER_ROLES:
            raise DomainError("share.invalid_role", status=400, role=req.role, allowed=", ".join(sorted(MEMBER_ROLES)))
        grantee = req.username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            if owner is not None and grantee == owner:
                raise DomainError("dataset.share.cannot_grant_self", status=400)
            existing = get_grant(session, dataset_id, grantee)
            if existing is not None:
                existing.role = req.role
                # Promote a link membership to a named (authoritative) invite so
                # it no longer tracks or gets revoked with the link.
                existing.created_by = current_user.username
            else:
                session.add(
                    DatasetShareGrantModel(
                        dataset_id=dataset_id,
                        grantee_username=grantee,
                        role=req.role,
                        created_by=current_user.username,
                        created_at=datetime.now(UTC),
                    )
                )
            session.commit()
            return _sharing_state(session, dataset_id, owner)

    @router.patch(
        "/datasets/library/{dataset_id}/sharing/members/{username}",
        response_model=DatasetSharingState,
        summary="Change an existing member's role",
    )
    def update_member(
        dataset_id: str,
        username: str,
        req: UpdateDatasetMemberRequest,
        current_user: AuthenticatedUserDep,
    ) -> DatasetSharingState:
        """Change an existing member's tier role.

        Args:
            dataset_id: Dataset to update.
            username: Grantee whose role changes.
            req: Body carrying the new ``role``.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`DatasetSharingState`.

        Raises:
            DomainError: 404/403 on access; 404 (``dataset.share.member_not_found``)
                when the member has no grant; 400 when the role is invalid or the
                caller targets their own grant.
        """
        if req.role not in MEMBER_ROLES:
            raise DomainError("share.invalid_role", status=400, role=req.role, allowed=", ".join(sorted(MEMBER_ROLES)))
        grantee = username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            if grantee == current_user.username.strip().lower():
                raise DomainError("dataset.share.cannot_modify_self", status=400)
            grant = get_grant(session, dataset_id, grantee)
            if grant is None:
                raise DomainError("dataset.share.member_not_found", status=404, username=grantee)
            grant.role = req.role
            session.commit()
            return _sharing_state(session, dataset_id, owner)

    @router.delete(
        "/datasets/library/{dataset_id}/sharing/members/{username}",
        response_model=DatasetSharingState,
        summary="Remove a member's grant",
    )
    def remove_member(
        dataset_id: str, username: str, current_user: AuthenticatedUserDep
    ) -> DatasetSharingState:
        """Remove a member's grant from the dataset.

        Args:
            dataset_id: Dataset to update.
            username: Grantee whose grant is removed.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`DatasetSharingState`.

        Raises:
            DomainError: 404/403 on access; 404 (``dataset.share.member_not_found``)
                when the member has no grant; 400 when the caller targets their
                own grant.
        """
        grantee = username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            if grantee == current_user.username.strip().lower():
                raise DomainError("dataset.share.cannot_modify_self", status=400)
            grant = get_grant(session, dataset_id, grantee)
            if grant is None:
                raise DomainError("dataset.share.member_not_found", status=404, username=grantee)
            session.delete(grant)
            session.commit()
            return _sharing_state(session, dataset_id, owner)

    @router.post(
        "/datasets/library/{dataset_id}/sharing/transfer",
        response_model=DatasetSharingState,
        summary="Transfer ownership to an existing member (old owner becomes an editor)",
    )
    def transfer_ownership(
        dataset_id: str, req: TransferDatasetOwnershipRequest, current_user: AuthenticatedUserDep
    ) -> DatasetSharingState:
        """Hand a dataset's ownership to an existing member.

        A dataset has exactly one owner, so ownership is reassigned outright: the
        ``owner_username`` column moves to the new owner, the previous owner is
        demoted to an ``editor`` grant, and the new owner's member grant is
        dropped. The new owner must already be a member.

        Args:
            dataset_id: Dataset whose ownership moves.
            req: Body carrying the new owner ``username`` (an existing member).
            current_user: Authenticated current owner/admin.

        Returns:
            The updated :class:`DatasetSharingState` — ``owner`` is the new owner
            and the previous owner now appears as an ``editor`` member.

        Raises:
            DomainError: 404/403 on access; 400 when transferring to the current
                owner; 404 (``dataset.share.member_not_found``) when the target
                is not a member.
        """
        new_owner = req.username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, dataset_id, current_user)
            if owner is not None and new_owner == owner:
                raise DomainError("dataset.share.cannot_grant_self", status=400)
            grant = get_grant(session, dataset_id, new_owner)
            if grant is None:
                raise DomainError("dataset.share.member_not_found", status=404, username=new_owner)
            store.reassign_owner(dataset_id, new_owner)
            session.delete(grant)
            if owner is not None:
                demoted = get_grant(session, dataset_id, owner)
                if demoted is not None:
                    demoted.role = str(ShareRole.editor)
                else:
                    session.add(
                        DatasetShareGrantModel(
                            dataset_id=dataset_id,
                            grantee_username=owner,
                            role=str(ShareRole.editor),
                            created_by=current_user.username,
                            created_at=datetime.now(UTC),
                        )
                    )
            session.commit()
            return _sharing_state(session, dataset_id, new_owner)

    @router.get(
        "/datasets/share/{token}",
        summary="Access-gated composite snapshot of a shared dataset (viewer+)",
    )
    def get_shared_dataset(token: str, current_user: AuthenticatedUserDep) -> dict[str, Any]:
        """Return the access-gated composite view of a shared dataset.

        Requires a signed-in caller; the floor is ``viewer``. Returns the
        dataset metadata, the caller's effective role, the owner, and the rows
        with their saved column schema for the read-only preview.

        Args:
            token: The public share token from the URL.
            current_user: Authenticated caller.

        Returns:
            ``{dataset_id, role, owner, name, source, row_count, column_count,
            byte_size, columns, rows, column_schema}``.

        Raises:
            DomainError: 404 when the token is unknown/revoked, the caller has no
                access, or the dataset no longer exists.
        """
        with Session(job_store.engine) as session:
            role = resolve_share_access(session, token, current_user)
            if role is None:
                raise DomainError("dataset.share.not_found", status=404)
            link = get_link_by_token(session, token)
            dataset_id = link.dataset_id

        record = store.get_dataset(dataset_id)
        rows = store.get_rows(dataset_id)
        if record is None or rows is None:
            raise DomainError("dataset.share.not_found", status=404)
        order = record.column_schema.get("column_order")
        columns = [str(c) for c in order] if isinstance(order, list) and order else (
            list(rows[0].keys()) if rows else []
        )
        return {
            "dataset_id": dataset_id,
            "role": role.value,
            "owner": record.owner_username,
            "name": record.name,
            "source": record.source,
            "row_count": record.row_count,
            "column_count": record.column_count,
            "byte_size": record.byte_size,
            "columns": columns,
            "rows": rows,
            "column_schema": record.column_schema,
        }

    @router.post(
        "/datasets/share/{token}/claim",
        response_model=ClaimDatasetResponse,
        summary="Resolve a restricted dataset link for a named member",
    )
    def claim_shared_dataset(token: str, current_user: AuthenticatedUserDep) -> ClaimDatasetResponse:
        """Resolve a restricted link for its owner, an admin, or a named member.

        Opening the link never creates access or a membership row.

        Args:
            token: The public share token from the URL.
            current_user: Authenticated caller redeeming the link.

        Returns:
            A :class:`ClaimDatasetResponse` with the target ``dataset_id`` and the
            caller's effective ``role`` after redemption.

        Raises:
            DomainError: 404 when the token is unknown/revoked, the dataset is
                gone, or the caller has no access under the link's policy.
        """
        with Session(job_store.engine) as session:
            role = resolve_share_access(session, token, current_user)
            if role is None:
                raise DomainError("dataset.share.not_found", status=404)
            link = get_link_by_token(session, token)
            dataset_id = link.dataset_id
            if store.get_dataset(dataset_id) is None:
                raise DomainError("dataset.share.not_found", status=404)
        return ClaimDatasetResponse(dataset_id=dataset_id, role=str(role))

    return router
