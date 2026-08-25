"""Google-Drive-style sharing for saved tagger sessions.

The tagging-session twin of :mod:`core.api.routers.dataset_share`. Owner-gated
management endpoints plus the access-gated claim:

* ``GET    /tagging-sessions/{id}/sharing`` — current sharing config (owner).
* ``PUT    /tagging-sessions/{id}/sharing`` — set the general-access policy.
* ``POST   /tagging-sessions/{id}/sharing/members`` — add/replace a member grant.
* ``PATCH  /tagging-sessions/{id}/sharing/members/{username}`` — change a role.
* ``DELETE /tagging-sessions/{id}/sharing/members/{username}`` — remove a grant.
* ``POST   /tagging-sessions/{id}/sharing/transfer`` — reassign ownership to an
  existing member (the previous owner is demoted to an editor).
* ``POST   /tagging-sessions/share/{token}/claim`` — redeem an ``anyone`` link,
  recording a link membership so the session lists in the caller's chooser.

There is no composite token read: once a caller is a member (invited or via
claim), the role-gated ``GET /tagging-sessions/{id}`` serves the session, so
the claim page just redeems and redirects. Two sharing modes coexist per
:mod:`core.api.tagging_session_access`: the active link's ``general_access``
and ``general_role`` combine with per-user member grants; effective access is
the highest the rules allow. The invite people-picker reuses the shared
``GET /users/search`` autocomplete.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...storage.models import (
    TaggingSessionModel,
    TaggingSessionShareGrantModel,
    TaggingSessionShareLinkModel,
)
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..sharing_access import (
    GENERAL_ACCESS_ANYONE,
    GENERAL_ACCESS_RESTRICTED,
    LINK_GRANT_MARKER,
    LINK_ROLES,
    MEMBER_ROLES,
    ShareRole,
)
from ..tagging_session_access import (
    get_active_link,
    get_grant,
    get_link_by_token,
    list_grants,
    require_role,
    resolve_share_access,
    session_owner,
)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

_GENERAL_ACCESS_VALUES = (GENERAL_ACCESS_RESTRICTED, GENERAL_ACCESS_ANYONE)
_LINK_ROLE_VALUES = tuple(sorted(LINK_ROLES))


class TaggingSessionSharingMember(BaseModel):
    """One invited member of a tagger session (username + tier role)."""

    username: str
    role: str


class TaggingSessionSharingState(BaseModel):
    """Owner-facing sharing config for one saved tagger session."""

    general_access: str
    general_role: str = "viewer"
    token: str | None = None
    share_path: str | None = None
    owner: str | None = None
    members: list[TaggingSessionSharingMember] = Field(default_factory=list)


class PutTaggingSessionSharingRequest(BaseModel):
    """Request body for ``PUT /tagging-sessions/{id}/sharing``."""

    general_access: str
    general_role: str | None = None


class AddTaggingSessionMemberRequest(BaseModel):
    """Request body for ``POST /tagging-sessions/{id}/sharing/members``."""

    username: str
    role: str


class UpdateTaggingSessionMemberRequest(BaseModel):
    """Request body for ``PATCH /tagging-sessions/{id}/sharing/members/{username}``."""

    role: str


class TransferTaggingSessionOwnershipRequest(BaseModel):
    """Request body for ``POST /tagging-sessions/{id}/sharing/transfer``."""

    username: str


class ClaimTaggingSessionResponse(BaseModel):
    """Envelope for ``POST /tagging-sessions/share/{token}/claim`` — the target session."""

    session_id: str
    role: str


def create_tagging_session_share_router(*, job_store) -> APIRouter:
    """Build the Google-Drive-style tagger-session sharing router.

    Args:
        job_store: Storage backend whose ``engine`` carries the
            ``tagging_sessions`` and session-share tables.

    Returns:
        A FastAPI ``APIRouter`` with the owner-gated management routes and the
        access-gated claim endpoint.
    """
    router = APIRouter()

    def _require_manage(session: Session, session_id: str, user: AuthenticatedUser) -> str | None:
        """Ensure ``user`` may manage sharing for ``session_id``, returning its owner.

        Management is owner-only: only the session owner or an admin may invite
        people, change roles, transfer ownership, and set general access.

        Args:
            session: Open DB session.
            session_id: Tagger session being managed.
            user: Authenticated caller.

        Returns:
            The session owner's lowercased username.

        Raises:
            DomainError: 404 when the caller has no access to the session; 403
                when the caller can reach it but is not the owner/admin.
        """
        require_role(session, session_id, user, ShareRole.owner)
        return session_owner(session, session_id)

    def _sharing_state(
        session: Session, session_id: str, owner: str | None
    ) -> TaggingSessionSharingState:
        """Assemble the current :class:`TaggingSessionSharingState` for a session.

        Args:
            session: Open DB session.
            session_id: Tagger session to describe.
            owner: The session owner username (shown to the manager).

        Returns:
            The populated :class:`TaggingSessionSharingState`.
        """
        link = get_active_link(session, session_id)
        general_access = link.general_access if link is not None else GENERAL_ACCESS_RESTRICTED
        general_role = link.general_role if link is not None else str(ShareRole.viewer)
        token = link.token if link is not None else None
        members = [
            TaggingSessionSharingMember(username=g.grantee_username, role=g.role)
            for g in list_grants(session, session_id)
            if g.created_by != LINK_GRANT_MARKER
        ]
        return TaggingSessionSharingState(
            general_access=general_access,
            general_role=general_role,
            token=token,
            share_path=f"/tagger/share/{token}" if token else None,
            owner=owner,
            members=members,
        )

    def _ensure_link(
        session: Session, session_id: str, created_by: str
    ) -> TaggingSessionShareLinkModel:
        """Return the active link, minting one if none exists.

        Args:
            session: Open DB session (caller commits).
            session_id: Tagger session the link belongs to.
            created_by: Username recorded as the link creator.

        Returns:
            The active :class:`TaggingSessionShareLinkModel`.
        """
        link = get_active_link(session, session_id)
        if link is None:
            link = TaggingSessionShareLinkModel(
                token=secrets.token_urlsafe(24),
                session_id=session_id,
                created_by=created_by,
                created_at=datetime.now(UTC),
                general_access=GENERAL_ACCESS_RESTRICTED,
                general_role=str(ShareRole.viewer),
            )
            session.add(link)
        return link

    def _sync_link_memberships(
        session: Session, session_id: str, link: TaggingSessionShareLinkModel
    ) -> None:
        """Reconcile link-derived memberships with the link's current policy.

        Drive-style live propagation: an ``anyone`` link re-points every link
        membership at its current tier; a ``restricted`` link (turning the link
        off) deletes them, revoking access and dropping the session from those
        users' choosers. Named invites are never touched. Caller commits.

        Args:
            session: Open DB session.
            session_id: Tagger session whose link memberships are reconciled.
            link: The just-updated active link row.
        """
        markers = session.scalars(
            select(TaggingSessionShareGrantModel).where(
                TaggingSessionShareGrantModel.session_id == session_id,
                TaggingSessionShareGrantModel.created_by == LINK_GRANT_MARKER,
            )
        )
        if link.general_access == GENERAL_ACCESS_ANYONE and link.general_role in MEMBER_ROLES:
            for grant in markers:
                grant.role = link.general_role
        else:
            for grant in markers:
                session.delete(grant)

    @router.get(
        "/tagging-sessions/{session_id}/sharing",
        response_model=TaggingSessionSharingState,
        summary="Get the sharing config (general access + members) for a session",
    )
    def get_sharing(
        session_id: str, current_user: AuthenticatedUserDep
    ) -> TaggingSessionSharingState:
        """Return the session's sharing config for its owner.

        Args:
            session_id: Tagger session to inspect.
            current_user: Authenticated owner/admin.

        Returns:
            The current :class:`TaggingSessionSharingState`.

        Raises:
            DomainError: 404 when unknown/inaccessible; 403 when the caller may
                not manage sharing.
        """
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            return _sharing_state(session, session_id, owner)

    @router.put(
        "/tagging-sessions/{session_id}/sharing",
        response_model=TaggingSessionSharingState,
        summary="Set the general-access policy (restricted vs anyone-with-link)",
    )
    def put_sharing(
        session_id: str,
        req: PutTaggingSessionSharingRequest,
        current_user: AuthenticatedUserDep,
    ) -> TaggingSessionSharingState:
        """Set the link's general-access policy and tier, minting a link if needed.

        ``general_role`` is the tier an ``anyone`` link grants a signed-in
        visitor (``viewer``/``editor``); omit it to leave the current tier
        unchanged.

        Args:
            session_id: Tagger session to update.
            req: Body carrying the new ``general_access`` policy and optional
                ``general_role`` tier.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`TaggingSessionSharingState`.

        Raises:
            DomainError: 404/403 on access; 400 when ``general_access`` or
                ``general_role`` is invalid.
        """
        if req.general_access not in _GENERAL_ACCESS_VALUES:
            raise DomainError(
                "share.invalid_general_access",
                status=400,
                allowed=", ".join(_GENERAL_ACCESS_VALUES),
            )
        if req.general_role is not None and req.general_role not in LINK_ROLES:
            raise DomainError(
                "share.invalid_role",
                status=400,
                role=req.general_role,
                allowed=", ".join(_LINK_ROLE_VALUES),
            )
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            link = _ensure_link(session, session_id, current_user.username)
            link.general_access = req.general_access
            if req.general_role is not None:
                link.general_role = req.general_role
            _sync_link_memberships(session, session_id, link)
            session.commit()
            return _sharing_state(session, session_id, owner)

    @router.post(
        "/tagging-sessions/{session_id}/sharing/members",
        response_model=TaggingSessionSharingState,
        summary="Invite a user (add or replace a member grant)",
    )
    def add_member(
        session_id: str,
        req: AddTaggingSessionMemberRequest,
        current_user: AuthenticatedUserDep,
    ) -> TaggingSessionSharingState:
        """Add or replace a member grant on the session.

        Args:
            session_id: Tagger session to share.
            req: Body carrying the grantee ``username`` and tier ``role``.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`TaggingSessionSharingState`.

        Raises:
            DomainError: 404/403 on access; 400 when the role is invalid or the
                caller tries to grant the owner themselves.
        """
        if req.role not in MEMBER_ROLES:
            raise DomainError(
                "share.invalid_role",
                status=400,
                role=req.role,
                allowed=", ".join(sorted(MEMBER_ROLES)),
            )
        grantee = req.username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            if owner is not None and grantee == owner:
                raise DomainError("tagger.session.share.cannot_grant_self", status=400)
            existing = get_grant(session, session_id, grantee)
            if existing is not None:
                existing.role = req.role
                # Promote a link membership to a named (authoritative) invite so
                # it no longer tracks or gets revoked with the link.
                existing.created_by = current_user.username
            else:
                session.add(
                    TaggingSessionShareGrantModel(
                        session_id=session_id,
                        grantee_username=grantee,
                        role=req.role,
                        created_by=current_user.username,
                        created_at=datetime.now(UTC),
                    )
                )
            session.commit()
            return _sharing_state(session, session_id, owner)

    @router.patch(
        "/tagging-sessions/{session_id}/sharing/members/{username}",
        response_model=TaggingSessionSharingState,
        summary="Change an existing member's role",
    )
    def update_member(
        session_id: str,
        username: str,
        req: UpdateTaggingSessionMemberRequest,
        current_user: AuthenticatedUserDep,
    ) -> TaggingSessionSharingState:
        """Change an existing member's tier role.

        Args:
            session_id: Tagger session to update.
            username: Grantee whose role changes.
            req: Body carrying the new ``role``.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`TaggingSessionSharingState`.

        Raises:
            DomainError: 404/403 on access; 404
                (``tagger.session.share.member_not_found``) when the member has
                no grant; 400 when the role is invalid or the caller targets
                their own grant.
        """
        if req.role not in MEMBER_ROLES:
            raise DomainError(
                "share.invalid_role",
                status=400,
                role=req.role,
                allowed=", ".join(sorted(MEMBER_ROLES)),
            )
        grantee = username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            if grantee == current_user.username.strip().lower():
                raise DomainError("tagger.session.share.cannot_modify_self", status=400)
            grant = get_grant(session, session_id, grantee)
            if grant is None:
                raise DomainError(
                    "tagger.session.share.member_not_found", status=404, username=grantee
                )
            grant.role = req.role
            session.commit()
            return _sharing_state(session, session_id, owner)

    @router.delete(
        "/tagging-sessions/{session_id}/sharing/members/{username}",
        response_model=TaggingSessionSharingState,
        summary="Remove a member's grant",
    )
    def remove_member(
        session_id: str, username: str, current_user: AuthenticatedUserDep
    ) -> TaggingSessionSharingState:
        """Remove a member's grant from the session.

        Args:
            session_id: Tagger session to update.
            username: Grantee whose grant is removed.
            current_user: Authenticated owner/admin.

        Returns:
            The updated :class:`TaggingSessionSharingState`.

        Raises:
            DomainError: 404/403 on access; 404
                (``tagger.session.share.member_not_found``) when the member has
                no grant; 400 when the caller targets their own grant.
        """
        grantee = username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            if grantee == current_user.username.strip().lower():
                raise DomainError("tagger.session.share.cannot_modify_self", status=400)
            grant = get_grant(session, session_id, grantee)
            if grant is None:
                raise DomainError(
                    "tagger.session.share.member_not_found", status=404, username=grantee
                )
            session.delete(grant)
            session.commit()
            return _sharing_state(session, session_id, owner)

    @router.post(
        "/tagging-sessions/{session_id}/sharing/transfer",
        response_model=TaggingSessionSharingState,
        summary="Transfer ownership to an existing member (old owner becomes an editor)",
    )
    def transfer_ownership(
        session_id: str,
        req: TransferTaggingSessionOwnershipRequest,
        current_user: AuthenticatedUserDep,
    ) -> TaggingSessionSharingState:
        """Hand a session's ownership to an existing member.

        A session has exactly one owner (its ``username`` column), so ownership
        is reassigned outright: the column moves to the new owner, the previous
        owner is demoted to an ``editor`` grant, and the new owner's member
        grant is dropped. The new owner must already be a member.

        Args:
            session_id: Tagger session whose ownership moves.
            req: Body carrying the new owner ``username`` (an existing member).
            current_user: Authenticated current owner/admin.

        Returns:
            The updated :class:`TaggingSessionSharingState` — ``owner`` is the
            new owner and the previous owner now appears as an ``editor``
            member.

        Raises:
            DomainError: 404/403 on access; 400 when transferring to the
                current owner; 404 (``tagger.session.share.member_not_found``)
                when the target is not a member.
        """
        new_owner = req.username.strip().lower()
        with Session(job_store.engine) as session:
            owner = _require_manage(session, session_id, current_user)
            if owner is not None and new_owner == owner:
                raise DomainError("tagger.session.share.cannot_grant_self", status=400)
            grant = get_grant(session, session_id, new_owner)
            if grant is None:
                raise DomainError(
                    "tagger.session.share.member_not_found", status=404, username=new_owner
                )
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            row.username = new_owner
            session.delete(grant)
            if owner is not None:
                demoted = get_grant(session, session_id, owner)
                if demoted is not None:
                    demoted.role = str(ShareRole.editor)
                else:
                    session.add(
                        TaggingSessionShareGrantModel(
                            session_id=session_id,
                            grantee_username=owner,
                            role=str(ShareRole.editor),
                            created_by=current_user.username,
                            created_at=datetime.now(UTC),
                        )
                    )
            session.commit()
            return _sharing_state(session, session_id, new_owner)

    @router.post(
        "/tagging-sessions/share/{token}/claim",
        response_model=ClaimTaggingSessionResponse,
        summary="Redeem a share link: record a link membership and return the session",
    )
    def claim_shared_session(
        token: str, current_user: AuthenticatedUserDep
    ) -> ClaimTaggingSessionResponse:
        """Redeem a share link, joining the caller to it, then point them at it.

        Opening an ``anyone`` link records a link membership for the signed-in
        caller at the link's current tier, so the session lists in their
        chooser and the normal session routes resolve them to that tier. The
        membership tracks the link (not frozen): a later tier change or
        restriction syncs or removes it. A ``restricted`` link grants nothing —
        the caller must be the owner or a named invitee, else 404.

        Args:
            token: The public share token from the URL.
            current_user: Authenticated caller redeeming the link.

        Returns:
            A :class:`ClaimTaggingSessionResponse` with the target
            ``session_id`` and the caller's effective ``role`` after
            redemption.

        Raises:
            DomainError: 404 when the token is unknown/revoked, the session is
                gone, or the caller has no access under the link's policy.
        """
        with Session(job_store.engine) as session:
            role = resolve_share_access(session, token, current_user)
            if role is None:
                raise DomainError("tagger.session.share.not_found", status=404)
            link = get_link_by_token(session, token)
            session_id = link.session_id
            if session.get(TaggingSessionModel, session_id) is None:
                raise DomainError("tagger.session.share.not_found", status=404)
            if (
                role != ShareRole.owner
                and link.general_access == GENERAL_ACCESS_ANYONE
                and link.general_role in MEMBER_ROLES
            ):
                username = current_user.username.strip().lower()
                existing = get_grant(session, session_id, username)
                if existing is None:
                    session.add(
                        TaggingSessionShareGrantModel(
                            session_id=session_id,
                            grantee_username=username,
                            role=link.general_role,
                            created_by=LINK_GRANT_MARKER,
                            created_at=datetime.now(UTC),
                        )
                    )
                    session.commit()
                elif existing.created_by == LINK_GRANT_MARKER and existing.role != link.general_role:
                    existing.role = link.general_role
                    session.commit()
        return ClaimTaggingSessionResponse(session_id=session_id, role=str(role))

    return router
