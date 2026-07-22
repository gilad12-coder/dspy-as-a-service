"""Access-control foundation for sharing saved tagger sessions.

The tagging-session twin of :mod:`core.api.dataset_access`. It reuses the
effective-role vocabulary from :mod:`core.api.sharing_access` (:class:`ShareRole`,
:func:`role_rank`, the ``GENERAL_ACCESS_*`` policies, ``MEMBER_ROLES`` /
``LINK_ROLES`` and the ``LINK_GRANT_MARKER`` sentinel) so a session grants the
same tiers as a dataset, and supplies the session-scoped reads of the
share-link and grant rows plus :func:`resolve_share_access` — the single
resolver every shared-session route consults.

A session's owner is its ``username`` column (there is no separate owner
column), so ownership is read straight off the ``tagging_sessions`` row. On
tagger sessions the tiers read as:

* ``viewer`` — open the session and browse its rows and labels read-only.
* ``editor`` — viewer + annotate, autosave progress and run AI assistance.
* ``owner`` — editor + rename + pin + delete + manage sharing + transfer.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..storage.models import (
    TaggingSessionModel,
    TaggingSessionShareGrantModel,
    TaggingSessionShareLinkModel,
)
from .auth import AuthenticatedUser, is_admin
from .errors import DomainError
from .sharing_access import (
    GENERAL_ACCESS_ANYONE,
    LINK_ROLES,
    MEMBER_ROLES,
    ShareRole,
    _normalize_username,
    role_rank,
)


def get_active_link(session: Session, session_id: str) -> TaggingSessionShareLinkModel | None:
    """Return the tagger session's live (non-revoked) sharing row, if any.

    Args:
        session: Open DB session.
        session_id: Tagger session to look up.

    Returns:
        The active :class:`TaggingSessionShareLinkModel`, or ``None`` when no
        live row exists.
    """
    return session.scalars(
        select(TaggingSessionShareLinkModel).where(
            TaggingSessionShareLinkModel.session_id == session_id,
            TaggingSessionShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def get_link_by_token(session: Session, token: str) -> TaggingSessionShareLinkModel | None:
    """Return the active sharing row for a public ``token``, if any.

    Args:
        session: Open DB session.
        token: The public share token from a ``/tagger/share/<token>`` URL.

    Returns:
        The active :class:`TaggingSessionShareLinkModel`, or ``None`` when the
        token is unknown or its row was revoked.
    """
    return session.scalars(
        select(TaggingSessionShareLinkModel).where(
            TaggingSessionShareLinkModel.token == token,
            TaggingSessionShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def list_grants(session: Session, session_id: str) -> list[TaggingSessionShareGrantModel]:
    """Return all member grants for a tagger session, ordered by username.

    Args:
        session: Open DB session.
        session_id: Tagger session whose member grants are listed.

    Returns:
        The session's :class:`TaggingSessionShareGrantModel` rows (possibly
        empty), ordered by ``grantee_username`` for stable rendering.
    """
    return list(
        session.scalars(
            select(TaggingSessionShareGrantModel)
            .where(TaggingSessionShareGrantModel.session_id == session_id)
            .order_by(TaggingSessionShareGrantModel.grantee_username)
        )
    )


def list_grants_for_user_all(session: Session, username: str) -> dict[str, str]:
    """Map ``session_id -> role`` for every grant a user holds.

    Backs the shared-with-me listing, where the candidate session ids are not
    known up front. Includes both named invites and link-derived memberships —
    the latter make a link-claimed session list in the claimer's chooser (stale
    link rows are pruned when the link is restricted, so what survives is
    always live access).

    Args:
        session: Open DB session.
        username: Grantee username (compared case-insensitively).

    Returns:
        ``{session_id: role}`` for every grant the user holds.
    """
    rows = session.scalars(
        select(TaggingSessionShareGrantModel).where(
            TaggingSessionShareGrantModel.grantee_username == _normalize_username(username),
        )
    )
    return {grant.session_id: grant.role for grant in rows}


def get_grant(
    session: Session, session_id: str, username: str
) -> TaggingSessionShareGrantModel | None:
    """Return a specific user's grant on a tagger session, if one exists.

    Args:
        session: Open DB session.
        session_id: Tagger session to look up.
        username: Grantee username (compared case-insensitively).

    Returns:
        The matching :class:`TaggingSessionShareGrantModel`, or ``None``.
    """
    return session.get(
        TaggingSessionShareGrantModel,
        {"session_id": session_id, "grantee_username": _normalize_username(username)},
    )


def session_owner(session: Session, session_id: str) -> str | None:
    """Return the normalized owner username for a tagger session.

    Args:
        session: Open DB session.
        session_id: Tagger session whose owner is resolved.

    Returns:
        The owner's lowercased username, or ``None`` when the session is
        unknown.
    """
    owner = session.scalars(
        select(TaggingSessionModel.username).where(TaggingSessionModel.id == session_id)
    ).first()
    return _normalize_username(owner) if owner is not None else None


def resolve_effective_role(
    session: Session, session_id: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve a logged-in caller's effective role on a tagger session, token-free.

    The owner/admin/member-grant core of :func:`resolve_share_access`, factored
    out so the logged-in session routes resolve access the same way the public
    token flow does — without requiring a share token.

    Args:
        session: Open DB session backing the sessions and grant tables.
        session_id: Tagger session to resolve access on.
        user: Authenticated caller.

    Returns:
        :attr:`ShareRole.owner` for the owner or an admin, the grant's role for
        an invited member, or ``None`` when the caller has neither.
    """
    username = _normalize_username(user.username)
    owner = session_owner(session, session_id)
    if (owner is not None and owner == username) or is_admin(user):
        return ShareRole.owner
    grant = get_grant(session, session_id, username)
    if grant is not None and grant.role in MEMBER_ROLES:
        return ShareRole(grant.role)
    return None


def resolve_share_access(
    session: Session, token: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve the effective role a caller has on a shared tagger session.

    The caller gets the *highest* tier any applicable rule grants, assembled
    from the active link by ``token``, the owner/admin/member resolution, and —
    under an ``'anyone'`` link — the link's ``general_role``. Access is
    login-gated, so a bare URL never resolves to access on its own.

    Args:
        session: Open DB session backing the sessions and share tables.
        token: The public share token from the ``/tagger/share/<token>`` URL.
        user: Authenticated caller.

    Returns:
        The caller's effective :class:`ShareRole`, or ``None`` when the token is
        invalid or the caller has no access under any rule.
    """
    link = get_link_by_token(session, token)
    if link is None:
        return None

    candidates: list[ShareRole] = []
    resolved = resolve_effective_role(session, link.session_id, user)
    if resolved is not None:
        candidates.append(resolved)

    if link.general_access == GENERAL_ACCESS_ANYONE:
        link_role = link.general_role if link.general_role in LINK_ROLES else ShareRole.viewer
        candidates.append(ShareRole(link_role))

    if not candidates:
        return None
    return max(candidates, key=role_rank)


def require_role(
    session: Session, session_id: str, user: AuthenticatedUser, minimum: ShareRole
) -> ShareRole:
    """Resolve and enforce a minimum effective role on a tagger session.

    The single access gate the logged-in session and sharing routes share. A
    caller with no access at all gets 404 (existence is never leaked); a caller
    who can reach the session but holds a lower tier than ``minimum`` gets 403.

    Args:
        session: Open DB session backing the sessions and grant tables.
        session_id: Tagger session being acted on.
        user: Authenticated caller.
        minimum: Lowest :class:`ShareRole` the route requires.

    Returns:
        The caller's effective :class:`ShareRole` (at least ``minimum``).

    Raises:
        DomainError: 404 when the caller has no access to the session; 403 when
            the caller's tier is below ``minimum``.
    """
    role = resolve_effective_role(session, session_id, user)
    if role is None:
        raise DomainError("tagger.session.not_found", status=404)
    if role_rank(role) < role_rank(minimum):
        raise DomainError("tagger.session.forbidden", status=403)
    return role
