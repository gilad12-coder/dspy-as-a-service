"""Access-control foundation for named optimization sharing.

Defines the effective-role vocabulary, the helpers that read and write the
per-optimization sharing config (the active share-link row plus its member
grants), and :func:`resolve_share_access` — the single resolver every
share-scoped route consults to turn a ``(token, user)`` pair into an effective
role (or ``None`` for no access).

An active opaque link can address one optimization, but the link alone grants
nothing. The authenticated caller must be the owner, an administrator, or a
named member with a ``viewer`` or ``editor`` grant. Ownership is reassigned only
by outright transfer, so ``owner`` is not a member-grant role. Deployment-wide
Explorer publication is a separate, authenticated read surface.

A caller's effective role is the highest any rule grants. The resolver is
intentionally pure to ``(session, token, user)``: it reads the job owner
straight from the ``jobs`` row on the same session so no router or job-store
wiring leaks into the access layer.
"""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..storage.models import JobModel, OptimizationShareGrantModel, OptimizationShareLinkModel
from .auth import AuthenticatedUser, is_admin
from .converters import job_owner


class ShareRole(StrEnum):
    """Effective access a caller resolves to on a shared optimization.

    Ordered loosest-to-tightest privilege:

    * ``viewer`` — read (real owner shown) + clone; no inference (that spends
      the owner's key, so it is reserved for ``editor`` and above).
    * ``editor`` — viewer + run inference/chat + rename + cancel/retry.
    * ``owner`` — editor + manage sharing + transfer + delete. The creator and
      admins are owners; an optimization has exactly one.

    Access is login-gated: every caller is authenticated, so the floor is
    ``viewer`` (which shows the real owner). There is no anonymous tier.
    """

    viewer = "viewer"
    editor = "editor"
    owner = "owner"


# General-access policy stored on the active share-link row.
GENERAL_ACCESS_RESTRICTED = "restricted"

# Roles a member grant may carry. ``owner`` is excluded — an optimization has
# exactly one owner (its creator), reassigned by outright transfer rather than
# granted (see the ``/sharing/transfer`` route).
MEMBER_ROLES: frozenset[str] = frozenset({ShareRole.viewer, ShareRole.editor})

# Roles retained in the sharing API schema for named-link compatibility.
# They do not grant access without a matching named member row.
LINK_ROLES: frozenset[str] = frozenset({ShareRole.viewer, ShareRole.editor})

# Earlier hosted builds marked link-derived grants with this sentinel. On-prem
# sharing never creates them and deletes any such row encountered during an
# update. Double underscores prevent collision with a real username.
LINK_GRANT_MARKER = "__link__"

# Loosest-to-tightest privilege ordering, used to compare an effective role
# against a route's minimum required tier.
_ROLE_RANK: dict[str, int] = {
    ShareRole.viewer: 0,
    ShareRole.editor: 1,
    ShareRole.owner: 2,
}


def role_rank(role: ShareRole | str) -> int:
    """Return the privilege rank of a role (higher is more privileged).

    Args:
        role: A :class:`ShareRole` (or its string value).

    Returns:
        The integer rank: ``viewer`` 0, ``editor`` 1, ``owner`` 2. Unknown
        roles rank ``-1`` so they never satisfy a minimum tier.
    """
    return _ROLE_RANK.get(str(role), -1)


def _normalize_username(username: str) -> str:
    """Return the canonical lowercased form used for owner/member comparison.

    Args:
        username: Raw username from a token claim or invite request.

    Returns:
        The stripped, lowercased username.
    """
    return username.strip().lower()


def get_active_link(session: Session, optimization_id: str) -> OptimizationShareLinkModel | None:
    """Return the optimization's live (non-revoked) sharing row, if any.

    Args:
        session: Open DB session.
        optimization_id: Optimization to look up.

    Returns:
        The active :class:`OptimizationShareLinkModel`, or ``None`` when no
        live row exists.
    """
    return session.scalars(
        select(OptimizationShareLinkModel).where(
            OptimizationShareLinkModel.optimization_id == optimization_id,
            OptimizationShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def get_link_by_token(session: Session, token: str) -> OptimizationShareLinkModel | None:
    """Return the active sharing row for a public ``token``, if any.

    Args:
        session: Open DB session.
        token: The public share token from a ``/share/<token>`` URL.

    Returns:
        The active :class:`OptimizationShareLinkModel`, or ``None`` when the
        token is unknown or its row was revoked.
    """
    return session.scalars(
        select(OptimizationShareLinkModel).where(
            OptimizationShareLinkModel.token == token,
            OptimizationShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def list_grants(session: Session, optimization_id: str) -> list[OptimizationShareGrantModel]:
    """Return all member grants for an optimization, ordered by username.

    Args:
        session: Open DB session.
        optimization_id: Optimization whose member grants are listed.

    Returns:
        The optimization's :class:`OptimizationShareGrantModel` rows (possibly
        empty), ordered by ``grantee_username`` for stable rendering.
    """
    return list(
        session.scalars(
            select(OptimizationShareGrantModel)
            .where(OptimizationShareGrantModel.optimization_id == optimization_id)
            .order_by(OptimizationShareGrantModel.grantee_username)
        )
    )


def list_grants_for_user(
    session: Session, optimization_ids: list[str], username: str
) -> dict[str, str]:
    """Map ``optimization_id -> role`` for one user's grants among given ids.

    A single batched query backing tier checks over a set of ids (bulk
    actions); returns only ids the user actually holds a grant on.

    Args:
        session: Open DB session.
        optimization_ids: Ids to scope the lookup to (empty -> ``{}``).
        username: Grantee username (compared case-insensitively).

    Returns:
        ``{optimization_id: role}`` for the user's grants intersecting the ids.
    """
    if not optimization_ids:
        return {}
    rows = session.scalars(
        select(OptimizationShareGrantModel).where(
            OptimizationShareGrantModel.grantee_username == _normalize_username(username),
            OptimizationShareGrantModel.optimization_id.in_(list(optimization_ids)),
        )
    )
    return {grant.optimization_id: grant.role for grant in rows}


def get_grant(
    session: Session, optimization_id: str, username: str
) -> OptimizationShareGrantModel | None:
    """Return a specific user's grant on an optimization, if one exists.

    Args:
        session: Open DB session.
        optimization_id: Optimization to look up.
        username: Grantee username (compared case-insensitively).

    Returns:
        The matching :class:`OptimizationShareGrantModel`, or ``None``.
    """
    return session.get(
        OptimizationShareGrantModel,
        {"optimization_id": optimization_id, "grantee_username": _normalize_username(username)},
    )


def _job_owner_for(session: Session, optimization_id: str) -> str | None:
    """Return the normalized owner username for a job, read on ``session``.

    Loads only the owner-bearing columns from the ``jobs`` row and reuses
    :func:`core.api.routers._helpers.job_owner` for identical normalization to
    the rest of the API (payload ``username`` first, then ``payload_overview``).

    Args:
        session: Open DB session.
        optimization_id: Optimization whose owner is resolved.

    Returns:
        The job owner's lowercased username, or ``None`` when the job is
        unknown or carries no owner.
    """
    row = session.execute(
        select(JobModel.payload, JobModel.payload_overview).where(
            JobModel.optimization_id == optimization_id
        )
    ).first()
    if row is None:
        return None
    payload, payload_overview = row
    return job_owner({"payload": payload, "payload_overview": payload_overview})


def resolve_effective_role(
    session: Session, optimization_id: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve a logged-in caller's effective role on an optimization, token-free.

    The owner/admin/member-grant core of :func:`resolve_share_access`, factored
    out so the logged-in app (the normal ``/optimizations/{id}`` routes and the
    sidebar) can resolve access the same way the ``/share/<token>`` page does —
    without requiring a share token. Returns ``None`` for a caller with no
    ownership and no grant.

    Args:
        session: Open DB session backing the jobs and grant tables.
        optimization_id: Optimization to resolve access on.
        user: Authenticated caller.

    Returns:
        :attr:`ShareRole.owner` for the creator or an admin, the grant's role
        for an invited member, or ``None`` when the caller has neither.
    """
    username = _normalize_username(user.username)
    owner = _job_owner_for(session, optimization_id)
    if (owner is not None and owner == username) or is_admin(user):
        return ShareRole.owner
    grant = get_grant(session, optimization_id, username)
    if grant is not None and grant.role in MEMBER_ROLES:
        return ShareRole(grant.role)
    return None


def resolve_share_access(
    session: Session, token: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve the effective role a caller has on a shared optimization.

    The active token identifies the optimization but grants no access by
    itself. The caller must resolve as its owner, an administrator, or a named
    member via :func:`resolve_effective_role`. With no applicable named rule the
    caller gets ``None`` (404).

    Args:
        session: Open DB session backing the jobs and share tables.
        token: The public share token from the ``/share/<token>`` URL.
        user: Authenticated caller.

    Returns:
        The caller's effective :class:`ShareRole`, or ``None`` when the token is
        invalid or the caller has no access under any rule.
    """
    link = get_link_by_token(session, token)
    if link is None:
        return None

    return resolve_effective_role(session, link.optimization_id, user)
