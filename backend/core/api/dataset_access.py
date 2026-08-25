"""Access-control foundation for sharing personal-library datasets.

The dataset-library twin of :mod:`core.api.sharing_access`. It reuses that
module's effective-role vocabulary (:class:`ShareRole`, :func:`role_rank`,
``MEMBER_ROLES`` and ``LINK_ROLES``) so a dataset and an optimization grant the
same tiers, and supplies the dataset-scoped reads of the share-link and grant
rows plus :func:`resolve_share_access` — the single resolver every
shared-dataset route consults.

The one structural difference from the optimization resolver: a dataset's owner
is a first-class column (``DatasetModel.owner_username``), so ownership is read
straight off the ``datasets`` row rather than reconstructed from a job payload.
On the dataset library the tiers read as:

* ``viewer`` — view metadata + read rows + use-in-a-run + clone.
* ``editor`` — viewer + edit rows.
* ``owner`` — editor + rename + delete + manage sharing + transfer.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..storage.models import DatasetModel, DatasetShareGrantModel, DatasetShareLinkModel
from .auth import AuthenticatedUser, is_admin
from .errors import DomainError
from .sharing_access import (
    MEMBER_ROLES,
    ShareRole,
    _normalize_username,
    role_rank,
)


def get_active_link(session: Session, dataset_id: str) -> DatasetShareLinkModel | None:
    """Return the dataset's live (non-revoked) sharing row, if any.

    Args:
        session: Open DB session.
        dataset_id: Dataset to look up.

    Returns:
        The active :class:`DatasetShareLinkModel`, or ``None`` when no live row
        exists.
    """
    return session.scalars(
        select(DatasetShareLinkModel).where(
            DatasetShareLinkModel.dataset_id == dataset_id,
            DatasetShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def get_link_by_token(session: Session, token: str) -> DatasetShareLinkModel | None:
    """Return the active sharing row for a public ``token``, if any.

    Args:
        session: Open DB session.
        token: The public share token from a ``/datasets/share/<token>`` URL.

    Returns:
        The active :class:`DatasetShareLinkModel`, or ``None`` when the token is
        unknown or its row was revoked.
    """
    return session.scalars(
        select(DatasetShareLinkModel).where(
            DatasetShareLinkModel.token == token,
            DatasetShareLinkModel.revoked_at.is_(None),
        )
    ).first()


def list_grants(session: Session, dataset_id: str) -> list[DatasetShareGrantModel]:
    """Return all member grants for a dataset, ordered by username.

    Args:
        session: Open DB session.
        dataset_id: Dataset whose member grants are listed.

    Returns:
        The dataset's :class:`DatasetShareGrantModel` rows (possibly empty),
        ordered by ``grantee_username`` for stable rendering.
    """
    return list(
        session.scalars(
            select(DatasetShareGrantModel)
            .where(DatasetShareGrantModel.dataset_id == dataset_id)
            .order_by(DatasetShareGrantModel.grantee_username)
        )
    )


def list_grants_for_user(session: Session, dataset_ids: list[str], username: str) -> dict[str, str]:
    """Map ``dataset_id -> role`` for one user's grants among given ids.

    A single batched query backing the shared-with-me listing; returns only ids
    the user actually holds a grant on.

    Args:
        session: Open DB session.
        dataset_ids: Ids to scope the lookup to (empty -> ``{}``).
        username: Grantee username (compared case-insensitively).

    Returns:
        ``{dataset_id: role}`` for the user's grants intersecting the ids.
    """
    if not dataset_ids:
        return {}
    rows = session.scalars(
        select(DatasetShareGrantModel).where(
            DatasetShareGrantModel.grantee_username == _normalize_username(username),
            DatasetShareGrantModel.dataset_id.in_(list(dataset_ids)),
        )
    )
    return {grant.dataset_id: grant.role for grant in rows}


def list_grants_for_user_all(session: Session, username: str) -> dict[str, str]:
    """Map ``dataset_id -> role`` for every grant a user holds.

    Backs the shared-with-me listing, where the candidate dataset ids are not
    known up front (unlike the optimization bulk-action path). Includes both
    named invites and link-derived memberships — the latter make a link-claimed
    dataset list in the claimer's library, exactly like the optimization side
    (stale link rows are pruned when the link is restricted, so what survives is
    always live access).

    Args:
        session: Open DB session.
        username: Grantee username (compared case-insensitively).

    Returns:
        ``{dataset_id: role}`` for every grant the user holds.
    """
    rows = session.scalars(
        select(DatasetShareGrantModel).where(
            DatasetShareGrantModel.grantee_username == _normalize_username(username),
        )
    )
    return {grant.dataset_id: grant.role for grant in rows}


def get_grant(
    session: Session, dataset_id: str, username: str
) -> DatasetShareGrantModel | None:
    """Return a specific user's grant on a dataset, if one exists.

    Args:
        session: Open DB session.
        dataset_id: Dataset to look up.
        username: Grantee username (compared case-insensitively).

    Returns:
        The matching :class:`DatasetShareGrantModel`, or ``None``.
    """
    return session.get(
        DatasetShareGrantModel,
        {"dataset_id": dataset_id, "grantee_username": _normalize_username(username)},
    )


def dataset_owner(session: Session, dataset_id: str) -> str | None:
    """Return the normalized owner username for a dataset, read on ``session``.

    Args:
        session: Open DB session.
        dataset_id: Dataset whose owner is resolved.

    Returns:
        The owner's lowercased username, or ``None`` when the dataset is
        unknown.
    """
    owner = session.scalars(
        select(DatasetModel.owner_username).where(DatasetModel.id == dataset_id)
    ).first()
    return _normalize_username(owner) if owner is not None else None


def resolve_effective_role(
    session: Session, dataset_id: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve a logged-in caller's effective role on a dataset, token-free.

    The owner/admin/member-grant core of :func:`resolve_share_access`, factored
    out so the logged-in library routes resolve access the same way the public
    token page does — without requiring a share token.

    Args:
        session: Open DB session backing the datasets and grant tables.
        dataset_id: Dataset to resolve access on.
        user: Authenticated caller.

    Returns:
        :attr:`ShareRole.owner` for the owner or an admin, the grant's role for
        an invited member, or ``None`` when the caller has neither.
    """
    username = _normalize_username(user.username)
    owner = dataset_owner(session, dataset_id)
    if (owner is not None and owner == username) or is_admin(user):
        return ShareRole.owner
    grant = get_grant(session, dataset_id, username)
    if grant is not None and grant.role in MEMBER_ROLES:
        return ShareRole(grant.role)
    return None


def resolve_share_access(
    session: Session, token: str, user: AuthenticatedUser
) -> ShareRole | None:
    """Resolve the effective role a caller has on a shared dataset.

    The token identifies the dataset but grants no access by itself. The caller
    must be its owner, an administrator, or a named member.

    Args:
        session: Open DB session backing the datasets and share tables.
        token: The public share token from the ``/datasets/share/<token>`` URL.
        user: Authenticated caller.

    Returns:
        The caller's effective :class:`ShareRole`, or ``None`` when the token is
        invalid or the caller has no access under any rule.
    """
    link = get_link_by_token(session, token)
    if link is None:
        return None

    return resolve_effective_role(session, link.dataset_id, user)


def require_role(
    session: Session, dataset_id: str, user: AuthenticatedUser, minimum: ShareRole
) -> ShareRole:
    """Resolve and enforce a minimum effective role on a dataset, token-free.

    The single access gate the logged-in library and sharing routes share. A
    caller with no access at all gets 404 (existence is never leaked); a caller
    who can reach the dataset but holds a lower tier than ``minimum`` gets 403.

    Args:
        session: Open DB session backing the datasets and grant tables.
        dataset_id: Dataset being acted on.
        user: Authenticated caller.
        minimum: Lowest :class:`ShareRole` the route requires.

    Returns:
        The caller's effective :class:`ShareRole` (at least ``minimum``).

    Raises:
        DomainError: 404 when the caller has no access to the dataset; 403 when
            the caller's tier is below ``minimum``.
    """
    role = resolve_effective_role(session, dataset_id, user)
    if role is None:
        raise DomainError("dataset.library.not_found", status=404)
    if role_rank(role) < role_rank(minimum):
        raise DomainError("dataset.library.forbidden", status=403)
    return role
