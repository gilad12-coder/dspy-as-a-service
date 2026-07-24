"""CRUD routes for persisted text-labeling (tagger) sessions. [INTERNAL]

Backs the tagger's save / restore so a user can resume annotating across
refreshes and devices, mirroring how optimizations persist. Access is resolved
through :mod:`core.api.tagging_session_access`: the row's ``username`` is the
owner, and share grants raise other callers to ``viewer`` (read) or ``editor``
(annotate); rename/pin/delete stay owner-only.

The dataset (``data``/``columns``/``config``) is uploaded once at create time;
annotation progress is autosaved through the lightweight PUT so the heavy JSON
payload is not re-shipped on every label.

Hidden from the public Scalar reference — wizard-internal flow.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...models import (
    BulkDeleteByIdsRequest,
    BulkDeleteByIdsResponse,
    BulkDeleteByIdsSkipped,
)
from ...storage.models import TaggingSessionModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..sharing_access import MEMBER_ROLES, ShareRole
from ..tagging_session_access import (
    list_grants_for_user_all,
    require_role,
    resolve_effective_role,
)

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

MAX_LIST = 200
DEFAULT_LIST = 100
MAX_NAME = 200


class TaggingSessionSummary(BaseModel):
    """List-row projection of a saved tagger session (no heavy JSON payload)."""

    id: str
    name: str
    phase: str
    row_count: int
    tagged_count: int
    pinned: bool
    created_at: datetime
    updated_at: datetime
    mode: str | None = None
    source_name: str | None = None
    # The caller's effective tier on the row ("owner" for their own sessions);
    # lets the chooser badge shared-in sessions and gate owner-only actions.
    role: str = "owner"


class TaggingSessionDetail(BaseModel):
    """Full tagger session — the complete state needed to resume annotating."""

    id: str
    name: str
    phase: str
    config: dict[str, Any]
    columns: list[str]
    data: list[dict[str, Any]]
    annotations: dict[str, Any]
    assist: dict[str, Any] | None = None
    current_index: int
    row_count: int
    tagged_count: int
    pinned: bool
    created_at: datetime
    updated_at: datetime
    role: str = "owner"


class TaggingSessionCreateRequest(BaseModel):
    """Create a session from the tagger's in-memory state when annotating begins."""

    name: str = Field(default="", max_length=MAX_NAME)
    phase: str = Field(default="annotating")
    config: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    data: list[dict[str, Any]] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)
    assist: dict[str, Any] | None = None
    current_index: int = Field(default=0, ge=0)


class TaggingSessionProgressRequest(BaseModel):
    """Autosave the mutable annotation progress without re-uploading the dataset."""

    annotations: dict[str, Any] = Field(default_factory=dict)
    # AI co-tagging state; ``None`` (absent) leaves the stored value untouched
    # so plain manual autosaves never clobber assist progress written elsewhere.
    assist: dict[str, Any] | None = None
    current_index: int = Field(default=0, ge=0)
    phase: str | None = None


class TaggingSessionPatchRequest(BaseModel):
    """Partial metadata update — at least one of ``name`` or ``pinned`` required."""

    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME)
    pinned: bool | None = None


class TaggingSessionListResponse(BaseModel):
    """Paginated sidebar list of the caller's tagger sessions."""

    items: list[TaggingSessionSummary]
    total: int


def _count_tagged(annotations: dict[str, Any]) -> int:
    """Count how many rows carry a present, non-empty annotation.

    Args:
        annotations: The ``{row_id: annotation}`` map; a value is a string
            (binary/freetext), a list of category ids (multiclass), or
            ``None``/empty when the row is not yet labeled.

    Returns:
        The number of rows whose annotation is present and non-empty.
    """
    total = 0
    for value in annotations.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        total += 1
    return total


def _summary_meta(config: Any) -> tuple[str | None, str | None]:
    """Extract the session-card metadata the frontend stores inside ``config``.

    Args:
        config: The session's ``config`` JSON value (may be ``None`` on old rows).

    Returns:
        ``(assist_mode, source_name)``; each ``None`` when absent so cards
        created before these keys existed degrade to showing nothing.
    """
    if not isinstance(config, dict):
        return None, None
    mode = str(config.get("assistMode") or "").strip() or None
    source = str(config.get("sourceName") or "").strip() or None
    return mode, source


def _row_to_summary(row: TaggingSessionModel, role: str = "owner") -> TaggingSessionSummary:
    """Project an ORM row into the lightweight list/summary response shape.

    Args:
        row: Loaded ``TaggingSessionModel``.
        role: The caller's effective tier on the row.

    Returns:
        The serializable summary the list / patch / put routes return.
    """
    mode, source_name = _summary_meta(row.config)
    return TaggingSessionSummary(
        id=cast(str, row.id),
        name=cast(str, row.name),
        phase=cast(str, row.phase),
        row_count=cast(int, row.row_count),
        tagged_count=cast(int, row.tagged_count),
        pinned=cast(bool, row.pinned),
        created_at=cast(datetime, row.created_at),
        updated_at=cast(datetime, row.updated_at),
        mode=mode,
        source_name=source_name,
        role=role,
    )


def _row_to_detail(row: TaggingSessionModel, role: str = "owner") -> TaggingSessionDetail:
    """Project an ORM row into the full restore-payload response shape.

    Args:
        row: Loaded ``TaggingSessionModel``.
        role: The caller's effective tier on the row.

    Returns:
        A ``TaggingSessionDetail`` carrying the complete session state.
    """
    return TaggingSessionDetail(
        id=cast(str, row.id),
        name=cast(str, row.name),
        phase=cast(str, row.phase),
        config=cast("dict[str, Any]", row.config),
        columns=cast("list[str]", row.columns),
        data=cast("list[dict[str, Any]]", row.data),
        annotations=cast("dict[str, Any]", row.annotations),
        assist=cast("dict[str, Any] | None", row.assist),
        current_index=cast(int, row.current_index),
        row_count=cast(int, row.row_count),
        tagged_count=cast(int, row.tagged_count),
        pinned=cast(bool, row.pinned),
        created_at=cast(datetime, row.created_at),
        updated_at=cast(datetime, row.updated_at),
        role=role,
    )


def create_tagging_session_router(*, job_store) -> APIRouter:
    """Build the tagger-session persistence router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.

    Returns:
        A FastAPI ``APIRouter`` exposing list / create / get / update / patch /
        delete routes for the caller's saved tagger sessions.
    """
    router = APIRouter()

    @router.get(
        "/tagging-sessions",
        response_model=TaggingSessionListResponse,
        operation_id="list_tagging_sessions_for_agent",
        summary="List the caller's saved tagger sessions",
        tags=["agent"],
    )
    def list_tagging_sessions(
        user: AuthenticatedUserDep,
        limit: int = Query(default=DEFAULT_LIST, ge=1, le=MAX_LIST),
        offset: int = Query(default=0, ge=0),
    ) -> TaggingSessionListResponse:
        """Return the caller's sessions, pinned first then newest activity.

        The heavy JSON columns (``data``/``annotations``/``assist``) are not
        selected so the sidebar list stays cheap as sessions accumulate;
        ``config`` is small (task shape + column names) and feeds the
        session-card metadata. Sessions shared with the caller are appended
        after the owned page, stamped with the granted role.

        Args:
            user: Authenticated caller.
            limit: Page size for the owned rows, clamped to ``MAX_LIST``.
            offset: Number of owned rows to skip for paging.

        Returns:
            A ``TaggingSessionListResponse`` with the page of summaries and the
            caller's total session count (owned + shared-in).
        """
        with Session(job_store.engine) as session:
            owned = session.query(TaggingSessionModel).filter(
                TaggingSessionModel.username == user.username
            )
            total = owned.with_entities(func.count(TaggingSessionModel.id)).scalar() or 0
            rows = (
                owned.with_entities(
                    TaggingSessionModel.id,
                    TaggingSessionModel.name,
                    TaggingSessionModel.phase,
                    TaggingSessionModel.row_count,
                    TaggingSessionModel.tagged_count,
                    TaggingSessionModel.pinned,
                    TaggingSessionModel.created_at,
                    TaggingSessionModel.updated_at,
                    TaggingSessionModel.config,
                )
                .order_by(
                    TaggingSessionModel.pinned.desc(),
                    TaggingSessionModel.updated_at.desc(),
                )
                .offset(offset)
                .limit(limit)
                .all()
            )
            items = []
            for r in rows:
                mode, source_name = _summary_meta(r.config)
                items.append(
                    TaggingSessionSummary(
                        id=cast(str, r.id),
                        name=cast(str, r.name),
                        phase=cast(str, r.phase),
                        row_count=cast(int, r.row_count),
                        tagged_count=cast(int, r.tagged_count),
                        pinned=cast(bool, r.pinned),
                        created_at=cast(datetime, r.created_at),
                        updated_at=cast(datetime, r.updated_at),
                        mode=mode,
                        source_name=source_name,
                    )
                )
            shared_roles = list_grants_for_user_all(session, user.username)
            owned_ids = {item.id for item in items}
            shared_ids = [
                sid
                for sid, role in shared_roles.items()
                if role in MEMBER_ROLES and sid not in owned_ids
            ]
            total = int(total) + len(shared_ids)
            # Shared-in rows are appended only on the first page — repeating
            # them at every offset would duplicate cards while paging. Same
            # light column projection as the owned page (no heavy JSON).
            if shared_ids and offset == 0:
                shared_rows = (
                    session.query(TaggingSessionModel)
                    .filter(TaggingSessionModel.id.in_(shared_ids))
                    .with_entities(
                        TaggingSessionModel.id,
                        TaggingSessionModel.name,
                        TaggingSessionModel.phase,
                        TaggingSessionModel.row_count,
                        TaggingSessionModel.tagged_count,
                        TaggingSessionModel.pinned,
                        TaggingSessionModel.created_at,
                        TaggingSessionModel.updated_at,
                        TaggingSessionModel.config,
                    )
                    .order_by(
                        TaggingSessionModel.pinned.desc(),
                        TaggingSessionModel.updated_at.desc(),
                    )
                    .all()
                )
                for r in shared_rows:
                    mode, source_name = _summary_meta(r.config)
                    items.append(
                        TaggingSessionSummary(
                            id=cast(str, r.id),
                            name=cast(str, r.name),
                            phase=cast(str, r.phase),
                            row_count=cast(int, r.row_count),
                            tagged_count=cast(int, r.tagged_count),
                            pinned=cast(bool, r.pinned),
                            created_at=cast(datetime, r.created_at),
                            updated_at=cast(datetime, r.updated_at),
                            mode=mode,
                            source_name=source_name,
                            role=shared_roles[cast(str, r.id)],
                        )
                    )
            return TaggingSessionListResponse(items=items, total=total)

    @router.post(
        "/tagging-sessions",
        response_model=TaggingSessionDetail,
        status_code=201,
        summary="Create (save) a tagger session from the current annotation state",
    )
    def create_tagging_session(
        req: TaggingSessionCreateRequest, user: AuthenticatedUserDep
    ) -> TaggingSessionDetail:
        """Persist a new tagger session owned by the caller.

        Args:
            req: The full session payload captured when annotating begins.
            user: Authenticated caller; recorded as the session owner.

        Returns:
            The created session as ``TaggingSessionDetail`` (carries the new id).
        """
        now = datetime.now(UTC)
        row = TaggingSessionModel(
            id=str(uuid4()),
            username=user.username,
            name=req.name.strip()[:MAX_NAME],
            phase=req.phase,
            config=req.config,
            columns=req.columns,
            data=req.data,
            annotations=req.annotations,
            assist=req.assist,
            current_index=req.current_index,
            row_count=len(req.data),
            tagged_count=_count_tagged(req.annotations),
            pinned=False,
            created_at=now,
            updated_at=now,
        )
        with Session(job_store.engine) as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_detail(row)

    @router.get(
        "/tagging-sessions/{session_id}",
        response_model=TaggingSessionDetail,
        summary="Fetch one session with its full state (restore)",
    )
    def get_tagging_session(
        session_id: str, user: AuthenticatedUserDep
    ) -> TaggingSessionDetail:
        """Return a single accessible session with everything needed to resume.

        Args:
            session_id: UUID of the session to load.
            user: Authenticated caller; needs at least ``viewer`` access.

        Returns:
            The session as ``TaggingSessionDetail`` stamped with the caller's
            effective role.

        Raises:
            DomainError: 404 when unknown or inaccessible.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            role = require_role(session, session_id, user, ShareRole.viewer)
            return _row_to_detail(row, role=str(role))

    @router.put(
        "/tagging-sessions/{session_id}",
        response_model=TaggingSessionSummary,
        summary="Autosave annotation progress for a session",
    )
    def update_tagging_progress(
        session_id: str,
        req: TaggingSessionProgressRequest,
        user: AuthenticatedUserDep,
    ) -> TaggingSessionSummary:
        """Write the mutable annotation progress without touching the dataset.

        Only ``annotations``, ``current_index``, ``phase`` and the derived
        ``tagged_count`` are updated, so Postgres need not rewrite the large
        ``data`` TOAST on every save.

        Args:
            session_id: UUID of the session to update.
            req: The latest annotations, cursor position and optional phase.
            user: Authenticated caller; needs at least ``editor`` access.

        Returns:
            The updated row projected as ``TaggingSessionSummary``.

        Raises:
            DomainError: 404 when unknown/inaccessible, 403 for a viewer,
                409 while the bulk auto-tag job is writing the row.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            role = require_role(session, session_id, user, ShareRole.editor)
            # The bulk auto-tag job owns the row while it runs — a stale tab's
            # autosave would otherwise overwrite the labels it already wrote.
            autotag = (cast("dict[str, Any]", row.assist) or {}).get("autotag") or {}
            if autotag.get("status") == "running":
                raise DomainError("tagger.assist.autotag_running", status=409)
            row.annotations = cast(Any, req.annotations)
            if req.assist is not None:
                row.assist = cast(Any, req.assist)
            row.current_index = cast(Any, req.current_index)
            row.tagged_count = cast(Any, _count_tagged(req.annotations))
            if req.phase is not None:
                row.phase = cast(Any, req.phase)
            row.updated_at = cast(Any, datetime.now(UTC))
            session.commit()
            session.refresh(row)
            return _row_to_summary(row, role=str(role))

    @router.patch(
        "/tagging-sessions/{session_id}",
        response_model=TaggingSessionSummary,
        summary="Rename or pin/unpin a session",
    )
    def patch_tagging_session(
        session_id: str,
        req: TaggingSessionPatchRequest,
        user: AuthenticatedUserDep,
    ) -> TaggingSessionSummary:
        """Patch ``name`` and/or ``pinned`` on an owned session.

        Args:
            session_id: UUID of the session to patch.
            req: Partial update body; at least one field must be supplied.
            user: Authenticated caller; must be the owner.

        Returns:
            The updated row projected as ``TaggingSessionSummary``.

        Raises:
            DomainError: 422 when no field supplied, 404 when unknown or
                inaccessible, 403 for a non-owner member.
        """
        if req.name is None and req.pinned is None:
            raise DomainError("tagger.session.patch_requires_field", status=422)
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            require_role(session, session_id, user, ShareRole.owner)
            if req.name is not None:
                row.name = cast(Any, req.name.strip()[:MAX_NAME])
            if req.pinned is not None:
                row.pinned = cast(Any, req.pinned)
            row.updated_at = cast(Any, datetime.now(UTC))
            session.commit()
            session.refresh(row)
            return _row_to_summary(row)

    @router.delete(
        "/tagging-sessions/{session_id}",
        summary="Permanently delete a session",
    )
    def delete_tagging_session(
        session_id: str, user: AuthenticatedUserDep
    ) -> dict[str, Any]:
        """Delete an owned session row.

        Returns a small JSON body (rather than 204) so the browser client's
        ``res.json()`` path stays uniform with the other mutations.

        Args:
            session_id: UUID of the session to delete.
            user: Authenticated caller; must be the owner.

        Returns:
            ``{"id": session_id, "deleted": True}`` on success.

        Raises:
            DomainError: 404 when unknown or inaccessible, 403 for a non-owner
                member.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            require_role(session, session_id, user, ShareRole.owner)
            session.delete(row)
            session.commit()
        return {"id": session_id, "deleted": True}

    @router.post(
        "/tagging-sessions/bulk-delete",
        response_model=BulkDeleteByIdsResponse,
        summary="Delete many sessions in a single request",
    )
    def bulk_delete_tagging_sessions(
        body: BulkDeleteByIdsRequest, user: AuthenticatedUserDep
    ) -> BulkDeleteByIdsResponse:
        """Delete a batch of owned sessions and report per-id outcomes.

        Duplicate ids are deduplicated. An id the caller cannot delete —
        unknown or owned by someone else — lands in ``skipped`` (both as
        ``not_found``, so a foreign id is indistinguishable from a missing
        one) rather than failing the whole request, mirroring the dataset
        library's bulk delete.

        Args:
            body: The bulk-delete request body carrying the session ids.
            user: Authenticated caller; only sessions they own are deleted.

        Returns:
            A :class:`BulkDeleteByIdsResponse` listing deleted and skipped ids.
        """
        deleted: list[str] = []
        skipped: list[BulkDeleteByIdsSkipped] = []
        with Session(job_store.engine) as session:
            for session_id in dict.fromkeys(body.ids):
                row = session.get(TaggingSessionModel, session_id)
                if row is None or resolve_effective_role(session, session_id, user) != ShareRole.owner:
                    skipped.append(BulkDeleteByIdsSkipped(id=session_id, reason="not_found"))
                    continue
                session.delete(row)
                deleted.append(session_id)
            session.commit()
        return BulkDeleteByIdsResponse(deleted=deleted, skipped=skipped)

    return router
