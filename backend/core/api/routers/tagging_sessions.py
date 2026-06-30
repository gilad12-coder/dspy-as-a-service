"""CRUD routes for persisted text-labeling (tagger) sessions. [INTERNAL]

Backs the tagger's save / restore so a user can resume annotating across
refreshes and devices, mirroring how optimizations persist. Each user owns
their own sessions; ownership is enforced on every read and write by comparing
the authenticated principal to the row's ``username``.

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

from ...storage.models import TaggingSessionModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError

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


class TaggingSessionDetail(BaseModel):
    """Full tagger session — the complete state needed to resume annotating."""

    id: str
    name: str
    phase: str
    config: dict[str, Any]
    columns: list[str]
    data: list[dict[str, Any]]
    annotations: dict[str, Any]
    current_index: int
    row_count: int
    tagged_count: int
    pinned: bool
    created_at: datetime
    updated_at: datetime


class TaggingSessionCreateRequest(BaseModel):
    """Create a session from the tagger's in-memory state when annotating begins."""

    name: str = Field(default="", max_length=MAX_NAME)
    phase: str = Field(default="annotating")
    config: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    data: list[dict[str, Any]] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)
    current_index: int = Field(default=0, ge=0)


class TaggingSessionProgressRequest(BaseModel):
    """Autosave the mutable annotation progress without re-uploading the dataset."""

    annotations: dict[str, Any] = Field(default_factory=dict)
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


def _row_to_summary(row: TaggingSessionModel) -> TaggingSessionSummary:
    """Project an ORM row into the lightweight list/summary response shape.

    Args:
        row: Loaded ``TaggingSessionModel``.

    Returns:
        The serializable summary the list / patch / put routes return.
    """
    return TaggingSessionSummary(
        id=cast(str, row.id),
        name=cast(str, row.name),
        phase=cast(str, row.phase),
        row_count=cast(int, row.row_count),
        tagged_count=cast(int, row.tagged_count),
        pinned=cast(bool, row.pinned),
        created_at=cast(datetime, row.created_at),
        updated_at=cast(datetime, row.updated_at),
    )


def _row_to_detail(row: TaggingSessionModel) -> TaggingSessionDetail:
    """Project an ORM row into the full restore-payload response shape.

    Args:
        row: Loaded ``TaggingSessionModel``.

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
        current_index=cast(int, row.current_index),
        row_count=cast(int, row.row_count),
        tagged_count=cast(int, row.tagged_count),
        pinned=cast(bool, row.pinned),
        created_at=cast(datetime, row.created_at),
        updated_at=cast(datetime, row.updated_at),
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
        summary="List the caller's saved tagger sessions",
    )
    def list_tagging_sessions(
        user: AuthenticatedUserDep,
        limit: int = Query(default=DEFAULT_LIST, ge=1, le=MAX_LIST),
        offset: int = Query(default=0, ge=0),
    ) -> TaggingSessionListResponse:
        """Return the caller's sessions, pinned first then newest activity.

        The heavy JSON columns (``data``/``annotations``/``config``) are not
        selected so the sidebar list stays cheap as sessions accumulate.

        Args:
            user: Authenticated caller; only their sessions are returned.
            limit: Page size, clamped to ``MAX_LIST``.
            offset: Number of rows to skip for paging.

        Returns:
            A ``TaggingSessionListResponse`` with the page of summaries and the
            caller's total session count.
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
                )
                .order_by(
                    TaggingSessionModel.pinned.desc(),
                    TaggingSessionModel.updated_at.desc(),
                )
                .offset(offset)
                .limit(limit)
                .all()
            )
            items = [
                TaggingSessionSummary(
                    id=cast(str, r.id),
                    name=cast(str, r.name),
                    phase=cast(str, r.phase),
                    row_count=cast(int, r.row_count),
                    tagged_count=cast(int, r.tagged_count),
                    pinned=cast(bool, r.pinned),
                    created_at=cast(datetime, r.created_at),
                    updated_at=cast(datetime, r.updated_at),
                )
                for r in rows
            ]
            return TaggingSessionListResponse(items=items, total=int(total))

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
        """Return a single owned session with everything needed to resume.

        Args:
            session_id: UUID of the session to load.
            user: Authenticated caller; must own the session.

        Returns:
            The session as ``TaggingSessionDetail``.

        Raises:
            DomainError: 404 when unknown, 403 when the caller does not own it.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            if row.username != user.username:
                raise DomainError("tagger.session.forbidden", status=403)
            return _row_to_detail(row)

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
            user: Authenticated caller; must own the session.

        Returns:
            The updated row projected as ``TaggingSessionSummary``.

        Raises:
            DomainError: 404 when unknown, 403 when the caller does not own it.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            if row.username != user.username:
                raise DomainError("tagger.session.forbidden", status=403)
            row.annotations = cast(Any, req.annotations)
            row.current_index = cast(Any, req.current_index)
            row.tagged_count = cast(Any, _count_tagged(req.annotations))
            if req.phase is not None:
                row.phase = cast(Any, req.phase)
            row.updated_at = cast(Any, datetime.now(UTC))
            session.commit()
            session.refresh(row)
            return _row_to_summary(row)

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
            user: Authenticated caller; must own the row.

        Returns:
            The updated row projected as ``TaggingSessionSummary``.

        Raises:
            DomainError: 422 when no field supplied, 404 when unknown,
                403 when the caller does not own the row.
        """
        if req.name is None and req.pinned is None:
            raise DomainError("tagger.session.patch_requires_field", status=422)
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            if row.username != user.username:
                raise DomainError("tagger.session.forbidden", status=403)
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
            user: Authenticated caller; must own the row.

        Returns:
            ``{"id": session_id, "deleted": True}`` on success.

        Raises:
            DomainError: 404 when unknown, 403 when the caller does not own it.
        """
        with Session(job_store.engine) as session:
            row = session.get(TaggingSessionModel, session_id)
            if row is None:
                raise DomainError("tagger.session.not_found", status=404)
            if row.username != user.username:
                raise DomainError("tagger.session.forbidden", status=403)
            session.delete(row)
            session.commit()
        return {"id": session_id, "deleted": True}

    return router
