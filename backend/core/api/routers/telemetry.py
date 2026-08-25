"""First-party product-telemetry ingestion and admin read endpoints.

The browser SDK batches interaction events (page views, labelled clicks, named
flow events) to ``POST /telemetry/events`` — a public endpoint that accepts both
authenticated and anonymous callers, so pre-login activity is captured too. The
caller's identity is taken from the request's auth token, never the request
body: an anonymous batch lands with ``username = None`` and is attributed only
by the SDK's opaque ``anonymous_id``. The admin-only read endpoints aggregate
the table into the "how is Skynet actually used" figures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...config import settings
from ...storage.models import TelemetryEventModel
from ..auth import AuthenticatedUser, get_authenticated_user, require_admin_user
from ..errors import DomainError
from ..posthog import export_telemetry_events

# Hard caps the public ingest endpoint enforces so a single request can't be
# used to bulk-load the table or smuggle large blobs in through ``properties``.
MAX_EVENTS_PER_BATCH = 50
MAX_PROPERTY_BYTES = 4096
MAX_EVENT_NAME = 80

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def _optional_user(request: Request, authorization: str | None = Header(default=None)) -> AuthenticatedUser | None:
    """Resolve the caller from the auth header, or ``None`` when absent/invalid.

    The ingest endpoint is public — logged-out and pre-login events are part of
    the funnel — so a missing or expired token is not an error here: it just
    means the batch is attributed anonymously. Mirrors
    :func:`core.api.auth.get_authenticated_user` but swallows its 401.

    Args:
        request: Incoming request (reached through to the PAT lookup path).
        authorization: HTTP Authorization header, when the SDK sent a token.

    Returns:
        The authenticated user, or ``None`` for an anonymous batch.
    """
    if not authorization:
        return None
    try:
        return get_authenticated_user(request, authorization)
    except DomainError:
        return None


OptionalUserDep = Annotated[AuthenticatedUser | None, Depends(_optional_user)]


class TelemetryEventIn(BaseModel):
    """One interaction event as sent by the browser SDK.

    ``name`` is the event identifier (``page_view``, ``run_submitted``, …) and
    ``ts`` the client-side epoch-millis time (best-effort, clock-skewed).
    ``properties`` carries structural, PII-free descriptors and ``context`` the
    page/referrer/viewport envelope. Identity fields are deliberately absent —
    the server sets ``username`` from the auth token, not from this body.
    """

    name: str = Field(min_length=1, max_length=MAX_EVENT_NAME)
    ts: int | None = Field(default=None, description="Client event time, epoch millis")
    path: str | None = Field(default=None, max_length=512)
    locale: str | None = Field(default=None, max_length=16)
    app_version: str | None = Field(default=None, max_length=40)
    properties: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class TelemetryBatchIn(BaseModel):
    """A batch of events sharing one browser session.

    ``anonymous_id`` is the SDK's stable per-browser id and ``session_id`` scopes
    a single visit; both are opaque and carry no personal data. The event count
    is capped so the public endpoint can't be used to bulk-load the table.
    """

    anonymous_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    events: list[TelemetryEventIn] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)


class TelemetryIngestResponse(BaseModel):
    """Outcome of an ingest call — the number of events persisted."""

    accepted: int


class TelemetryTopEvent(BaseModel):
    """One row of the top-events leaderboard: an event name and its count."""

    name: str
    count: int


class TelemetrySummaryResponse(BaseModel):
    """Windowed usage rollup for the admin console.

    Counts cover the trailing ``window_hours``. ``users`` counts distinct
    authenticated identities; ``visitors`` counts distinct ``anonymous_id``s
    (logged-out included). ``top_events`` is the most frequent event names.
    """

    window_hours: int
    total_events: int
    users: int
    visitors: int
    top_events: list[TelemetryTopEvent]


class TelemetryRecentEvent(BaseModel):
    """One stored event row for the admin recent-activity feed."""

    name: str
    received_at: datetime
    occurred_at: datetime | None
    username: str | None
    anonymous_id: str | None
    session_id: str | None
    path: str | None
    locale: str | None
    app_version: str | None
    properties: dict[str, Any]


class TelemetryRecentResponse(BaseModel):
    """Envelope for the admin recent-activity feed, newest first."""

    events: list[TelemetryRecentEvent]


def _occurred_at(ts_millis: int | None) -> datetime | None:
    """Convert a client epoch-millis timestamp to an aware UTC datetime.

    Args:
        ts_millis: Client-reported event time in epoch milliseconds, or ``None``.

    Returns:
        A UTC ``datetime``, or ``None`` when no/invalid timestamp was sent. An
        out-of-range value is dropped rather than raised — a bad client clock
        must not fail the whole batch.
    """
    if ts_millis is None:
        return None
    try:
        return datetime.fromtimestamp(ts_millis / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _clip(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` if it serializes under the byte cap, else a marker.

    Oversized property/context blobs are replaced wholesale (not truncated mid
    structure) so an abusive or buggy client can't bloat a row, while a normal
    event is stored verbatim.

    Args:
        payload: The event ``properties`` or ``context`` mapping.

    Returns:
        The original mapping, or a small ``{"_dropped": …}`` marker when it
        exceeds :data:`MAX_PROPERTY_BYTES` serialized or cannot be serialized.
    """
    try:
        size = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return {"_dropped": "unserializable"}
    if size > MAX_PROPERTY_BYTES:
        return {"_dropped": "oversize"}
    return payload


def create_telemetry_router(*, job_store) -> APIRouter:
    """Build the telemetry ingest + admin-read router.

    Args:
        job_store: Storage backend exposing a SQLAlchemy ``engine`` the events
            table is written to and read from.

    Returns:
        A configured :class:`APIRouter` with the public ingest endpoint and the
        admin-only read endpoints.
    """
    router = APIRouter()

    @router.post(
        "/telemetry/events",
        response_model=TelemetryIngestResponse,
        summary="Ingest a batch of product-telemetry events (public, best-effort)",
    )
    def ingest_events(
        batch: TelemetryBatchIn,
        background_tasks: BackgroundTasks,
        current_user: OptionalUserDep,
    ) -> TelemetryIngestResponse:
        """Persist a batch of interaction events, attributing the caller server-side.

        Public so logged-out and pre-login events are captured; when a valid
        token is present the rows are tagged with that identity, otherwise they
        are anonymous. When ``settings.telemetry_enabled`` is off the call is a
        silent no-op (nothing stored) so ingestion can be killed without a
        client change.

        Args:
            batch: The session-scoped batch of events from the browser SDK.
            background_tasks: Post-response task queue for optional PostHog export.
            current_user: The token-resolved caller, or ``None`` when anonymous.

        Returns:
            A :class:`TelemetryIngestResponse` with the number of rows persisted
            (``0`` when telemetry is disabled).
        """
        if not settings.telemetry_enabled:
            return TelemetryIngestResponse(accepted=0)
        username = current_user.username if current_user else None
        received_at = datetime.now(UTC)
        rows = [
            TelemetryEventModel(
                event_name=event.name,
                occurred_at=_occurred_at(event.ts),
                received_at=received_at,
                username=username,
                anonymous_id=batch.anonymous_id,
                session_id=batch.session_id,
                path=event.path,
                locale=event.locale,
                app_version=event.app_version,
                properties=_clip(event.properties),
                context=_clip(event.context),
            )
            for event in batch.events
        ]
        with Session(job_store.engine) as session:
            session.add_all(rows)
            session.commit()
        if settings.posthog_project_api_key is not None:
            export_events = [
                {
                    "name": event.name,
                    "timestamp": (_occurred_at(event.ts) or received_at).isoformat(),
                    "path": event.path,
                    "locale": event.locale,
                    "app_version": event.app_version,
                    "properties": _clip(event.properties),
                    "context": _clip(event.context),
                }
                for event in batch.events
            ]
            background_tasks.add_task(
                export_telemetry_events,
                username=username,
                anonymous_id=batch.anonymous_id,
                session_id=batch.session_id,
                events=export_events,
            )
        return TelemetryIngestResponse(accepted=len(rows))

    @router.get(
        "/telemetry/summary",
        response_model=TelemetrySummaryResponse,
        summary="Admin: windowed usage rollup (events, users, visitors, top events)",
    )
    def get_summary(
        current_user: AuthenticatedUserDep,
        window_hours: Annotated[int, Query(ge=1, le=720)] = 24,
        top: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> TelemetrySummaryResponse:
        """Aggregate the trailing window into the admin usage figures.

        Args:
            current_user: Authenticated caller; must be a backend admin.
            window_hours: Trailing window to roll up, clamped to ``1..720`` (30d).
            top: Number of leaderboard rows to return, clamped to ``1..100``.

        Returns:
            A :class:`TelemetrySummaryResponse` over the window.

        Raises:
            DomainError: 403 when the caller is not a backend admin.
        """
        require_admin_user(current_user)
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        with Session(job_store.engine) as session:
            in_window = session.query(TelemetryEventModel).filter(TelemetryEventModel.received_at >= cutoff)
            total = in_window.with_entities(func.count(TelemetryEventModel.id)).scalar() or 0
            users = (
                in_window.with_entities(func.count(func.distinct(TelemetryEventModel.username)))
                .filter(TelemetryEventModel.username.isnot(None))
                .scalar()
                or 0
            )
            visitors = (
                in_window.with_entities(func.count(func.distinct(TelemetryEventModel.anonymous_id)))
                .filter(TelemetryEventModel.anonymous_id.isnot(None))
                .scalar()
                or 0
            )
            top_rows = (
                session.query(TelemetryEventModel.event_name, func.count(TelemetryEventModel.id))
                .filter(TelemetryEventModel.received_at >= cutoff)
                .group_by(TelemetryEventModel.event_name)
                .order_by(func.count(TelemetryEventModel.id).desc())
                .limit(top)
                .all()
            )
        return TelemetrySummaryResponse(
            window_hours=window_hours,
            total_events=int(total),
            users=int(users),
            visitors=int(visitors),
            top_events=[TelemetryTopEvent(name=name, count=int(count)) for name, count in top_rows],
        )

    @router.get(
        "/telemetry/events/recent",
        response_model=TelemetryRecentResponse,
        summary="Admin: most recent telemetry events, optionally filtered by name",
    )
    def get_recent(
        current_user: AuthenticatedUserDep,
        name: Annotated[str | None, Query(max_length=MAX_EVENT_NAME)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> TelemetryRecentResponse:
        """Return the most recent events for the admin activity feed.

        Args:
            current_user: Authenticated caller; must be a backend admin.
            name: Optional exact event-name filter.
            limit: Maximum rows to return, newest first, clamped to ``1..200``.

        Returns:
            A :class:`TelemetryRecentResponse` ordered newest-first.

        Raises:
            DomainError: 403 when the caller is not a backend admin.
        """
        require_admin_user(current_user)
        with Session(job_store.engine) as session:
            query = session.query(TelemetryEventModel)
            if name:
                query = query.filter(TelemetryEventModel.event_name == name)
            rows = (
                query.order_by(TelemetryEventModel.received_at.desc(), TelemetryEventModel.id.desc()).limit(limit).all()
            )
        return TelemetryRecentResponse(
            events=[
                TelemetryRecentEvent(
                    name=row.event_name,
                    received_at=row.received_at,
                    occurred_at=row.occurred_at,
                    username=row.username,
                    anonymous_id=row.anonymous_id,
                    session_id=row.session_id,
                    path=row.path,
                    locale=row.locale,
                    app_version=row.app_version,
                    properties=row.properties or {},
                )
                for row in rows
            ]
        )

    return router
