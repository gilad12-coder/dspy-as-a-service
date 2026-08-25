"""Tests for the telemetry router (ingest + admin reads).

Mounts ``create_telemetry_router`` on an in-memory SQLite store (the sibling
routers' pattern: a ``RemoteDBJobStore`` subclass that skips the pgvector
bootstrap so ``Base.metadata.create_all`` stands up every table). Covers the
public ingest contract — anonymous acceptance, server-trusted attribution, the
batch/property caps, and the kill switch — plus the admin-only aggregates and
their authorization.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ...config import settings
from ...storage.models import Base, TelemetryEventModel
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..routers import telemetry as telemetry_module
from ..routers.telemetry import (
    MAX_EVENTS_PER_BATCH,
    MAX_PROPERTY_BYTES,
    _optional_user,
    create_telemetry_router,
)

_ADMIN = AuthenticatedUser(username="alice", role="admin", groups=())
_NONADMIN = AuthenticatedUser(username="bob", role="user", groups=())


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite store for telemetry-router tests (no pgvector)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


def _client(
    *,
    optional_user: AuthenticatedUser | None = None,
    admin: AuthenticatedUser | None = None,
) -> tuple[TestClient, _MemStore]:
    """Mount the telemetry router on a fresh store with optional auth overrides.

    Args:
        optional_user: When set, the ingest endpoint resolves to this caller
            (simulating an authenticated batch); otherwise ingest is anonymous.
        admin: When set, the admin read endpoints resolve to this caller.

    Returns:
        A ``(client, store)`` pair sharing one in-memory store.
    """
    store = _MemStore()
    app = FastAPI()
    app.include_router(create_telemetry_router(job_store=store))
    if optional_user is not None:
        app.dependency_overrides[_optional_user] = lambda: optional_user
    if admin is not None:
        app.dependency_overrides[get_authenticated_user] = lambda: admin
    return TestClient(app), store


def _batch(
    names: list[str],
    *,
    anonymous_id: str | None = "anon-1",
    session_id: str | None = "sess-1",
    **event_fields: object,
) -> dict[str, object]:
    """Build an ingest batch body whose events all share ``event_fields``.

    Args:
        names: One event name per event in the batch.
        anonymous_id: Batch-level opaque per-browser id.
        session_id: Batch-level per-visit id.
        **event_fields: Extra per-event fields (e.g. ``properties``, ``ts``).

    Returns:
        A JSON-serializable request body for ``POST /telemetry/events``.
    """
    return {
        "anonymous_id": anonymous_id,
        "session_id": session_id,
        "events": [{"name": name, **event_fields} for name in names],
    }


def test_anonymous_ingest_is_accepted_and_unattributed() -> None:
    """A batch with no auth is stored with ``username`` null and its anon id set."""
    client, store = _client()
    resp = client.post("/telemetry/events", json=_batch(["page_view", "run_submitted"]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": 2}
    with store._session_factory() as session:
        rows = session.query(TelemetryEventModel).order_by(TelemetryEventModel.id).all()
    assert [row.event_name for row in rows] == ["page_view", "run_submitted"]
    assert all(row.username is None for row in rows)
    assert all(row.anonymous_id == "anon-1" for row in rows)
    assert all(row.received_at is not None for row in rows)


def test_authenticated_ingest_attributes_username_server_side() -> None:
    """An authenticated batch is tagged with the token identity, not the body."""
    client, store = _client(optional_user=_NONADMIN)
    resp = client.post("/telemetry/events", json=_batch(["page_view"], anonymous_id="anon-x"))
    assert resp.status_code == 200, resp.text
    with store._session_factory() as session:
        row = session.query(TelemetryEventModel).one()
    assert row.username == "bob"
    assert row.anonymous_id == "anon-x"


def test_ingest_exports_accepted_batch_when_posthog_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured PostHog key schedules export after durable persistence."""
    exported: dict[str, object] = {}
    monkeypatch.setattr(settings, "posthog_project_api_key", SecretStr("phc_test"))

    def capture_export(**kwargs: object) -> None:
        """Capture the background export arguments."""
        exported.update(kwargs)

    monkeypatch.setattr(telemetry_module, "export_telemetry_events", capture_export)
    client, store = _client(optional_user=_NONADMIN)

    response = client.post(
        "/telemetry/events",
        json=_batch(["run_submitted"], properties={"react": True}),
    )

    assert response.status_code == 200
    assert exported["username"] == "bob"
    assert exported["anonymous_id"] == "anon-1"
    events = exported["events"]
    assert isinstance(events, list)
    assert events[0]["name"] == "run_submitted"
    with store._session_factory() as session:
        assert session.query(TelemetryEventModel).count() == 1


def test_ingest_records_client_timestamp() -> None:
    """The client epoch-millis ``ts`` is stored as ``occurred_at``."""
    client, store = _client()
    client.post("/telemetry/events", json=_batch(["page_view"], ts=1_700_000_000_000))
    with store._session_factory() as session:
        row = session.query(TelemetryEventModel).one()
    assert row.occurred_at is not None
    assert row.occurred_at.year == 2023


def test_ingest_rejects_oversized_batch() -> None:
    """More than the per-batch cap of events is a validation error."""
    client, _ = _client()
    body = _batch([f"e{i}" for i in range(MAX_EVENTS_PER_BATCH + 1)])
    assert client.post("/telemetry/events", json=body).status_code == 422


def test_ingest_rejects_empty_batch() -> None:
    """A batch with no events is a validation error."""
    client, _ = _client()
    assert client.post("/telemetry/events", json={"events": []}).status_code == 422


def test_ingest_clips_oversized_properties() -> None:
    """An oversized ``properties`` blob is replaced with a drop marker, not stored."""
    client, store = _client()
    huge = {"blob": "x" * (MAX_PROPERTY_BYTES + 100)}
    resp = client.post("/telemetry/events", json=_batch(["page_view"], properties=huge))
    assert resp.status_code == 200, resp.text
    with store._session_factory() as session:
        row = session.query(TelemetryEventModel).one()
    assert row.properties == {"_dropped": "oversize"}


def test_ingest_is_a_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the kill switch off, ingest accepts nothing and stores nothing."""
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    client, store = _client()
    resp = client.post("/telemetry/events", json=_batch(["page_view"]))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": 0}
    with store._session_factory() as session:
        assert session.query(TelemetryEventModel).count() == 0


def test_summary_aggregates_events_visitors_and_top_events() -> None:
    """The summary rolls up totals, distinct visitors, and the top events."""
    client, _ = _client(admin=_ADMIN)
    client.post("/telemetry/events", json=_batch(["page_view", "page_view"], anonymous_id="a1"))
    client.post("/telemetry/events", json=_batch(["page_view", "run_submitted"], anonymous_id="a2"))
    body = client.get("/telemetry/summary", params={"window_hours": 24}).json()
    assert body["total_events"] == 4
    assert body["visitors"] == 2
    assert body["users"] == 0
    counts = {event["name"]: event["count"] for event in body["top_events"]}
    assert counts == {"page_view": 3, "run_submitted": 1}
    assert body["top_events"][0]["name"] == "page_view"


def test_summary_window_excludes_events_older_than_cutoff() -> None:
    """A stale event outside the window is not counted (tz-aware cutoff on SQLite)."""
    client, store = _client(admin=_ADMIN)
    client.post("/telemetry/events", json=_batch(["page_view"], anonymous_id="a1"))
    with store._session_factory() as session:
        session.add(
            TelemetryEventModel(
                event_name="old_event",
                anonymous_id="a0",
                received_at=datetime.now(UTC) - timedelta(hours=48),
            )
        )
        session.commit()
    body = client.get("/telemetry/summary", params={"window_hours": 24}).json()
    assert body["total_events"] == 1
    assert [event["name"] for event in body["top_events"]] == ["page_view"]


def test_summary_forbidden_for_non_admin() -> None:
    """A non-admin caller cannot read the usage summary."""
    client, _ = _client(admin=_NONADMIN)
    assert client.get("/telemetry/summary").status_code == 403


def test_recent_is_newest_first_and_filters_by_name() -> None:
    """The recent feed returns newest-first and honors an exact name filter."""
    client, store = _client(admin=_ADMIN)
    with store._session_factory() as session:
        session.add(
            TelemetryEventModel(
                event_name="page_view",
                anonymous_id="a1",
                received_at=datetime.now(UTC) - timedelta(minutes=5),
            )
        )
        session.add(
            TelemetryEventModel(
                event_name="run_submitted",
                anonymous_id="a1",
                received_at=datetime.now(UTC),
            )
        )
        session.commit()
    events = client.get("/telemetry/events/recent").json()["events"]
    assert [event["name"] for event in events] == ["run_submitted", "page_view"]
    filtered = client.get("/telemetry/events/recent", params={"name": "page_view"}).json()
    assert [event["name"] for event in filtered["events"]] == ["page_view"]


def test_recent_forbidden_for_non_admin() -> None:
    """A non-admin caller cannot read the recent-events feed."""
    client, _ = _client(admin=_NONADMIN)
    assert client.get("/telemetry/events/recent").status_code == 403
