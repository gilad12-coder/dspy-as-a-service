"""Tests for server-emitted telemetry milestones."""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...config import settings
from ...storage.models import Base, TelemetryEventModel
from .. import server_events
from ..server_events import record_server_event


def _engine():
    """Build an in-memory SQLite engine with the telemetry table created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def test_record_persists_row_with_last_known_browser_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The milestone lands in the table, attributed to the user's latest browser id."""
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "posthog_project_api_key", None)
    engine = _engine()
    with Session(engine) as session:
        session.add(TelemetryEventModel(event_name="page_view", username="alice", anonymous_id="old-browser"))
        session.add(TelemetryEventModel(event_name="page_view", username="alice", anonymous_id="new-browser"))
        session.add(TelemetryEventModel(event_name="page_view", username="bob", anonymous_id="bobs-browser"))
        session.commit()

    record_server_event(
        engine,
        username="alice",
        name="run_completed",
        properties={"status": "success", "optimizer": "gepa"},
    )

    with Session(engine) as session:
        row = session.query(TelemetryEventModel).filter_by(event_name="run_completed").one()
    assert row.username == "alice"
    assert row.anonymous_id == "new-browser"
    assert row.properties == {"status": "success", "optimizer": "gepa"}
    assert row.context == {"source": "server"}
    assert row.occurred_at is not None
    assert row.received_at is not None


def test_record_exports_to_posthog_off_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """With PostHog configured the milestone is forwarded with the browser id."""
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "posthog_project_api_key", "phc_test")
    engine = _engine()
    done = threading.Event()
    captured: dict[str, object] = {}

    def fake_export(**kwargs: object) -> None:
        """Capture the export call and release the waiting test."""
        captured.update(kwargs)
        done.set()

    monkeypatch.setattr(server_events, "export_telemetry_events", fake_export)

    record_server_event(engine, username="alice", name="purchase_completed", properties={"credits": 500})

    assert done.wait(timeout=5)
    assert captured["username"] == "alice"
    assert captured["anonymous_id"] is None
    events = captured["events"]
    assert isinstance(events, list)
    assert events[0]["name"] == "purchase_completed"
    assert events[0]["properties"] == {"credits": 500}
    assert events[0]["context"] == {"source": "server"}


def test_record_is_noop_when_disabled_or_engineless(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill switch and a store without an engine both skip the write."""
    monkeypatch.setattr(settings, "posthog_project_api_key", None)
    engine = _engine()

    monkeypatch.setattr(settings, "telemetry_enabled", False)
    record_server_event(engine, username="alice", name="run_failed")
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    record_server_event(None, username="alice", name="run_failed")

    with Session(engine) as session:
        assert session.query(TelemetryEventModel).count() == 0


def test_record_swallows_storage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken table never raises into the webhook or worker caller."""
    monkeypatch.setattr(settings, "telemetry_enabled", True)
    monkeypatch.setattr(settings, "posthog_project_api_key", None)
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)

    record_server_event(engine, username="alice", name="run_failed")
