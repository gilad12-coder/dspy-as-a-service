"""Tests for the AI co-tagging router (interview / predict / autotag).

Mounts the assist router (plus the session router for setup) on an in-memory
SQLite store, mirroring ``test_tagging_sessions_router``. LLM entry points on
the engine module are monkeypatched so no network is touched. Covers the
interview turn, prediction with exclusion semantics, the bulk auto-tag job's
full lifecycle (progress persistence, provenance, phase flip, resume guard),
the autosave 409 while a job runs, and the ownership guard.
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...service_gateway import tagging
from ...storage.models import Base, TaggingSessionModel
from ...storage.remote import RemoteDBJobStore
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers.tagger_assist import create_tagger_assist_router
from ..routers.tagging_sessions import create_tagging_session_router

_ALICE = AuthenticatedUser(username="alice", role="user", groups=())
_BOB = AuthenticatedUser(username="bob", role="user", groups=())

_SESSION_BODY = {
    "name": "Sentiment pass",
    "phase": "interview",
    "config": {"mode": "binary", "inputColumns": ["text"], "question": "Positive?"},
    "columns": ["text"],
    "data": [
        {"id": 1, "text": "great"},
        {"id": 2, "text": "awful"},
        {"id": 3, "text": "meh"},
        {"id": 4, "text": "fine"},
    ],
    "annotations": {"1": "yes"},
    "assist": {
        "mode": "copilot",
        "calibrationStyle": "blind",
        "interview": {"turns": [], "done": False},
        "rubric": ["Sarcasm counts as negative."],
        "calibrationIds": ["1", "2"],
        "predictions": {},
        "provenance": {"1": "human"},
        "rounds": [],
    },
    "current_index": 0,
}


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store for assist-router tests (no pgvector)."""

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
    user: AuthenticatedUser, store: _MemStore | None = None
) -> tuple[TestClient, _MemStore]:
    """Mount the session + assist routers on a shared store, authed as ``user``.

    Args:
        user: Identity the auth dependency resolves to for every request.
        store: Existing store to reuse (so two users can share one DB).

    Returns:
        A ``(client, store)`` pair sharing one in-memory store.
    """
    store = store or _MemStore()
    app = FastAPI()
    app.include_router(create_tagging_session_router(job_store=store))
    app.include_router(create_tagger_assist_router(job_store=store))
    app.dependency_overrides[get_authenticated_user] = lambda: user

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request, exc: DomainError) -> JSONResponse:
        """Mirror the app-level envelope so tests can assert on ``code``."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code, "params": exc.params},
        )

    return TestClient(app), store


def _create(client: TestClient) -> str:
    """Create an assist session via the API and return its new id.

    Args:
        client: The mounted test client.

    Returns:
        The created session's id.
    """
    resp = client.post("/tagging-sessions", json=_SESSION_BODY)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _wait_terminal(client: TestClient, session_id: str, timeout: float = 5.0) -> dict:
    """Poll the autotag status route until the job leaves ``running``.

    Args:
        client: The mounted test client.
        session_id: The session whose job is polled.
        timeout: Seconds before the poll gives up.

    Returns:
        The final status payload.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("autotag job never reached a terminal status")


def test_create_roundtrips_assist_state() -> None:
    """The assist JSON persists on create and comes back on detail GET."""
    client, _ = _client(_ALICE)
    session_id = _create(client)
    detail = client.get(f"/tagging-sessions/{session_id}").json()
    assert detail["assist"]["mode"] == "copilot"
    assert detail["assist"]["rubric"] == ["Sarcasm counts as negative."]
    assert detail["phase"] == "interview"


def test_interview_returns_turn(monkeypatch) -> None:
    """The interview route forwards the transcript and returns the turn."""
    seen: dict = {}

    def fake_turn(config, columns, data, turns, locale):
        """Capture the forwarded arguments and return a canned turn."""
        seen.update({"turns": turns, "locale": locale, "rows": len(data)})
        return {
            "message": "Does sarcasm count?",
            "quick_replies": ["Yes", "No"],
            "rubric": [],
            "done": False,
        }

    monkeypatch.setattr(tagging, "interview_turn", fake_turn)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/interview",
        json={"turns": [{"role": "user", "content": "hi"}], "locale": "he"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["message"] == "Does sarcasm count?"
    assert resp.json()["done"] is False
    assert seen["locale"] == "he"
    assert seen["rows"] == 4
    assert seen["turns"] == [{"role": "user", "content": "hi"}]


def test_predict_excludes_requested_rows_from_examples(monkeypatch) -> None:
    """Predictions come back per row; requested ids never leak into examples."""
    captured: dict = {}

    def fake_predict(config, instructions, rows, on_batch=None, cancel=None):
        """Return a canned prediction for every requested row."""
        captured["instructions"] = instructions
        return (
            {str(r["id"]): {"value": "no", "confidence": 0.8, "reason": "test"} for r in rows},
            2,
        )

    monkeypatch.setattr(tagging, "predict_rows", fake_predict)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/predict", json={"row_ids": ["2", "3"]}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["credits"] == 2
    assert set(body["predictions"]) == {"2", "3"}
    # Row 1 is the only labeled row, so it is the only candidate example; the
    # requested rows must not appear as examples even if labeled.
    assert "great" in captured["instructions"]
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/predict", json={"row_ids": ["999"]}
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "tagger.assist.rows_not_found"


def test_optimize_requires_examples(monkeypatch) -> None:
    """Optimize 422s without labels and returns the improved rubric with them."""
    monkeypatch.setattr(
        tagging, "refine_rubric", lambda config, rubric, examples, locale: ["Better rule."]
    )
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/optimize", json={"locale": "en"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["rubric"] == ["Better rule."]

    empty_body = dict(_SESSION_BODY, annotations={})
    resp = client.post("/tagging-sessions", json=empty_body)
    bare_id = resp.json()["id"]
    resp = client.post(f"/tagging-sessions/{bare_id}/assist/optimize", json={})
    assert resp.status_code == 422
    assert resp.json()["code"] == "tagger.assist.no_examples"


def test_estimate_counts_untagged_rows() -> None:
    """The estimate covers exactly the rows without a final label."""
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/estimate")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows"] == 3
    assert body["credits_high"] >= body["credits_low"] >= 0


def test_autotag_lifecycle_tags_writes_provenance_and_flips_phase(monkeypatch) -> None:
    """The bulk job labels untagged rows, persists progress, and completes."""

    def fake_predict(config, instructions, rows, on_batch=None, cancel=None):
        """Emit one canned batch through on_batch, like the real engine."""
        batch = {
            str(r["id"]): {"value": "no", "confidence": 0.4, "reason": "test"} for r in rows
        }
        if on_batch is not None:
            on_batch(batch)
        return batch, 5

    monkeypatch.setattr(tagging, "predict_rows", fake_predict)
    client, _store = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 202, resp.text
    assert resp.json()["total"] == 3

    final = _wait_terminal(client, session_id)
    assert final["status"] == "done"
    assert final["done"] == 3
    assert final["credits_spent"] == 5
    assert final["live"] is False

    detail = client.get(f"/tagging-sessions/{session_id}").json()
    assert detail["phase"] == "complete"
    assert detail["tagged_count"] == 4
    assert detail["annotations"]["2"] == "no"
    assert detail["assist"]["provenance"]["2"] == "ai_auto"
    # The pre-existing human label was never overwritten.
    assert detail["annotations"]["1"] == "yes"
    assert detail["assist"]["provenance"]["1"] == "human"

    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 422
    assert resp.json()["code"] == "tagger.assist.nothing_to_tag"


def test_autosave_blocked_while_autotag_running() -> None:
    """A stale tab's autosave cannot clobber a running job's writes."""
    client, store = _client(_ALICE)
    session_id = _create(client)
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, session_id)
        row.assist = {**row.assist, "autotag": {"status": "running", "total": 3, "done": 1}}
        db.commit()
    resp = client.put(
        f"/tagging-sessions/{session_id}",
        json={"annotations": {}, "current_index": 0},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "tagger.assist.autotag_running"


def test_stale_running_job_reports_not_live() -> None:
    """A row claiming 'running' with no local thread reports live=false."""
    client, store = _client(_ALICE)
    session_id = _create(client)
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, session_id)
        row.assist = {**row.assist, "autotag": {"status": "running", "total": 3, "done": 1}}
        db.commit()
    body = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
    assert body["status"] == "running"
    assert body["live"] is False


def test_ownership_enforced_on_assist_routes() -> None:
    """Another user gets 403 on every assist surface."""
    alice_client, store = _client(_ALICE)
    session_id = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)
    resp = bob_client.post(
        f"/tagging-sessions/{session_id}/assist/interview", json={"turns": []}
    )
    assert resp.status_code == 403
    resp = bob_client.post(f"/tagging-sessions/{session_id}/assist/estimate")
    assert resp.status_code == 403
    resp = bob_client.get(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 403
