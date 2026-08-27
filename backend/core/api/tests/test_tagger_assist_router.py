"""Tests for the AI co-tagging router (interview / predict / autotag).

Mounts the assist router (plus the session router for setup) on an in-memory
SQLite store, mirroring ``test_tagging_sessions_router``. LLM entry points on
the engine module are monkeypatched so no network is touched, and the
background worker is a recording fake — the job loop itself is covered in
``core.worker.tests.test_tagging_job``. Covers the interview turn, prediction
with exclusion semantics, bulk-job submission mechanics (job row + overview +
payload + session mirror), status reconciliation against the job row, cancel,
the autosave 409 while a job runs, and the ownership guard.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...constants import OPTIMIZATION_TYPE_TAGGING
from ...service_gateway import tagging
from ...storage.models import Base, TaggingSessionModel
from ...storage.remote import RemoteDBJobStore
from ...worker.tagging_job import TaggingAutotagPayload, run_autotag_job
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..routers.tagger_assist import _effective_config, create_tagger_assist_router
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
    """In-memory SQLite job store for assist-router tests (no pgvector).

    Beyond the tagging-session tables this one also services the job-row
    methods (``create_job`` / ``set_payload_overview`` / status reads), so the
    instance attributes those methods touch are seeded here.
    """

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        self._code_version = "test"
        self._max_progress_events = 100
        self._max_log_entries = 100
        self._progress_counter_lock = threading.Lock()


class _FakeWorker:
    """Recording stand-in for the background worker."""

    def __init__(self) -> None:
        """Start with empty submission/cancellation logs."""
        self.submitted: list[tuple[str, TaggingAutotagPayload]] = []
        self.cancelled: list[str] = []

    def submit_job(self, job_id: str, payload: TaggingAutotagPayload) -> None:
        """Record the submission instead of enqueueing it."""
        self.submitted.append((job_id, payload))

    def cancel_job(self, job_id: str) -> bool:
        """Record the cancel request and report it was found locally."""
        self.cancelled.append(job_id)
        return True


def _client(
    user: AuthenticatedUser,
    store: _MemStore | None = None,
    worker: _FakeWorker | None = None,
    *,
    worker_available: bool = True,
) -> tuple[TestClient, _MemStore]:
    """Mount the session + assist routers on a shared store, authed as ``user``.

    Args:
        user: Identity the auth dependency resolves to for every request.
        store: Existing store to reuse (so two users can share one DB).
        worker: Fake worker the assist router submits bulk jobs to.
        worker_available: Whether the API process has an in-process worker.

    Returns:
        A ``(client, store)`` pair sharing one in-memory store.
    """
    store = store or _MemStore()
    worker = (worker or _FakeWorker()) if worker_available else None
    app = FastAPI()
    app.include_router(create_tagging_session_router(job_store=store))
    app.include_router(
        create_tagger_assist_router(job_store=store, get_worker_ref=lambda: worker)
    )
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


def test_create_roundtrips_assist_state() -> None:
    """The assist JSON persists on create and comes back on detail GET."""
    client, _ = _client(_ALICE)
    session_id = _create(client)
    detail = client.get(f"/tagging-sessions/{session_id}").json()
    assert detail["assist"]["mode"] == "copilot"
    assert detail["assist"]["rubric"] == ["Sarcasm counts as negative."]
    assert detail["phase"] == "interview"


def test_effective_config_applies_inferred_mode() -> None:
    """The interview's inferred answer style overrides a provisional config."""
    client, store = _client(_ALICE)
    body = dict(_SESSION_BODY)
    body["config"] = {"mode": "freetext", "modeProvisional": True, "inputColumns": ["text"]}
    body["assist"] = {
        **_SESSION_BODY["assist"],
        "taskOverride": {
            "mode": "multiclass",
            "categories": [
                {"id": "cat1", "label": "Billing"},
                {"id": "cat2", "label": "Support"},
            ],
        },
    }
    resp = client.post("/tagging-sessions", json=body)
    assert resp.status_code == 201, resp.text
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, resp.json()["id"])
        config = _effective_config(row)
    assert config["mode"] == "multiclass"
    assert "modeProvisional" not in config
    assert [c["label"] for c in config["categories"]] == ["Billing", "Support"]


def test_interview_returns_turn(monkeypatch) -> None:
    """The interview route forwards the transcript and returns the turn."""
    seen: dict = {}

    def fake_turn(
        config, columns, data, turns, locale, model=None, reasoning_effort=None, lm_extra_body=None, usage_sink=None
    ):
        """Capture the forwarded arguments and return a canned turn."""
        seen.update(
            {
                "turns": turns,
                "locale": locale,
                "rows": len(data),
                "config": config,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "lm_extra_body": lm_extra_body,
            }
        )
        return {
            "message": "Does sarcasm count?",
            "options": [
                {"label": "Yes", "description": "Treat sarcasm as positive."},
                {"label": "No", "description": "Judge on literal wording."},
            ],
            "rubric": [],
            "done": False,
        }

    monkeypatch.setattr(tagging, "interview_turn", fake_turn)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/interview",
        json={
            "turns": [{"role": "user", "content": "hi"}],
            "locale": "he",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["message"] == "Does sarcasm count?"
    assert resp.json()["done"] is False
    assert resp.json()["options"] == [
        {"label": "Yes", "description": "Treat sarcasm as positive."},
        {"label": "No", "description": "Judge on literal wording."},
    ]
    assert seen["locale"] == "he"
    assert seen["rows"] == 4
    assert seen["turns"] == [{"role": "user", "content": "hi"}]
    assert seen["config"]["_assist_mode"] == "copilot"
    assert seen["model"] is None
    assert seen["reasoning_effort"] is None
    assert seen["lm_extra_body"] is None


def test_interview_ignores_client_model(monkeypatch) -> None:
    """A stale client's model/effort fields never reach the engine."""
    seen: dict = {}

    def fake_turn(
        config, columns, data, turns, locale, model=None, reasoning_effort=None, lm_extra_body=None, usage_sink=None
    ):
        """Capture the forwarded model kwargs and return a canned turn."""
        seen.update({"model": model, "reasoning_effort": reasoning_effort})
        return {"message": "hi", "options": [], "rubric": [], "done": False}

    monkeypatch.setattr(tagging, "interview_turn", fake_turn)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/interview",
        json={"turns": [], "model": "openai/not-a-model", "reasoning_effort": "high"},
    )
    assert resp.status_code == 200, resp.text
    assert seen["model"] is None
    assert seen["reasoning_effort"] is None
    assert "model" not in resp.json()


def test_interview_stream_ignores_client_model(monkeypatch) -> None:
    """The SSE interview route runs the operator default, whatever the client sends."""
    seen: dict = {}

    async def fake_stream(
        config, columns, data, turns, locale, model=None, reasoning_effort=None, lm_extra_body=None, usage_sink=None
    ):
        """Capture the forwarded kwargs and finish immediately."""
        seen.update(
            {
                "turns": turns,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "lm_extra_body": lm_extra_body,
            }
        )
        yield {"event": "interview_done", "data": {"message": "hi", "done": False}}

    monkeypatch.setattr(tagging, "interview_turn_stream", fake_stream)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/interview/stream",
        json={"turns": [], "model": "openai/gpt-test", "reasoning_effort": "low"},
    )
    assert resp.status_code == 200
    assert "event: interview_done" in resp.text
    assert seen["model"] is None
    assert seen["reasoning_effort"] is None
    assert seen["lm_extra_body"] is None


def test_interview_stream_tolerates_turn_bookkeeping_fields(monkeypatch) -> None:
    """Echoed turns with extra (even null) bookkeeping fields still validate."""
    seen: dict = {}

    async def fake_stream(
        config, columns, data, turns, locale, model=None, reasoning_effort=None, lm_extra_body=None, usage_sink=None
    ):
        """Capture the forwarded turns and finish immediately."""
        seen["turns"] = turns
        yield {"event": "interview_done", "data": {"message": "hi", "done": False}}

    monkeypatch.setattr(tagging, "interview_turn_stream", fake_stream)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/interview/stream",
        json={
            "turns": [
                {
                    "role": "assistant",
                    "content": "What counts as positive?",
                    "model": "openai/gpt-test",
                    "servedModel": None,
                },
                {"role": "user", "content": "Praise of the product."},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert seen["turns"] == [
        {"role": "assistant", "content": "What counts as positive?"},
        {"role": "user", "content": "Praise of the product."},
    ]


def test_predict_excludes_requested_rows_from_examples(monkeypatch) -> None:
    """Predictions come back per row; requested ids never leak into examples."""
    captured: dict = {}

    def fake_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
    ):
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
    assert body["total_tokens"] == 2
    assert set(body["predictions"]) == {"2", "3"}
    # Row 1 is the only labeled row, so it is the only candidate example; the
    # requested rows must not appear as examples even if labeled.
    assert "great" in captured["instructions"]
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/predict", json={"row_ids": ["999"]}
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "tagger.assist.rows_not_found"


def test_predict_stream_emits_rows_then_done(monkeypatch) -> None:
    """The SSE predict route relays per-row events, then the terminal summary."""
    captured: dict = {}

    async def fake_stream(config, instructions, rows, usage_sink=None):
        """Yield one event per requested row, then the merged summary."""
        captured["instructions"] = instructions
        captured["ids"] = [str(r["id"]) for r in rows]
        merged = {}
        for r in rows:
            prediction = {"value": "no", "confidence": 0.8, "reason": "test"}
            merged[str(r["id"])] = prediction
            yield {"event": "prediction", "data": {"id": str(r["id"]), "prediction": prediction}}
        yield {"event": "predict_done", "data": {"predictions": merged, "credits": 2}}

    monkeypatch.setattr(tagging, "predict_rows_stream", fake_stream)
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/predict/stream", json={"row_ids": ["2", "3"]}
    )
    assert resp.status_code == 200, resp.text
    assert captured["ids"] == ["2", "3"]
    # Same exclusion semantics as the non-streaming route: row 1 is the only
    # labeled candidate, and the requested rows never appear as examples.
    assert "great" in captured["instructions"]
    assert resp.text.count("event: prediction") == 2
    assert resp.text.index("event: predict_done") > resp.text.rindex("event: prediction")
    resp = client.post(
        f"/tagging-sessions/{session_id}/assist/predict/stream", json={"row_ids": ["999"]}
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "tagger.assist.rows_not_found"


def test_estimate_counts_untagged_rows() -> None:
    """The estimate covers exactly the rows without a final label."""
    client, _ = _client(_ALICE)
    session_id = _create(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/estimate")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows"] == 3
    assert body["estimated_input_tokens"] >= 0
    assert body["estimated_output_tokens"] >= 0


def _create_with_client_model(client: TestClient) -> str:
    """Create a session whose assist state smuggles a model choice and BYOK params."""
    body = dict(_SESSION_BODY)
    body["assist"] = {
        **_SESSION_BODY["assist"],
        "model": "smuggled/frontier-xl",
        "modelParams": {
            "temperature": 0.1,
            "token_source": "byok",
            "byok_provider": "openrouter",
        },
    }
    resp = client.post("/tagging-sessions", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_estimate_never_echoes_a_model() -> None:
    """The estimate carries no model id: the tagging model is operator config."""
    client, _ = _client(_ALICE)
    session_id = _create_with_client_model(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/estimate")
    assert resp.status_code == 200, resp.text
    assert "model" not in resp.json()


def test_client_model_on_assist_never_reaches_the_engine(monkeypatch) -> None:
    """A model written onto ``assist`` is ignored on every spend route."""
    captured: dict = {}

    def fake_predict(config, instructions, rows, on_batch=None, cancel=None, usage_sink=None):
        """Record the effective config instead of calling an LM."""
        captured["config"] = config
        return (
            {str(r["id"]): {"value": "no", "confidence": 0.8, "reason": "test"} for r in rows},
            1,
        )

    monkeypatch.setattr(tagging, "predict_rows", fake_predict)
    worker = _FakeWorker()
    client, store = _client(_ALICE, worker=worker)
    session_id = _create_with_client_model(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/predict", json={"row_ids": ["2"]})
    assert resp.status_code == 200, resp.text
    assert "model" not in captured["config"]
    assert "modelParams" not in captured["config"]
    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 202, resp.text
    assert len(worker.submitted) == 1
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, session_id)
        assert "model" not in _effective_config(row)


def test_autotag_start_submits_worker_job(monkeypatch) -> None:
    """Start creates the job row, submits to the worker, and mirrors state."""
    worker = _FakeWorker()
    client, store = _client(_ALICE, worker=worker)
    session_id = _create(client)
    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 202, resp.text
    assert resp.json()["total"] == 3

    (job_id, payload) = worker.submitted[0]
    assert payload.session_id == session_id
    assert payload.username == "alice"
    job = store.get_job(job_id)
    assert job["status"] == "pending"
    assert job["optimization_type"] == OPTIMIZATION_TYPE_TAGGING
    assert job["username"] == "alice"

    detail = client.get(f"/tagging-sessions/{session_id}").json()
    assert detail["phase"] == "autotagging"
    assert detail["assist"]["autotag"]["status"] == "running"
    assert detail["assist"]["autotag"]["job_id"] == job_id
    status = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
    assert status["status"] == "running"
    assert status["live"] is True

    # A second start while the job row is still active is rejected.
    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 409
    assert resp.json()["code"] == "tagger.assist.autotag_running"

    # Run the job body the worker would execute; the session row completes.
    def fake_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
    ):
        """Emit one canned batch through on_batch, like the real engine."""
        batch = {
            str(r["id"]): {"value": "no", "confidence": 0.4, "reason": "test"} for r in rows
        }
        if on_batch is not None:
            on_batch(batch)
        return batch, 5

    monkeypatch.setattr(tagging, "predict_rows", fake_predict)
    outcome = run_autotag_job(
        store, job_id, session_id, cancel_event=threading.Event(), heartbeat=lambda: None
    )
    assert outcome == {"status": "done", "rows_tagged": 3, "total_tokens": 5}

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


def test_autotag_start_persists_payload_without_local_worker() -> None:
    """Queue a durable bulk job when the API and worker processes are split."""
    client, store = _client(_ALICE, worker_available=False)
    session_id = _create(client)

    resp = client.post(f"/tagging-sessions/{session_id}/assist/autotag")

    assert resp.status_code == 202, resp.text
    detail = client.get(f"/tagging-sessions/{session_id}").json()
    job_id = detail["assist"]["autotag"]["job_id"]
    job = store.get_job(job_id)
    assert job["status"] == "pending"
    assert job["payload"] == {"session_id": session_id, "username": "alice"}


def test_autotag_cancel_flips_job_row_and_reconciles_status() -> None:
    """Cancel signals the worker, CAS-cancels the job row, and status maps it."""
    worker = _FakeWorker()
    client, store = _client(_ALICE, worker=worker)
    session_id = _create(client)
    client.post(f"/tagging-sessions/{session_id}/assist/autotag")
    (job_id, _) = worker.submitted[0]

    resp = client.delete(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.json()["cancelled"] is True
    assert worker.cancelled == [job_id]
    assert store.get_job_status_fields(job_id)["status"] == "cancelled"

    # The session mirror still says running; the status route reconciles.
    status = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
    assert status["status"] == "canceled"
    assert status["live"] is False


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


def test_stale_running_mirror_reports_not_live_or_failed() -> None:
    """A 'running' mirror without an active job reconciles honestly."""
    client, store = _client(_ALICE)
    session_id = _create(client)
    # No job_id at all (e.g. legacy in-process session): running but dead.
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, session_id)
        row.assist = {**row.assist, "autotag": {"status": "running", "total": 3, "done": 1}}
        db.commit()
    body = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
    assert body["status"] == "running"
    assert body["live"] is False

    # A job row that failed (e.g. exhausted orphan-recovery attempts) surfaces.
    store.create_job("job-dead", username="alice")
    store.update_job("job-dead", status="failed")
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, session_id)
        row.assist = {
            **row.assist,
            "autotag": {"status": "running", "total": 3, "done": 1, "job_id": "job-dead"},
        }
        db.commit()
    body = client.get(f"/tagging-sessions/{session_id}/assist/autotag").json()
    assert body["status"] == "failed"
    assert body["live"] is False


def test_ownership_enforced_on_assist_routes() -> None:
    """A user with no grant gets 404 on every assist surface (no leak)."""
    alice_client, store = _client(_ALICE)
    session_id = _create(alice_client)
    bob_client, _ = _client(_BOB, store=store)
    resp = bob_client.post(
        f"/tagging-sessions/{session_id}/assist/interview", json={"turns": []}
    )
    assert resp.status_code == 404
    resp = bob_client.post(f"/tagging-sessions/{session_id}/assist/estimate")
    assert resp.status_code == 404
    resp = bob_client.get(f"/tagging-sessions/{session_id}/assist/autotag")
    assert resp.status_code == 404
