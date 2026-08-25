"""Tests for the bulk auto-tag worker job (loop + engine dispatch).

Runs :func:`run_autotag_job` against an in-memory SQLite ``RemoteDBJobStore``
with the LLM predictor monkeypatched, covering the done / user-cancel /
lease-loss / failure paths and the session-row writes each one performs. The
final test drives the real ``BackgroundWorker._process_job`` dispatch to prove
a ``tagging_autotag`` job row runs in the worker thread and lands ``success``
with its result persisted.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ...constants import OPTIMIZATION_TYPE_TAGGING, PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE
from ...service_gateway import tagging
from ...storage.models import Base, BillingCustomerModel, TaggingSessionModel
from ...storage.remote import RemoteDBJobStore
from .. import tagging_job
from ..engine import BackgroundWorker
from ..tagging_job import run_autotag_job

_SESSION_ID = "sess-1"


class _MemStore(RemoteDBJobStore):
    """In-memory SQLite job store (no pgvector) with job-method attrs seeded."""

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


def _seed_session(store: _MemStore) -> None:
    """Insert a binary tagging session with one human label and three untagged rows.

    Args:
        store: The store whose engine backs the session table.
    """
    with Session(store.engine) as db:
        db.add(
            TaggingSessionModel(
                id=_SESSION_ID,
                username="alice",
                name="Sentiment pass",
                phase="autotagging",
                config={"mode": "binary", "inputColumns": ["text"], "question": "Positive?"},
                columns=["text"],
                data=[
                    {"id": 1, "text": "great"},
                    {"id": 2, "text": "awful"},
                    {"id": 3, "text": "meh"},
                    {"id": 4, "text": "fine"},
                ],
                annotations={"1": "yes"},
                assist={
                    "mode": "copilot",
                    "rubric": ["Sarcasm counts as negative."],
                    "predictions": {},
                    "provenance": {"1": "human"},
                    "autotag": {"status": "running", "total": 3, "done": 0, "credits_spent": 0},
                },
                row_count=4,
                tagged_count=1,
            )
        )
        db.commit()


def _fund(store: _MemStore, username: str = "alice", credits: int = 10_000) -> None:
    """Seed a funded billing row so a job run under ``username`` passes the credit gate.

    Args:
        store: The store whose engine backs the billing tables.
        username: Account to fund.
        credits: Paid balance to seed.
    """
    with Session(store.engine) as db:
        db.add(
            BillingCustomerModel(
                username=username,
                stripe_customer_id=f"cus_{username}",
                credit_balance=credits,
                grant_remaining=0,
            )
        )
        db.commit()


def _seed_job(store: _MemStore, job_id: str, status: str | None = None) -> None:
    """Create a tagging job row, optionally forcing its status.

    Args:
        store: The store to create the row in.
        job_id: The job row's id.
        status: When set, the status the row is updated to after creation.
    """
    store.create_job(job_id, username="alice")
    store.set_payload_overview(
        job_id,
        {PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_TAGGING, "username": "alice"},
    )
    if status is not None:
        store.update_job(job_id, status=status)


def _fake_predict_all(
    config,
    instructions,
    rows,
    on_batch=None,
    cancel=None,
    usage_sink=None,
    model_config=None,
):
    """Label every row 'no' through on_batch, like the real engine."""
    batch = {str(r["id"]): {"value": "no", "confidence": 0.9, "reason": "t"} for r in rows}
    if on_batch is not None and batch:
        on_batch(batch)
    return batch, 7


def _autotag_state(store: _MemStore) -> dict:
    """Read the session row's autotag mirror plus phase.

    Args:
        store: The store whose engine backs the session table.
    """
    with Session(store.engine) as db:
        row = db.get(TaggingSessionModel, _SESSION_ID)
        return {"phase": row.phase, **dict((row.assist or {}).get("autotag") or {})}


def test_run_autotag_job_done_path(monkeypatch) -> None:
    """A clean run labels everything, flips the phase, and reports credits."""
    monkeypatch.setattr(tagging, "predict_rows", _fake_predict_all)
    store = _MemStore()
    _seed_session(store)
    _seed_job(store, "job-1", status="running")

    beats: list[int] = []
    outcome = run_autotag_job(
        store,
        "job-1",
        _SESSION_ID,
        cancel_event=threading.Event(),
        heartbeat=lambda: beats.append(1),
    )
    assert outcome == {"status": "done", "rows_tagged": 3, "credits_spent": 7}
    state = _autotag_state(store)
    assert state["status"] == "done"
    assert state["done"] == 3
    assert state["credits_spent"] == 7
    assert state["phase"] == "complete"


def test_run_autotag_job_user_cancel(monkeypatch) -> None:
    """A cancelled job row stops the loop and marks the session canceled."""
    monkeypatch.setattr(tagging_job, "MONITOR_TICK_SECONDS", 0.01)

    def waiting_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
        model_config=None,
    ):
        """Block until the stop signal arrives, like a long real run."""
        assert cancel is not None
        cancel.wait(5.0)
        return {}, 2

    monkeypatch.setattr(tagging, "predict_rows", waiting_predict)
    store = _MemStore()
    _seed_session(store)
    # The cancel route already flipped the job row before the loop notices.
    _seed_job(store, "job-1", status="cancelled")

    outcome = run_autotag_job(
        store, "job-1", _SESSION_ID, cancel_event=threading.Event(), heartbeat=lambda: None
    )
    assert outcome == {"status": "cancelled"}
    state = _autotag_state(store)
    assert state["status"] == "canceled"
    assert state["phase"] == "autotagging"


def test_run_autotag_job_lease_loss_abandons_silently(monkeypatch) -> None:
    """A stolen lease stops the loop without writing a terminal session state."""
    monkeypatch.setattr(tagging_job, "MONITOR_TICK_SECONDS", 0.01)

    def waiting_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
        model_config=None,
    ):
        """Block until the stop signal arrives, like a long real run."""
        assert cancel is not None
        cancel.wait(5.0)
        return {}, 0

    monkeypatch.setattr(tagging, "predict_rows", waiting_predict)
    store = _MemStore()
    _seed_session(store)
    _seed_job(store, "job-1", status="running")

    # The worker's heartbeat sets the cancel event itself on lease loss; the
    # job row stays active because a peer pod now owns it.
    stolen = threading.Event()
    stolen.set()
    outcome = run_autotag_job(
        store, "job-1", _SESSION_ID, cancel_event=stolen, heartbeat=lambda: None
    )
    assert outcome == {"status": "aborted"}
    assert _autotag_state(store)["status"] == "running"


def test_run_autotag_job_failure_marks_session(monkeypatch) -> None:
    """A batch-loop failure marks the session failed and re-raises."""

    def broken_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
        model_config=None,
    ):
        """Blow up like an exhausted provider."""
        raise RuntimeError("provider down")

    monkeypatch.setattr(tagging, "predict_rows", broken_predict)
    store = _MemStore()
    _seed_session(store)
    _seed_job(store, "job-1", status="running")

    with pytest.raises(RuntimeError, match="provider down"):
        run_autotag_job(
            store, "job-1", _SESSION_ID, cancel_event=threading.Event(), heartbeat=lambda: None
        )
    assert _autotag_state(store)["status"] == "failed"


def test_run_autotag_job_depleted_account_stops_before_first_call(monkeypatch) -> None:
    """A job for a zero-balance account cancels up front without one LLM call."""

    def never_called(config, instructions, rows, on_batch=None, cancel=None, usage_sink=None):
        """Fail the test if the batch loop is ever reached."""
        raise AssertionError("predict_rows must not run for a depleted account")

    monkeypatch.setattr(tagging, "predict_rows", never_called)
    store = _MemStore()
    _seed_session(store)
    _seed_job(store, "job-1", status="running")

    outcome = run_autotag_job(
        store,
        "job-1",
        _SESSION_ID,
        username="alice",
        cancel_event=threading.Event(),
        heartbeat=lambda: None,
    )
    assert outcome == {"status": "cancelled", "reason": "credits_exhausted"}
    state = _autotag_state(store)
    assert state["status"] == "canceled"
    assert state["reason"] == "credits_exhausted"


def test_run_autotag_job_credit_watch_stops_mid_run(monkeypatch) -> None:
    """The monitor stops the loop once accrued cost reaches the balance."""
    monkeypatch.setattr(tagging_job, "MONITOR_TICK_SECONDS", 0.01)
    monkeypatch.setattr(
        tagging_job, "estimate_run_credits", lambda sink, token_source="managed": 10_000
    )

    def waiting_predict(
        config,
        instructions,
        rows,
        on_batch=None,
        cancel=None,
        usage_sink=None,
        model_config=None,
    ):
        """Block until the stop signal arrives, like a long real run."""
        assert cancel is not None
        cancel.wait(5.0)
        return {}, 3

    monkeypatch.setattr(tagging, "predict_rows", waiting_predict)
    store = _MemStore()
    _seed_session(store)
    _fund(store, credits=10)
    _seed_job(store, "job-1", status="running")

    outcome = run_autotag_job(
        store,
        "job-1",
        _SESSION_ID,
        username="alice",
        cancel_event=threading.Event(),
        heartbeat=lambda: None,
    )
    assert outcome == {"status": "cancelled", "reason": "credits_exhausted"}
    state = _autotag_state(store)
    assert state["status"] == "canceled"
    assert state["reason"] == "credits_exhausted"


def test_process_job_dispatches_tagging_type(monkeypatch) -> None:
    """The worker runs a tagging_autotag row in-thread and lands success."""
    monkeypatch.setattr(tagging, "predict_rows", _fake_predict_all)
    store = _MemStore()
    _seed_session(store)
    _fund(store)
    _seed_job(store, "job-1")
    store.update_job(
        "job-1", payload={"session_id": _SESSION_ID, "username": "alice"}
    )

    worker = BackgroundWorker(job_store=store, num_workers=1)
    worker._process_job("job-1", worker_id=0)

    job = store.get_job("job-1")
    assert job["status"] == "success"
    assert job["result"] == {"status": "done", "rows_tagged": 3, "credits_spent": 7}
    assert job["message"] == "Tagged 3 rows"
    assert _autotag_state(store)["phase"] == "complete"
