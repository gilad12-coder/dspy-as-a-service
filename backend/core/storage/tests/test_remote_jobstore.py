"""Unit tests for RemoteDBJobStore against an in-memory SQLite engine.

SQLite is used to avoid requiring a live Postgres instance for the fast CI gate.
``SQLiteJobStore`` bypasses ``RemoteDBJobStore.__init__`` so the fast suite can
avoid PostgreSQL-only engine options and pgvector bootstrap calls.

Two additional behaviours depend on a real Postgres instance and are explicitly skipped
here with a clear reason:
  - UTC timezone offset is NOT preserved by SQLite's ``DateTime`` column (it stores the
    wall-clock value as a naive string).  Tests that need timezone-aware comparisons are
    structured to compare the time *value* only.
  - Production filters ``optimization_type`` through a top-level indexed column;
    SQLite exercises the same ORM column but not PostgreSQL's planner behaviour.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import core.storage.remote as remote_mod
from core.storage.base import JobStore
from core.storage.models import Base, JobModel
from core.storage.remote import RemoteDBJobStore


class SQLiteJobStore(RemoteDBJobStore):
    """RemoteDBJobStore wired to an in-memory SQLite engine.

    Overrides only ``__init__`` to:
    - Use ``StaticPool`` (required for sqlite:///:memory: across multiple
      session factory calls to share one connection).
    - Skip PostgreSQL-only pgvector bootstrap calls.
    """

    def __init__(self, db_url: str = "sqlite:///:memory:") -> None:
        """Build an in-memory SQLite engine and create the ORM tables.

        Args:
            db_url: SQLAlchemy URL; defaults to in-memory SQLite.
        """
        self._engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        self._checkpoints = remote_mod.PostgresCheckpointBlobStore(self._engine)
        self._grid_pair_results = remote_mod.PostgresGridPairResultStore(self._engine)


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO timestamp that may use a trailing ``Z`` UTC marker."""
    return datetime.fromisoformat(value)


@pytest.fixture
def store() -> Iterator[SQLiteJobStore]:
    """Yield a fresh ``SQLiteJobStore`` and drop its tables afterwards."""
    s = SQLiteJobStore()
    yield s
    # Drop all tables so state doesn't leak between tests that share the same
    # in-memory engine handle (StaticPool reuses the connection).
    Base.metadata.drop_all(s.engine)


def test_create_job_returns_pending_status(store: SQLiteJobStore) -> None:
    """Create job returns pending status."""
    result = store.create_job("job-create-1")
    assert result["status"] == "pending"
    assert result["optimization_id"] == "job-create-1"


def test_create_job_with_estimate_stored(store: SQLiteJobStore) -> None:
    """Create job with estimate stored."""
    result = store.create_job("job-create-2", estimated_remaining_seconds=60.0)
    assert result["estimated_remaining_seconds"] == 60.0


def test_create_job_is_retrievable(store: SQLiteJobStore) -> None:
    """Create job is retrievable."""
    store.create_job("job-create-3")
    assert store.job_exists("job-create-3")


def test_create_job_created_at_is_iso_parseable(store: SQLiteJobStore) -> None:
    """Create job created at is iso parseable."""
    result = store.create_job("job-create-4")
    # Real Postgres TIMESTAMPTZ preserves the +00:00 offset; SQLite's DateTime
    # column stores naive datetimes, so we only assert the value round-trips.
    assert result["created_at"] is not None
    datetime.fromisoformat(result["created_at"])


def test_create_job_default_fields_are_empty(store: SQLiteJobStore) -> None:
    """Create job default fields are empty."""
    result = store.create_job("job-create-5")
    assert result["latest_metrics"] == {}
    assert result["payload_overview"] == {}
    assert result["result"] is None
    assert result["message"] is None
    assert result["started_at"] is None
    assert result["completed_at"] is None
    assert result["attempts"] == 0


def test_get_job_raises_key_error_for_unknown(store: SQLiteJobStore) -> None:
    """Get job raises key error for unknown."""
    with pytest.raises(KeyError, match="no-such-job"):
        store.get_job("no-such-job")


def test_get_job_returns_correct_record(store: SQLiteJobStore) -> None:
    """Get job returns correct record."""
    store.create_job("job-get-1")
    store.update_job("job-get-1", status="running")
    job = store.get_job("job-get-1")
    assert job["optimization_id"] == "job-get-1"
    assert job["status"] == "running"


def test_job_exists_true_after_create(store: SQLiteJobStore) -> None:
    """Job exists true after create."""
    store.create_job("job-exists-1")
    assert store.job_exists("job-exists-1") is True


def test_job_exists_false_for_missing(store: SQLiteJobStore) -> None:
    """Job exists false for missing."""
    assert store.job_exists("missing-id") is False


def test_update_job_status_field(store: SQLiteJobStore) -> None:
    """Update job status field."""
    store.create_job("job-upd-1")
    store.update_job("job-upd-1", status="running")
    assert store.get_job("job-upd-1")["status"] == "running"


def test_update_job_message_field(store: SQLiteJobStore) -> None:
    """Update job message field."""
    store.create_job("job-upd-2")
    store.update_job("job-upd-2", message="all good")
    assert store.get_job("job-upd-2")["message"] == "all good"


def test_update_job_multiple_fields_at_once(store: SQLiteJobStore) -> None:
    """Update job multiple fields at once."""
    store.create_job("job-upd-3")
    store.update_job("job-upd-3", status="success", message="done", estimated_remaining_seconds=0.0)
    job = store.get_job("job-upd-3")
    assert job["status"] == "success"
    assert job["message"] == "done"
    assert job["estimated_remaining_seconds"] == 0.0


def test_update_job_datetime_string_parsed_and_stored(store: SQLiteJobStore) -> None:
    """Update job datetime string parsed and stored."""
    store.create_job("job-upd-4")
    ts_str = "2026-04-14T10:00:00+00:00"
    store.update_job("job-upd-4", started_at=ts_str)
    job = store.get_job("job-upd-4")
    # The value round-trips through SQLite which loses timezone suffix,
    # but the wall-clock time must be preserved.
    started_at = job["started_at"]
    assert isinstance(started_at, str)
    stored = _parse_iso_datetime(started_at)
    expected = datetime(2026, 4, 14, 10, 0, 0)
    assert stored.replace(tzinfo=None) == expected


def test_update_job_latest_metrics_merges(store: SQLiteJobStore) -> None:
    """Update job latest metrics merges."""
    store.create_job("job-upd-5")
    store.update_job("job-upd-5", latest_metrics={"acc": 0.8})
    store.update_job("job-upd-5", latest_metrics={"loss": 0.2})
    metrics = store.get_job("job-upd-5")["latest_metrics"]
    # RemoteDBJobStore merges dicts rather than replacing
    assert metrics.get("acc") == 0.8
    assert metrics.get("loss") == 0.2


def test_update_job_json_result_roundtrips(store: SQLiteJobStore) -> None:
    """Update job json result roundtrips."""
    store.create_job("job-upd-6")
    payload: dict[str, Any] = {"weights": [1, 2, 3], "config": {"k": "v"}}
    store.update_job("job-upd-6", result=payload)
    assert store.get_job("job-upd-6")["result"] == payload


def test_delete_job_removes_the_job(store: SQLiteJobStore) -> None:
    """Delete job removes the job."""
    store.create_job("job-del-1")
    store.delete_job("job-del-1")
    assert not store.job_exists("job-del-1")


def test_delete_job_cascades_logs(store: SQLiteJobStore) -> None:
    """Delete job cascades logs."""
    store.create_job("job-del-2")
    store.append_log("job-del-2", level="INFO", logger_name="t", message="hi")
    store.delete_job("job-del-2")
    assert store.get_logs("job-del-2") == []


def test_delete_job_cascades_progress_events(store: SQLiteJobStore) -> None:
    """Delete job cascades progress events."""
    store.create_job("job-del-3")
    store.record_progress("job-del-3", "step", {"x": 1})
    store.delete_job("job-del-3")
    assert store.get_progress_events("job-del-3") == []


def test_delete_job_tolerates_nonexistent_id(store: SQLiteJobStore) -> None:
    """Delete job tolerates nonexistent id."""
    store.delete_job("ghost-id")  # must not raise


def test_requeue_for_rerun_resets_row_and_clears_children(store: SQLiteJobStore) -> None:
    """A from-scratch rerun flips the row to pending and wipes the prior attempt in place."""
    store.create_job("rr1")
    store.update_job("rr1", status="failed", result={"score": 0.9}, attempts=3)
    store.append_log("rr1", level="INFO", logger_name="t", message="boom")
    store.record_progress("rr1", "step", {"x": 1})
    store.save_gepa_checkpoint("rr1", b"GEPA-STATE", iteration=7)

    assert store.requeue_for_rerun("rr1") is True

    assert store.job_exists("rr1")
    job = store.get_job("rr1")
    assert job["status"] == "pending"
    assert job["result"] is None
    assert job["attempts"] == 0
    assert store.get_logs("rr1") == []
    assert store.get_progress_events("rr1") == []
    assert store.has_gepa_checkpoint("rr1") is False


def test_requeue_for_rerun_missing_id_returns_false(store: SQLiteJobStore) -> None:
    """Re-running an unknown id is a no-op that reports the miss."""
    assert store.requeue_for_rerun("ghost") is False


def test_get_jobs_status_by_ids_returns_map(store: SQLiteJobStore) -> None:
    """Get jobs status by ids returns map."""
    store.create_job("s1")
    store.create_job("s2")
    store.update_job("s2", status="running")
    result = store.get_jobs_status_by_ids(["s1", "s2"])
    assert result == {"s1": "pending", "s2": "running"}


def test_get_jobs_status_by_ids_omits_missing(store: SQLiteJobStore) -> None:
    """Get jobs status by ids omits missing."""
    store.create_job("s3")
    result = store.get_jobs_status_by_ids(["s3", "missing-xyz"])
    assert "missing-xyz" not in result
    assert result["s3"] == "pending"


def test_get_jobs_status_by_ids_empty_input(store: SQLiteJobStore) -> None:
    """Get jobs status by ids empty input."""
    assert store.get_jobs_status_by_ids([]) == {}


def test_delete_jobs_returns_count_of_deleted(store: SQLiteJobStore) -> None:
    """Delete jobs returns count of deleted."""
    store.create_job("b1")
    store.create_job("b2")
    assert store.delete_jobs(["b1", "b2"]) == 2


def test_delete_jobs_actually_removes_jobs(store: SQLiteJobStore) -> None:
    """Delete jobs actually removes jobs."""
    store.create_job("b3")
    store.create_job("b4")
    store.delete_jobs(["b3", "b4"])
    assert not store.job_exists("b3")
    assert not store.job_exists("b4")


def test_delete_jobs_removes_associated_logs_and_progress(store: SQLiteJobStore) -> None:
    """Delete jobs removes associated logs and progress."""
    store.create_job("b5")
    store.append_log("b5", level="INFO", logger_name="t", message="log entry")
    store.record_progress("b5", "step", {"m": 1})
    store.delete_jobs(["b5"])
    assert store.get_logs("b5") == []
    assert store.get_progress_events("b5") == []


def test_delete_jobs_tolerates_missing_ids(store: SQLiteJobStore) -> None:
    """Delete jobs tolerates missing ids."""
    store.create_job("b6")
    removed = store.delete_jobs(["b6", "no-such-id"])
    assert removed == 1


def test_delete_jobs_tolerates_empty_list(store: SQLiteJobStore) -> None:
    """Delete jobs tolerates empty list."""
    assert store.delete_jobs([]) == 0


def test_record_progress_creates_event(store: SQLiteJobStore) -> None:
    """Record progress creates event."""
    store.create_job("p1")
    store.record_progress("p1", "started", {"loss": 0.5})
    events = store.get_progress_events("p1")
    assert len(events) == 1
    assert events[0]["event"] == "started"
    assert events[0]["metrics"] == {"loss": 0.5}


def test_record_progress_none_message_allowed(store: SQLiteJobStore) -> None:
    """Record progress none message allowed."""
    store.create_job("p2")
    store.record_progress("p2", None, {})
    assert store.get_progress_count("p2") == 1


def test_record_progress_replaces_latest_metrics_snapshot(store: SQLiteJobStore) -> None:
    """Record progress replaces latest metrics snapshot."""
    store.create_job("p3")
    store.record_progress("p3", "step", {"acc": 0.9})
    store.record_progress("p3", "next", {"loss": 0.2})
    job = store.get_job("p3")
    assert job["latest_metrics"] == {"loss": 0.2}


def test_record_progress_carries_tqdm_progress_across_non_tqdm_events(store: SQLiteJobStore) -> None:
    """Optimizer tqdm progress persists in latest_metrics across interleaved non-tqdm events."""
    store.create_job("p3-tqdm")
    store.record_progress("p3-tqdm", "optimizer_progress", {"tqdm_percent": 30.0, "tqdm_n": 3, "tqdm_total": 10})
    store.record_progress("p3-tqdm", "minibatch_feedback", {"score": 1.0, "iteration": 2})
    job = store.get_job("p3-tqdm")
    assert job["latest_metrics"]["score"] == 1.0
    assert job["latest_metrics"]["iteration"] == 2
    assert job["latest_metrics"]["tqdm_percent"] == 30.0
    assert job["latest_metrics"]["tqdm_n"] == 3
    assert job["latest_metrics"]["tqdm_total"] == 10


def test_record_progress_refreshes_sticky_tqdm_on_new_optimizer_tick(store: SQLiteJobStore) -> None:
    """A newer optimizer_progress event replaces the carried tqdm snapshot without accumulating fields."""
    store.create_job("p3-tqdm2")
    store.record_progress("p3-tqdm2", "optimizer_progress", {"tqdm_percent": 30.0})
    store.record_progress("p3-tqdm2", "minibatch_feedback", {"score": 0.5})
    store.record_progress("p3-tqdm2", "optimizer_progress", {"tqdm_percent": 70.0})
    job = store.get_job("p3-tqdm2")
    assert job["latest_metrics"]["tqdm_percent"] == 70.0
    assert "score" not in job["latest_metrics"]


def test_record_progress_json_metrics_roundtrip(store: SQLiteJobStore) -> None:
    """Record progress json metrics roundtrip."""
    store.create_job("p4")
    metrics: dict[str, Any] = {"nested": {"a": 1}, "values": [1, 2, 3]}
    store.record_progress("p4", "step", metrics)
    events = store.get_progress_events("p4")
    assert events[0]["metrics"] == metrics


def test_get_progress_events_chronological_order(store: SQLiteJobStore) -> None:
    """Get progress events chronological order."""
    store.create_job("p5")
    for i in range(3):
        store.record_progress("p5", f"step-{i}", {"i": i})
    events = store.get_progress_events("p5")
    names = [e["event"] for e in events]
    assert names == ["step-0", "step-1", "step-2"]


def test_get_progress_count_matches(store: SQLiteJobStore) -> None:
    """Get progress count matches."""
    store.create_job("p6")
    store.record_progress("p6", "a", {})
    store.record_progress("p6", "b", {})
    assert store.get_progress_count("p6") == 2


def test_get_progress_events_empty_for_unknown_job(store: SQLiteJobStore) -> None:
    """Get progress events empty for unknown job."""
    assert store.get_progress_events("unknown-job") == []


def test_progress_silently_ignored_for_deleted_job(store: SQLiteJobStore) -> None:
    """Progress silently ignored for deleted job."""
    store.create_job("p-del")
    store.delete_job("p-del")
    # Logging / progress for a deleted job must not raise
    store.record_progress("p-del", "after delete", {})


def test_append_log_makes_entry_retrievable(store: SQLiteJobStore) -> None:
    """Append log makes entry retrievable."""
    store.create_job("l1")
    store.append_log("l1", level="INFO", logger_name="mylogger", message="hello")
    logs = store.get_logs("l1")
    assert len(logs) == 1
    assert logs[0]["message"] == "hello"
    assert logs[0]["level"] == "INFO"
    assert logs[0]["logger"] == "mylogger"


def test_append_log_pair_index_stored(store: SQLiteJobStore) -> None:
    """Append log pair index stored."""
    store.create_job("l2")
    store.append_log("l2", level="DEBUG", logger_name="lg", message="msg", pair_index=42)
    assert store.get_logs("l2")[0]["pair_index"] == 42


def test_append_log_pair_index_null_when_absent(store: SQLiteJobStore) -> None:
    """Append log pair index null when absent."""
    store.create_job("l3")
    store.append_log("l3", level="INFO", logger_name="lg", message="no pair")
    assert store.get_logs("l3")[0]["pair_index"] is None


def test_append_log_warns_for_deleted_job(store: SQLiteJobStore, caplog: pytest.LogCaptureFixture) -> None:
    """Append log warns and skips the entry for deleted jobs."""
    store.create_job("l-del")
    store.delete_job("l-del")
    store.append_log("l-del", level="INFO", logger_name="lg", message="after delete")
    assert store.get_logs("l-del") == []
    assert "Discarding log entry for missing job l-del" in caplog.text


def test_get_logs_level_filter(store: SQLiteJobStore) -> None:
    """Get logs level filter."""
    store.create_job("l4")
    store.append_log("l4", level="INFO", logger_name="lg", message="info msg")
    store.append_log("l4", level="ERROR", logger_name="lg", message="err msg")
    errors = store.get_logs("l4", level="ERROR")
    assert len(errors) == 1
    assert errors[0]["level"] == "ERROR"


def test_get_logs_offset_pagination(store: SQLiteJobStore) -> None:
    """Get logs offset pagination."""
    store.create_job("l5")
    for i in range(5):
        store.append_log("l5", level="INFO", logger_name="lg", message=f"msg {i}")
    page = store.get_logs("l5", offset=2)
    assert len(page) == 3
    assert page[0]["message"] == "msg 2"


def test_get_logs_limit_pagination(store: SQLiteJobStore) -> None:
    """Get logs limit pagination."""
    store.create_job("l6")
    for i in range(5):
        store.append_log("l6", level="INFO", logger_name="lg", message=f"msg {i}")
    page = store.get_logs("l6", limit=2)
    assert len(page) == 2


def test_get_logs_limit_and_offset_combined(store: SQLiteJobStore) -> None:
    """Get logs limit and offset combined."""
    store.create_job("l7")
    for i in range(10):
        store.append_log("l7", level="INFO", logger_name="lg", message=f"msg {i}")
    page = store.get_logs("l7", limit=3, offset=4)
    assert len(page) == 3
    assert page[0]["message"] == "msg 4"


def test_get_log_count_no_filter(store: SQLiteJobStore) -> None:
    """Get log count no filter."""
    store.create_job("l8")
    store.append_log("l8", level="INFO", logger_name="lg", message="a")
    store.append_log("l8", level="WARN", logger_name="lg", message="b")
    assert store.get_log_count("l8") == 2


def test_get_log_count_with_level_filter(store: SQLiteJobStore) -> None:
    """Get log count with level filter."""
    store.create_job("l9")
    store.append_log("l9", level="INFO", logger_name="lg", message="i")
    store.append_log("l9", level="ERROR", logger_name="lg", message="e1")
    store.append_log("l9", level="ERROR", logger_name="lg", message="e2")
    assert store.get_log_count("l9", level="ERROR") == 2


def test_get_logs_returns_empty_for_unknown_job(store: SQLiteJobStore) -> None:
    """Get logs returns empty for unknown job."""
    assert store.get_logs("unknown-job") == []


def test_set_payload_overview_stores_and_retrieves_data(store: SQLiteJobStore) -> None:
    """Set payload overview stores and retrieves data."""
    store.create_job("o1")
    overview: dict[str, Any] = {"username": "alice", "optimization_type": "gepa", "extra": [1, 2]}
    store.set_payload_overview("o1", overview)
    job = store.get_job("o1")
    assert job["payload_overview"]["username"] == "alice"
    assert job["payload_overview"]["extra"] == [1, 2]


def test_set_payload_overview_overwrites_previous(store: SQLiteJobStore) -> None:
    """Set payload overview overwrites previous."""
    store.create_job("o2")
    store.set_payload_overview("o2", {"username": "old"})
    store.set_payload_overview("o2", {"username": "new"})
    assert store.get_job("o2")["payload_overview"]["username"] == "new"


def test_set_payload_overview_sets_username_column(store: SQLiteJobStore) -> None:
    """Set payload overview sets username column."""
    store.create_job("o3")
    store.set_payload_overview("o3", {"username": "carol"})
    # The store writes to the job.username column as well; verify via list_jobs filter
    jobs = store.list_jobs(username="carol")
    assert any(j["optimization_id"] == "o3" for j in jobs)


def test_set_payload_overview_without_username_preserves_column(store: SQLiteJobStore) -> None:
    """Payload overview updates without username keep the indexed owner."""
    store.create_job("o4")
    store.set_payload_overview("o4", {"username": "dana", "optimization_type": "gepa"})
    store.set_payload_overview("o4", {"optimization_type": "bootstrap"})
    assert store.get_job("o4")["username"] == "dana"
    assert store.list_jobs(username="dana")[0]["optimization_id"] == "o4"


def test_list_jobs_no_filter_returns_all(store: SQLiteJobStore) -> None:
    """List jobs no filter returns all."""
    store.create_job("lj1")
    store.create_job("lj2")
    assert len(store.list_jobs()) == 2


def test_list_jobs_status_filter(store: SQLiteJobStore) -> None:
    """List jobs status filter."""
    store.create_job("lj-pend")
    store.create_job("lj-run")
    store.update_job("lj-run", status="running")
    result = store.list_jobs(status="pending")
    ids = {j["optimization_id"] for j in result}
    assert ids == {"lj-pend"}


def test_list_jobs_username_filter(store: SQLiteJobStore) -> None:
    """List jobs username filter."""
    store.create_job("lj-alice")
    store.set_payload_overview("lj-alice", {"username": "alice"})
    store.create_job("lj-bob")
    store.set_payload_overview("lj-bob", {"username": "bob"})
    result = store.list_jobs(username="alice")
    assert all(j["optimization_id"] == "lj-alice" for j in result)


def test_list_jobs_optimization_type_filter(store: SQLiteJobStore) -> None:
    """List jobs optimization type filter."""
    store.create_job("lj-opt-a")
    store.set_payload_overview("lj-opt-a", {"optimization_type": "opt_a"})
    store.create_job("lj-bs")
    store.set_payload_overview("lj-bs", {"optimization_type": "bootstrap"})
    result = store.list_jobs(optimization_type="opt_a")
    ids = {j["optimization_id"] for j in result}
    assert ids == {"lj-opt-a"}


def test_list_jobs_limit(store: SQLiteJobStore) -> None:
    """List jobs limit."""
    for i in range(5):
        store.create_job(f"lim-{i}")
    assert len(store.list_jobs(limit=3)) == 3


def test_list_jobs_offset(store: SQLiteJobStore) -> None:
    """List jobs offset."""
    for i in range(5):
        store.create_job(f"off-{i}")
    all_jobs = store.list_jobs()
    paged = store.list_jobs(offset=2)
    assert len(paged) == len(all_jobs) - 2


def test_list_jobs_includes_progress_and_log_counts(store: SQLiteJobStore) -> None:
    """List jobs includes progress and log counts."""
    store.create_job("lj-counts")
    store.append_log("lj-counts", level="INFO", logger_name="t", message="m")
    store.record_progress("lj-counts", "step", {})
    jobs = store.list_jobs()
    job = next(j for j in jobs if j["optimization_id"] == "lj-counts")
    assert job["log_count"] == 1
    assert job["progress_count"] == 1


def test_count_jobs_total(store: SQLiteJobStore) -> None:
    """Count jobs total."""
    store.create_job("cj1")
    store.create_job("cj2")
    assert store.count_jobs() == 2


def test_count_jobs_status_filter(store: SQLiteJobStore) -> None:
    """Count jobs status filter."""
    store.create_job("cj-pend")
    store.create_job("cj-done")
    store.update_job("cj-done", status="success")
    assert store.count_jobs(status="pending") == 1
    assert store.count_jobs(status="success") == 1


def test_count_jobs_username_filter(store: SQLiteJobStore) -> None:
    """Count jobs username filter."""
    store.create_job("cj-alice")
    store.set_payload_overview("cj-alice", {"username": "alice"})
    store.create_job("cj-bob")
    store.set_payload_overview("cj-bob", {"username": "bob"})
    assert store.count_jobs(username="alice") == 1


def test_count_jobs_zero_when_empty(store: SQLiteJobStore) -> None:
    """Count jobs zero when empty."""
    assert store.count_jobs() == 0


def test_recover_orphaned_jobs_requeues_running_job(store: SQLiteJobStore) -> None:
    """Recover orphaned jobs requeues a first-attempt running job."""
    store.create_job("r1")
    store.update_job("r1", status="running")
    store.recover_orphaned_jobs()
    job = store.get_job("r1")
    assert job["status"] == "pending"
    assert job["attempts"] == 1
    assert job["message"] == "Re-queued after pod failure (attempt 1)"


def test_recover_orphaned_jobs_requeues_validating_job(store: SQLiteJobStore) -> None:
    """Recover orphaned jobs requeues a first-attempt validating job."""
    store.create_job("r2")
    store.update_job("r2", status="validating")
    store.recover_orphaned_jobs()
    assert store.get_job("r2")["status"] == "pending"


def test_recover_orphaned_jobs_leaves_terminal_jobs_intact(store: SQLiteJobStore) -> None:
    """Recover orphaned jobs leaves terminal jobs intact."""
    store.create_job("r3")
    store.update_job("r3", status="success")
    store.create_job("r4")
    store.update_job("r4", status="failed")
    store.recover_orphaned_jobs()
    assert store.get_job("r3")["status"] == "success"
    assert store.get_job("r4")["status"] == "failed"


def test_recover_orphaned_jobs_returns_count(store: SQLiteJobStore) -> None:
    """Recover orphaned jobs returns count."""
    store.create_job("r5")
    store.update_job("r5", status="running")
    store.create_job("r6")
    store.update_job("r6", status="validating")
    store.create_job("r7")
    store.update_job("r7", status="success")
    assert store.recover_orphaned_jobs() == 2


def test_recover_orphaned_jobs_fails_at_max_attempts(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover orphaned jobs fails the row when the retry cap is reached."""
    monkeypatch.setattr(remote_mod.settings, "job_max_attempts", 3)
    store.create_job("r8")
    store.update_job("r8", status="running", attempts=2)
    store.recover_orphaned_jobs()
    job = store.get_job("r8")
    assert job["status"] == "failed"
    assert job["attempts"] == 3
    assert job["message"] == "Job failed after pod failure (attempt 3)"
    assert job["completed_at"] is not None


def test_recover_orphaned_jobs_reports_version_skew_at_max_attempts(
    store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover orphaned jobs explains incompatible code versions at the retry cap."""
    monkeypatch.setattr(remote_mod.settings, "job_max_attempts", 3)
    monkeypatch.setattr(remote_mod.settings, "code_version", "v2", raising=False)
    store.create_job("r-version")
    store.update_job("r-version", status="running", attempts=2, code_version="v1")

    store.recover_orphaned_jobs()

    job = store.get_job("r-version")
    assert job["status"] == "failed"
    assert job["message"] == "No compatible worker version available (job=v1, fleet=v2)"


def test_recover_orphaned_jobs_returns_zero_when_none_present(store: SQLiteJobStore) -> None:
    """Recover orphaned jobs returns zero when none present."""
    store.create_job("r9")
    store.update_job("r9", status="success")
    assert store.recover_orphaned_jobs() == 0


def test_recover_pending_jobs_returns_only_pending(store: SQLiteJobStore) -> None:
    """Recover pending jobs returns only pending."""
    store.create_job("rp1")
    store.create_job("rp2")
    store.update_job("rp2", status="success")
    pending = store.recover_pending_jobs()
    assert "rp1" in pending
    assert "rp2" not in pending


def test_recover_pending_jobs_skips_incompatible_code_version(
    store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending local restart hints exclude jobs for a different code version."""
    monkeypatch.setattr(remote_mod.settings, "code_version", "v2", raising=False)
    store.create_job("rp-versioned")
    store.update_job("rp-versioned", code_version="v1")

    assert "rp-versioned" not in store.recover_pending_jobs()


def test_recover_pending_jobs_returns_list_of_strings(store: SQLiteJobStore) -> None:
    """Recover pending jobs returns list of strings."""
    store.create_job("rp3")
    result = store.recover_pending_jobs()
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)


def test_recover_pending_jobs_empty_when_none_pending(store: SQLiteJobStore) -> None:
    """Recover pending jobs empty when none pending."""
    store.create_job("rp4")
    store.update_job("rp4", status="success")
    assert store.recover_pending_jobs() == []


def test_duplicate_optimization_id_raises_on_insert(store: SQLiteJobStore) -> None:
    """Duplicate optimization id raises on insert."""
    store.create_job("dup-1")
    with pytest.raises(IntegrityError):
        # Bypass create_job's session so we insert a raw duplicate row
        session = store._get_session()
        try:
            job = JobModel(
                optimization_id="dup-1",
                status="pending",
                created_at=datetime.now(UTC),
                latest_metrics={},
                payload_overview={},
                attempts=0,
            )
            session.add(job)
            session.commit()
        finally:
            session.rollback()
            session.close()


def test_remote_db_job_store_satisfies_job_store_protocol() -> None:
    """Remote db job store satisfies job store protocol."""
    assert isinstance(SQLiteJobStore(), JobStore)


def test_claim_next_job_skips_incompatible_code_version(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """A v2 worker cannot claim a job submitted by v1."""
    monkeypatch.setattr(remote_mod.settings, "code_version", "v2", raising=False)
    store.create_job("claim-v1")
    store.update_job("claim-v1", code_version="v1")

    assert store.claim_next_job("pod-v2", lease_seconds=60.0) is None


def test_claim_next_job_accepts_null_code_version(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Jobs without a code version remain claimable by any worker."""
    monkeypatch.setattr(remote_mod.settings, "code_version", "v2", raising=False)
    store.create_job("claim-null")
    store.update_job("claim-null", code_version=None)

    row = store.claim_next_job("pod-v2", lease_seconds=60.0)

    assert row is not None
    assert row["optimization_id"] == "claim-null"


def test_update_job_unknown_field_raises_value_error(store: SQLiteJobStore) -> None:
    """Update job unknown field raises value error."""
    store.create_job("uf-1")
    with pytest.raises(ValueError, match="Unknown field"):
        store.update_job("uf-1", bogus_field="x")


def test_update_job_unknown_field_message_contains_field_name(store: SQLiteJobStore) -> None:
    """Update job unknown field message contains field name."""
    store.create_job("uf-2")
    with pytest.raises(ValueError, match="totally_bogus"):
        store.update_job("uf-2", totally_bogus="value")


def test_update_job_unknown_field_does_not_corrupt_existing_data(store: SQLiteJobStore) -> None:
    """Update job unknown field does not corrupt existing data."""
    store.create_job("uf-3")
    store.update_job("uf-3", status="running")
    with pytest.raises(ValueError):
        store.update_job("uf-3", not_a_real_field=42)
    # The earlier valid update must still be in place
    assert store.get_job("uf-3")["status"] == "running"


def test_log_eviction_oldest_entry_removed_when_cap_reached(
    store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log eviction oldest entry removed when cap reached."""
    monkeypatch.setattr(remote_mod, "MAX_LOG_ENTRIES", 3)

    store.create_job("evict-log-1")
    for i in range(3):
        store.append_log("evict-log-1", level="INFO", logger_name="lg", message=f"msg-{i}")

    # Adding a 4th entry must evict msg-0 (the oldest)
    store.append_log("evict-log-1", level="INFO", logger_name="lg", message="msg-3")

    logs = store.get_logs("evict-log-1")
    messages = [lg["message"] for lg in logs]
    assert "msg-0" not in messages
    assert "msg-3" in messages


def test_log_eviction_count_stays_at_cap(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Log eviction count stays at cap."""
    monkeypatch.setattr(remote_mod, "MAX_LOG_ENTRIES", 3)

    store.create_job("evict-log-2")
    for i in range(5):
        store.append_log("evict-log-2", level="INFO", logger_name="lg", message=f"msg-{i}")

    assert store.get_log_count("evict-log-2") == 3


def test_progress_eviction_oldest_entry_removed_when_cap_reached(
    store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Progress eviction oldest entry removed when cap reached."""
    monkeypatch.setattr(remote_mod, "MAX_PROGRESS_EVENTS", 3)
    monkeypatch.setattr(remote_mod, "PROGRESS_TRIM_SAMPLE_RATE", 1)

    store.create_job("evict-prog-1")
    for i in range(3):
        store.record_progress("evict-prog-1", f"step-{i}", {"i": i})

    # Adding a 4th entry must evict step-0 (the oldest)
    store.record_progress("evict-prog-1", "step-3", {"i": 3})

    events = store.get_progress_events("evict-prog-1")
    event_names = [e["event"] for e in events]
    assert "step-0" not in event_names
    assert "step-3" in event_names


def test_progress_eviction_count_stays_at_cap(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress eviction count stays at cap."""
    monkeypatch.setattr(remote_mod, "MAX_PROGRESS_EVENTS", 3)
    monkeypatch.setattr(remote_mod, "PROGRESS_TRIM_SAMPLE_RATE", 1)

    store.create_job("evict-prog-2")
    for i in range(5):
        store.record_progress("evict-prog-2", f"step-{i}", {"i": i})

    assert store.get_progress_count("evict-prog-2") == 3


def test_progress_eviction_preserves_structural_events(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Structural phase markers survive eviction while high-volume rows get dropped."""
    monkeypatch.setattr(remote_mod, "MAX_PROGRESS_EVENTS", 3)
    monkeypatch.setattr(remote_mod, "PROGRESS_TRIM_SAMPLE_RATE", 1)

    store.create_job("evict-prog-3")
    store.record_progress("evict-prog-3", "grid_pair_started", {"pair_index": 0})
    store.record_progress("evict-prog-3", "optimizer_progress", {"step": 0})
    store.record_progress("evict-prog-3", "optimizer_progress", {"step": 1})
    # Adding a 4th entry should evict the oldest optimizer_progress, not the
    # structural grid_pair_started marker.
    store.record_progress("evict-prog-3", "grid_pair_completed", {"pair_index": 0})

    events = store.get_progress_events("evict-prog-3")
    names = [e["event"] for e in events]
    assert "grid_pair_started" in names
    assert "grid_pair_completed" in names
    # Exactly one optimizer_progress survived — the newer one.
    assert names.count("optimizer_progress") == 1
    surviving = next(e for e in events if e["event"] == "optimizer_progress")
    assert surviving["metrics"] == {"step": 1}


def test_progress_eviction_structural_dropped_only_as_last_resort(
    store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every existing row is structural, the oldest structural row is evicted."""
    monkeypatch.setattr(remote_mod, "MAX_PROGRESS_EVENTS", 2)
    monkeypatch.setattr(remote_mod, "PROGRESS_TRIM_SAMPLE_RATE", 1)

    store.create_job("evict-prog-4")
    store.record_progress("evict-prog-4", "grid_pair_started", {"pair_index": 0})
    store.record_progress("evict-prog-4", "baseline_evaluated", {"pair_index": 0})
    # Cap is full and no non-structural rows exist — oldest structural gets evicted.
    store.record_progress("evict-prog-4", "optimized_evaluated", {"pair_index": 0})

    events = store.get_progress_events("evict-prog-4")
    names = [e["event"] for e in events]
    assert "grid_pair_started" not in names
    assert "baseline_evaluated" in names
    assert "optimized_evaluated" in names


def test_record_progress_does_not_lock_parent_job_row() -> None:
    """``record_progress`` does not use ``SELECT ... FOR UPDATE`` on jobs."""
    source = inspect.getsource(RemoteDBJobStore.record_progress)

    assert "with_for_update" not in source


def test_update_job_does_not_lock_parent_job_row() -> None:
    """``update_job`` does not use ``SELECT ... FOR UPDATE`` on jobs.

    Row-level locks held across a SELECT+UPDATE round-trip serialise every
    writer touching the same id and become a contention hotspot at scale.
    ``record_progress`` already uses a non-locking pattern; ``update_job``
    must too.
    """
    source = inspect.getsource(RemoteDBJobStore.update_job)

    assert "with_for_update" not in source


def test_progress_trim_is_sampled_and_batched(store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampled trimming allows a small overshoot but keeps rows near the cap."""
    monkeypatch.setattr(remote_mod, "MAX_PROGRESS_EVENTS", 5000)
    monkeypatch.setattr(remote_mod, "PROGRESS_TRIM_SAMPLE_RATE", 100)

    store.create_job("sampled-trim")
    for i in range(5200):
        store.record_progress("sampled-trim", "optimizer_progress", {"i": i})

    assert 5000 <= store.get_progress_count("sampled-trim") <= 5100


def test_engine_kwargs_use_configured_pool_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine kwargs read DB pool settings from configuration."""
    monkeypatch.setattr(remote_mod.settings, "db_pool_size", 12)
    monkeypatch.setattr(remote_mod.settings, "db_pool_max_overflow", 7)
    monkeypatch.setattr(remote_mod.settings, "db_pool_recycle_seconds", 600)

    kwargs = remote_mod._build_engine_kwargs("postgresql://user:pass@localhost/skynet")

    assert kwargs["pool_size"] == 12
    assert kwargs["max_overflow"] == 7
    assert kwargs["pool_recycle"] == 600


def test_pgbouncer_transaction_mode_disables_psycopg_prepared_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Psycopg3 URLs disable server-side prepared statements for PgBouncer."""
    monkeypatch.setattr(remote_mod.settings, "db_pgbouncer_transaction_mode", True)

    connect_args = remote_mod._build_connect_args("postgresql+psycopg://user:pass@localhost/skynet")

    assert connect_args["prepare_threshold"] is None


def test_create_job_persists_username_and_idempotency_key(store: SQLiteJobStore) -> None:
    """``create_job`` records the submitter and dedup key when supplied."""
    store.create_job("idem-create-1", username="alice", idempotency_key="abc-123")

    row = store.get_job("idem-create-1")
    assert row["username"] == "alice"


def test_find_job_by_idempotency_key_returns_prior_submission(store: SQLiteJobStore) -> None:
    """Looking up a prior ``(username, idempotency_key)`` pair returns its job id."""
    store.create_job("idem-find-1", username="bob", idempotency_key="key-xyz")

    found = store.find_job_by_idempotency_key("bob", "key-xyz")

    assert found == "idem-find-1"


def test_find_job_by_idempotency_key_scopes_by_username(store: SQLiteJobStore) -> None:
    """Two users sharing the same key resolve to distinct jobs."""
    store.create_job("idem-scope-a", username="alice", idempotency_key="shared")
    store.create_job("idem-scope-b", username="bob", idempotency_key="shared")

    assert store.find_job_by_idempotency_key("alice", "shared") == "idem-scope-a"
    assert store.find_job_by_idempotency_key("bob", "shared") == "idem-scope-b"


def test_find_job_by_idempotency_key_returns_none_for_unknown(store: SQLiteJobStore) -> None:
    """Unknown key returns ``None`` rather than raising."""
    assert store.find_job_by_idempotency_key("alice", "never-seen") is None


def test_find_job_by_idempotency_key_rejects_empty_inputs(store: SQLiteJobStore) -> None:
    """Empty username or key short-circuits to ``None`` (no wildcard match)."""
    store.create_job("idem-empty-1", username="alice", idempotency_key="real-key")

    assert store.find_job_by_idempotency_key("", "real-key") is None
    assert store.find_job_by_idempotency_key("alice", "") is None


def test_claim_completion_notification_first_caller_wins(store: SQLiteJobStore) -> None:
    """The first caller to claim a job's notification wins."""
    store.create_job("notify-claim-1")

    assert store.claim_completion_notification("notify-claim-1") is True


def test_claim_completion_notification_second_caller_loses(store: SQLiteJobStore) -> None:
    """The second caller for the same job loses the CAS race."""
    store.create_job("notify-claim-2")

    store.claim_completion_notification("notify-claim-2")

    assert store.claim_completion_notification("notify-claim-2") is False


def test_claim_completion_notification_missing_job_returns_false(store: SQLiteJobStore) -> None:
    """A missing job id reports ``False`` rather than raising."""
    assert store.claim_completion_notification("no-such-job") is False


def test_immutable_columns_include_notification_and_idempotency_fields() -> None:
    """``_IMMUTABLE_JOB_COLUMNS`` guards the new dedup columns from ``update_job``."""
    assert "notified_at" in remote_mod._IMMUTABLE_JOB_COLUMNS
    assert "idempotency_key" in remote_mod._IMMUTABLE_JOB_COLUMNS


def test_gepa_checkpoint_save_get_delete_roundtrip(store: SQLiteJobStore) -> None:
    """A checkpoint saves, reads back with its iteration/size, replaces, and deletes."""
    store.create_job("ck1")
    assert store.has_gepa_checkpoint("ck1") is False
    assert store.get_gepa_checkpoint("ck1") is None

    store.save_gepa_checkpoint("ck1", b"STATE-A", iteration=3)
    assert store.has_gepa_checkpoint("ck1") is True
    cp = store.get_gepa_checkpoint("ck1")
    assert cp is not None
    assert cp.data == b"STATE-A"
    assert cp.iteration == 3
    assert cp.stored_bytes == 7

    store.save_gepa_checkpoint("ck1", b"STATE-B-longer", iteration=9)
    cp = store.get_gepa_checkpoint("ck1")
    assert cp.data == b"STATE-B-longer"
    assert cp.iteration == 9

    store.delete_gepa_checkpoint("ck1")
    assert store.has_gepa_checkpoint("ck1") is False
    store.delete_gepa_checkpoint("ck1")  # idempotent no-op


def test_delete_job_removes_its_checkpoint(store: SQLiteJobStore) -> None:
    """Deleting a job drops its checkpoint row too (no orphaned bytes)."""
    store.create_job("ck2")
    store.save_gepa_checkpoint("ck2", b"STATE", iteration=1)
    store.delete_job("ck2")
    assert store.has_gepa_checkpoint("ck2") is False


def test_requeue_for_resume_flips_to_pending_and_bumps_attempts(store: SQLiteJobStore) -> None:
    """Resume re-queues in place: same id, pending, claim cleared, attempts+1."""
    store.create_job("ck3")
    store.update_job("ck3", status="failed", attempts=0, claimed_by="pod-x", completed_at="2026-01-01T00:00:00Z")

    new_attempt = store.requeue_for_resume("ck3")
    assert new_attempt == 1
    job = store.get_job("ck3")
    assert job["status"] == "pending"
    assert job["attempts"] == 1
    assert job.get("claimed_by") is None
    assert job.get("completed_at") is None


def test_requeue_for_resume_missing_job_returns_none(store: SQLiteJobStore) -> None:
    """Re-queuing an unknown id reports ``None`` rather than raising."""
    assert store.requeue_for_resume("no-such-job") is None


def test_requeue_for_resume_banks_finished_leg_runtime(store: SQLiteJobStore) -> None:
    """Each resume folds the finished leg's duration in; the paused gap is excluded."""
    store.create_job("ck-acc")
    store.update_job(
        "ck-acc",
        status="failed",
        attempts=0,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:05Z",  # leg 1 = 65s
    )

    store.requeue_for_resume("ck-acc")
    job = store.get_job("ck-acc")
    assert job["accumulated_runtime_seconds"] == 65.0
    assert job.get("started_at") is None
    assert job.get("completed_at") is None

    store.update_job(
        "ck-acc",
        status="failed",
        started_at="2026-01-01T00:30:00Z",  # ~29 min paused before this leg
        completed_at="2026-01-01T00:30:30Z",  # leg 2 = 30s
    )
    store.requeue_for_resume("ck-acc")
    job = store.get_job("ck-acc")
    assert job["accumulated_runtime_seconds"] == 95.0  # 65 + 30, the pause is never counted


def test_gepa_checkpoints_are_isolated_per_grid_pair(store: SQLiteJobStore) -> None:
    """A grid keeps an independent checkpoint per pair index; delete-all clears them."""
    store.create_job("g1")
    store.save_gepa_checkpoint("g1", b"P0", iteration=1, pair_index=0)
    store.save_gepa_checkpoint("g1", b"P1", iteration=2, pair_index=1)
    assert {c.pair_index for c in store.list_gepa_checkpoints("g1")} == {0, 1}
    assert store.get_gepa_checkpoint("g1", 1).data == b"P1"

    store.delete_gepa_checkpoint("g1", 0)
    assert {c.pair_index for c in store.list_gepa_checkpoints("g1")} == {1}
    assert store.has_gepa_checkpoint("g1") is True

    store.delete_all_gepa_checkpoints("g1")
    assert store.list_gepa_checkpoints("g1") == []
    assert store.has_gepa_checkpoint("g1") is False


def test_grid_pair_results_roundtrip(store: SQLiteJobStore) -> None:
    """Completed grid pairs are stored, read as ``{index: result}``, and cleared."""
    store.create_job("g2")
    assert store.has_grid_pair_results("g2") is False
    store.save_grid_pair_result("g2", 0, {"pair_index": 0, "optimized_test_metric": 0.8})
    store.save_grid_pair_result("g2", 3, {"pair_index": 3})
    assert store.has_grid_pair_results("g2") is True

    results = store.get_grid_pair_results("g2")
    assert set(results) == {0, 3}
    assert results[0]["optimized_test_metric"] == 0.8

    store.delete_grid_pair_results("g2")
    assert store.has_grid_pair_results("g2") is False


def test_delete_job_removes_grid_pair_results_and_pair_checkpoints(store: SQLiteJobStore) -> None:
    """Deleting a grid job drops its per-pair checkpoints and stored pair results."""
    store.create_job("g3")
    store.save_gepa_checkpoint("g3", b"P0", iteration=1, pair_index=0)
    store.save_grid_pair_result("g3", 1, {"pair_index": 1})
    store.delete_job("g3")
    assert store.has_gepa_checkpoint("g3") is False
    assert store.has_grid_pair_results("g3") is False
