"""Tests for the ``/optimizations`` router.

Covers list, get, cancel, delete, bulk delete, dashboard streams, dataset
splits, and grid-search pair operations.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...config import settings
from ...models.artifacts import ProgramArtifact
from ...models.common import SplitCounts
from ...models.results import RunResponse
from ..errors import DomainError
from ..routers.optimizations import create_optimizations_router
from ..routers.optimizations._local import clone_payload
from .conftest import bypass_auth
from .mocks import _BaseFakeJobStore, real_grid_response_dict

# _ExtendedFakeJobStore is just _BaseFakeJobStore (which already has bulk-delete).
_ExtendedFakeJobStore = _BaseFakeJobStore


@pytest.fixture
def store() -> _ExtendedFakeJobStore:
    """Provide a fresh fake job store for each test.

    Returns:
        A new ``_ExtendedFakeJobStore`` instance.
    """
    return _ExtendedFakeJobStore()


@pytest.fixture
def opt_client(store: _ExtendedFakeJobStore) -> TestClient:
    """Build a ``TestClient`` exposing only the optimizations router.

    Args:
        store: Fake job store wired into the router factory.

    Returns:
        A ``TestClient`` over a minimal FastAPI app.
    """
    app = FastAPI()
    app.include_router(create_optimizations_router(job_store=store, get_worker_ref=lambda: None))
    bypass_auth(app)
    return TestClient(app, raise_server_exceptions=False)


def test_list_jobs_returns_empty_when_store_is_empty(opt_client: TestClient) -> None:
    """An empty store yields a zero-total list with no items."""
    resp = opt_client.get("/optimizations")

    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["items"] == []


def test_list_jobs_invalid_status_returns_422(opt_client: TestClient) -> None:
    """Unknown ``status`` query values are rejected with 422."""
    resp = opt_client.get("/optimizations?status=bogus")

    assert resp.status_code == 422


def test_list_jobs_invalid_optimization_type_returns_422(opt_client: TestClient) -> None:
    """Unknown ``optimization_type`` values are rejected with 422."""
    resp = opt_client.get("/optimizations?optimization_type=bad_type")

    assert resp.status_code == 422


def test_list_jobs_valid_status_filter_returns_matching_jobs(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """The ``status`` filter narrows results to matching jobs only."""
    store.seed_job("j1", status="success")
    store.seed_job("j2", status="failed")

    resp = opt_client.get("/optimizations?status=success")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["optimization_id"] == "j1"


def test_get_job_returns_404_for_unknown_id(opt_client: TestClient) -> None:
    """A get against an unknown id returns 404."""
    resp = opt_client.get("/optimizations/does-not-exist")

    assert resp.status_code == 404


def test_get_job_returns_404_for_internal_tagger_job(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Internal tagger worker rows cannot render as optimization details."""
    store.seed_job("internal-tagger", payload_overview={"optimization_type": "tagging_autotag"})

    resp = opt_client.get("/optimizations/internal-tagger")

    assert resp.status_code == 404


def test_get_job_returns_404_for_unknown_internal_job_type(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Unknown worker job types cannot render as optimization details."""
    store.seed_job("future-internal", payload_overview={"optimization_type": "future_internal_job"})

    resp = opt_client.get("/optimizations/future-internal")

    assert resp.status_code == 404


def test_get_job_returns_200_for_existing_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """An existing job can be fetched by id."""
    store.seed_job("abc", status="success")

    resp = opt_client.get("/optimizations/abc")

    assert resp.status_code == 200
    assert resp.json()["optimization_id"] == "abc"


def test_get_job_returns_304_when_etag_matches(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A matching ``If-None-Match`` header yields a 304 from the get endpoint."""
    store.seed_job("e1", status="success")

    first = opt_client.get("/optimizations/e1")
    etag = first.headers.get("etag")
    assert etag is not None

    second = opt_client.get("/optimizations/e1", headers={"if-none-match": etag})

    assert second.status_code == 304
    # A 304 must carry no body: a ``JSONResponse(content=None)`` would emit
    # ``b"null"``, which real uvicorn/h11 rejects with "Too much data for
    # declared Content-Length". The TestClient transport doesn't enforce that
    # framing, so assert the empty body directly to guard the regression.
    assert second.content == b""
    assert second.headers.get("etag") == etag


def _seed_streamed_job(
    store: _ExtendedFakeJobStore, optimization_id: str, *, n_progress: int, n_logs: int
) -> None:
    """Seed a running job with synthetic, ordered progress events and log rows.

    Args:
        store: Fake job store to populate.
        optimization_id: Identifier for the synthesised job.
        n_progress: Number of progress events to attach (tagged ``ev0``..).
        n_logs: Number of log rows to attach (messaged ``log0``..).
    """
    store.seed_job(optimization_id, status="running")
    store._progress[optimization_id] = [
        {"timestamp": f"2026-01-01T00:00:{i:02d}+00:00", "event": f"ev{i}", "metrics": {}}
        for i in range(n_progress)
    ]
    store._logs[optimization_id] = [
        {"timestamp": f"2026-01-01T00:00:{i:02d}+00:00", "level": "INFO", "logger": "test", "message": f"log{i}"}
        for i in range(n_logs)
    ]


def test_get_job_full_fetch_reports_zero_offsets(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A cursorless fetch returns the whole stream with zero offsets."""
    _seed_streamed_job(store, "d0", n_progress=5, n_logs=4)

    resp = opt_client.get("/optimizations/d0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["progress_offset"] == 0
    assert body["logs_offset"] == 0
    assert len(body["progress_events"]) == 5
    assert len(body["logs"]) == 4


def test_get_job_delta_cursor_returns_tail_only(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A valid since_* cursor returns only the rows past the offset, echoing it back."""
    _seed_streamed_job(store, "d1", n_progress=5, n_logs=4)

    resp = opt_client.get("/optimizations/d1?since_progress=3&since_log=2")

    assert resp.status_code == 200
    body = resp.json()
    assert body["progress_offset"] == 3
    assert body["logs_offset"] == 2
    assert [e["event"] for e in body["progress_events"]] == ["ev3", "ev4"]
    assert [log["message"] for log in body["logs"]] == ["log2", "log3"]


def test_get_job_stale_cursor_falls_back_to_full(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A cursor past the current count re-sends the whole stream at offset 0."""
    _seed_streamed_job(store, "d2", n_progress=3, n_logs=3)

    resp = opt_client.get("/optimizations/d2?since_progress=99&since_log=99")

    assert resp.status_code == 200
    body = resp.json()
    assert body["progress_offset"] == 0
    assert body["logs_offset"] == 0
    assert len(body["progress_events"]) == 3
    assert len(body["logs"]) == 3


def test_get_job_delta_cursor_falls_back_at_retention_cap(
    opt_client: TestClient, store: _ExtendedFakeJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the eviction cap positions stop being stable, so cursors re-send everything."""
    monkeypatch.setattr(settings, "progress_events_per_job_cap", 5)
    _seed_streamed_job(store, "d3", n_progress=5, n_logs=4)

    resp = opt_client.get("/optimizations/d3?since_progress=3")

    assert resp.status_code == 200
    body = resp.json()
    assert body["progress_offset"] == 0
    assert len(body["progress_events"]) == 5


def test_get_summary_returns_404_for_unknown_id(opt_client: TestClient) -> None:
    """A summary request against an unknown id returns 404."""
    resp = opt_client.get("/optimizations/missing/summary")

    assert resp.status_code == 404


def test_get_summary_returns_200_for_existing_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """An existing job exposes a summary keyed by ``optimization_id``."""
    store.seed_job("s1")

    resp = opt_client.get("/optimizations/s1/summary")

    assert resp.status_code == 200
    assert resp.json()["optimization_id"] == "s1"


def test_cancel_job_returns_404_for_unknown_id(opt_client: TestClient) -> None:
    """Cancelling an unknown job returns 404."""
    resp = opt_client.post("/optimizations/ghost/cancel")

    assert resp.status_code == 404


def test_cancel_job_returns_409_when_already_terminal(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Cancelling a job already in a terminal state returns 409."""
    store.seed_job("done", status="success")

    resp = opt_client.post("/optimizations/done/cancel")

    assert resp.status_code == 409


@pytest.mark.parametrize("terminal_status", ["success", "failed", "cancelled"])
def test_cancel_job_returns_409_for_all_terminal_statuses(
    opt_client: TestClient,
    store: _ExtendedFakeJobStore,
    terminal_status: str,
) -> None:
    """Every terminal status (success/failed/cancelled) blocks cancel with 409."""
    store.seed_job(f"j_{terminal_status}", status=terminal_status)

    resp = opt_client.post(f"/optimizations/j_{terminal_status}/cancel")

    assert resp.status_code == 409


def test_cancel_job_returns_200_and_sets_cancelled_for_pending(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Cancelling a pending job returns 200 with status ``cancelled``."""
    store.seed_job("active", status="pending")

    resp = opt_client.post("/optimizations/active/cancel")

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_cancel_job_updates_store_status(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Cancelling a running job persists the new ``cancelled`` status to the store."""
    store.seed_job("running_job", status="running")

    opt_client.post("/optimizations/running_job/cancel")

    assert store.get_job("running_job")["status"] == "cancelled"


def test_delete_job_returns_404_for_unknown_id(opt_client: TestClient) -> None:
    """Deleting an unknown job returns 404."""
    resp = opt_client.delete("/optimizations/ghost")

    assert resp.status_code == 404


def test_delete_job_returns_409_for_active_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Deleting a still-running job returns 409."""
    store.seed_job("live", status="running")

    resp = opt_client.delete("/optimizations/live")

    assert resp.status_code == 409


@pytest.mark.parametrize("active_status", ["pending", "validating", "running"])
def test_delete_job_returns_409_for_non_terminal_statuses(
    opt_client: TestClient,
    store: _ExtendedFakeJobStore,
    active_status: str,
) -> None:
    """Every non-terminal status blocks delete with 409."""
    store.seed_job(f"j_{active_status}", status=active_status)

    resp = opt_client.delete(f"/optimizations/j_{active_status}")

    assert resp.status_code == 409


def test_delete_job_returns_200_and_removes_terminal_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Deleting a terminal job removes it from the store and reports ``deleted=True``."""
    store.seed_job("done", status="success")

    resp = opt_client.delete("/optimizations/done")

    assert resp.status_code == 200
    body = resp.json()
    assert body["optimization_id"] == "done"
    assert body["deleted"] is True
    assert not store.job_exists("done")


def test_bulk_delete_empty_list_returns_zero_deleted(opt_client: TestClient) -> None:
    """An empty bulk-delete returns empty ``deleted`` and ``skipped`` lists."""
    resp = opt_client.post("/optimizations/bulk-delete", json={"optimization_ids": []})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == []
    assert body["skipped"] == []


def test_bulk_delete_missing_ids_are_skipped(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Unknown ids are reported in ``skipped`` with a ``not_found`` reason."""
    resp = opt_client.post(
        "/optimizations/bulk-delete",
        json={"optimization_ids": ["ghost1", "ghost2"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == []
    assert {s["reason"] for s in body["skipped"]} == {"not_found"}


def test_bulk_delete_active_ids_are_skipped_with_status_reason(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Non-terminal jobs are skipped and report their current status as the reason."""
    store.seed_job("r1", status="running")

    resp = opt_client.post(
        "/optimizations/bulk-delete",
        json={"optimization_ids": ["r1"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == []
    assert body["skipped"][0]["reason"] == "running"


def test_bulk_delete_deletes_terminal_jobs(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Terminal jobs are removed from the store and listed in ``deleted``."""
    store.seed_job("t1", status="success")
    store.seed_job("t2", status="failed")

    resp = opt_client.post(
        "/optimizations/bulk-delete",
        json={"optimization_ids": ["t1", "t2"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert set(body["deleted"]) == {"t1", "t2"}
    assert body["skipped"] == []
    assert not store.job_exists("t1")
    assert not store.job_exists("t2")


def test_bulk_delete_deduplicates_ids(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Duplicate ids are collapsed before deletion."""
    store.seed_job("dup", status="success")

    resp = opt_client.post(
        "/optimizations/bulk-delete",
        json={"optimization_ids": ["dup", "dup", "dup"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == ["dup"]
    assert body["skipped"] == []


def test_bulk_delete_mixed_batch_reports_partial_results(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Mixed batches return partial deletions and per-id skip reasons."""
    store.seed_job("ok", status="cancelled")
    store.seed_job("busy", status="pending")

    resp = opt_client.post(
        "/optimizations/bulk-delete",
        json={"optimization_ids": ["ok", "busy", "missing"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == ["ok"]
    skip_reasons = {s["optimization_id"]: s["reason"] for s in body["skipped"]}
    assert skip_reasons["busy"] == "pending"
    assert skip_reasons["missing"] == "not_found"


def test_artifact_returns_404_for_unknown_job(opt_client: TestClient) -> None:
    """An artifact request against an unknown job returns 404."""
    resp = opt_client.get("/optimizations/ghost/artifact")

    assert resp.status_code == 404


def test_artifact_returns_409_for_pending_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A pending job has no artifact yet and returns 409."""
    store.seed_job("p", status="pending")

    resp = opt_client.get("/optimizations/p/artifact")

    assert resp.status_code == 409


def test_artifact_returns_409_for_failed_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A failed job has no artifact and returns 409."""
    store.seed_job("f", status="failed", message="OOM")

    resp = opt_client.get("/optimizations/f/artifact")

    assert resp.status_code == 409


def test_artifact_returns_409_for_cancelled_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A cancelled job has no artifact and returns 409."""
    store.seed_job("c", status="cancelled")

    resp = opt_client.get("/optimizations/c/artifact")

    assert resp.status_code == 409


def test_artifact_returns_404_for_grid_search_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The single-program artifact endpoint returns 404 for grid-search jobs."""
    store.seed_job("gs", status="success", payload_overview={"optimization_type": "grid_search"})

    resp = opt_client.get("/optimizations/gs/artifact")

    assert resp.status_code == 404


_EXPORT_WORKFLOW_SPEC = {
    "nodes": [
        {"id": "start", "kind": "input", "fields": [{"name": "question"}]},
        {
            "id": "draft",
            "kind": "signature",
            "signature_code": "import dspy\n\n\nclass QA(dspy.Signature):\n"
            "    question: str = dspy.InputField()\n    answer: str = dspy.OutputField()\n",
        },
        {"id": "end", "kind": "output", "fields": [{"name": "answer"}]},
    ],
    "edges": [
        {"source": "start", "source_port": "question", "target": "draft", "target_port": "question"},
        {"source": "draft", "source_port": "answer", "target": "end", "target_port": "answer"},
    ],
}


def test_program_export_serves_a_workflow_run(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A workflow run carries no top-level signature_code and is still exportable."""
    result_data = _make_run_response_dict(
        module_name="workflow",
        program_artifact=ProgramArtifact(program_state_json={"n_draft": {"demos": []}}).model_dump(),
    )
    store.seed_job(
        "wf",
        status="success",
        payload_overview={"module_name": "workflow", "workflow": _EXPORT_WORKFLOW_SPEC},
        result=result_data,
    )

    resp = opt_client.get("/optimizations/wf/program-export")

    assert resp.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert "workflow.json" in names
    assert "signature.py" not in names


def test_program_export_409s_without_any_program_definition(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A non-workflow run with no signature_code still has nothing to rebuild from."""
    result_data = _make_run_response_dict(
        program_artifact=ProgramArtifact(program_state_json={"demos": []}).model_dump()
    )
    store.seed_job("nosig", status="success", payload_overview={"module_name": "predict"}, result=result_data)

    resp = opt_client.get("/optimizations/nosig/program-export")

    assert resp.status_code == 409


def test_grid_result_returns_404_for_unknown_job(opt_client: TestClient) -> None:
    """A grid-result request against an unknown job returns 404."""
    resp = opt_client.get("/optimizations/ghost/grid-result")

    assert resp.status_code == 404


def test_grid_result_returns_404_for_non_grid_search_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The grid-result endpoint returns 404 for non-grid jobs."""
    store.seed_job("run_job", status="success", payload_overview={"optimization_type": "run"})

    resp = opt_client.get("/optimizations/run_job/grid-result")

    assert resp.status_code == 404


def test_grid_result_returns_409_for_still_running_grid(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A still-running grid search has no result yet; the endpoint returns 409."""
    store.seed_job("gs", status="running", payload_overview={"optimization_type": "grid_search"})

    resp = opt_client.get("/optimizations/gs/grid-result")

    assert resp.status_code == 409


def test_counts_empty_store_returns_all_zeros(opt_client: TestClient) -> None:
    """An empty store returns zero counts across every status bucket."""
    resp = opt_client.get("/optimizations/counts")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    for key in ("pending", "validating", "running", "success", "failed", "cancelled"):
        assert body[key] == 0


def test_counts_reflects_mixed_statuses(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Counts reflect a mixed set of seeded statuses."""
    store.seed_job("j1", status="success")
    store.seed_job("j2", status="success")
    store.seed_job("j3", status="failed")
    store.seed_job("j4", status="pending")

    resp = opt_client.get("/optimizations/counts")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4
    assert body["success"] == 2
    assert body["failed"] == 1
    assert body["pending"] == 1
    assert body["running"] == 0
    assert body["cancelled"] == 0


def test_counts_username_filter_restricts_to_single_user(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The ``username`` filter restricts counts to that user."""
    store.seed_job("alice_job", status="success", payload_overview={"username": "alice"})
    store.seed_job("bob_job", status="failed", payload_overview={"username": "bob"})

    resp = opt_client.get("/optimizations/counts?username=alice")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["success"] == 1
    assert body["failed"] == 0


def test_sidebar_empty_store_returns_empty_items(opt_client: TestClient) -> None:
    """An empty store returns an empty sidebar."""
    resp = opt_client.get("/optimizations/sidebar")

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_sidebar_returns_minimal_fields_per_job(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Sidebar items expose only the minimal fields required by the UI."""
    store.seed_job(
        "s1",
        status="running",
        payload_overview={"name": "My Run", "optimizer_name": "gepa", "model_name": "gpt-4o"},
    )

    resp = opt_client.get("/optimizations/sidebar")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["optimization_id"] == "s1"
    assert item["status"] == "running"
    assert "name" in item
    assert "optimizer_name" in item
    assert "model_name" in item


def test_sidebar_limit_caps_returned_items(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The ``limit`` query caps the number of returned items but not ``total``."""
    for i in range(5):
        store.seed_job(f"job_{i}", status="success")

    resp = opt_client.get("/optimizations/sidebar?limit=2")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5


def test_sidebar_pinned_flag_is_propagated(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The ``pinned`` flag is propagated to sidebar items."""
    store.seed_job("pinned_job", status="success", payload_overview={"pinned": True})
    store.seed_job("normal_job", status="success", payload_overview={"pinned": False})

    resp = opt_client.get("/optimizations/sidebar")

    assert resp.status_code == 200
    items = {item["optimization_id"]: item for item in resp.json()["items"]}
    assert items["pinned_job"]["pinned"] is True
    assert items["normal_job"]["pinned"] is False


def test_sidebar_username_filter_restricts_results(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The ``username`` filter restricts sidebar items to that user."""
    store.seed_job("alice_1", status="success", payload_overview={"username": "alice"})
    store.seed_job("alice_2", status="running", payload_overview={"username": "alice"})
    store.seed_job("bob_1", status="failed", payload_overview={"username": "bob"})

    resp = opt_client.get("/optimizations/sidebar?username=alice")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    ids = {item["optimization_id"] for item in body["items"]}
    assert ids == {"alice_1", "alice_2"}


def test_dashboard_stream_returns_text_event_stream_content_type(opt_client: TestClient) -> None:
    """The dashboard stream uses an SSE content type."""
    # With no active jobs the generator emits one data event then an idle event and closes.
    with opt_client.stream("GET", "/optimizations/stream") as resp:
        assert "text/event-stream" in resp.headers["content-type"]


def test_dashboard_stream_first_event_has_active_jobs_key(opt_client: TestClient) -> None:
    """The first SSE event includes ``active_jobs`` and ``active_count`` fields."""
    with opt_client.stream("GET", "/optimizations/stream") as resp:
        for chunk in resp.iter_lines():
            if chunk.startswith("data:"):
                payload = json.loads(chunk[len("data:") :].strip())
                assert "active_jobs" in payload
                assert "active_count" in payload
                break


def test_dashboard_stream_empty_store_sends_idle_event(opt_client: TestClient) -> None:
    """An empty store emits a single data event followed by an idle event."""
    # With no active jobs the generator emits one data event then an idle event
    # and closes — no sleep between events, so this completes quickly.
    events: list[dict] = []
    idle_seen = False
    with opt_client.stream("GET", "/optimizations/stream") as resp:
        for chunk in resp.iter_lines():
            if chunk.startswith("data:"):
                events.append(json.loads(chunk[len("data:") :].strip()))
            if chunk.startswith("event: idle"):
                idle_seen = True

    # There should be exactly one data event (active_count=0) followed by idle.
    assert len(events) >= 1
    assert events[0]["active_count"] == 0
    assert idle_seen


def test_job_stream_returns_404_for_unknown_job(opt_client: TestClient) -> None:
    """A per-job stream against an unknown id returns 404."""
    resp = opt_client.get("/optimizations/no-such-id/stream")

    assert resp.status_code == 404


def test_job_stream_returns_text_event_stream_for_known_job(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A known job exposes an SSE stream with the standard content type."""
    store.seed_job("done_job", status="success")

    with opt_client.stream("GET", "/optimizations/done_job/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


def test_job_stream_first_event_contains_expected_fields(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """The first per-job SSE event carries id, status, metrics, and counters."""
    store.seed_job("done_job2", status="success")

    with opt_client.stream("GET", "/optimizations/done_job2/stream") as resp:
        for chunk in resp.iter_lines():
            if chunk.startswith("data:"):
                payload = json.loads(chunk[len("data:") :].strip())
                assert payload["optimization_id"] == "done_job2"
                assert "status" in payload
                assert "latest_metrics" in payload
                assert "log_count" in payload
                assert "progress_count" in payload
                break


def _make_run_response_dict(**kwargs) -> dict:
    """Build a minimal ``RunResponse`` dict with overrides.

    Args:
        **kwargs: Field overrides applied on top of the default response.

    Returns:
        A serialised ``RunResponse`` ready to seed into a fake job store.
    """
    base = RunResponse(
        module_name="predict",
        optimizer_name="BootstrapFewShot",
        metric_name="accuracy",
        split_counts=SplitCounts(train=7, val=1, test=2),
    ).model_dump()
    base.update(kwargs)
    return base


def _make_grid_response_dict(**kwargs) -> dict:
    """Build a grid-search response dict from the canned fixture, with overrides.

    Args:
        **kwargs: Field overrides applied on top of the default grid response.

    Returns:
        A serialised grid-search response ready to seed into a fake job store.
    """
    base = real_grid_response_dict()
    base.update(kwargs)
    return base


def _seed_job_with_dataset(
    store: _ExtendedFakeJobStore,
    optimization_id: str = "ds1",
    num_rows: int = 10,
    **extra_payload,
) -> None:
    """Seed a fake job with a synthetic dataset and column mapping.

    Args:
        store: Fake store to seed into.
        optimization_id: Job id to seed under.
        num_rows: Number of rows in the synthetic dataset.
        **extra_payload: Extra fields merged into the job payload.
    """
    dataset = [{"q": f"q{i}", "a": f"a{i}"} for i in range(num_rows)]
    payload = {
        "dataset": dataset,
        "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "shuffle": False,
        **extra_payload,
    }
    store.seed_job(optimization_id, status="success", payload=payload)


# Module-level picklable program stubs used by evaluate-examples tests.


class _SuccessProg:
    """A picklable program stub that always returns a fixed prediction."""

    def __call__(self, **kwargs):
        """Return a stub prediction object with ``answer='x'``."""

        class _Pred:
            answer = "x"

        return _Pred()


class _ErrorProg:
    """A picklable program stub that raises on every call."""

    def __call__(self, **kwargs):
        """Raise ``RuntimeError`` to simulate a program-level failure."""
        raise RuntimeError("program blew up")


def test_dataset_returns_404_when_payload_missing(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A job with no payload returns 404 from the dataset endpoint."""
    store.seed_job("nopayload", status="success", payload=None)

    resp = opt_client.get("/optimizations/nopayload/dataset")

    assert resp.status_code == 404


def test_dataset_returns_404_when_payload_not_a_dict(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A non-dict payload is treated as missing and returns 404."""
    store.seed_job("badpayload", status="success", payload="not-a-dict")

    resp = opt_client.get("/optimizations/badpayload/dataset")

    assert resp.status_code == 404


def test_dataset_returns_404_when_dataset_empty(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """An empty dataset is treated as missing and returns 404."""
    store.seed_job(
        "emptyds",
        status="success",
        payload={
            "dataset": [],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        },
    )

    resp = opt_client.get("/optimizations/emptyds/dataset")

    assert resp.status_code == 404


def test_dataset_returns_404_when_dataset_not_a_list(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """A non-list dataset is treated as missing and returns 404."""
    store.seed_job(
        "notlist",
        status="success",
        payload={
            "dataset": "not-a-list",
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
        },
    )

    resp = opt_client.get("/optimizations/notlist/dataset")

    assert resp.status_code == 404


def test_dataset_returns_500_when_column_mapping_invalid(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """An invalid stored column mapping surfaces as a 500."""
    # empty inputs violates ColumnMapping._ensure_non_empty
    store.seed_job(
        "badmap",
        status="success",
        payload={
            "dataset": [{"q": "hi"}],
            "column_mapping": {"inputs": {}, "outputs": {}},
        },
    )

    resp = opt_client.get("/optimizations/badmap/dataset")

    assert resp.status_code == 500


def test_dataset_split_fractions_fallback_when_invalid(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Invalid stored split fractions fall back to the default 70/15/15 split."""
    # invalid fractions (sum != 1.0) → falls back to SplitFractions defaults (0.7/0.15/0.15)
    store.seed_job(
        "badfrac",
        status="success",
        payload={
            "dataset": [{"q": f"q{i}", "a": f"a{i}"} for i in range(10)],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "split_fractions": {"train": 0.5, "val": 0.5, "test": 0.5},
            "shuffle": False,
        },
    )

    resp = opt_client.get("/optimizations/badfrac/dataset")

    assert resp.status_code == 200
    body = resp.json()
    # defaults: 0.7 → 7 train, floor(10*0.15)=1 val, 2 test
    assert body["split_counts"]["train"] == 7


def test_dataset_deterministic_shuffle_produces_stable_splits(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Two calls against the same job yield identical deterministic splits."""
    _seed_job_with_dataset(store, "det1", num_rows=10)

    body1 = opt_client.get("/optimizations/det1/dataset").json()
    body2 = opt_client.get("/optimizations/det1/dataset").json()

    sc = body1["split_counts"]
    assert sc["train"] + sc["val"] + sc["test"] == 10
    # Two calls with the same derived seed must yield identical splits
    assert body1["splits"]["train"] == body2["splits"]["train"]


def test_dataset_split_counts_match_stored_fractions(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Split counts match the stored fractions for a deterministic ordering."""
    _seed_job_with_dataset(store, "frac2", num_rows=20, shuffle=False)

    body = opt_client.get("/optimizations/frac2/dataset").json()

    sc = body["split_counts"]
    # floor(0.7*20)=14, floor(0.15*20)=3, remainder=3
    assert sc["train"] == 14
    assert sc["val"] == 3
    assert sc["test"] == 3


def test_evaluate_examples_404_when_payload_missing(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Evaluating examples on a job with no payload returns 404."""
    store.seed_job("nopayload_eval", status="success", payload=None)

    resp = opt_client.post(
        "/optimizations/nopayload_eval/evaluate-examples",
        json={"indices": [0], "program_type": "optimized"},
    )

    assert resp.status_code == 404


def test_evaluate_examples_400_when_metric_code_empty(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Evaluation requires metric code; an empty metric returns 400."""
    store.seed_job(
        "nometric",
        status="success",
        payload={
            "dataset": [{"q": "hi", "a": "yes"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "",
        },
    )

    resp = opt_client.post(
        "/optimizations/nometric/evaluate-examples",
        json={"indices": [0], "program_type": "optimized"},
    )

    assert resp.status_code == 400


def test_evaluate_examples_400_when_no_model_config(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Evaluation requires a model config; ``None`` returns 400."""
    store.seed_job(
        "nomodel",
        status="success",
        payload={
            "dataset": [{"q": "hi", "a": "yes"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): return 1.0",
            "model_config": None,
        },
        payload_overview={},
    )

    resp = opt_client.post(
        "/optimizations/nomodel/evaluate-examples",
        json={"indices": [0], "program_type": "optimized"},
    )

    assert resp.status_code == 400


def test_evaluate_examples_409_when_result_empty_optimized(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Evaluation against ``optimized`` requires a stored result; missing is 409."""
    store.seed_job(
        "noresult",
        status="success",
        payload={
            "dataset": [{"q": "hi", "a": "yes"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): return 1.0",
            "model_config": {"name": "gpt-4o"},
        },
        result=None,
    )

    resp = opt_client.post(
        "/optimizations/noresult/evaluate-examples",
        json={"indices": [0], "program_type": "optimized"},
    )

    assert resp.status_code == 409


def test_evaluate_examples_409_when_artifact_missing(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Evaluating ``optimized`` requires a program artifact; missing is 409."""
    result_data = _make_run_response_dict(program_artifact=None)
    store.seed_job(
        "noartifact",
        status="success",
        payload={
            "dataset": [{"q": "hi", "a": "yes"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): return 1.0",
            "model_config": {"name": "gpt-4o"},
        },
        result=result_data,
    )

    resp = opt_client.post(
        "/optimizations/noartifact/evaluate-examples",
        json={"indices": [0], "program_type": "optimized"},
    )

    assert resp.status_code == 409


def test_evaluate_examples_baseline_branch_calls_module_factory(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Evaluating ``baseline`` invokes the module factory rather than a pickled artifact."""
    # Baseline path requires invoking the module factory (no pickled artifact yet) — this checks the wiring.
    fake_prediction = MagicMock()
    fake_prediction.answer = "yes"
    fake_program = MagicMock(return_value=fake_prediction)
    fake_module_factory = MagicMock(return_value=fake_program)

    store.seed_job(
        "baseline1",
        status="success",
        payload={
            "dataset": [{"q": "hello", "a": "world"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): return 1.0",
            "model_config": {"name": "gpt-4o"},
            "signature_code": "",
            "module_name": "predict",
            "module_kwargs": {},
        },
    )

    with (
        patch("core.api.routers.optimizations.detail.build_language_model", return_value=MagicMock()),
        patch("core.api.routers.optimizations.detail.load_metric_from_code", return_value=lambda ex, pred: 1.0),
        patch("core.api.routers.optimizations.detail.load_signature_from_code", return_value=MagicMock()),
        patch(
            "core.api.routers.optimizations.detail.resolve_module_factory",
            return_value=(fake_module_factory, True),
        ),
        patch("dspy.context"),
    ):
        resp = opt_client.post(
            "/optimizations/baseline1/evaluate-examples",
            json={"indices": [0], "program_type": "baseline"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["program_type"] == "baseline"
    assert len(body["results"]) == 1


def test_evaluate_examples_metric_raises_records_zero_score(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Metric exceptions are swallowed and the row records a score of 0.0."""
    # Metric exceptions are swallowed: row score=0.0, no `error` key (program ran fine).
    artifact_b64 = base64.b64encode(pickle.dumps(_SuccessProg())).decode()
    result_data = _make_run_response_dict(
        program_artifact=ProgramArtifact(program_pickle_base64=artifact_b64).model_dump()
    )
    store.seed_job(
        "metricfail",
        status="success",
        payload={
            "dataset": [{"q": "hello", "a": "world"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): raise ValueError('oops')",
            "model_config": {"name": "gpt-4o"},
        },
        result=result_data,
    )

    def _boom(ex, pred):
        """Stand-in metric that always raises to exercise the error path."""
        raise ValueError("oops")

    with (
        patch("core.api.routers.optimizations.detail.build_language_model", return_value=MagicMock()),
        patch("core.api.routers.optimizations.detail.load_metric_from_code", return_value=_boom),
        patch("dspy.context"),
    ):
        resp = opt_client.post(
            "/optimizations/metricfail/evaluate-examples",
            json={"indices": [0], "program_type": "optimized"},
        )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert "error" not in results[0]


def test_evaluate_examples_program_raises_records_error(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Program-level exceptions populate the row's ``error`` and do not abort the batch."""
    # Program-level exceptions populate `error` and DO NOT short-circuit the batch — later rows still run.
    artifact_b64 = base64.b64encode(pickle.dumps(_ErrorProg())).decode()
    result_data = _make_run_response_dict(
        program_artifact=ProgramArtifact(program_pickle_base64=artifact_b64).model_dump()
    )
    store.seed_job(
        "progfail",
        status="success",
        payload={
            "dataset": [{"q": "hello", "a": "world"}, {"q": "world", "a": "hello"}],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "metric_code": "def metric(ex, pred): return 1.0",
            "model_config": {"name": "gpt-4o"},
        },
        result=result_data,
    )

    with (
        patch("core.api.routers.optimizations.detail.build_language_model", return_value=MagicMock()),
        patch("core.api.routers.optimizations.detail.load_metric_from_code", return_value=lambda ex, pred: 1.0),
        patch("dspy.context"),
    ):
        resp = opt_client.post(
            "/optimizations/progfail/evaluate-examples",
            json={"indices": [0, 1], "program_type": "optimized"},
        )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    for r in results:
        assert r["score"] == 0.0
        assert r["pass"] is False
        assert "program blew up" in r["error"]


def test_test_results_409_when_no_result(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Test results require a stored result; missing returns 409."""
    store.seed_job("nores", status="success", result=None)

    resp = opt_client.get("/optimizations/nores/test-results")

    assert resp.status_code == 409


def test_test_results_happy_path_returns_baseline_and_optimized(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Test results expose both baseline and optimized arrays with mapped indices."""
    result_data = _make_run_response_dict(
        baseline_test_results=[{"index": 0, "score": 0.4, "pass": False}],
        optimized_test_results=[{"index": 0, "score": 0.9, "pass": True}],
    )
    store.seed_job(
        "tr1",
        status="success",
        result=result_data,
        payload={
            "dataset": [{"q": f"q{i}", "a": f"a{i}"} for i in range(10)],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
            "shuffle": False,
        },
    )

    resp = opt_client.get("/optimizations/tr1/test-results")

    assert resp.status_code == 200
    body = resp.json()
    assert "baseline" in body
    assert "optimized" in body
    assert len(body["baseline"]) == 1
    assert len(body["optimized"]) == 1
    # both seq_idx=0 → same global index
    assert body["baseline"][0]["index"] == body["optimized"][0]["index"]


def test_test_results_index_remapping_uses_test_split(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Stored ``seq_idx`` values are remapped to the global indices of the test split."""
    # seq_idx 0 in stored results must remap to the first GLOBAL index of the test split.
    result_data = _make_run_response_dict(
        baseline_test_results=[{"index": 0, "score": 0.5, "pass": True}],
        optimized_test_results=[{"index": 0, "score": 0.8, "pass": True}],
    )
    store.seed_job(
        "remap1",
        status="success",
        result=result_data,
        payload={
            "dataset": list(range(10)),
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
            "shuffle": False,
            "seed": None,
        },
    )

    body = opt_client.get("/optimizations/remap1/test-results").json()

    # shuffle=False → ordered=[0..9], train_end=7, val_end=8, test_indices=[8,9]
    assert body["baseline"][0]["index"] == 8
    assert body["optimized"][0]["index"] == 8


def test_pair_test_results_409_when_not_grid_search(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Pair test-results require a grid-search job; non-grid jobs return 409."""
    store.seed_job("notgs", status="success", payload_overview={"optimization_type": "run"})

    resp = opt_client.get("/optimizations/notgs/pair/0/test-results")

    assert resp.status_code == 409


def test_pair_test_results_409_when_status_not_success(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Pair test-results require a successful job; running jobs return 409."""
    store.seed_job(
        "gsrunning2",
        status="running",
        payload_overview={"optimization_type": "grid_search"},
    )

    resp = opt_client.get("/optimizations/gsrunning2/pair/0/test-results")

    assert resp.status_code == 409


def test_pair_test_results_404_for_out_of_range_pair_index(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Pair test-results return 404 when the requested pair index doesn't exist."""
    result_data = _make_grid_response_dict()
    store.seed_job(
        "gsood",
        status="success",
        payload_overview={"optimization_type": "grid_search"},
        result=result_data,
        payload={
            "dataset": [{"q": f"q{i}", "a": f"a{i}"} for i in range(10)],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
            "shuffle": False,
        },
    )

    resp = opt_client.get("/optimizations/gsood/pair/99/test-results")

    assert resp.status_code == 404


def test_pair_test_results_happy_path(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Pair test-results return baseline and optimized arrays with remapped indices."""
    result_data = _make_grid_response_dict()
    store.seed_job(
        "gs_happy",
        status="success",
        payload_overview={"optimization_type": "grid_search"},
        result=result_data,
        payload={
            "dataset": [{"q": f"q{i}", "a": f"a{i}"} for i in range(10)],
            "column_mapping": {"inputs": {"question": "q"}, "outputs": {"answer": "a"}},
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
            "shuffle": False,
        },
    )

    resp = opt_client.get("/optimizations/gs_happy/pair/0/test-results")

    assert resp.status_code == 200
    body = resp.json()
    assert "baseline" in body
    assert "optimized" in body
    # fixture has 3 test results per pair; 10-row dataset → 2 test slots → all 3 remapped
    assert len(body["baseline"]) == 3
    assert len(body["optimized"]) == 3
    # shuffle=False, train_end=7, val_end=8 → test_indices=[8,9]; seq_idx=0 → global 8
    assert body["baseline"][0]["index"] == 8


def test_artifact_returns_500_when_result_blob_is_corrupted(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """A corrupted result blob fails validation and surfaces as a 500."""
    # A dict that passes isinstance(result_data, dict) but fails RunResponse.model_validate
    corrupt_result = {"garbage": True, "not_a_valid_run_response": 42}
    store.seed_job("corrupted", status="success", result=corrupt_result)

    resp = opt_client.get("/optimizations/corrupted/artifact")

    assert resp.status_code == 500


def test_delete_pair_returns_404_for_unknown_job(opt_client: TestClient) -> None:
    """Deleting a pair on an unknown job returns 404."""
    resp = opt_client.delete("/optimizations/missing/pair/0")

    assert resp.status_code == 404


def test_delete_pair_returns_404_for_non_grid_search(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Deleting a pair on a non-grid job returns 404."""
    store.seed_job(
        "run_job",
        status="success",
        payload_overview={"optimization_type": "run"},
    )

    resp = opt_client.delete("/optimizations/run_job/pair/0")

    assert resp.status_code == 404


def test_delete_pair_returns_409_when_job_not_terminal(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Deleting a pair on a still-running grid job returns 409."""
    store.seed_job(
        "gs_running",
        status="running",
        payload_overview={"optimization_type": "grid_search"},
    )

    resp = opt_client.delete("/optimizations/gs_running/pair/0")

    assert resp.status_code == 409


def test_delete_pair_returns_404_for_missing_pair_index(opt_client: TestClient, store: _ExtendedFakeJobStore) -> None:
    """Deleting a missing pair index returns 404."""
    result_data = _make_grid_response_dict()
    store.seed_job(
        "gs_missing_pair",
        status="success",
        payload_overview={"optimization_type": "grid_search"},
        result=result_data,
    )

    resp = opt_client.delete("/optimizations/gs_missing_pair/pair/99")

    assert resp.status_code == 404


def test_delete_pair_happy_path_removes_pair_and_recomputes_counts(
    opt_client: TestClient, store: _ExtendedFakeJobStore
) -> None:
    """Deleting a pair removes it and recomputes the totals in the response."""
    result_data = _make_grid_response_dict()
    original_total = result_data["total_pairs"]
    removed_index = result_data["pair_results"][0]["pair_index"]

    store.seed_job(
        "gs_delete_ok",
        status="success",
        payload_overview={"optimization_type": "grid_search"},
        result=result_data,
    )

    resp = opt_client.delete(f"/optimizations/gs_delete_ok/pair/{removed_index}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_pairs"] == original_total - 1
    remaining_indices = [pr["pair_index"] for pr in body["pair_results"]]
    assert removed_index not in remaining_indices

    stored = store.get_job("gs_delete_ok")
    stored_indices = [pr["pair_index"] for pr in stored["result"]["pair_results"]]
    assert removed_index not in stored_indices


def test_clone_payload_raises_409_when_saved_payload_no_longer_validates() -> None:
    """A stored payload that fails its request model surfaces a 409, not a 500.

    Regression: schema drift (a now-required field missing from an older saved
    payload) was raised as an opaque HTTP 500 ``internal_error`` from both the
    retry and clone routes. It is a state condition on the saved resource, so it
    must be a 4xx the caller can act on — not a server fault.
    """
    with pytest.raises(DomainError) as exc_info:
        clone_payload({"username": "alice"}, optimization_type="run", new_name="retry")

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "optimization.cannot_resubmit_payload"
