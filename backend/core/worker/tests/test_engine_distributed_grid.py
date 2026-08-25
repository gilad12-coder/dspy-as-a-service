"""Distributed grid-search execution: fan-out, pair derivation, finalization.

Runs against the real store implementation on in-memory SQLite (the
``FakeJobStore`` used by the other engine tests lacks the pair-row surface,
which is itself the switch that keeps legacy tests on the classic path).
No subprocesses are spawned: the fan-out path returns before the spawn, and
finalization is exercised by seeding terminal children directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from core.api.routers.optimizations.lifecycle import _cancel_grid_pair_children
from core.constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_USERNAME,
    PROGRESS_GRID_PAIR_COMPLETED,
    PROGRESS_GRID_PAIR_FAILED,
)
from core.models import GridSearchRequest
from core.storage.base import JobStore
from core.storage.tests.test_remote_jobstore import SQLiteJobStore

from ..engine import GRID_ENVELOPE_PAIR_INDEX, BackgroundWorker, reset_worker_for_tests
from .mocks import REAL_GRID_PAYLOAD

PARENT_ID = "grid-parent-1"
USERNAME = "fixture-grid"


@pytest.fixture(autouse=True)
def _reset_global_worker() -> Iterator[None]:
    """Reset the module-level singleton before and after each test."""
    reset_worker_for_tests()
    yield
    reset_worker_for_tests()


@pytest.fixture
def store() -> SQLiteJobStore:
    """Return a fresh SQLite-backed real store (own in-memory engine per test)."""
    return SQLiteJobStore()


@pytest.fixture
def worker(store: SQLiteJobStore) -> BackgroundWorker:
    """Build an unstarted worker whose subprocess machinery must never fire."""
    w = BackgroundWorker(job_store=cast(JobStore, store), num_workers=1, poll_interval=1.0)
    w._mp_ctx = MagicMock()
    w._mp_ctx.Process.side_effect = AssertionError("distributed-grid tests must not spawn subprocesses")
    return w


def _grid_payload(**overrides: Any) -> dict[str, Any]:
    """Return the 2-pair grid payload dump without runner-internal keys."""
    payload = {k: v for k, v in REAL_GRID_PAYLOAD.items() if not k.startswith("_")}
    payload.update(overrides)
    return payload


def _seed_parent(store: SQLiteJobStore, payload: dict[str, Any] | None = None) -> None:
    """Create the pending grid parent row with payload and overview."""
    store.create_job(PARENT_ID, username=USERNAME)
    store.set_payload_overview(
        PARENT_ID,
        {
            PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_GRID_SEARCH,
            PAYLOAD_OVERVIEW_USERNAME: USERNAME,
        },
    )
    store.update_job(PARENT_ID, payload=payload or _grid_payload())


def _seed_distributed(store: SQLiteJobStore) -> list[str]:
    """Seed a fanned-out parent and return its two child ids."""
    _seed_parent(store)
    return store.create_grid_pair_jobs(
        PARENT_ID,
        2,
        username=USERNAME,
        payload_overview={PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_GRID_SEARCH},
    )


def _pair_result(index: int, score: float | None = 0.9, error: str | None = None) -> dict[str, Any]:
    """Build a minimal serialized PairResult."""
    return {
        "pair_index": index,
        "generation_model": "openai/gpt-4o-mini",
        "reflection_model": "openai/gpt-4o-mini",
        "baseline_test_metric": 0.5 if error is None else None,
        "optimized_test_metric": score if error is None else None,
        "error": error,
    }


_ENVELOPE = {
    "split_counts": {"train": 1, "val": 0, "test": 0},
    "metric_name": "metric",
    "module_name": "predict",
    "optimizer_name": "gepa",
}


def test_claimed_multi_pair_grid_fans_out(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """Processing a claimed multi-pair grid creates pair rows instead of running."""
    _seed_parent(store)
    claimed = store.claim_next_job("pod", lease_seconds=60.0)
    assert claimed is not None

    worker._process_job(PARENT_ID, 0)

    children = store.get_grid_pair_children(PARENT_ID)
    assert [c["pair_index"] for c in children] == [0, 1]
    assert all(c["status"] == "pending" for c in children)
    assert all(c["parent_optimization_id"] == PARENT_ID for c in children)
    parent = store.get_job(PARENT_ID)
    assert parent["status"] == "running"
    # Children are real queue rows: they claim like any job.
    first = store.claim_next_job("pod", lease_seconds=60.0)
    assert first is not None
    assert first["parent_optimization_id"] == PARENT_ID


def test_should_distribute_respects_flag_and_pair_count(
    worker: BackgroundWorker, store: SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-pair grids and a disabled flag keep the classic in-child path."""
    _seed_parent(store)
    two_pair = GridSearchRequest.model_validate(_grid_payload())
    one_pair = GridSearchRequest.model_validate(
        _grid_payload(reflection_models=[_grid_payload()["reflection_models"][0]])
    )

    assert worker._should_distribute_grid(PARENT_ID, two_pair) is True
    assert worker._should_distribute_grid(PARENT_ID, one_pair) is False

    monkeypatch.setattr("core.worker.engine.settings.grid_distributed_pairs", False)
    assert worker._should_distribute_grid(PARENT_ID, two_pair) is False
    # A parent that already fanned out stays distributed even with the flag off.
    store.create_grid_pair_jobs(PARENT_ID, 2, username=USERNAME, payload_overview={})
    assert worker._should_distribute_grid(PARENT_ID, two_pair) is True


def test_derive_pair_payload_selects_the_right_pair(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """Derivation picks generation-major pairs and carries the parent total."""
    gens = [{"name": "gen-a"}, {"name": "gen-b"}]
    refs = [{"name": "ref-x"}, {"name": "ref-y"}]
    _seed_parent(store, payload=_grid_payload(generation_models=gens, reflection_models=refs))
    store.update_job(PARENT_ID, status="running")

    derived = worker._derive_pair_payload(PARENT_ID, 2)

    assert derived is not None
    child_dict, validated = derived
    assert child_dict["generation_models"] == [{"name": "gen-b"}]
    assert child_dict["reflection_models"] == [{"name": "ref-x"}]
    assert child_dict["_parent_total_pairs"] == 4
    assert validated.generation_models[0].name == "gen-b"


def test_derive_pair_payload_refuses_non_running_parent(
    worker: BackgroundWorker, store: SQLiteJobStore
) -> None:
    """A cancelled/pending parent (or a missing one) yields no pair payload."""
    _seed_parent(store)
    assert worker._derive_pair_payload(PARENT_ID, 0) is None
    store.update_job(PARENT_ID, status="cancelled")
    assert worker._derive_pair_payload(PARENT_ID, 0) is None
    assert worker._derive_pair_payload("no-such-parent", 0) is None


def test_finalize_assembles_success_once(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """All-terminal children finalize the parent exactly once with a full result."""
    children = _seed_distributed(store)
    store.save_grid_pair_result(PARENT_ID, 0, _pair_result(0, score=0.7))
    store.save_grid_pair_result(PARENT_ID, 1, _pair_result(1, score=0.9))
    store.save_grid_pair_result(PARENT_ID, GRID_ENVELOPE_PAIR_INDEX, _ENVELOPE)
    for child_id in children:
        store.update_job(child_id, status="success")

    with (
        patch("core.worker.engine.notify_job_completed") as notify,
        patch.object(worker, "_schedule_embedding_indexing"),
    ):
        worker._maybe_finalize_grid(PARENT_ID)
        worker._maybe_finalize_grid(PARENT_ID)  # second call must be a no-op

    parent = store.get_job(PARENT_ID)
    assert parent["status"] == "success"
    result = parent["result"]
    assert result["total_pairs"] == 2
    assert result["completed_pairs"] == 2
    assert result["best_pair"]["pair_index"] == 1
    assert result["metric_name"] == "metric"
    assert result["split_counts"] == {"train": 1, "val": 0, "test": 0}
    assert notify.call_count == 1
    # The parent's pair archive is retained to seed per-pair re-runs.
    assert set(store.get_grid_pair_results(PARENT_ID)) == {0, 1, GRID_ENVELOPE_PAIR_INDEX}


def test_finalize_synthesizes_missing_pair_from_child_row(
    worker: BackgroundWorker, store: SQLiteJobStore
) -> None:
    """A child that died without recording a result becomes an errored pair."""
    children = _seed_distributed(store)
    store.save_grid_pair_result(PARENT_ID, 0, _pair_result(0, score=0.7))
    store.save_grid_pair_result(PARENT_ID, GRID_ENVELOPE_PAIR_INDEX, _ENVELOPE)
    store.update_job(children[0], status="success")
    store.update_job(children[1], status="failed", message="OOM-killed")

    with (
        patch("core.worker.engine.notify_job_completed"),
        patch.object(worker, "_schedule_embedding_indexing"),
    ):
        worker._maybe_finalize_grid(PARENT_ID)

    result = store.get_job(PARENT_ID)["result"]
    assert store.get_job(PARENT_ID)["status"] == "success"
    assert result["completed_pairs"] == 1
    assert result["failed_pairs"] == 1
    assert result["pair_results"][1]["error"] == "OOM-killed"


def test_finalize_all_failed_marks_parent_failed(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """A grid whose every pair failed completes as failed, not success."""
    children = _seed_distributed(store)
    for child_id in children:
        store.update_job(child_id, status="failed", message="boom")

    with patch("core.worker.engine.notify_job_completed") as notify:
        worker._maybe_finalize_grid(PARENT_ID)

    parent = store.get_job(PARENT_ID)
    assert parent["status"] == "failed"
    assert "All 2 model pairs failed" in parent["message"]
    assert notify.call_count == 1


def test_finalize_noops_while_a_child_still_runs(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """No assembly happens while any sibling is non-terminal."""
    children = _seed_distributed(store)
    store.update_job(children[0], status="success")
    store.update_job(children[1], status="running")

    worker._maybe_finalize_grid(PARENT_ID)

    assert store.get_job(PARENT_ID)["status"] == "running"


def test_record_pair_outcome_stores_result_and_envelope(
    worker: BackgroundWorker, store: SQLiteJobStore
) -> None:
    """A child's 1-pair grid response lands on the parent under its global index."""
    _seed_distributed(store)
    child_response = {
        "pair_results": [_pair_result(1, score=0.8)],
        "split_counts": _ENVELOPE["split_counts"],
        "metric_name": "metric",
        "module_name": "predict",
        "optimizer_name": "gepa",
    }

    worker._record_pair_outcome(PARENT_ID, child_response)

    stored = store.get_grid_pair_results(PARENT_ID)
    assert stored[1]["optimized_test_metric"] == 0.8
    assert stored[GRID_ENVELOPE_PAIR_INDEX]["metric_name"] == "metric"


def test_pair_progress_rewrite_reports_grid_wide_counters(
    worker: BackgroundWorker, store: SQLiteJobStore
) -> None:
    """Pair-terminal events carry sibling-derived counts, not the child's 1/1 view."""
    children = _seed_distributed(store)
    store.update_job(children[0], status="success")

    rewrite = worker._pair_progress_rewrite(PARENT_ID)

    completed = rewrite(PROGRESS_GRID_PAIR_COMPLETED, {"pair_index": 1})
    assert completed["completed_so_far"] == 2
    assert completed["failed_so_far"] == 0
    failed = rewrite(PROGRESS_GRID_PAIR_FAILED, {"pair_index": 1})
    assert failed["completed_so_far"] == 1
    assert failed["failed_so_far"] == 1
    untouched = rewrite("candidate", {"pair_index": 1})
    assert "completed_so_far" not in untouched


def test_stuck_grid_backstop_finalizes(worker: BackgroundWorker, store: SQLiteJobStore) -> None:
    """An idle sweep completes a grid whose last finisher died mid-handoff."""
    children = _seed_distributed(store)
    store.save_grid_pair_result(PARENT_ID, 0, _pair_result(0))
    store.save_grid_pair_result(PARENT_ID, 1, _pair_result(1))
    store.save_grid_pair_result(PARENT_ID, GRID_ENVELOPE_PAIR_INDEX, _ENVELOPE)
    for child_id in children:
        store.update_job(child_id, status="success")

    with (
        patch("core.worker.engine.notify_job_completed"),
        patch.object(worker, "_schedule_embedding_indexing"),
    ):
        worker._finalize_stuck_grids()

    assert store.get_job(PARENT_ID)["status"] == "success"


def test_cancel_fans_out_to_pair_children(store: SQLiteJobStore) -> None:
    """Cancelling a distributed parent cancels its live children only."""
    children = _seed_distributed(store)
    store.update_job(children[0], status="success")

    _cancel_grid_pair_children(store, None, PARENT_ID)

    assert store.get_job(children[0])["status"] == "success"
    assert store.get_job(children[1])["status"] == "cancelled"
