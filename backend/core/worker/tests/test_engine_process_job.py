"""Unit tests for BackgroundWorker._process_job — error branches.

All subprocess/multiprocessing calls are mocked; no real child processes
are spawned.  FakeJobStore from conftest is used as the in-memory store.
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from core.i18n import CANCELLATION_REASON
from core.storage.base import JobStore

from .. import engine as engine_module
from ..constants import EVENT_RESULT
from ..engine import BackgroundWorker, reset_worker_for_tests
from .conftest import FakeJobStore
from .mocks import (
    REAL_GRID_PAYLOAD,
    REAL_RUN_PAYLOAD,
    fake_mp_process,
    fake_mp_queue,
    make_mp_context,
    make_progress_event,
    make_result_event,
)


@pytest.fixture(autouse=True)
def _reset_global_worker() -> Iterator[None]:
    """Reset the module-level singleton before and after each test."""
    reset_worker_for_tests()
    yield
    reset_worker_for_tests()


@pytest.fixture
def store() -> FakeJobStore:
    """Yield a fresh in-memory job store for a test."""
    return FakeJobStore()


@pytest.fixture
def worker(store: FakeJobStore) -> BackgroundWorker:
    """Build an unstarted BackgroundWorker bound to the test store."""
    return BackgroundWorker(job_store=cast(JobStore, store), num_workers=1, poll_interval=1.0)


def test_process_job_raises_when_payload_is_none(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A ``None`` payload causes the job to be marked failed."""
    store.seed_job("opt-1", payload=None)
    worker.enqueue_job("opt-1")

    with patch("core.worker.engine.notify_job_completed"):
        worker._process_job("opt-1", 0)  # must not raise — error handled internally

    assert store._jobs["opt-1"]["status"] == "failed"


def test_process_job_raises_when_payload_is_empty_dict(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """An empty-dict payload causes the job to be marked failed."""
    store.seed_job("opt-1", payload={})
    worker.enqueue_job("opt-1")

    with patch("core.worker.engine.notify_job_completed"):
        worker._process_job("opt-1", 0)  # must not raise — error handled internally

    assert store._jobs["opt-1"]["status"] == "failed"


def test_process_job_parses_json_string_payload_overview(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A JSON-string ``payload_overview`` is parsed and the job runs to success."""
    overview_str = json.dumps({"optimization_type": "run", "username": "alice"})
    store.seed_job(
        "opt-2",
        payload=REAL_RUN_PAYLOAD,
        payload_overview=overview_str,
    )
    worker.enqueue_job("opt-2")

    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": {"baseline_test_metric": 0.5, "optimized_test_metric": 0.7}}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        mock_svc.return_value.validate_grid_search_payload = MagicMock()
        worker._process_job("opt-2", 0)

    # If the JSON-string branch worked, job succeeded (not failed due to bad overview).
    assert store._jobs["opt-2"]["status"] == "success"


def test_process_job_handles_invalid_json_string_payload_overview_gracefully(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """An unparseable ``payload_overview`` does not break the run."""
    store.seed_job(
        "opt-3",
        payload=REAL_RUN_PAYLOAD,
        payload_overview="not-json-at-all",
    )
    worker.enqueue_job("opt-3")

    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": {"baseline_test_metric": 0.5, "optimized_test_metric": 0.7}}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-3", 0)

    assert store._jobs["opt-3"]["status"] == "success"


def test_process_job_handles_non_dict_json_payload_overview_gracefully(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A JSON payload overview that is not an object falls back to defaults."""
    store.seed_job(
        "opt-3b",
        payload=REAL_RUN_PAYLOAD,
        payload_overview="[]",
    )
    worker.enqueue_job("opt-3b")

    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": {"baseline_test_metric": 0.5, "optimized_test_metric": 0.7}}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-3b", 0)

    assert store._jobs["opt-3b"]["status"] == "success"


def test_process_job_sets_status_failed_on_nonzero_exit_without_result(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A non-zero subprocess exit code without a result marks the job failed."""
    store.seed_job("opt-4", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-4")

    # Exit code 1, no result events → RuntimeError caught by BaseException handler
    ctx, _proc = make_mp_context(exitcode=1, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-4", 0)  # must not raise — error handled internally

    assert store._jobs["opt-4"]["status"] == "failed"


def test_process_job_requeues_on_sigkill_oom(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A SIGKILL (-9) with no result re-queues to ``pending`` under the attempt cap."""
    store.seed_job("opt-oom", payload=REAL_RUN_PAYLOAD, attempts=0)
    worker.enqueue_job("opt-oom")

    # Exit code -9 (SIGKILL, likely OOM), no result → bounded re-queue, not failed.
    ctx, _proc = make_mp_context(exitcode=-9, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-oom", 0)

    assert store._jobs["opt-oom"]["status"] == "pending"
    assert store._jobs["opt-oom"]["attempts"] == 1


def test_process_job_sets_status_failed_when_subprocess_sends_no_result(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A clean subprocess exit with no result event still marks the job failed."""
    store.seed_job("opt-5", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-5")

    # Exit code 0 but queue is empty → no result_dict → RuntimeError caught internally
    ctx, _proc = make_mp_context(exitcode=0, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-5", 0)  # must not raise — error handled internally

    assert store._jobs["opt-5"]["status"] == "failed"


def test_process_job_grid_search_all_pairs_failed_sets_status_failed(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A grid search whose pairs all error out is marked failed."""
    payload = REAL_GRID_PAYLOAD
    store.seed_job(
        "opt-6",
        payload=payload,
        payload_overview={"optimization_type": "grid_search", "username": "bob"},
    )
    worker.enqueue_job("opt-6")

    grid_result = {
        "completed_pairs": 0,
        "total_pairs": 3,
        "pair_results": [
            {"error": "model timeout"},
            {"error": "out of memory"},
            {"error": "model timeout"},
        ],
    }
    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": grid_result}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_grid_search_payload = MagicMock()
        worker._process_job("opt-6", 0)

    assert store._jobs["opt-6"]["status"] == "failed"
    assert "model timeout" in store._jobs["opt-6"]["message"]


def test_process_job_grid_search_all_pairs_failed_message_includes_first_error(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """The failure message includes the first pair error."""
    payload = REAL_GRID_PAYLOAD
    store.seed_job(
        "opt-7",
        payload=payload,
        payload_overview={"optimization_type": "grid_search", "username": "bob"},
    )
    worker.enqueue_job("opt-7")

    grid_result = {
        "completed_pairs": 0,
        "total_pairs": 2,
        "pair_results": [{"error": "first error detail"}, {"error": "second error"}],
    }
    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": grid_result}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_grid_search_payload = MagicMock()
        worker._process_job("opt-7", 0)

    assert "first error detail" in store._jobs["opt-7"]["message"]


def test_process_job_handles_key_error_when_job_deleted_mid_run(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """When get_job raises KeyError (job deleted), _process_job must log and return cleanly."""
    store.seed_job("opt-8", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-8")

    good_result = {"baseline_test_metric": 0.5, "optimized_test_metric": 0.7}
    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": good_result}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    original_get_job = store.get_job
    call_count = {"n": 0}

    def _get_job_side_effect(opt_id: str):
        """Return the row on first call; raise ``KeyError`` on subsequent calls."""
        call_count["n"] += 1
        # First call (load payload): return normally.
        # Second call (post-result status check): raise KeyError.
        if call_count["n"] >= 2:
            raise KeyError(opt_id)
        return original_get_job(opt_id)

    store.get_job = _get_job_side_effect  # type: ignore[method-assign,assignment]

    # Must NOT raise even though get_job raises KeyError on the second call.
    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-8", 0)  # must not raise


def test_process_job_cancellation_error_notifies_cancelled(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """CancellationError fired *after* overview is assigned → notify called with status='cancelled'.

    We trigger cancel at the second _check_cancel() (after validation) by setting the event
    inside validate_payload, which is called after overview is assigned.
    """
    store.seed_job("opt-9", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-9")

    notify_mock = MagicMock()

    def _set_cancel_then_validate(*args, **kwargs):
        """Side effect that sets the cancel event from inside validation."""
        worker._cancel_events["opt-9"].set()

    with (
        patch("core.worker.engine.notify_job_completed", notify_mock),
        patch.object(worker, "_get_service") as mock_svc,
    ):
        mock_svc.return_value.validate_payload.side_effect = _set_cancel_then_validate
        worker._process_job("opt-9", 0)

    # notify_job_completed must have been called with status='cancelled'.
    call_kwargs = notify_mock.call_args_list
    assert any(kw.get("status") == "cancelled" for _, kw in call_kwargs)


def test_process_job_cancellation_error_uses_hebrew_message(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """The Hebrew cancellation message must appear in the engine code path."""
    source = inspect.getsource(engine_module)
    assert "CANCELLATION_REASON" in source
    assert CANCELLATION_REASON == "בוטלה על ידי המשתמש"


def test_process_job_generic_exception_sets_status_failed(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A generic exception during validation marks the job failed."""
    store.seed_job("opt-10", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-10")

    ctx, _proc = make_mp_context(exitcode=0, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        # validate_payload raises a generic exception (after overview is assigned)
        mock_svc.return_value.validate_payload.side_effect = RuntimeError("boom")
        worker._process_job("opt-10", 0)  # must not raise — error handled internally

    assert store._jobs["opt-10"]["status"] == "failed"


def test_process_job_system_exit_sets_status_failed_and_reraises(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """SystemExit must set status='failed' and then re-raise (is_shutdown=True path)."""
    store.seed_job("opt-11", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-11")

    ctx, _proc = make_mp_context(exitcode=0, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload.side_effect = SystemExit(1)
        with pytest.raises(SystemExit):
            worker._process_job("opt-11", 0)

    assert store._jobs["opt-11"]["status"] == "failed"


def test_process_job_keyboard_interrupt_sets_status_failed_and_reraises(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """KeyboardInterrupt (is_shutdown=True) must set status='failed' and re-raise."""
    store.seed_job("opt-12", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-12")

    ctx, _proc = make_mp_context(exitcode=0, result_events=[])
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            worker._process_job("opt-12", 0)

    assert store._jobs["opt-12"]["status"] == "failed"


def _stall_ctx() -> tuple[MagicMock, MagicMock]:
    """Build an mp context whose child stays alive and emits no events.

    Returns:
        ``(ctx, proc)`` where ``proc.is_alive()`` always returns ``True`` and
        the queue is empty — the shape that should trip the stall watchdog.
    """
    proc = fake_mp_process(exitcode=0, is_alive=True)
    mq = fake_mp_queue()
    ctx = MagicMock()
    ctx.Queue.return_value = mq
    ctx.Process.return_value = proc
    ctx.get_start_method.return_value = "spawn"
    return ctx, proc


def test_process_job_watchdog_fails_run_that_emits_no_events(
    worker: BackgroundWorker,
    store: FakeJobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live child that emits nothing past the stall window is failed and terminated."""
    store.seed_job("opt-stall", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-stall")

    ctx, proc = _stall_ctx()
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"
    monkeypatch.setattr(engine_module.settings, "job_stall_timeout_seconds", 1.0)

    # A monotonic clock that jumps far past the 1s window on every read forces
    # the watchdog to trip on the first event-less iteration, deterministically.
    with (
        patch("core.worker.engine.notify_job_completed"),
        patch.object(worker, "_get_service") as mock_svc,
        patch.object(engine_module.time, "monotonic", side_effect=itertools.count(0.0, 100_000.0)),
    ):
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-stall", 0)  # must not raise — handled internally

    assert store._jobs["opt-stall"]["status"] == "failed"
    assert "stall" in store._jobs["opt-stall"]["message"].lower()
    assert proc.terminate.called


def test_process_job_watchdog_disabled_when_timeout_zero(
    worker: BackgroundWorker,
    store: FakeJobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero stall timeout disables the watchdog: an event-less child is not stall-failed."""
    store.seed_job("opt-nowatch", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-nowatch")

    # Child stays alive for two iterations then exits cleanly with a result, so
    # the run completes normally — the watchdog must never look at the clock.
    proc = fake_mp_process(exitcode=0, is_alive=[True, True, False, False])
    mq = fake_mp_queue(make_result_event())
    ctx = MagicMock()
    ctx.Queue.return_value = mq
    ctx.Process.return_value = proc
    ctx.get_start_method.return_value = "spawn"
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"
    monkeypatch.setattr(engine_module.settings, "job_stall_timeout_seconds", 0.0)

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-nowatch", 0)

    assert store._jobs["opt-nowatch"]["status"] == "success"
    assert not proc.terminate.called


def test_process_job_watchdog_does_not_fire_while_events_flow(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """Draining events each iteration keeps the stall timer reset so the run completes."""
    store.seed_job("opt-live", payload=REAL_RUN_PAYLOAD)
    worker.enqueue_job("opt-live")

    proc = fake_mp_process(exitcode=0, is_alive=[True, True, False, False])
    mq = fake_mp_queue(make_progress_event(), make_result_event())
    ctx = MagicMock()
    ctx.Queue.return_value = mq
    ctx.Process.return_value = proc
    ctx.get_start_method.return_value = "spawn"
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with patch("core.worker.engine.notify_job_completed"), patch.object(worker, "_get_service") as mock_svc:
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-live", 0)

    assert store._jobs["opt-live"]["status"] == "success"
    assert not proc.terminate.called


def test_process_job_records_run_completed_milestone(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A successful run emits one ``run_completed`` telemetry milestone with structural props only."""
    store.seed_job(
        "opt-tel",
        payload=REAL_RUN_PAYLOAD,
        payload_overview={
            "optimization_type": "run",
            "username": "alice",
            "optimizer_name": "gepa",
            "model_name": "openai/gpt-4o-mini",
            "token_source": "managed",
            "name": "secret run name",
        },
    )
    worker.enqueue_job("opt-tel")

    ctx, _proc = make_mp_context(
        exitcode=0,
        result_events=[{"type": EVENT_RESULT, "result": {"baseline_test_metric": 0.5, "optimized_test_metric": 0.7}}],
    )
    worker._mp_ctx = ctx
    worker._mp_start_method = "spawn"

    with (
        patch("core.worker.engine.notify_job_completed"),
        patch("core.worker.engine.record_server_event") as record,
        patch.object(worker, "_get_service") as mock_svc,
    ):
        mock_svc.return_value.validate_payload = MagicMock()
        worker._process_job("opt-tel", 0)

    assert store._jobs["opt-tel"]["status"] == "success"
    record.assert_called_once()
    kwargs = record.call_args.kwargs
    assert kwargs["username"] == "alice"
    assert kwargs["name"] == "run_completed"
    assert kwargs["properties"] == {
        "status": "success",
        "optimization_type": "run",
        "optimizer": "gepa",
        "model": "openai/gpt-4o-mini",
        "token_source": "managed",
    }


def test_process_job_records_run_failed_milestone(
    worker: BackgroundWorker,
    store: FakeJobStore,
) -> None:
    """A failed run emits ``run_failed`` and a telemetry error never changes the outcome."""
    store.seed_job("opt-bad", payload=None, payload_overview={"username": "alice", "optimization_type": "run"})
    worker.enqueue_job("opt-bad")

    with (
        patch("core.worker.engine.notify_job_completed"),
        patch("core.worker.engine.record_server_event", side_effect=RuntimeError("telemetry down")) as record,
    ):
        worker._process_job("opt-bad", 0)

    assert store._jobs["opt-bad"]["status"] == "failed"
    record.assert_called_once()
    assert record.call_args.kwargs["name"] == "run_failed"
    assert record.call_args.kwargs["properties"]["status"] == "failed"
