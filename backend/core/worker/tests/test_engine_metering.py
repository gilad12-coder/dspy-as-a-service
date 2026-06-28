"""Tests for the worker's billing hooks at run completion.

Covers ``_report_run_usage_best_effort`` (Stripe metering) and
``_debit_run_credits`` (the local credit-ledger debit). Both must never affect
job status. Metering is a no-op unless the store exposes a SQL engine, Stripe is
configured, the caller is known, and the run reported token usage; the local
debit drops the Stripe-configured requirement (the ledger is the credit source of
truth even on a key-less deploy) but otherwise gates the same way.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from core.config import settings
from core.storage import JobStore
from core.worker.engine import BackgroundWorker


class _Store:
    """Stand-in job store; ``engine`` is present only when ``engine`` is passed."""

    def __init__(self, engine: object | None = None) -> None:
        """Optionally expose a SQL engine, mirroring RemoteDBJobStore.

        Args:
            engine: Engine sentinel to expose, or ``None`` to omit the attribute
                entirely (as a legacy/in-memory store would).
        """
        if engine is not None:
            self.engine = engine
        self.updates: list[dict[str, Any]] = []

    def update_job(self, optimization_id: str, **fields: Any) -> None:
        """Capture a re-persist so the billing-outcome stamp can be asserted."""
        self.updates.append({"id": optimization_id, **fields})


class _SyncThread:
    """Thread stand-in that runs its target inline on ``start`` for assertions."""

    def __init__(self, target: Any = None, args: tuple = (), **_kwargs: Any) -> None:
        """Capture the target and its args.

        Args:
            target: Callable the real thread would run.
            args: Positional args for ``target``.
            **_kwargs: Ignored ``Thread`` kwargs (``name``, ``daemon``).
        """
        self._target = target
        self._args = args

    def start(self) -> None:
        """Invoke the target synchronously."""
        self._target(*self._args)


def _worker(store: _Store) -> BackgroundWorker:
    """Build a worker over ``store`` without starting any threads.

    Args:
        store: The stand-in store to bind.

    Returns:
        An unstarted ``BackgroundWorker``.
    """
    return BackgroundWorker(job_store=cast(JobStore, store), num_workers=1, poll_interval=1.0)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake Stripe secret key so the hook is not short-circuited."""
    monkeypatch.setattr(settings, "stripe_secret_key", SecretStr("sk_test_dummy"))


def test_hook_meters_tokens_for_successful_run(configured: None) -> None:
    """With an engine, Stripe configured, and tokens present, usage is reported."""
    engine = object()
    worker = _worker(_Store(engine=engine))
    with (
        patch("core.worker.engine.threading.Thread", _SyncThread),
        patch("core.worker.engine.StripeBillingService") as billing_cls,
    ):
        worker._report_run_usage_best_effort("u@x.com", {"total_tokens": 5000})
    billing_cls.assert_called_once_with(engine=engine)
    billing_cls.return_value.report_run_usage.assert_called_once_with("u@x.com", 5000)


def test_hook_noop_without_engine(configured: None) -> None:
    """A store without a SQL engine (legacy/in-memory) meters nothing."""
    worker = _worker(_Store(engine=None))
    with patch("core.worker.engine.threading.Thread") as thread:
        worker._report_run_usage_best_effort("u@x.com", {"total_tokens": 5000})
    thread.assert_not_called()


def test_hook_noop_when_stripe_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No engine work is dispatched when Stripe is unconfigured."""
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.threading.Thread") as thread:
        worker._report_run_usage_best_effort("u@x.com", {"total_tokens": 5000})
    thread.assert_not_called()


def test_hook_noop_without_token_usage(configured: None) -> None:
    """A run that reported no token total (None or absent) meters nothing."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.threading.Thread") as thread:
        worker._report_run_usage_best_effort("u@x.com", {"total_tokens": None})
        worker._report_run_usage_best_effort("u@x.com", {})
        worker._report_run_usage_best_effort("u@x.com", None)
    thread.assert_not_called()


def test_meter_run_usage_swallows_failures(configured: None) -> None:
    """A Stripe failure on the daemon thread never propagates to the worker."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        billing_cls.return_value.report_run_usage.side_effect = RuntimeError("stripe down")
        worker._meter_run_usage(object(), "u@x.com", 5000)  # must not raise


def test_debit_hook_charges_credits_for_successful_run() -> None:
    """With an engine, a known caller, and tokens present, the run is debited."""
    engine = object()
    worker = _worker(_Store(engine=engine))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._debit_run_credits(
            "u@x.com", {"total_tokens": 5000}, run_name="sentiment v3", model="m1"
        )
    billing_cls.assert_called_once_with(engine=engine)
    billing_cls.return_value.debit_run.assert_called_once_with(
        "u@x.com", 5000, model="m1", description="sentiment v3"
    )


def test_debit_hook_runs_without_stripe_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local debit fires even when Stripe is unconfigured (ledger is local truth)."""
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    engine = object()
    worker = _worker(_Store(engine=engine))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._debit_run_credits("u@x.com", {"total_tokens": 5000}, run_name="r", model=None)
    billing_cls.return_value.debit_run.assert_called_once()


def test_debit_hook_noop_without_engine() -> None:
    """A store without a SQL engine (legacy/in-memory) debits nothing."""
    worker = _worker(_Store(engine=None))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._debit_run_credits("u@x.com", {"total_tokens": 5000}, run_name="r", model=None)
    billing_cls.assert_not_called()


def test_debit_hook_noop_without_token_usage() -> None:
    """A run that reported no token total debits nothing."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._debit_run_credits("u@x.com", {"total_tokens": None}, run_name="r", model=None)
        worker._debit_run_credits("u@x.com", {}, run_name="r", model=None)
        worker._debit_run_credits("u@x.com", None, run_name="r", model=None)
    billing_cls.assert_not_called()


def test_debit_hook_swallows_failures() -> None:
    """A debit failure never propagates to the worker (job status is untouched)."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        billing_cls.return_value.debit_run.side_effect = RuntimeError("db down")
        worker._debit_run_credits("u@x.com", {"total_tokens": 5000}, run_name="r", model=None)


def _overview(**extra: Any) -> dict[str, Any]:
    """Build a payload overview carrying the guarantee inputs."""
    base = {"task_fingerprint": "task-1", "token_source": "managed", "name": "sentiment v3"}
    base.update(extra)
    return base


def test_guarantee_hook_adjudicates_for_successful_run() -> None:
    """With an engine, a fingerprint, and tokens, the guarantee is adjudicated."""
    engine = object()
    worker = _worker(_Store(engine=engine))
    result = {"total_tokens": 5000, "guarantee": {"basis": "test", "baseline": 0.7, "optimized": 0.7}}
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        billing_cls.return_value.adjudicate_guarantee.return_value = 5
        worker._apply_guarantee_best_effort("u@x.com", result, _overview(), "opt-1")
    billing_cls.assert_called_once_with(engine=engine)
    call = billing_cls.return_value.adjudicate_guarantee.call_args
    assert call.args[:3] == ("u@x.com", "task-1", "opt-1")
    assert call.args[3] == result["guarantee"]
    assert call.kwargs["token_source"] == "managed"
    assert call.kwargs["total_tokens"] == 5000


def test_guarantee_hook_noop_without_task_fingerprint() -> None:
    """A run with no task fingerprint can't claim the per-task slot — no-op."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._apply_guarantee_best_effort(
            "u@x.com", {"total_tokens": 5000}, _overview(task_fingerprint=None), "opt-1"
        )
    billing_cls.assert_not_called()


def test_guarantee_hook_noop_without_engine() -> None:
    """A store without a SQL engine adjudicates nothing."""
    worker = _worker(_Store(engine=None))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._apply_guarantee_best_effort("u@x.com", {"total_tokens": 5000}, _overview(), "opt-1")
    billing_cls.assert_not_called()


def test_guarantee_hook_noop_without_token_usage() -> None:
    """A run that reported no token total adjudicates nothing."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        worker._apply_guarantee_best_effort("u@x.com", {"total_tokens": None}, _overview(), "opt-1")
        worker._apply_guarantee_best_effort("u@x.com", {}, _overview(), "opt-1")
    billing_cls.assert_not_called()


def test_guarantee_hook_defaults_token_source_to_managed() -> None:
    """An overview missing token_source falls back to managed for the guarantee."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        billing_cls.return_value.adjudicate_guarantee.return_value = 0
        worker._apply_guarantee_best_effort(
            "u@x.com", {"total_tokens": 5000}, _overview(token_source=None), "opt-1"
        )
    assert billing_cls.return_value.adjudicate_guarantee.call_args.kwargs["token_source"] == "managed"


def test_guarantee_hook_swallows_failures() -> None:
    """A guarantee failure never propagates to the worker (job status untouched)."""
    worker = _worker(_Store(engine=object()))
    with patch("core.worker.engine.StripeBillingService") as billing_cls:
        billing_cls.return_value.adjudicate_guarantee.side_effect = RuntimeError("db down")
        worker._apply_guarantee_best_effort("u@x.com", {"total_tokens": 5000}, _overview(), "opt-1")


def test_stamp_billing_records_billed_outcome() -> None:
    """A billed run stamps the charged credits as the proof receipt."""
    store = _Store(engine=object())
    worker = _worker(store)
    result: dict[str, Any] = {"total_tokens": 5000}
    worker._stamp_billing_outcome("opt-1", result, billed=5, refunded=0)
    assert result["details"]["billing"] == {"outcome": "billed", "credits": 5}
    assert store.updates == [{"id": "opt-1", "result": result}]


def test_stamp_billing_refund_wins_over_bill() -> None:
    """A refund (guarantee held) frames the outcome as a free run, not a charge."""
    store = _Store(engine=object())
    worker = _worker(store)
    result: dict[str, Any] = {"total_tokens": 5000, "details": {"existing": True}}
    worker._stamp_billing_outcome("opt-1", result, billed=5, refunded=5)
    assert result["details"]["billing"] == {"outcome": "refunded", "credits": 5}
    assert result["details"]["existing"] is True


def test_stamp_billing_noop_when_nothing_charged() -> None:
    """A free-grant run that cost zero credits stamps nothing."""
    store = _Store(engine=object())
    worker = _worker(store)
    result: dict[str, Any] = {"total_tokens": 0}
    worker._stamp_billing_outcome("opt-1", result, billed=0, refunded=0)
    assert "details" not in result
    assert store.updates == []


def test_stamp_billing_skips_grid_results() -> None:
    """A grid envelope has no per-run billing receipt — skip it."""
    store = _Store(engine=object())
    worker = _worker(store)
    result: dict[str, Any] = {"pair_results": [], "total_tokens": 5000}
    worker._stamp_billing_outcome("opt-1", result, billed=5, refunded=0)
    assert "details" not in result
    assert store.updates == []


def test_stamp_billing_swallows_persist_failures() -> None:
    """A re-persist failure never propagates to the worker (job status untouched)."""
    store = _Store(engine=object())
    worker = _worker(store)
    with patch.object(store, "update_job", side_effect=RuntimeError("db down")):
        worker._stamp_billing_outcome("opt-1", {"total_tokens": 5000}, billed=5, refunded=0)
