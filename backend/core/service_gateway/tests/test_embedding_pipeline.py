"""Tests for the explore search embedding pipeline.

These tests stub out the embedder, summariser, and Session so the pure
Python logic (metadata extraction, upsert branches, backfill queueing)
is covered without a live pgvector instance.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from core.service_gateway.embedding_pipeline import core as pipeline


class _FakeEmbedder:
    """Minimal embedder double used by the tests.

    ``encode`` returns a small deterministic vector so call-count and
    argument shape can be asserted without a live embedding API.
    """

    def __init__(self, *, available: bool = True, vector: list[float] | None = None) -> None:
        """Configure availability and the canned vector returned by ``encode``."""
        self._available = available
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self.encode_calls: list[tuple[str, str | None]] = []

    def available(self) -> bool:
        """Return whether this embedder is configured as available."""
        return self._available

    def encode(self, text: str, *, task: str | None = None) -> list[float] | None:
        """Record the call and return the canned vector (or ``None`` for blanks)."""
        self.encode_calls.append((text, task))
        if not text or not text.strip():
            return None
        return list(self._vector)


class _FakeJobStore:
    """Carries just the attributes the pipeline touches."""

    def __init__(self, jobs: dict[str, dict[str, Any]] | None = None) -> None:
        """Seed the in-memory job map and an opaque engine handle."""
        self._jobs = jobs or {}
        self.engine = MagicMock(name="engine")

    def get_job(self, optimization_id: str) -> dict[str, Any]:
        """Return the seeded job dict, raising ``KeyError`` if unknown."""
        if optimization_id not in self._jobs:
            raise KeyError(optimization_id)
        return self._jobs[optimization_id]


def _success_job(**overrides: Any) -> dict[str, Any]:
    """Build a minimal successful run-job dict with optional overrides."""
    job: dict[str, Any] = {
        "status": "success",
        "payload_overview": {
            "username": "alice",
            "model_name": "openai/gpt-4o-mini",
            "optimization_type": "run",
            "name": "Sentiment task",
        },
        "payload": {
            "signature_code": "class S(dspy.Signature):\n    q = dspy.InputField()\n    a = dspy.OutputField()",
            "metric_code": "def metric(gold, pred):\n    return gold.a == pred.a",
            "column_mapping": {"inputs": {"q": "question"}, "outputs": {"a": "answer"}},
            "dataset": [{"question": "hi", "answer": "hello"}],
            "module_name": "dspy.Predict",
            "optimizer_name": "MIPROv2",
        },
        "latest_metrics": {"baseline_test_metric": 50.0, "optimized_test_metric": 72.5},
        "result": None,
    }
    job.update(overrides)
    return job


def test_extract_metadata_run_returns_overview_model() -> None:
    """A run job's winner is the configured model name on the overview."""
    job = _success_job()
    model, job_type = pipeline._extract_metadata(job)
    assert model == "openai/gpt-4o-mini"
    assert job_type == "run"


def test_extract_metadata_grid_search_uses_best_pair() -> None:
    """A grid-search job's winner is ``result.best_pair.generation_model``."""
    job = _success_job(
        payload_overview={
            "username": "alice",
            "optimization_type": "grid_search",
            "name": "Grid task",
        },
        result={"best_pair": {"generation_model": "openai/gpt-4o"}},
    )
    model, job_type = pipeline._extract_metadata(job)
    assert model == "openai/gpt-4o"
    assert job_type == "grid_search"


def test_extract_metadata_grid_search_without_best_pair_returns_none() -> None:
    """Missing ``best_pair`` falls back to ``None`` model, preserving the type."""
    job = _success_job(
        payload_overview={"username": "alice", "optimization_type": "grid_search"},
        result={},
    )
    model, job_type = pipeline._extract_metadata(job)
    assert model is None
    assert job_type == "grid_search"


def test_extract_scores_run_uses_latest_metrics() -> None:
    """A run job reads ``latest_metrics.baseline/optimized``."""
    job = _success_job()
    baseline, optimized = pipeline._extract_scores(job)
    assert baseline == 50.0
    assert optimized == 72.5


def test_extract_scores_run_falls_back_to_result_baseline() -> None:
    """A run job recovers the baseline from ``result`` when latest_metrics drops it.

    GEPA run jobs only keep ``optimized_test_metric`` in latest_metrics at
    completion; the baseline survives in ``result``. Without the fallback the
    embedded baseline is NULL and the gain sort collapses to raw score.
    """
    job = _success_job(
        latest_metrics={"optimized_test_metric": 90.0},
        result={"baseline_test_metric": 85.0, "optimized_test_metric": 90.0},
    )
    baseline, optimized = pipeline._extract_scores(job)
    assert baseline == 85.0
    assert optimized == 90.0


def test_extract_scores_grid_search_uses_best_pair() -> None:
    """A grid-search job prefers the winning pair's scores over latest_metrics."""
    job = _success_job(
        payload_overview={"username": "alice", "optimization_type": "grid_search"},
        latest_metrics={"baseline_test_metric": 10.0, "optimized_test_metric": 12.0},
        result={
            "best_pair": {
                "generation_model": "openai/gpt-4o",
                "baseline_test_metric": 60.0,
                "optimized_test_metric": 88.0,
            }
        },
    )
    baseline, optimized = pipeline._extract_scores(job)
    assert baseline == 60.0
    assert optimized == 88.0


def test_extract_scores_handles_non_numeric_metrics() -> None:
    """Non-numeric metric values coerce to ``None`` instead of raising."""
    job = _success_job(latest_metrics={"baseline_test_metric": "n/a", "optimized_test_metric": None})
    baseline, optimized = pipeline._extract_scores(job)
    assert baseline is None
    assert optimized is None


def test_extract_display_fields_prefers_overview_name() -> None:
    """``task_name`` falls back to ``payload.name`` only when overview lacks it."""
    job = _success_job()
    fields = pipeline._extract_display_fields(job)
    assert fields == {
        "task_name": "Sentiment task",
        "module_name": "dspy.Predict",
        "optimizer_name": "MIPROv2",
    }


def test_extract_display_fields_falls_back_to_payload_name() -> None:
    """A missing overview name falls back to ``payload.name``."""
    job = _success_job(
        payload_overview={"username": "alice", "optimization_type": "run"},
        payload={"name": "Fallback name", "module_name": "M", "optimizer_name": "O"},
    )
    fields = pipeline._extract_display_fields(job)
    assert fields["task_name"] == "Fallback name"


def test_embed_finished_job_skips_when_disabled() -> None:
    """``embeddings_enabled=False`` short-circuits the pipeline."""
    store = _FakeJobStore({"job-1": _success_job()})
    with patch.object(pipeline.settings, "embeddings_enabled", False):
        assert pipeline.embed_finished_job("job-1", job_store=store) is False


def test_embed_finished_job_skips_when_embedder_unavailable() -> None:
    """An unavailable embedder short-circuits before touching the DB."""
    store = _FakeJobStore({"job-1": _success_job()})
    embedder = _FakeEmbedder(available=False)
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is False
    assert embedder.encode_calls == []


def test_embed_finished_job_skips_missing_job() -> None:
    """A ``KeyError`` from ``get_job`` is swallowed; nothing is written."""
    store = _FakeJobStore({})
    embedder = _FakeEmbedder()
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
    ):
        assert pipeline.embed_finished_job("missing", job_store=store) is False


def test_embed_finished_job_skips_non_success_status() -> None:
    """A job that did not succeed is ignored — backfill won't surface it either."""
    store = _FakeJobStore({"job-1": _success_job(status="failed")})
    embedder = _FakeEmbedder()
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is False


def test_embed_finished_job_skips_when_summary_empty() -> None:
    """An empty summary (no usable text) skips the upsert entirely."""
    store = _FakeJobStore({"job-1": _success_job()})
    embedder = _FakeEmbedder()
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
        patch.object(pipeline, "summarize_task", return_value=""),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is False
    assert embedder.encode_calls == []


def test_embed_finished_job_writes_summary_row(monkeypatch) -> None:
    """A happy-path call inserts a new row with summary embedding + display fields."""
    store = _FakeJobStore({"job-1": _success_job()})
    embedder = _FakeEmbedder(vector=[0.5, 0.5, 0.5])

    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.first.return_value = None

    added: list[Any] = []
    session.add.side_effect = added.append

    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
        patch.object(pipeline, "summarize_task", return_value="A short task summary."),
        patch.object(pipeline, "Session", return_value=session),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is True

    assert len(added) == 1
    row = added[0]
    assert row.optimization_id == "job-1"
    assert row.embedding_summary == [0.5, 0.5, 0.5]
    assert row.task_name == "Sentiment task"
    assert row.module_name == "dspy.Predict"
    assert row.optimizer_name == "MIPROv2"
    assert row.winning_model == "openai/gpt-4o-mini"
    assert row.optimization_type == "run"
    assert row.summary_text == "A short task summary."
    session.commit.assert_called_once()


def test_embed_finished_job_updates_existing_row() -> None:
    """An existing row is mutated in-place rather than re-inserted."""
    store = _FakeJobStore({"job-1": _success_job()})
    embedder = _FakeEmbedder(vector=[0.9, 0.0, 0.0])

    existing = MagicMock(name="existing_row")
    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.first.return_value = existing

    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
        patch.object(pipeline, "summarize_task", return_value="Refreshed summary."),
        patch.object(pipeline, "Session", return_value=session),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is True

    session.add.assert_not_called()
    assert existing.embedding_summary == [0.9, 0.0, 0.0]
    assert existing.summary_text == "Refreshed summary."
    assert existing.updated_at.tzinfo is not None
    session.commit.assert_called_once()


def test_embed_finished_job_swallows_session_failure() -> None:
    """A DB error during commit returns ``False`` instead of raising."""
    store = _FakeJobStore({"job-1": _success_job()})
    embedder = _FakeEmbedder()

    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.first.return_value = None
    session.commit.side_effect = RuntimeError("pgvector connection lost")

    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "get_embedder", return_value=embedder),
        patch.object(pipeline, "summarize_task", return_value="A task summary."),
        patch.object(pipeline, "Session", return_value=session),
    ):
        assert pipeline.embed_finished_job("job-1", job_store=store) is False


def test_fetch_missing_embedding_ids_returns_ids() -> None:
    """The scan unwraps ``optimization_id`` values from the LEFT JOIN."""
    store = _FakeJobStore()
    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.execute.return_value.mappings.return_value.all.return_value = [
        {"optimization_id": "a"},
        {"optimization_id": "b"},
    ]
    with patch.object(pipeline, "Session", return_value=session):
        ids = pipeline._fetch_missing_embedding_ids(store)
    assert ids == ["a", "b"]


def test_fetch_missing_embedding_ids_swallows_db_failure() -> None:
    """A SQL-side failure degrades to an empty list."""
    store = _FakeJobStore()
    session = MagicMock(name="session")
    session.__enter__ = MagicMock(side_effect=RuntimeError("DB down"))
    with patch.object(pipeline, "Session", return_value=session):
        assert pipeline._fetch_missing_embedding_ids(store) == []


def test_backfill_returns_zero_when_disabled() -> None:
    """``embeddings_enabled=False`` skips even the scan."""
    store = _FakeJobStore()
    with patch.object(pipeline.settings, "embeddings_enabled", False):
        assert pipeline.backfill_missing_embeddings(store) == 0


def test_backfill_returns_zero_when_nothing_missing() -> None:
    """When the scan returns no IDs, no thread is started and 0 is returned."""
    store = _FakeJobStore()
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "_fetch_missing_embedding_ids", return_value=[]),
        patch.object(pipeline.threading, "Thread") as thread_cls,
    ):
        assert pipeline.backfill_missing_embeddings(store) == 0
        thread_cls.assert_not_called()


def test_backfill_queues_drain_thread() -> None:
    """When IDs exist, a daemon thread is started and the count is returned."""
    store = _FakeJobStore()
    fake_thread = MagicMock(name="thread")
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "_fetch_missing_embedding_ids", return_value=["a", "b", "c"]),
        patch.object(pipeline.threading, "Thread", return_value=fake_thread) as thread_cls,
    ):
        assert pipeline.backfill_missing_embeddings(store) == 3
        thread_cls.assert_called_once()
        kwargs = thread_cls.call_args.kwargs
        assert kwargs["daemon"] is True
        assert kwargs["name"] == "embed-backfill"
        fake_thread.start.assert_called_once()


def test_drain_backfill_queue_processes_each_id() -> None:
    """The drain calls ``embed_finished_job`` for every queued id in order."""
    store = _FakeJobStore()
    calls: list[str] = []

    def _fake_embed(optimization_id: str, *, job_store: Any) -> bool:
        """Stub embedder that records each call and reports success."""
        calls.append(optimization_id)
        return True

    with patch.object(pipeline, "embed_finished_job", side_effect=_fake_embed):
        pipeline._drain_backfill_queue(store, ["a", "b", "c"])
    assert calls == ["a", "b", "c"]


def test_drain_backfill_queue_continues_after_embed_failure() -> None:
    """A raised embed_finished_job does not abort the drain."""
    store = _FakeJobStore()
    seen: list[str] = []

    def _fake_embed(optimization_id: str, *, job_store: Any) -> bool:
        """Stub embedder that raises on the middle id."""
        seen.append(optimization_id)
        if optimization_id == "b":
            raise RuntimeError("flaky LLM")
        return True

    with patch.object(pipeline, "embed_finished_job", side_effect=_fake_embed):
        pipeline._drain_backfill_queue(store, ["a", "b", "c"])
    assert seen == ["a", "b", "c"]


def test_embedding_index_sweeper_repairs_a_bounded_batch() -> None:
    """A repair pass purges orphans and drains only the configured batch."""
    store = _FakeJobStore()
    store.engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    sweeper = pipeline.EmbeddingIndexSweeper(store, interval_seconds=5.0, batch_size=2)
    with (
        patch.object(pipeline.settings, "embeddings_enabled", True),
        patch.object(pipeline, "purge_orphan_embeddings", return_value=0) as purge,
        patch.object(pipeline, "_fetch_missing_embedding_ids", return_value=["a", "b"]) as fetch,
        patch.object(pipeline, "_drain_backfill_queue", return_value=2) as drain,
    ):
        assert sweeper.sweep_once() == 2
    purge.assert_called_once_with(store)
    fetch.assert_called_once_with(store, limit=2)
    drain.assert_called_once_with(store, ["a", "b"])


def test_set_embedding_task_name_updates_row() -> None:
    """A rename propagates the new display name onto the embedding row."""
    store = _FakeJobStore()
    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    with patch.object(pipeline, "Session", return_value=session):
        pipeline.set_embedding_task_name(store, "job-1", "Renamed task")

    update = session.query.return_value.filter.return_value.update
    update.assert_called_once()
    values = update.call_args.args[0]
    assert values[pipeline.JobEmbeddingModel.task_name] == "Renamed task"
    assert values[pipeline.JobEmbeddingModel.updated_at].tzinfo is not None
    session.commit.assert_called_once()


def test_set_embedding_task_name_noop_without_engine() -> None:
    """A store with no SQLAlchemy engine (offline mode) is left untouched."""
    store = _FakeJobStore()
    store.engine = None

    with patch.object(pipeline, "Session") as session_factory:
        pipeline.set_embedding_task_name(store, "job-1", "Renamed task")

    session_factory.assert_not_called()
