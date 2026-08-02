"""Per-job embedding pipeline backing public explore search.

Pipeline:

1. After a job finishes successfully, the worker fires
   ``embed_finished_job(optimization_id, job_store)`` on a daemon thread.
2. That task: fetches the job payload, asks an LLM for a 2-3 sentence
   task summary, embeds the summary, and upserts the vector + display
   metadata (task / module / optimizer name, baseline / optimized score,
   winning model) into ``job_embeddings``.
3. ``EmbeddingIndexSweeper`` scans for missing or stale success jobs on a
   bounded, advisory-locked loop so crashed worker threads, provider outages,
   and resumed optimizations heal without requiring a restart.

Failures never raise — embedding is isolated from job completion. Missing
embedding API credentials, LLM hiccups, or a flaky pgvector connection all
degrade to "skip this job" and the row is retried by the repair loop.

Only the ``summary`` aspect is embedded. The code / schema aspects from
the original recommendations design were dropped: the explore page's only
consumer is dashboard.py, which reads ``embedding_summary`` exclusively.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ...config import settings
from ...constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    PAYLOAD_OVERVIEW_IS_PRIVATE,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_USERNAME,
)
from ...storage.models import JobEmbeddingModel
from .embeddings import get_embedder
from .summarizer import summarize_task

logger = logging.getLogger(__name__)

_EMBEDDING_SWEEP_LOCK_KEY = 742137000003
_EMBEDDING_IN_FLIGHT_LOCK = threading.Lock()
_EMBEDDING_IN_FLIGHT: set[str] = set()


def _extract_metadata(job: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(winning_model, optimization_type)`` for a finished job.

    For a ``run`` job the winner is the single configured model. For a
    grid search the winner is ``result.best_pair.generation_model``.

    Args:
        job: The job-store record (with ``payload_overview`` and ``result``
            sub-dicts) for a finished optimization.

    Returns:
        A 2-tuple of ``(winning_model, optimization_type)`` where each
        entry may be ``None`` when the corresponding field is absent.
    """
    overview = job.get("payload_overview") or {}
    result = job.get("result") or {}
    job_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE)
    if job_type == OPTIMIZATION_TYPE_GRID_SEARCH:
        best = result.get("best_pair") if isinstance(result, dict) else None
        if isinstance(best, dict):
            return best.get("generation_model"), job_type
        return None, job_type
    return overview.get(PAYLOAD_OVERVIEW_MODEL_NAME), job_type


def _extract_scores(job: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return ``(baseline_metric, optimized_metric)`` for a finished job.

    Run jobs publish the score pair to ``latest_metrics`` while running, but
    only ``optimized_test_metric`` survives there at completion — the baseline
    lands in ``result`` — so fall back to ``result`` for either missing half.
    Grid jobs keep the winning pair under ``result.best_pair``.

    Scores are passed through verbatim on the canonical 0-100 percentage
    scale — the scale ``dspy.Evaluate`` reports and the one every job persists
    after the metric-scale normalization migration — so no rescaling happens
    here.

    A missing baseline matters: the gain sort ranks on
    ``optimized - baseline``, so a ``None`` baseline would silently collapse
    the ranking to raw optimized score and float an unimproved-but-high run
    above one that actually gained.

    Args:
        job: The job-store record for a finished optimization.

    Returns:
        A 2-tuple of ``(baseline_metric, optimized_metric)`` where each entry
        is the metric's float on the 0-100 scale, or ``None`` when the score
        is missing.
    """
    overview = job.get("payload_overview") or {}
    job_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE)
    metrics = job.get("latest_metrics") or {}
    result = job.get("result")
    result = result if isinstance(result, dict) else {}
    baseline = metrics.get("baseline_test_metric")
    optimized = metrics.get("optimized_test_metric")
    if job_type == OPTIMIZATION_TYPE_GRID_SEARCH:
        best = result.get("best_pair")
        if isinstance(best, dict):
            baseline = best.get("baseline_test_metric", baseline)
            optimized = best.get("optimized_test_metric", optimized)
    else:
        if baseline is None:
            baseline = result.get("baseline_test_metric")
        if optimized is None:
            optimized = result.get("optimized_test_metric")
    try:
        baseline_f = float(baseline) if baseline is not None else None
    except (TypeError, ValueError):
        baseline_f = None
    try:
        optimized_f = float(optimized) if optimized is not None else None
    except (TypeError, ValueError):
        optimized_f = None
    return baseline_f, optimized_f


def _extract_display_fields(job: dict[str, Any]) -> dict[str, str | None]:
    """Pull the human-readable labels shown in explore search results.

    Args:
        job: The job-store record with ``payload`` and ``payload_overview``
            sub-dicts.

    Returns:
        A dict with ``task_name``, ``module_name``, and ``optimizer_name``
        keys; values may be ``None`` when the field is absent.
    """
    payload = job.get("payload") or {}
    overview = job.get("payload_overview") or {}
    return {
        "task_name": overview.get("name") or payload.get("name"),
        "module_name": payload.get("module_name"),
        "optimizer_name": payload.get("optimizer_name"),
    }


def _claim_embedding(optimization_id: str) -> bool:
    """Claim an in-process embedding attempt for ``optimization_id``.

    Args:
        optimization_id: Job identifier to claim.

    Returns:
        ``True`` when this process did not already start the same attempt.
    """
    with _EMBEDDING_IN_FLIGHT_LOCK:
        if optimization_id in _EMBEDDING_IN_FLIGHT:
            return False
        _EMBEDDING_IN_FLIGHT.add(optimization_id)
        return True


def _release_embedding(optimization_id: str) -> None:
    """Release the in-process embedding claim for ``optimization_id``.

    Args:
        optimization_id: Job identifier whose attempt finished.
    """
    with _EMBEDDING_IN_FLIGHT_LOCK:
        _EMBEDDING_IN_FLIGHT.discard(optimization_id)


def embed_finished_job(optimization_id: str, *, job_store: Any) -> bool:
    """Compute and upsert one finished job's summary embedding.

    Duplicate attempts in the same process are coalesced so the periodic
    repair loop cannot race the worker's fire-and-forget task. Failures return
    ``False`` and remain eligible for a later repair pass.

    Args:
        optimization_id: ID of the finished job whose embedding should be
            (re)computed.
        job_store: Job-store handle used to load the payload and to open
            a SQLAlchemy session for the upsert.

    Returns:
        ``True`` when a row was written, else ``False``.
    """
    if not _claim_embedding(optimization_id):
        return False
    try:
        return _embed_finished_job_once(optimization_id, job_store=job_store)
    finally:
        _release_embedding(optimization_id)


def _embed_finished_job_once(optimization_id: str, *, job_store: Any) -> bool:
    """Compute and upsert the summary embedding for a finished job.

    Called on a daemon thread from the worker and repair loop — must never
    raise.

    Args:
        optimization_id: ID of the finished job whose embedding should be
            (re)computed.
        job_store: Job-store handle used to load the payload and to open
            a SQLAlchemy session for the upsert.

    Returns:
        True when a row was written; False when the pipeline skipped the
        job (disabled, embedder unavailable, job missing, no usable text,
        DB error). The return value drives backfill progress logging.
    """
    if not settings.embeddings_enabled:
        logger.info("Embedding skipped for %s: SEARCH_BACKEND is not 'semantic'.", optimization_id)
        return False

    embedder = get_embedder()
    if not embedder.available():
        return False

    try:
        job = job_store.get_job(optimization_id)
    except KeyError:
        logger.debug("embed_finished_job: job %s not found", optimization_id)
        return False
    except Exception as exc:
        logger.warning("embed_finished_job: could not fetch job %s: %s", optimization_id, exc)
        return False

    if job.get("status") != "success":
        return False

    payload = job.get("payload") or {}
    overview = job.get("payload_overview") or {}
    signature_code = payload.get("signature_code")
    metric_code = payload.get("metric_code")
    column_mapping = payload.get("column_mapping")
    dataset = payload.get("dataset") or []

    summary_text = summarize_task(
        signature_code=signature_code,
        metric_code=metric_code,
        column_mapping=column_mapping,
        dataset_sample=dataset,
    )

    emb_summary = embedder.encode(summary_text, task="retrieval.passage") if summary_text else None
    if emb_summary is None:
        logger.debug("embed_finished_job: no usable summary for %s, skipping", optimization_id)
        return False

    winning_model, optimization_type = _extract_metadata(job)
    baseline, optimized = _extract_scores(job)
    display = _extract_display_fields(job)
    user_id = overview.get(PAYLOAD_OVERVIEW_USERNAME)
    is_private = bool(overview.get(PAYLOAD_OVERVIEW_IS_PRIVATE, False))

    try:
        with Session(job_store.engine) as session:
            existing = (
                session.query(JobEmbeddingModel).filter(JobEmbeddingModel.optimization_id == optimization_id).first()
            )
            fields: dict[str, Any] = {
                "user_id": user_id,
                "optimization_type": optimization_type,
                "winning_model": winning_model,
                "embedding_summary": emb_summary,
                "is_private": is_private,
                "baseline_metric": baseline,
                "optimized_metric": optimized,
                "summary_text": summary_text or None,
                "task_name": display["task_name"],
                "module_name": display["module_name"],
                "optimizer_name": display["optimizer_name"],
                # Persist signature_code so identity dedup on the explore
                # page can collapse repeated submissions of the same task.
                "signature_code": signature_code,
            }
            indexed_at = datetime.now(UTC)
            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
                existing.updated_at = indexed_at
            else:
                session.add(
                    JobEmbeddingModel(
                        optimization_id=optimization_id,
                        created_at=indexed_at,
                        updated_at=indexed_at,
                        **fields,
                    )
                )
            session.commit()
    except Exception as exc:
        logger.warning("embed_finished_job upsert failed for %s: %s", optimization_id, exc)
        return False

    logger.info(
        "Embedding indexed for %s (type=%s, winner=%s, baseline=%s, optimized=%s)",
        optimization_id,
        optimization_type,
        winning_model,
        baseline,
        optimized,
    )
    return True


def set_embedding_privacy(job_store: Any, optimization_id: str, is_private: bool) -> None:
    """Sync the denormalized ``is_private`` flag on a job's embedding row.

    The public Explore corpus filters on the indexed ``job_embeddings.is_private``
    column, so flipping a job's visibility after it was embedded must update that
    row — not just ``payload_overview`` — or the change stays invisible to search
    and the explore page until the row is re-embedded. A no-op when the job has no
    embedding row yet (not-yet-embedded or non-success): the pipeline reads
    ``is_private`` from the overview at embed time and picks up the new value then.

    Args:
        job_store: Job-store exposing the SQLAlchemy ``engine``.
        optimization_id: Optimization whose embedding row should be updated.
        is_private: New visibility flag (``True`` hides it from the public corpus).
    """
    with Session(job_store.engine) as session:
        session.query(JobEmbeddingModel).filter(
            JobEmbeddingModel.optimization_id == optimization_id
        ).update(
            {
                JobEmbeddingModel.is_private: is_private,
                JobEmbeddingModel.updated_at: datetime.now(UTC),
            }
        )
        session.commit()


def set_embedding_task_name(job_store: Any, optimization_id: str, task_name: str | None) -> None:
    """Sync the denormalized display name on a job's embedding row.

    Explore search and the corpus list resolve a job's label from
    ``job_embeddings.task_name``; renaming a job after it was embedded updates
    ``payload_overview`` but leaves this snapshot stale, so the rename handler
    must propagate the new name here too. A no-op when the store exposes no
    SQLAlchemy ``engine`` (local/offline mode, which has no embedding table) or
    the job has no embedding row yet — the pipeline snapshots ``name`` from the
    overview at embed time and picks up the current value then.

    Args:
        job_store: Job-store exposing the SQLAlchemy ``engine``.
        optimization_id: Optimization whose embedding row should be updated.
        task_name: New display name to denormalize onto the embedding row.
    """
    if getattr(job_store, "engine", None) is None:
        return
    with Session(job_store.engine) as session:
        session.query(JobEmbeddingModel).filter(
            JobEmbeddingModel.optimization_id == optimization_id
        ).update(
            {
                JobEmbeddingModel.task_name: task_name,
                JobEmbeddingModel.updated_at: datetime.now(UTC),
            }
        )
        session.commit()


def _fetch_missing_embedding_ids(job_store: Any, *, limit: int | None = None) -> list[str]:
    """Return success-state job IDs with missing or stale embeddings.

    A row is stale when its source job completed after the embedding was
    written. This catches resumed optimizations whose existing embedding row
    must be refreshed rather than merely inserted once.

    Args:
        job_store: Job-store handle exposing a SQLAlchemy engine.
        limit: Optional upper bound for one repair pass.

    Returns:
        A list of optimization IDs ordered by the oldest source/index
        timestamp first.
    """
    try:
        with Session(job_store.engine) as session:
            rows = (
                session.execute(
                    text(
                        "SELECT j.optimization_id "
                        "FROM jobs j "
                        "LEFT JOIN job_embeddings e ON e.optimization_id = j.optimization_id "
                        "WHERE j.status = 'success' "
                        "AND ("
                        "e.optimization_id IS NULL "
                        "OR e.embedding_summary IS NULL "
                        "OR e.updated_at IS NULL "
                        "OR e.updated_at < COALESCE(j.completed_at, j.created_at)"
                        ") "
                        "ORDER BY COALESCE(e.updated_at, j.completed_at, j.created_at) ASC, "
                        "j.created_at ASC"
                        + (" LIMIT :limit" if limit is not None else "")
                    ),
                    {"limit": limit} if limit is not None else {},
                )
                .mappings()
                .all()
            )
            return [row["optimization_id"] for row in rows]
    except Exception as exc:
        logger.warning("Could not scan for missing embeddings: %s", exc)
        return []


def _drain_backfill_queue(job_store: Any, ids: list[str]) -> int:
    """Embed each pending job sequentially, logging progress per row.

    Sequential (not fan-out) so backfill never thunders the embedding API
    on a cold start. The summary LLM call inside ``embed_finished_job``
    is the slow leg; running it serially also keeps the LM provider's
    rate limiter happy.

    Args:
        job_store: Job-store handle forwarded to ``embed_finished_job``.
        ids: Optimization IDs to embed, in the order returned by
            ``_fetch_missing_embedding_ids``.

    Returns:
        The number of rows written successfully.
    """
    total = len(ids)
    if total == 0:
        return 0
    logger.info("Embedding backfill: starting drain of %d job(s)", total)
    ok = 0
    for idx, optimization_id in enumerate(ids, start=1):
        try:
            written = embed_finished_job(optimization_id, job_store=job_store)
        except Exception as exc:
            logger.warning("Embedding backfill: %s raised: %s", optimization_id, exc)
            written = False
        if written:
            ok += 1
        logger.info("Embedding backfill: %d/%d processed (%d written)", idx, total, ok)
    logger.info("Embedding backfill: drain complete (%d/%d written)", ok, total)
    return ok


def backfill_missing_embeddings(job_store: Any) -> int:
    """Scan for jobs missing a summary embedding and queue a drain thread.

    Returns immediately after queueing so the API lifespan does not block
    on the LLM. The drain itself runs on a single daemon thread that logs
    progress every row.

    Args:
        job_store: Job-store handle forwarded to the drain.

    Returns:
        The number of jobs queued (0 when none are missing or the scan
        failed). Logged by the caller for operator visibility.
    """
    if not settings.embeddings_enabled:
        return 0
    ids = _fetch_missing_embedding_ids(job_store)
    if not ids:
        return 0
    thread = threading.Thread(
        target=_drain_backfill_queue,
        args=(job_store, ids),
        name="embed-backfill",
        daemon=True,
    )
    thread.start()
    return len(ids)


class EmbeddingIndexSweeper:
    """Continuously repair missing and stale optimization embeddings.

    Every API and standalone-worker process runs one instance. PostgreSQL
    session-level advisory locking ensures only one replica performs a repair
    pass at a time, while the bounded batch keeps provider outages and large
    backlogs from monopolizing a process.
    """

    def __init__(
        self,
        job_store: Any,
        interval_seconds: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the repair loop.

        Args:
            job_store: Store exposing the SQLAlchemy engine and job payloads.
            interval_seconds: Optional override for the repair interval.
            batch_size: Optional maximum number of jobs per pass.
        """
        self._job_store = job_store
        self._engine = getattr(job_store, "engine", None)
        resolved_interval = (
            interval_seconds
            if interval_seconds is not None
            else settings.embedding_index_sweep_interval_seconds
        )
        resolved_batch = (
            batch_size
            if batch_size is not None
            else settings.embedding_index_sweep_batch_size
        )
        self._interval_seconds = max(5.0, float(resolved_interval))
        self._batch_size = max(1, int(resolved_batch))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the periodic repair thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="embedding-index-sweeper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the periodic repair thread and wait briefly for shutdown."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sweep_once(self) -> int:
        """Run one leader-elected repair pass.

        Returns:
            The number of embeddings written by this pass, or ``0`` when
            embeddings are disabled, the database is unavailable, or another
            replica owns the advisory lock.
        """
        if not settings.embeddings_enabled or self._engine is None:
            return 0
        try:
            if getattr(self._engine.dialect, "name", None) != "postgresql":
                return self._repair()
            with self._engine.connect() as connection:
                acquired = connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"),
                    {"key": _EMBEDDING_SWEEP_LOCK_KEY},
                ).scalar()
                if not acquired:
                    return 0
                try:
                    return self._repair()
                finally:
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:key)"),
                        {"key": _EMBEDDING_SWEEP_LOCK_KEY},
                    )
                    connection.commit()
        except Exception:
            logger.warning("Embedding index repair pass failed", exc_info=True)
            return 0

    def _repair(self) -> int:
        """Purge orphan rows and process one bounded stale-ID batch.

        Returns:
            The number of embeddings written successfully.
        """
        purge_orphan_embeddings(self._job_store)
        ids = _fetch_missing_embedding_ids(self._job_store, limit=self._batch_size)
        return _drain_backfill_queue(self._job_store, ids)

    def _run(self) -> None:
        """Run one repair immediately, then continue at the configured interval."""
        self.sweep_once()
        while not self._stop_event.wait(self._interval_seconds):
            self.sweep_once()


def start_embedding_index_sweeper(job_store: Any) -> EmbeddingIndexSweeper:
    """Start and return an embedding index sweeper for ``job_store``.

    Args:
        job_store: Store whose successful jobs should be indexed.

    Returns:
        The started sweeper; callers should invoke ``stop()`` during shutdown.
    """
    sweeper = EmbeddingIndexSweeper(job_store)
    sweeper.start()
    return sweeper


def purge_orphan_embeddings(job_store: Any) -> int:
    """Delete embedding rows whose underlying ``jobs`` row no longer exists.

    Pre-existing orphans accumulated before the deletion path cascaded into
    ``job_embeddings``. This sweep runs during startup and every repair pass,
    keeping the index honest even after a migration / restore that drops jobs
    out from under the embedding table.

    Args:
        job_store: Job-store handle exposing a SQLAlchemy engine.

    Returns:
        The number of orphan rows deleted (0 when none were found or the
        delete failed).
    """
    try:
        with Session(job_store.engine) as session:
            result = session.execute(
                text(
                    "DELETE FROM job_embeddings WHERE optimization_id IN ("
                    "SELECT je.optimization_id FROM job_embeddings je "
                    "LEFT JOIN jobs j ON j.optimization_id = je.optimization_id "
                    "WHERE j.optimization_id IS NULL"
                    ")"
                )
            )
            session.commit()
            deleted = int(result.rowcount or 0)
            if deleted:
                logger.info("Embedding sweep: removed %d orphan row(s)", deleted)
            return deleted
    except Exception as exc:
        logger.warning("Embedding orphan sweep failed: %s", exc)
        return 0
