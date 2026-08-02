"""Standalone job-worker entrypoint (no HTTP API).

Runs the same claim → fork → optimize pipeline as the in-process worker,
against the shared Postgres queue, without loading the FastAPI app or its
routers. Deploy as its own service — with the API pods set to
``WORKER_ENABLED=0`` — to split job memory from request traffic:

    python worker_main.py

Scale by adding replicas: ``claim_next_job`` (``FOR UPDATE SKIP LOCKED``)
already coordinates peers, and the orphan sweeper is advisory-locked so only
one replica per tick does recovery work.
"""

from __future__ import annotations

import gc
import logging
import signal
import threading
from pathlib import Path

from dotenv import load_dotenv

from core.api.observability import (
    configure_logging,
    start_orphan_recovery_sweeper,
    start_queue_metrics_refresher,
)
from core.config import settings
from core.service_gateway.embedding_pipeline import start_embedding_index_sweeper
from core.service_gateway.service_builder import build_default_service
from core.storage import get_job_store
from core.worker.engine import get_worker

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


def run_worker() -> None:
    """Boot the worker and its sweepers, then block until SIGTERM/SIGINT."""
    configure_logging()
    job_store = get_job_store()
    job_store.recover_orphaned_jobs()
    pending_ids = job_store.recover_pending_jobs()
    worker = get_worker(
        job_store,
        service=build_default_service(),
        pending_optimization_ids=pending_ids,
    )
    sweepers = [
        start_queue_metrics_refresher(job_store),
        start_orphan_recovery_sweeper(job_store),
    ]
    if settings.embeddings_enabled:
        sweepers.append(start_embedding_index_sweeper(job_store))
    if pending_ids:
        logger.info("Re-queued %d pending jobs from previous run (local hint)", len(pending_ids))

    # Same rationale as the API lifespan: jobs fork this process, and GC
    # header writes would dirty the copy-on-write pages of every live child.
    # Freeze the fully-initialized startup heap so it stays physically shared.
    gc.collect()
    gc.freeze()

    stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        """Request a graceful stop on SIGTERM/SIGINT."""
        logger.info("Signal %d received; stopping worker", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.info("Standalone worker started (threads=%d)", settings.worker_threads)

    stop.wait()
    worker.stop()
    for sweeper in sweepers:
        sweeper.stop()
    logger.info("Standalone worker stopped")


if __name__ == "__main__":
    run_worker()
