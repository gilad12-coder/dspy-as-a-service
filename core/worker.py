"""Background worker for DSPy optimization jobs.

Simple threaded worker that polls the job store for pending jobs
and processes them sequentially or with configurable concurrency.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .jobs import RemoteDBJobStore
from .logging_utils import JobLogHandler
from .models import RunRequest, RunResponse
from .registry import ServiceRegistry
from .service_gateway import DspyService

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """Background worker that processes optimization jobs.

    Uses threading to run jobs in the background while the API
    continues to accept requests.

    Args:
        job_store: Job store instance for tracking job state.
        num_workers: Number of concurrent worker threads.
        poll_interval: Seconds between polling for new jobs.
    """

    def __init__(
        self,
        job_store: RemoteDBJobStore,
        num_workers: int = 2,
        poll_interval: float = 2.0,
    ) -> None:
        self._job_store = job_store
        self._num_workers = num_workers
        self._poll_interval = poll_interval
        self._running = False
        self._threads: list[threading.Thread] = []
        self._service: Optional[DspyService] = None
        self._registry: Optional[ServiceRegistry] = None
        self._lock = threading.Lock()

        self._pending_jobs: list[str] = []
        self._processing_jobs: set[str] = set()
        self._queue_lock = threading.Lock()

    def _get_service(self) -> DspyService:
        """Get or create the DspyService instance.

        Returns:
            DspyService: Shared service instance for this worker.
        """
        if self._service is None:
            self._registry = ServiceRegistry()
            self._service = DspyService(self._registry)
        return self._service

    def submit_job(self, job_id: str, payload: RunRequest) -> None:
        """Submit a job for background processing.

        Args:
            job_id: Unique job identifier.
            payload: The optimization request payload.
        """
        self._job_store.update_job(
            job_id,
            payload=payload.model_dump(mode="json"),
        )

        with self._queue_lock:
            if job_id not in self._pending_jobs and job_id not in self._processing_jobs:
                self._pending_jobs.append(job_id)
                logger.info("Job %s added to queue", job_id)

    def _get_next_job(self) -> Optional[str]:
        """Get the next pending job from the queue.

        Returns:
            Optional[str]: Job ID if available, None otherwise.
        """
        with self._queue_lock:
            if self._pending_jobs:
                job_id = self._pending_jobs.pop(0)
                self._processing_jobs.add(job_id)
                return job_id
        return None

    def _mark_job_done(self, job_id: str) -> None:
        """Mark a job as no longer processing.

        Args:
            job_id: The job identifier to mark as done.
        """
        with self._queue_lock:
            self._processing_jobs.discard(job_id)

    def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop that processes jobs.

        Args:
            worker_id: Identifier for this worker thread.
        """
        logger.info("Worker %d started", worker_id)

        while self._running:
            job_id = self._get_next_job()

            if job_id is None:
                time.sleep(self._poll_interval)
                continue

            try:
                self._process_job(job_id)
            except Exception as exc:
                logger.exception("Worker %d failed processing job %s: %s", worker_id, job_id, exc)
            finally:
                self._mark_job_done(job_id)

        logger.info("Worker %d stopped", worker_id)

    def _process_job(self, job_id: str) -> None:
        """Process a single optimization job.

        Args:
            job_id: The job identifier to process.

        Raises:
            ValueError: If job has no payload.
        """
        logger.info("Processing job %s", job_id)

        try:
            job_data = self._job_store.get_job(job_id)
            payload_dict = job_data.get("payload")

            if not payload_dict:
                raise ValueError(f"Job {job_id} has no payload")

            payload = RunRequest.model_validate(payload_dict)

            self._job_store.update_job(
                job_id,
                status="validating",
                message="Validating payload",
            )

            service = self._get_service()
            service.validate_payload(payload)

            self._job_store.update_job(
                job_id,
                status="running",
                message="Running optimization",
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            def progress_callback(message: str, metrics: Dict[str, Any]) -> None:
                logger.debug("Job %s progress: %s %s", job_id, message, metrics)
                self._job_store.record_progress(job_id, message, metrics)

            log_handler = JobLogHandler(job_id, self._job_store)
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            tracked_loggers = [logging.getLogger("dspy")]
            previous_levels: Dict[logging.Logger, int] = {}

            for tracked in tracked_loggers:
                previous_levels[tracked] = tracked.level
                if tracked.level == 0 or tracked.level > logging.INFO:
                    tracked.setLevel(logging.INFO)
                tracked.addHandler(log_handler)

            try:
                result = service.run(
                    payload,
                    artifact_id=job_id,
                    progress_callback=progress_callback,
                )

                result_dict = result.model_dump(mode="json")
                self._job_store.update_job(
                    job_id,
                    status="success",
                    message="Optimization completed successfully",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    result=result_dict,
                )
                logger.info("Job %s completed successfully", job_id)

            finally:
                for tracked in tracked_loggers:
                    tracked.removeHandler(log_handler)
                    tracked.setLevel(previous_levels.get(tracked, tracked.level))

        except Exception as exc:
            error_message = str(exc)
            logger.exception("Job %s failed: %s", job_id, error_message)
            self._job_store.update_job(
                job_id,
                status="failed",
                message=error_message,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

    def start(self) -> None:
        """Start the background worker threads."""
        if self._running:
            return

        self._running = True
        for i in range(self._num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"dspy-worker-{i}",
            )
            thread.start()
            self._threads.append(thread)

        logger.info("Started %d background workers", self._num_workers)

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the background worker threads.

        Args:
            timeout: Maximum seconds to wait for threads to finish.
        """
        if not self._running:
            return

        self._running = False
        for thread in self._threads:
            thread.join(timeout=timeout / len(self._threads))

        self._threads.clear()
        logger.info("Stopped background workers")

    def is_running(self) -> bool:
        """Check if the worker is running.

        Returns:
            bool: True if worker threads are active.
        """
        return self._running

    def queue_size(self) -> int:
        """Get the number of pending jobs.

        Returns:
            int: Count of jobs waiting to be processed.
        """
        with self._queue_lock:
            return len(self._pending_jobs)

    def active_jobs(self) -> int:
        """Get the number of jobs currently being processed.

        Returns:
            int: Count of jobs actively running.
        """
        with self._queue_lock:
            return len(self._processing_jobs)


# Global worker instance
_worker: Optional[BackgroundWorker] = None
_worker_lock = threading.Lock()


def get_worker(job_store: RemoteDBJobStore) -> BackgroundWorker:
    """Get or create the global background worker.

    Args:
        job_store: Job store for the worker to use.

    Returns:
        BackgroundWorker: The global worker instance.
    """
    global _worker

    with _worker_lock:
        if _worker is None:
            num_workers = int(os.getenv("WORKER_CONCURRENCY", "2"))
            poll_interval = float(os.getenv("WORKER_POLL_INTERVAL", "2.0"))
            _worker = BackgroundWorker(
                job_store=job_store,
                num_workers=num_workers,
                poll_interval=poll_interval,
            )
            _worker.start()

    return _worker
