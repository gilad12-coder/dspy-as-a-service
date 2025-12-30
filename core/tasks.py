"""Celery task definitions for DSPy optimization workflows."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery import current_task

from .celery_app import celery_app
from .constants import JOB_SUCCESS_MESSAGE
from .jobs import RedisJobStore
from .logging_utils import JobLogHandler
from .models import RunRequest, RunResponse
from .registry import ServiceRegistry
from .service_gateway import DspyService

logger = logging.getLogger(__name__)

# Lazily initialized service instance for the worker process
_service: Optional[DspyService] = None
_registry: Optional[ServiceRegistry] = None


def get_service() -> DspyService:
    """Get or create the DspyService instance for this worker."""
    global _service, _registry
    if _service is None:
        _registry = ServiceRegistry()
        _service = DspyService(_registry)
    return _service


def get_registry() -> ServiceRegistry:
    """Get the ServiceRegistry instance for this worker."""
    global _registry
    if _registry is None:
        get_service()
    return _registry


class TaskState:
    """Custom task states for DSPy optimization jobs."""

    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@celery_app.task(bind=True, name="core.tasks.run_optimization")
def run_optimization(
    self,
    payload_dict: Dict[str, Any],
    job_store_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute a DSPy optimization workflow as a Celery task.

    Args:
        self: Celery task instance (bound).
        payload_dict: Serialized RunRequest payload.
        job_store_config: Optional Redis connection config for job store.

    Returns:
        Dict containing the optimization result or error information.
    """
    task_id = self.request.id
    logger.info("Starting optimization task %s", task_id)

    job_store = RedisJobStore(config=job_store_config)

    try:
        payload = RunRequest.model_validate(payload_dict)

        self.update_state(
            state=TaskState.VALIDATING,
            meta={"message": "Validating payload", "progress": 0},
        )
        job_store.update_job(
            task_id,
            status=TaskState.VALIDATING,
            message="Validating payload",
        )

        service = get_service()
        service.validate_payload(payload)

        self.update_state(
            state=TaskState.RUNNING,
            meta={"message": "Running optimization", "progress": 0},
        )
        job_store.update_job(
            task_id,
            status=TaskState.RUNNING,
            message="Running optimization",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        def progress_callback(message: str, metrics: Dict[str, Any]) -> None:
            """Forward progress updates to Celery state and Redis."""
            logger.debug("Task %s progress: %s %s", task_id, message, metrics)
            self.update_state(
                state=TaskState.RUNNING,
                meta={"message": message, "metrics": metrics},
            )
            job_store.record_progress(task_id, message, metrics)

        log_handler = JobLogHandler(task_id, job_store)
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
                artifact_id=task_id,
                progress_callback=progress_callback,
            )

            # Convert result to dict for serialization
            result_dict = result.model_dump(mode="json")

            job_store.update_job(
                task_id,
                status=TaskState.SUCCESS,
                message=JOB_SUCCESS_MESSAGE,
                completed_at=datetime.now(timezone.utc).isoformat(),
                result=result_dict,
            )

            logger.info("Task %s completed successfully", task_id)
            return {
                "status": "success",
                "result": result_dict,
            }

        finally:
            for tracked in tracked_loggers:
                tracked.removeHandler(log_handler)
                tracked.setLevel(previous_levels.get(tracked, tracked.level))

    except Exception as exc:
        error_message = str(exc)
        logger.exception("Task %s failed: %s", task_id, error_message)

        job_store.update_job(
            task_id,
            status=TaskState.FAILURE,
            message=error_message,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Re-raise to let Celery handle the failure
        raise


@celery_app.task(bind=True, name="core.tasks.health_check")
def health_check(self) -> Dict[str, Any]:
    """Simple health check task to verify worker connectivity.

    Returns:
        Dict with worker status information.
    """
    registry = get_registry()
    return {
        "status": "healthy",
        "worker_id": self.request.hostname,
        "registered_assets": registry.snapshot(),
    }
