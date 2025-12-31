from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterable, List, Optional
from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .constants import (
    PAYLOAD_OVERVIEW_COMPILE_KWARGS,
    PAYLOAD_OVERVIEW_DATASET_ROWS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_SEED,
    PAYLOAD_OVERVIEW_SHUFFLE,
    PAYLOAD_OVERVIEW_SPLIT_FRACTIONS
)
from .jobs import RemoteDBJobStore
from .models import (
    HEALTH_STATUS_OK,
    HealthResponse,
    JobLogEntry,
    JobStatus,
    JobStatusResponse,
    JobSummaryResponse,
    JobSubmissionResponse,
    ProgramArtifactResponse,
    RunRequest,
    RunResponse,
)
from .registry import RegistryError, ServiceRegistry
from .service_gateway import DspyService, ServiceError
from .worker import BackgroundWorker, get_worker

logger = logging.getLogger(__name__)


def _status_to_job_status(status: str) -> JobStatus:
    """Map status string to JobStatus enum.

    Args:
        status: Status string from job store.

    Returns:
        JobStatus: Corresponding enum value.
    """
    status_map = {
        "pending": JobStatus.pending,
        "validating": JobStatus.validating,
        "running": JobStatus.running,
        "success": JobStatus.success,
        "failed": JobStatus.failed,
    }
    return status_map.get(status, JobStatus.pending)


def create_app(
    registry: ServiceRegistry | None = None,
    *,
    service: DspyService | None = None,
    service_kwargs: dict | None = None,
) -> FastAPI:
    """Create a FastAPI app wired up with the supplied registry.

    Args:
        registry: Registry instance containing user-registered assets.
        service: Optional preconstructed ``DspyService``.
        service_kwargs: Keyword arguments forwarded to ``DspyService`` when
            ``service_gateway`` is not supplied.

    Returns:
        FastAPI: Configured application instance.
    """

    registry = registry or ServiceRegistry()
    service = service or DspyService(
        registry,
        **(service_kwargs or {}),
    )

    # TODO: Configure your remote DB API endpoint via environment variables:
    #   REMOTE_DB_URL - Base URL for your remote DB API
    #   REMOTE_DB_API_KEY - Optional API key for authentication
    job_store = RemoteDBJobStore()

    # Background worker for processing jobs
    worker: Optional[BackgroundWorker] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle - start/stop worker."""
        nonlocal worker
        worker = get_worker(job_store)
        logger.info("Background worker started")
        yield
        if worker:
            worker.stop()
            logger.info("Background worker stopped")

    app = FastAPI(title="DSPy as a Service", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Transform Pydantic validation errors into a consistent response payload.

        Args:
            request: Incoming HTTP request that triggered the validation error.
            exc: Pydantic ``RequestValidationError`` raised by FastAPI.

        Returns:
            JSONResponse: Structured error response consumed by API clients.
        """

        def _format_field(loc: Iterable[Any]) -> str:
            """Render dotted field paths for structured validation errors.

            Args:
                loc: Iterable describing the nested field path reported by Pydantic.

            Returns:
                str: Dotted path (or ``body`` when unspecified) for error reporting.
            """

            parts: list[str] = []
            for entry in loc:
                if entry in {"body", "__root__"}:
                    continue
                if isinstance(entry, int):
                    if parts:
                        parts[-1] = f"{parts[-1]}[{entry}]"
                    else:
                        parts.append(f"[{entry}]")
                else:
                    parts.append(str(entry))
            return ".".join(parts) if parts else "body"

        issues = []
        for error in exc.errors():
            issues.append(
                {
                    "field": _format_field(error.get("loc", [])),
                    "message": error.get("msg", "Invalid value"),
                    "type": error.get("type", "validation_error"),
                }
            )
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "detail": issues,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def healthcheck() -> HealthResponse:
        """Expose a snapshot of registered assets.

        Returns:
            HealthResponse: Status payload used for readiness checks.
        """
        snapshot = registry.snapshot()
        logger.debug("Health check requested; registered assets: %s", snapshot)
        return HealthResponse(status=HEALTH_STATUS_OK, registered_assets=snapshot)

    @app.post("/run", response_model=JobSubmissionResponse, status_code=200)
    def submit_job(payload: RunRequest) -> JobSubmissionResponse:
        """Validate and queue a DSPy optimization request.

        Args:
            payload: Parsed request containing dataset and optimizer settings.

        Returns:
            JobSubmissionResponse: Job identifier and scheduling metadata.

        Raises:
            HTTPException: If validation fails.
        """

        try:
            service.validate_payload(payload)
        except (ServiceError, RegistryError) as exc:
            logger.warning("Payload validation failed: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = str(uuid4())

        job_store.create_job(job_id)
        job_store.set_payload_overview(
            job_id,
            {
                PAYLOAD_OVERVIEW_MODULE_NAME: payload.module_name,
                PAYLOAD_OVERVIEW_OPTIMIZER_NAME: payload.optimizer_name,
                PAYLOAD_OVERVIEW_DATASET_ROWS: len(payload.dataset),
                PAYLOAD_OVERVIEW_SPLIT_FRACTIONS: payload.split_fractions.model_dump(),
                PAYLOAD_OVERVIEW_SHUFFLE: payload.shuffle,
                PAYLOAD_OVERVIEW_SEED: payload.seed,
                PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS: dict(payload.optimizer_kwargs),
                PAYLOAD_OVERVIEW_COMPILE_KWARGS: dict(payload.compile_kwargs),
            },
        )

        current_worker = get_worker(job_store)
        current_worker.submit_job(job_id, payload)

        logger.info(
            "Enqueued job %s for module=%s optimizer=%s",
            job_id,
            payload.module_name,
            payload.optimizer_name,
        )

        return JobSubmissionResponse(
            job_id=job_id,
            status=JobStatus.pending,
            estimated_total_seconds=None,
        )

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str) -> JobStatusResponse:
        """Return the status of a queued or running job.

        Args:
            job_id: Identifier returned during submission.

        Returns:
            JobStatusResponse: Current job metadata and latest metrics.

        Raises:
            HTTPException: If the job is not found.
        """

        try:
            job_data = job_store.get_job(job_id)
        except KeyError:
            logger.warning("Job status requested for unknown job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")

        status_str = job_data.get("status", "pending")
        status = _status_to_job_status(status_str)

        def parse_timestamp(val: Any) -> Optional[datetime]:
            """Convert value to datetime, handling None and ISO strings."""
            if val is None or val == "":
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return None

        progress_events = job_store.get_progress_events(job_id)
        logs = job_store.get_logs(job_id)

        result = None
        if status == JobStatus.success:
            result_data = job_data.get("result")
            if result_data and isinstance(result_data, dict):
                result = RunResponse.model_validate(result_data)

        logger.debug("Returning status for job_id=%s state=%s", job_id, status)
        return JobStatusResponse(
            job_id=job_id,
            status=status,
            created_at=parse_timestamp(job_data.get("created_at")) or datetime.now(),
            started_at=parse_timestamp(job_data.get("started_at")),
            completed_at=parse_timestamp(job_data.get("completed_at")),
            message=job_data.get("message"),
            latest_metrics=job_data.get("latest_metrics", {}),
            progress_events=progress_events,
            logs=[JobLogEntry(**log) for log in logs],
            estimated_seconds_remaining=None,
            result=result,
        )

    @app.get("/jobs/{job_id}/summary", response_model=JobSummaryResponse)
    def get_job_summary(job_id: str) -> JobSummaryResponse:
        """Return a coarse summary of job progress and metadata.

        Args:
            job_id: Identifier for the job returned during submission.

        Returns:
            JobSummaryResponse: Aggregated job metadata and timing information.
        """

        try:
            job_data = job_store.get_job(job_id)
        except KeyError:
            logger.warning("Job summary requested for unknown job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")

        def parse_timestamp(val: Any) -> Optional[datetime]:
            """Convert value to datetime, handling None and ISO strings."""
            if val is None or val == "":
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return None

        status_str = job_data.get("status", "pending")
        status = _status_to_job_status(status_str)
        created_at = parse_timestamp(job_data.get("created_at")) or datetime.now()
        completed_at = parse_timestamp(job_data.get("completed_at"))

        end_time = completed_at or datetime.now()
        elapsed_seconds = max(0.0, (end_time - created_at).total_seconds())

        overview = job_data.get("payload_overview", {})
        if isinstance(overview, str):
            import json
            try:
                overview = json.loads(overview)
            except json.JSONDecodeError:
                overview = {}

        return JobSummaryResponse(
            job_id=job_id,
            status=status,
            message=job_data.get("message"),
            created_at=created_at,
            started_at=parse_timestamp(job_data.get("started_at")),
            completed_at=completed_at,
            elapsed_seconds=elapsed_seconds,
            estimated_seconds_remaining=None,
            module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME),
            optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
            dataset_rows=overview.get(PAYLOAD_OVERVIEW_DATASET_ROWS),
            split_fractions=overview.get(PAYLOAD_OVERVIEW_SPLIT_FRACTIONS),
            shuffle=overview.get(PAYLOAD_OVERVIEW_SHUFFLE),
            seed=overview.get(PAYLOAD_OVERVIEW_SEED),
            optimizer_kwargs=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS, {}),
            compile_kwargs=overview.get(PAYLOAD_OVERVIEW_COMPILE_KWARGS, {}),
            latest_metrics=job_data.get("latest_metrics", {}),
        )

    @app.get("/jobs/{job_id}/logs", response_model=List[JobLogEntry])
    def get_job_logs(job_id: str) -> List[JobLogEntry]:
        """Return the chronological run log for the job.

        Args:
            job_id: Identifier for the job returned during submission.

        Returns:
            List[JobLogEntry]: Ordered log entries captured during execution.
        """

        if not job_store.job_exists(job_id):
            logger.warning("Job logs requested for unknown job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")

        log_entries = job_store.get_logs(job_id)
        return [JobLogEntry(**entry) for entry in log_entries]

    @app.get("/jobs/{job_id}/artifact", response_model=ProgramArtifactResponse)
    def get_job_artifact(job_id: str) -> ProgramArtifactResponse:
        """Return the serialized artifact once the job succeeds.

        Args:
            job_id: Identifier for the job returned during submission.

        Returns:
            ProgramArtifactResponse: Serialized program artifact payload.
        """

        try:
            job_data = job_store.get_job(job_id)
        except KeyError:
            logger.warning("Artifact requested for unknown job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")

        status_str = job_data.get("status", "pending")
        status = _status_to_job_status(status_str)

        if status == JobStatus.success:
            result_data = job_data.get("result")
            if result_data and isinstance(result_data, dict):
                result = RunResponse.model_validate(result_data)
                return ProgramArtifactResponse(
                    program_artifact_path=result.program_artifact_path,
                    program_artifact=result.program_artifact,
                )
            raise HTTPException(status_code=404, detail="Job did not produce an artifact.")

        if status in {JobStatus.pending, JobStatus.validating, JobStatus.running}:
            raise HTTPException(status_code=409, detail="Job has not finished yet.")

        raise HTTPException(status_code=404, detail="Job did not produce an artifact.")

    return app
