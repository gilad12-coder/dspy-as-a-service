"""Job storage backends for DSPy optimization service.

Provides RemoteDBJobStore for persisting job state to a remote database API.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from .constants import (
    JOB_SUCCESS_MESSAGE,
    PAYLOAD_OVERVIEW_COMPILE_KWARGS,
    PAYLOAD_OVERVIEW_DATASET_ROWS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_SEED,
    PAYLOAD_OVERVIEW_SHUFFLE,
    PAYLOAD_OVERVIEW_SPLIT_FRACTIONS,
    TQDM_ELAPSED_KEY,
    TQDM_N_KEY,
    TQDM_TOTAL_KEY
)
from .models import JobLogEntry, JobStatus, RunResponse

MAX_PROGRESS_EVENTS = 5000
MAX_LOG_ENTRIES = 5000


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    """Track the lifecycle of a single optimization request."""

    job_id: str
    status: JobStatus = JobStatus.pending
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_total_seconds: Optional[float] = None
    message: Optional[str] = None
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    result: Optional[RunResponse] = None
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    payload_overview: Dict[str, Any] = field(default_factory=dict)

    def seconds_elapsed(self) -> float:
        """Return elapsed seconds since job creation."""
        end_time = self.completed_at or _utcnow()
        return max(0.0, (end_time - self.created_at).total_seconds())

    def seconds_remaining(self) -> Optional[float]:
        """Return estimated seconds remaining based on simple heuristics."""
        if self.estimated_total_seconds is None:
            return None
        elapsed = self.seconds_elapsed()
        return max(0.0, self.estimated_total_seconds - elapsed)


class JobManager:
    """In-memory registry for background optimization jobs.

    Simple in-memory job storage. Useful for development/testing.
    Data is lost on restart.
    """

    def __init__(self) -> None:
        """Initialize the job store."""
        self._records: Dict[str, JobRecord] = {}
        self._lock = Lock()
        self._max_progress_events = MAX_PROGRESS_EVENTS
        self._max_log_entries = MAX_LOG_ENTRIES

    def create_job(self, estimated_total_seconds: Optional[float]) -> JobRecord:
        """Create a new job record and return it."""
        job_id = str(uuid4())
        record = JobRecord(job_id=job_id, estimated_total_seconds=estimated_total_seconds)
        with self._lock:
            self._records[job_id] = record
        return record

    def get_job(self, job_id: str) -> JobRecord:
        """Fetch a job record by identifier."""
        with self._lock:
            return self._records[job_id]

    def mark_validating(self, job_id: str) -> None:
        """Mark the job as validating user inputs."""
        self._update_status(job_id, JobStatus.validating)

    def mark_running(self, job_id: str) -> None:
        """Mark the job as running optimization logic."""
        with self._lock:
            record = self._records[job_id]
            record.status = JobStatus.running
            record.started_at = record.started_at or _utcnow()

    def mark_succeeded(self, job_id: str, result: RunResponse) -> None:
        """Mark the job as completed successfully."""
        with self._lock:
            record = self._records[job_id]
            record.status = JobStatus.success
            record.completed_at = _utcnow()
            record.result = result
            record.latest_metrics = result.details
            record.message = JOB_SUCCESS_MESSAGE
            result.run_log = [JobLogEntry(**entry) for entry in record.logs]

    def mark_failed(self, job_id: str, message: str) -> None:
        """Mark the job as failed."""
        with self._lock:
            record = self._records[job_id]
            record.status = JobStatus.failed
            record.completed_at = _utcnow()
            record.message = message

    def record_progress(
        self,
        job_id: str,
        message: Optional[str],
        metrics: Dict[str, Any],
        *,
        update_job_message: bool = True,
    ) -> None:
        """Record intermediate progress for a running job."""
        event_payload = {
            "timestamp": _utcnow(),
            "event": message,
            "metrics": dict(metrics or {}),
        }
        with self._lock:
            record = self._records[job_id]
            if message and update_job_message:
                record.message = message
            if metrics:
                record.latest_metrics.update(metrics)
                self._update_estimate(record, metrics)
            record.progress_events.append(event_payload)
            if len(record.progress_events) > self._max_progress_events:
                record.progress_events.pop(0)

    def append_log(
        self,
        job_id: str,
        *,
        level: str,
        logger_name: str,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append a log entry for the job."""
        entry = {
            "timestamp": timestamp or _utcnow(),
            "level": level,
            "logger": logger_name,
            "message": message,
        }
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.logs.append(entry)
            if len(record.logs) > self._max_log_entries:
                record.logs.pop(0)

    def set_payload_overview(self, job_id: str, overview: Dict[str, Any]) -> None:
        """Persist lightweight metadata about the submitted payload."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.payload_overview = dict(overview or {})

    def snapshot_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """Return a chronological copy of the accumulated logs."""
        with self._lock:
            record = self._records[job_id]
            return list(record.logs)

    def build_summary(self, job_id: str) -> Dict[str, Any]:
        """Produce an aggregated summary for the requested job."""
        with self._lock:
            record = self._records[job_id]
            overview = dict(record.payload_overview)
            summary = {
                "job_id": record.job_id,
                "status": record.status,
                "message": record.message,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "elapsed_seconds": record.seconds_elapsed(),
                "estimated_seconds_remaining": record.seconds_remaining(),
                "module_name": overview.get(PAYLOAD_OVERVIEW_MODULE_NAME),
                "optimizer_name": overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME),
                "dataset_rows": overview.get(PAYLOAD_OVERVIEW_DATASET_ROWS),
                "split_fractions": overview.get(PAYLOAD_OVERVIEW_SPLIT_FRACTIONS),
                "shuffle": overview.get(PAYLOAD_OVERVIEW_SHUFFLE),
                "seed": overview.get(PAYLOAD_OVERVIEW_SEED),
                "optimizer_kwargs": overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS, {}),
                "compile_kwargs": overview.get(PAYLOAD_OVERVIEW_COMPILE_KWARGS, {}),
                "latest_metrics": dict(record.latest_metrics),
            }
            return summary

    def _update_status(self, job_id: str, status: JobStatus) -> None:
        """Set the status for a job without touching other fields."""
        with self._lock:
            record = self._records[job_id]
            record.status = status

    @staticmethod
    def _update_estimate(record: JobRecord, metrics: Dict[str, Any]) -> None:
        """Recompute the estimated duration using tqdm metrics when available."""
        total = metrics.get(TQDM_TOTAL_KEY)
        current = metrics.get(TQDM_N_KEY)
        elapsed = metrics.get(TQDM_ELAPSED_KEY)
        if not total or not current or not elapsed:
            return
        fraction = current / total
        if fraction <= 0:
            return
        record.estimated_total_seconds = elapsed / fraction


class JobStoreProtocol(Protocol):
    """Protocol defining the interface for job storage backends."""

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update job fields."""
        ...

    def record_progress(
        self, job_id: str, message: Optional[str], metrics: Dict[str, Any]
    ) -> None:
        """Record progress event."""
        ...

    def append_log(
        self,
        job_id: str,
        *,
        level: str,
        logger_name: str,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append a log entry."""
        ...

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job data."""
        ...


class RemoteDBJobStore:
    """Remote database-backed job storage via API.

    This store persists job state to a remote database service, enabling:
    - Permanent job history and auditing
    - Centralized storage across deployments
    - Custom data retention policies

    Currently uses in-memory cache until you implement the _api_request method.

    ============================================================================
    TODO: IMPLEMENTATION GUIDE
    ============================================================================

    To connect to your remote DB API, you need to implement these methods using
    your DB's `insert`, `update`, `query`, `delete` operations.

    REQUIRED DB OPERATIONS:
    ─────────────────────────────────────────────────────────────────────────────
    | Method              | DB Operation | Table                | Filter        |
    ─────────────────────────────────────────────────────────────────────────────
    | create_job()        | insert       | jobs                 | -             |
    | update_job()        | update       | jobs                 | job_id=X      |
    | get_job()           | query        | jobs                 | job_id=X      |
    | job_exists()        | query        | jobs                 | job_id=X      |
    | delete_job()        | delete       | jobs                 | job_id=X      |
    |                     | delete       | job_progress_events  | job_id=X      |
    |                     | delete       | job_logs             | job_id=X      |
    | record_progress()   | insert       | job_progress_events  | -             |
    | get_progress_events | query        | job_progress_events  | job_id=X      |
    | append_log()        | insert       | job_logs             | -             |
    | get_logs()          | query        | job_logs             | job_id=X      |
    | set_payload_overview| update       | jobs                 | job_id=X      |
    ─────────────────────────────────────────────────────────────────────────────

    EXPECTED DATA SCHEMAS:

       Job:
       {
           "job_id": "uuid-string",
           "status": "pending|validating|running|success|failed",
           "created_at": "2024-01-01T00:00:00Z",
           "started_at": "2024-01-01T00:00:00Z" | null,
           "completed_at": "2024-01-01T00:00:00Z" | null,
           "message": "string" | null,
           "latest_metrics": {...},
           "result": {...} | null,
           "payload_overview": {...},
           "payload": {...}  # Full request payload for worker
       }

       Progress Event:
       {
           "timestamp": "2024-01-01T00:00:00Z",
           "event": "string" | null,
           "metrics": {...}
       }

       Log Entry:
       {
           "timestamp": "2024-01-01T00:00:00Z",
           "level": "INFO|WARNING|ERROR",
           "logger": "logger.name",
           "message": "string"
       }

    4. Environment variables:
       - REMOTE_DB_URL: Base URL for your API (e.g., "https://api.yourdb.com")
       - REMOTE_DB_API_KEY: Optional authentication token

    5. DATABASE SCHEMA (3 tables):

       ┌─────────────────────────────────────────────────────────────────────┐
       │ TABLE: jobs                                                         │
       ├─────────────────────────────────────────────────────────────────────┤
       │ job_id                  VARCHAR(36)   PRIMARY KEY                   │
       │ status                  VARCHAR(20)   NOT NULL  -- pending/running/ │
       │                                                 -- validating/      │
       │                                                 -- success/failed   │
       │ created_at              TIMESTAMP     NOT NULL                      │
       │ started_at              TIMESTAMP     NULL                          │
       │ completed_at            TIMESTAMP     NULL                          │
       │ estimated_total_seconds FLOAT         NULL                          │
       │ message                 TEXT          NULL                          │
       │ latest_metrics          JSONB         DEFAULT '{}'                  │
       │ result                  JSONB         NULL      -- optimization     │
       │                                                 -- result when done │
       │ payload_overview        JSONB         DEFAULT '{}'                  │
       │ payload                 JSONB         NULL      -- full request     │
       │                                                 -- for worker       │
       └─────────────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────────────┐
       │ TABLE: job_progress_events                                          │
       ├─────────────────────────────────────────────────────────────────────┤
       │ id                      SERIAL        PRIMARY KEY                   │
       │ job_id                  VARCHAR(36)   NOT NULL  FK -> jobs.job_id   │
       │ timestamp               TIMESTAMP     NOT NULL                      │
       │ event                   VARCHAR(255)  NULL      -- event name       │
       │ metrics                 JSONB         DEFAULT '{}'                  │
       │                                                                     │
       │ INDEX: idx_progress_job_id ON job_id                                │
       └─────────────────────────────────────────────────────────────────────┘

       ┌─────────────────────────────────────────────────────────────────────┐
       │ TABLE: job_logs                                                     │
       ├─────────────────────────────────────────────────────────────────────┤
       │ id                      SERIAL        PRIMARY KEY                   │
       │ job_id                  VARCHAR(36)   NOT NULL  FK -> jobs.job_id   │
       │ timestamp               TIMESTAMP     NOT NULL                      │
       │ level                   VARCHAR(20)   NOT NULL  -- INFO/WARNING/    │
       │                                                 -- ERROR/DEBUG      │
       │ logger                  VARCHAR(255)  NOT NULL  -- logger name      │
       │ message                 TEXT          NOT NULL                      │
       │                                                                     │
       │ INDEX: idx_logs_job_id ON job_id                                    │
       └─────────────────────────────────────────────────────────────────────┘

       SQL CREATE STATEMENTS:

       CREATE TABLE jobs (
           job_id VARCHAR(36) PRIMARY KEY,
           status VARCHAR(20) NOT NULL DEFAULT 'pending',
           created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
           started_at TIMESTAMP,
           completed_at TIMESTAMP,
           estimated_total_seconds FLOAT,
           message TEXT,
           latest_metrics JSONB DEFAULT '{}',
           result JSONB,
           payload_overview JSONB DEFAULT '{}',
           payload JSONB
       );

       CREATE TABLE job_progress_events (
           id SERIAL PRIMARY KEY,
           job_id VARCHAR(36) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
           timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
           event VARCHAR(255),
           metrics JSONB DEFAULT '{}'
       );
       CREATE INDEX idx_progress_job_id ON job_progress_events(job_id);

       CREATE TABLE job_logs (
           id SERIAL PRIMARY KEY,
           job_id VARCHAR(36) NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
           timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
           level VARCHAR(20) NOT NULL,
           logger VARCHAR(255) NOT NULL,
           message TEXT NOT NULL
       );
       CREATE INDEX idx_logs_job_id ON job_logs(job_id);

    ============================================================================
    """

    def __init__(self, db_client: Any = None) -> None:
        """Initialize remote DB connection.

        Args:
            db_client: Your database client instance with insert/update/query/delete methods.

        TODO: Initialize your DB client here. Example:
            from your_db_module import DatabaseClient
            self.db = db_client or DatabaseClient(
                url=os.getenv("REMOTE_DB_URL"),
                api_key=os.getenv("REMOTE_DB_API_KEY"),
            )
        """
        # TODO: Replace None with your actual DB client
        # self.db = db_client or YourDatabaseClient()
        self.db = db_client  # Currently None - implement your DB client

        # In-memory cache (works without DB implementation for testing)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._progress_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._logs_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = Lock()

    def create_job(
        self,
        job_id: str,
        estimated_total_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new job record in the remote DB.

        Args:
            job_id: Unique job identifier.
            estimated_total_seconds: Optional time estimate.

        Returns:
            Dict containing the initial job state.
        """
        now = datetime.now(timezone.utc).isoformat()
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "estimated_total_seconds": estimated_total_seconds,
            "message": None,
            "latest_metrics": {},
            "result": None,
            "payload_overview": {},
        }

        # TODO: Insert into 'jobs' table
        # self.db.insert(table="jobs", data=job_data)

        with self._lock:
            self._cache[job_id] = job_data
            self._progress_cache[job_id] = []
            self._logs_cache[job_id] = []

        return job_data

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update job fields in the remote DB.

        Args:
            job_id: Job identifier.
            **kwargs: Fields to update.
        """
        # TODO: Update 'jobs' table where job_id=job_id
        # self.db.update(table="jobs", filter={"job_id": job_id}, data=kwargs)

        with self._lock:
            if job_id in self._cache:
                self._cache[job_id].update(kwargs)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job data from the remote DB.

        Args:
            job_id: Job identifier.

        Returns:
            Dict containing job state.

        Raises:
            KeyError: If job does not exist.
        """
        with self._lock:
            if job_id in self._cache:
                return dict(self._cache[job_id])

        # TODO: Query 'jobs' table where job_id=job_id
        # rows = self.db.query(table="jobs", filter={"job_id": job_id})
        # if rows:
        #     data = rows[0]
        #     with self._lock:
        #         self._cache[job_id] = data
        #     return data

        raise KeyError(f"Job '{job_id}' not found")

    def record_progress(
        self,
        job_id: str,
        message: Optional[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Record a progress event for a job.

        Args:
            job_id: Job identifier.
            message: Optional status message.
            metrics: Metric payload.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": message,
            "metrics": metrics or {},
        }

        # TODO: Insert into 'job_progress_events' table
        # self.db.insert(table="job_progress_events", data={"job_id": job_id, **event})
        #
        # TODO: Also update 'jobs' table with latest_metrics and message
        # self.db.update(table="jobs", filter={"job_id": job_id}, data={
        #     "latest_metrics": merged_metrics,
        #     "message": message
        # })

        with self._lock:
            if job_id not in self._progress_cache:
                self._progress_cache[job_id] = []
            self._progress_cache[job_id].append(event)

            if len(self._progress_cache[job_id]) > MAX_PROGRESS_EVENTS:
                self._progress_cache[job_id].pop(0)

            if metrics and job_id in self._cache:
                if "latest_metrics" not in self._cache[job_id]:
                    self._cache[job_id]["latest_metrics"] = {}
                self._cache[job_id]["latest_metrics"].update(metrics)

            if message and job_id in self._cache:
                self._cache[job_id]["message"] = message

    def get_progress_events(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all progress events for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of progress event dictionaries.
        """
        # TODO: Query 'job_progress_events' table where job_id=job_id, order by timestamp
        # return self.db.query(table="job_progress_events", filter={"job_id": job_id}, order_by="timestamp")

        with self._lock:
            return list(self._progress_cache.get(job_id, []))

    def append_log(
        self,
        job_id: str,
        *,
        level: str,
        logger_name: str,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append a log entry for the job.

        Args:
            job_id: Job identifier.
            level: Log level string.
            logger_name: Logger name.
            message: Log message.
            timestamp: Optional timestamp.
        """
        entry = {
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "level": level,
            "logger": logger_name,
            "message": message,
        }

        # TODO: Insert into 'job_logs' table
        # self.db.insert(table="job_logs", data={"job_id": job_id, **entry})

        with self._lock:
            if job_id not in self._logs_cache:
                self._logs_cache[job_id] = []
            self._logs_cache[job_id].append(entry)

            if len(self._logs_cache[job_id]) > MAX_LOG_ENTRIES:
                self._logs_cache[job_id].pop(0)

    def get_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all log entries for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of log entry dictionaries.
        """
        # TODO: Query 'job_logs' table where job_id=job_id, order by timestamp
        # return self.db.query(table="job_logs", filter={"job_id": job_id}, order_by="timestamp")

        with self._lock:
            return list(self._logs_cache.get(job_id, []))

    def set_payload_overview(self, job_id: str, overview: Dict[str, Any]) -> None:
        """Store payload overview metadata.

        Args:
            job_id: Job identifier.
            overview: Payload overview dictionary.
        """
        # TODO: Update 'jobs' table where job_id=job_id
        # self.db.update(table="jobs", filter={"job_id": job_id}, data={"payload_overview": overview})

        with self._lock:
            if job_id in self._cache:
                self._cache[job_id]["payload_overview"] = overview or {}

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists in the remote DB.

        Args:
            job_id: Job identifier.

        Returns:
            bool: True if job exists.
        """
        with self._lock:
            if job_id in self._cache:
                return True

        # TODO: Query 'jobs' table where job_id=job_id, check if any rows returned
        # rows = self.db.query(table="jobs", filter={"job_id": job_id}, limit=1)
        # if rows:
        #     return True

        return False

    def delete_job(self, job_id: str) -> None:
        """Delete all data for a job.

        Args:
            job_id: Job identifier.
        """
        # TODO: Delete from all 3 tables where job_id=job_id
        # self.db.delete(table="job_logs", filter={"job_id": job_id})
        # self.db.delete(table="job_progress_events", filter={"job_id": job_id})
        # self.db.delete(table="jobs", filter={"job_id": job_id})

        with self._lock:
            self._cache.pop(job_id, None)
            self._progress_cache.pop(job_id, None)
            self._logs_cache.pop(job_id, None)
