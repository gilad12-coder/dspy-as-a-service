import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

import redis

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
        """Return elapsed seconds since job creation.

        Args:
            None.

        Returns:
            float: Elapsed seconds using completion timestamp when available.
        """

        end_time = self.completed_at or _utcnow()
        return max(0.0, (end_time - self.created_at).total_seconds())

    def seconds_remaining(self) -> Optional[float]:
        """Return estimated seconds remaining based on simple heuristics.

        Args:
            None.

        Returns:
            Optional[float]: Seconds remaining when an estimate exists.
        """

        if self.estimated_total_seconds is None:
            return None
        elapsed = self.seconds_elapsed()
        return max(0.0, self.estimated_total_seconds - elapsed)


class JobManager:
    """In-memory registry for background optimization jobs."""

    def __init__(self) -> None:
        """Initialize the job store.

        Args:
            None.

        Returns:
            None
        """

        self._records: Dict[str, JobRecord] = {}
        self._lock = Lock()
        self._max_progress_events = MAX_PROGRESS_EVENTS
        self._max_log_entries = MAX_LOG_ENTRIES

    def create_job(self, estimated_total_seconds: Optional[float]) -> JobRecord:
        """Create a new job record and return it.

        Args:
            estimated_total_seconds: Optional wall-clock estimate for completion.

        Returns:
            JobRecord: Newly tracked job.
        """

        job_id = str(uuid4())
        record = JobRecord(job_id=job_id, estimated_total_seconds=estimated_total_seconds)
        with self._lock:
            self._records[job_id] = record
        return record

    def get_job(self, job_id: str) -> JobRecord:
        """Fetch a job record by identifier.

        Args:
            job_id: Job identifier returned by the submission endpoint.

        Returns:
            JobRecord: Stored job.

        Raises:
            KeyError: If the identifier does not exist.
        """

        with self._lock:
            return self._records[job_id]

    def mark_validating(self, job_id: str) -> None:
        """Mark the job as validating user inputs.

        Args:
            job_id: Identifier to update.

        Returns:
            None
        """

        self._update_status(job_id, JobStatus.validating)

    def mark_running(self, job_id: str) -> None:
        """Mark the job as running optimization logic.

        Args:
            job_id: Identifier to update.

        Returns:
            None
        """

        with self._lock:
            record = self._records[job_id]
            record.status = JobStatus.running
            record.started_at = record.started_at or _utcnow()

    def mark_succeeded(self, job_id: str, result: RunResponse) -> None:
        """Mark the job as completed successfully.

        Args:
            job_id: Identifier to update.
            result: Final optimization response.

        Returns:
            None
        """

        with self._lock:
            record = self._records[job_id]
            record.status = JobStatus.success
            record.completed_at = _utcnow()
            record.result = result
            record.latest_metrics = result.details
            record.message = JOB_SUCCESS_MESSAGE
            result.run_log = [JobLogEntry(**entry) for entry in record.logs]

    def mark_failed(self, job_id: str, message: str) -> None:
        """Mark the job as failed.

        Args:
            job_id: Identifier to update.
            message: Human-readable error description.

        Returns:
            None
        """

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
        """Record intermediate progress for a running job.

        Args:
            job_id: Identifier to update.
            message: Optional status message describing the progress event.
            metrics: Arbitrary metric payload.
            update_job_message: Whether to update the job's top-level message.

        Returns:
            None
        """

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
        """Append a log entry for the job.

        Args:
            job_id: Identifier of the job receiving the log entry.
            level: Log level string (e.g., INFO, ERROR).
            logger_name: Fully qualified logger name of the emitter.
            message: Rendered log message.
            timestamp: Optional explicit timestamp; defaults to ``datetime.now(timezone.utc)``.

        Returns:
            None
        """

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
        """Persist lightweight metadata about the submitted payload.

        Args:
            job_id: Identifier of the job being updated.
            overview: Dictionary containing summarized request attributes.

        Returns:
            None
        """

        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.payload_overview = dict(overview or {})

    def snapshot_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """Return a chronological copy of the accumulated logs.

        Args:
            job_id: Identifier of the job to inspect.

        Returns:
            List[Dict[str, Any]]: Copy of stored log dictionaries ordered by time.
        """

        with self._lock:
            record = self._records[job_id]
            return list(record.logs)

    def build_summary(self, job_id: str) -> Dict[str, Any]:
        """Produce an aggregated summary for the requested job.

        Args:
            job_id: Identifier of the job to summarize.

        Returns:
            Dict[str, Any]: Serializable summary payload consumed by API responses.
        """

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
        """Set the status for a job without touching other fields.

        Args:
            job_id: Identifier to update.
            status: New job status.

        Returns:
            None
        """

        with self._lock:
            record = self._records[job_id]
            record.status = status

    @staticmethod
    def _update_estimate(record: JobRecord, metrics: Dict[str, Any]) -> None:
        """Recompute the estimated duration using tqdm metrics when available.

        Args:
            record: Job record to update.
            metrics: Progress metrics emitted by DSPy optimizers.

        Returns:
            None
        """

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


class RedisJobStore:
    """Redis-backed job storage for distributed task tracking.

    This store persists job state to Redis, enabling:
    - Job state persistence across restarts
    - Distributed access from multiple workers
    - Real-time progress tracking
    """

    # Redis key prefixes
    JOB_PREFIX = "dspy:job:"
    PROGRESS_PREFIX = "dspy:progress:"
    LOGS_PREFIX = "dspy:logs:"

    # Limits
    MAX_PROGRESS_EVENTS = 5000
    MAX_LOG_ENTRIES = 5000

    # Default TTL for job data (7 days)
    DEFAULT_TTL = 7 * 24 * 60 * 60

    def __init__(
        self,
        config: Optional[Dict[str, str]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Initialize Redis connection.

        Args:
            config: Optional dict with host, port, db keys.
            ttl: Time-to-live for job data in seconds.
        """
        config = config or {}
        host = config.get("host") or os.getenv("REDIS_HOST", "localhost")
        port = int(config.get("port") or os.getenv("REDIS_PORT", "6379"))
        db = int(config.get("db") or os.getenv("REDIS_DB_JOBS", "2"))

        self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._ttl = ttl or self.DEFAULT_TTL

    def _job_key(self, job_id: str) -> str:
        """Generate Redis key for job data."""
        return f"{self.JOB_PREFIX}{job_id}"

    def _progress_key(self, job_id: str) -> str:
        """Generate Redis key for progress events."""
        return f"{self.PROGRESS_PREFIX}{job_id}"

    def _logs_key(self, job_id: str) -> str:
        """Generate Redis key for log entries."""
        return f"{self.LOGS_PREFIX}{job_id}"

    def create_job(
        self,
        job_id: str,
        estimated_total_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new job record in Redis.

        Args:
            job_id: Unique job identifier (typically Celery task ID).
            estimated_total_seconds: Optional time estimate.

        Returns:
            Dict containing the initial job state.
        """
        now = datetime.now(timezone.utc).isoformat()
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "created_at": now,
            "started_at": "",
            "completed_at": "",
            "estimated_total_seconds": str(estimated_total_seconds) if estimated_total_seconds else "",
            "message": "",
            "latest_metrics": "{}",
            "result": "",
            "payload_overview": "{}",
        }
        self._redis.hset(self._job_key(job_id), mapping=job_data)
        self._redis.expire(self._job_key(job_id), self._ttl)
        return job_data

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update job fields in Redis.

        Args:
            job_id: Job identifier.
            **kwargs: Fields to update.
        """
        key = self._job_key(job_id)

        # Serialize complex types
        updates = {}
        for field_name, value in kwargs.items():
            if isinstance(value, dict):
                updates[field_name] = json.dumps(value)
            elif value is None:
                updates[field_name] = ""
            else:
                updates[field_name] = str(value) if not isinstance(value, str) else value

        if updates:
            self._redis.hset(key, mapping=updates)
            self._redis.expire(key, self._ttl)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job data from Redis.

        Args:
            job_id: Job identifier.

        Returns:
            Dict containing job state.

        Raises:
            KeyError: If job does not exist.
        """
        key = self._job_key(job_id)
        data = self._redis.hgetall(key)

        if not data:
            raise KeyError(f"Job '{job_id}' not found")

        # Deserialize JSON fields
        for field_name in ("latest_metrics", "payload_overview", "result"):
            if field_name in data and data[field_name]:
                try:
                    data[field_name] = json.loads(data[field_name])
                except json.JSONDecodeError:
                    data[field_name] = {}

        # Convert empty strings back to None
        for field_name in ("started_at", "completed_at", "message", "result"):
            if field_name in data and data[field_name] == "":
                data[field_name] = None

        return data

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

        key = self._progress_key(job_id)
        self._redis.rpush(key, json.dumps(event))
        self._redis.ltrim(key, -self.MAX_PROGRESS_EVENTS, -1)
        self._redis.expire(key, self._ttl)

        if metrics:
            job_key = self._job_key(job_id)
            existing = self._redis.hget(job_key, "latest_metrics")
            try:
                current_metrics = json.loads(existing) if existing else {}
            except json.JSONDecodeError:
                current_metrics = {}
            current_metrics.update(metrics)
            self._redis.hset(job_key, "latest_metrics", json.dumps(current_metrics))

        if message:
            self._redis.hset(self._job_key(job_id), "message", message)

    def get_progress_events(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all progress events for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of progress event dictionaries.
        """
        key = self._progress_key(job_id)
        events = self._redis.lrange(key, 0, -1)
        return [json.loads(e) for e in events]

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

        key = self._logs_key(job_id)
        self._redis.rpush(key, json.dumps(entry))
        self._redis.ltrim(key, -self.MAX_LOG_ENTRIES, -1)
        self._redis.expire(key, self._ttl)

    def get_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all log entries for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of log entry dictionaries.
        """
        key = self._logs_key(job_id)
        logs = self._redis.lrange(key, 0, -1)
        return [json.loads(entry) for entry in logs]

    def set_payload_overview(self, job_id: str, overview: Dict[str, Any]) -> None:
        """Store payload overview metadata.

        Args:
            job_id: Job identifier.
            overview: Payload overview dictionary.
        """
        self._redis.hset(
            self._job_key(job_id),
            "payload_overview",
            json.dumps(overview or {}),
        )

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists in Redis.

        Args:
            job_id: Job identifier.

        Returns:
            True if job exists.
        """
        return self._redis.exists(self._job_key(job_id)) > 0

    def delete_job(self, job_id: str) -> None:
        """Delete all data for a job.

        Args:
            job_id: Job identifier.
        """
        self._redis.delete(
            self._job_key(job_id),
            self._progress_key(job_id),
            self._logs_key(job_id),
        )
