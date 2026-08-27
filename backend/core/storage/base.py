"""JobStore Protocol defining the contract for all storage backends.

Declares the typed dictionary records returned by stores and the
``JobStore`` runtime-checkable protocol every backend must satisfy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, TypedDict, runtime_checkable


class JobRecord(TypedDict, total=False):
    """Dict representation of a persisted job returned by the store.

    Matches the shape produced by the PostgreSQL backend's ``_job_to_dict``.
    ``progress_count`` and ``log_count`` are only populated by ``list_jobs``.
    """

    optimization_id: str
    status: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    estimated_remaining_seconds: float | None
    message: str | None
    latest_metrics: dict[str, Any]
    result: dict[str, Any] | None
    payload_overview: dict[str, Any]
    payload: dict[str, Any] | None
    username: str | None
    optimization_type: str | None
    attempts: int
    code_version: str | None
    progress_count: int
    log_count: int
    stored_bytes: int
    accumulated_runtime_seconds: float
    parent_optimization_id: str | None
    pair_index: int | None


class ProgressEventRecord(TypedDict):
    """Dict representation of a progress event."""

    timestamp: str | None
    event: str | None
    metrics: dict[str, Any]


class LogEntryRecord(TypedDict):
    """Dict representation of a log entry."""

    timestamp: str | None
    level: str
    logger: str
    message: str
    pair_index: int | None


@runtime_checkable
class JobStore(Protocol):
    """Protocol that all job storage backends must satisfy."""

    def create_job(self, optimization_id: str, estimated_remaining_seconds: float | None = None) -> JobRecord:
        """Create a new job record and return its dict representation.

        Args:
            optimization_id: Unique identifier for the new job.
            estimated_remaining_seconds: Initial ETA, or ``None`` if unknown.

        Returns:
            The newly created job as a ``JobRecord`` mapping.
        """
        ...

    def update_job(self, optimization_id: str, **kwargs: Any) -> None:
        """Update fields on an existing job.

        Implementations must reject unknown fields rather than silently
        swallow them.

        Args:
            optimization_id: ID of the job to update.
            **kwargs: Fields to overwrite on the row.

        Raises:
            KeyError: When the job ID is missing.
            ValueError: When ``kwargs`` contains an unknown column name.
        """
        ...

    def get_job(self, optimization_id: str, *, include_payload: bool = True) -> JobRecord:
        """Retrieve a job by its ID.

        Args:
            optimization_id: ID of the job to fetch.
            include_payload: When ``False``, the record's ``payload`` is
                reported as ``None`` so hot polling paths skip materializing
                the (potentially multi-MB) training dataset.

        Returns:
            The matching ``JobRecord``.

        Raises:
            KeyError: When no job is filed under ``optimization_id``.
        """
        ...

    def get_job_status_fields(self, optimization_id: str) -> JobRecord:
        """Retrieve only the live-polling fields for a job.

        A narrow projection (``status`` / ``message`` / ``latest_metrics``) for
        high-frequency pollers such as the per-job SSE loop, which must not
        re-read the potentially multi-MB ``payload`` JSONB that
        :meth:`get_job` returns.

        Args:
            optimization_id: ID of the job to read.

        Returns:
            A partial ``JobRecord`` carrying ``status``, ``message`` and
            ``latest_metrics``.

        Raises:
            KeyError: When no job is filed under ``optimization_id``.
        """
        ...

    def job_exists(self, optimization_id: str) -> bool:
        """Return ``True`` when ``optimization_id`` is present in the store.

        Args:
            optimization_id: ID to check.

        Returns:
            Whether the job exists.
        """
        ...

    def delete_job(self, optimization_id: str) -> None:
        """Delete a job and all its associated data (logs, progress events).

        Missing IDs are treated as a no-op by implementations.

        Args:
            optimization_id: ID of the job to delete.
        """
        ...

    def get_jobs_status_by_ids(self, optimization_ids: list[str]) -> dict[str, str]:
        """Return a ``{id: status}`` map for the requested IDs.

        Used by the bulk-delete endpoint to partition a batch into
        deletable / non-terminal / not-found in a single round trip,
        instead of running N individual ``get_job`` calls. IDs that don't
        exist are simply absent from the returned dict.

        Args:
            optimization_ids: IDs to look up.

        Returns:
            Mapping of present IDs to their status strings.
        """
        ...

    def delete_jobs(self, optimization_ids: list[str]) -> int:
        """Hard-delete a batch of jobs and their associated rows.

        Implementations should run this in a single transaction with
        bulk ``DELETE ... WHERE id IN (...)`` queries so the cost is
        O(1) round trips instead of O(n). Duplicates and missing IDs are
        tolerated.

        Args:
            optimization_ids: IDs to remove.

        Returns:
            The number of job rows actually removed.
        """
        ...

    def record_progress(self, optimization_id: str, message: str | None, metrics: dict[str, Any]) -> None:
        """Record a progress event for a job.

        Args:
            optimization_id: ID of the job emitting the event.
            message: Human-readable event marker, or ``None``.
            metrics: Metric snapshot to merge into ``latest_metrics``.
        """
        ...

    def get_progress_events(self, optimization_id: str, *, since: int = 0) -> list[ProgressEventRecord]:
        """Retrieve progress events for a job in chronological order.

        Args:
            optimization_id: ID of the job to inspect.
            since: Number of leading events to skip, for tail (delta) fetches;
                ``0`` returns the full history.

        Returns:
            Events ordered oldest-first, starting at offset ``since``.
        """
        ...

    def get_progress_count(self, optimization_id: str) -> int:
        """Return the number of progress events recorded for a job.

        Args:
            optimization_id: ID of the job to inspect.

        Returns:
            Number of stored events.
        """
        ...

    def append_log(
        self,
        optimization_id: str,
        *,
        level: str,
        logger_name: str,
        message: str,
        timestamp: datetime | None = None,
        pair_index: int | None = None,
    ) -> None:
        """Append a log entry to a job's log history.

        Implementations default ``timestamp`` to the current UTC time
        when ``None``.

        Args:
            optimization_id: ID of the job emitting the log.
            level: Log level string (``INFO``, ``ERROR``, ...).
            logger_name: Originating logger name.
            message: Log line content.
            timestamp: Optional override for the entry timestamp.
            pair_index: Optional grid-pair index when emitted from a sweep.
        """
        ...

    def get_logs(
        self,
        optimization_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        level: str | None = None,
    ) -> list[LogEntryRecord]:
        """Retrieve log entries for a job with optional level filter and pagination.

        Args:
            optimization_id: ID of the job to inspect.
            limit: Maximum number of entries to return; ``None`` means no cap.
            offset: Number of entries to skip from the start.
            level: When set, restricts results to the given level.

        Returns:
            Matching log entries in chronological order.
        """
        ...

    def get_log_count(self, optimization_id: str, *, level: str | None = None) -> int:
        """Return the number of log entries for a job, optionally filtered by level.

        Args:
            optimization_id: ID of the job to inspect.
            level: When set, counts only entries at this level.

        Returns:
            Number of matching log entries.
        """
        ...

    def set_payload_overview(self, optimization_id: str, overview: dict[str, Any]) -> None:
        """Store a summary overview of the job payload.

        Args:
            optimization_id: ID of the job to update.
            overview: Summary fields to persist.
        """
        ...

    def list_jobs(
        self,
        *,
        status: str | None = None,
        username: str | None = None,
        optimization_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List jobs with optional filtering and pagination, newest first.

        Args:
            status: Restrict to jobs with this status when set.
            username: Restrict to jobs owned by this user when set.
            optimization_type: Restrict to a particular run type when set.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip from the start.

        Returns:
            Matching ``JobRecord`` rows in newest-first order. Implementations
            may prune ``result`` down to the summary scalars list surfaces
            read (test metrics, runtime, pair counters, ``best_pair`` labels);
            use :meth:`get_job` for the full blob.
        """
        ...

    def count_jobs(
        self,
        *,
        status: str | None = None,
        username: str | None = None,
        optimization_type: str | None = None,
    ) -> int:
        """Count jobs matching the given filters.

        Args:
            status: Restrict count to this status when set.
            username: Restrict count to this owner when set.
            optimization_type: Restrict count to this run type when set.

        Returns:
            Number of matching rows.
        """
        ...

    def recover_orphaned_jobs(self) -> int:
        """Reclaim jobs whose worker lease has expired.

        Used on boot and as a periodic janitor: any ``running``/``validating``
        row whose ``lease_expires_at`` is in the past is transitioned to
        ``failed`` (the previous owner is presumed dead).

        Returns:
            Number of jobs that were transitioned to ``failed``.
        """
        ...

    def recover_pending_jobs(self) -> list[str]:
        """Return IDs of still-pending jobs ordered by creation time, oldest first.

        Kept for the legacy in-memory queue path. The DB-backed claim queue
        does not need this — workers discover pending rows via
        ``claim_next_job`` on every idle poll.

        Returns:
            Pending job IDs in FIFO order suitable for re-enqueue.
        """
        ...

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: float,
    ) -> JobRecord | None:
        """Atomically claim the oldest pending or expired-lease job.

        Implementations must use ``SELECT ... FOR UPDATE SKIP LOCKED`` (or an
        equivalent atomic primitive) so that two workers running concurrently
        cannot claim the same row. On a successful claim the row's
        ``status`` is moved to ``validating``, ``claimed_by`` is set to
        ``worker_id``, and ``lease_expires_at`` is set to ``now + lease_seconds``.

        Args:
            worker_id: Identifier of the worker (typically the pod name).
            lease_seconds: How long the worker may hold the lease before it
                must call :meth:`extend_lease` to renew it.

        Returns:
            The claimed ``JobRecord`` (``payload`` may be omitted — the worker
            re-reads it via :meth:`get_job` when the run starts), or ``None``
            if no claimable job exists.
        """
        ...

    def extend_lease(
        self,
        optimization_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> bool:
        """Extend the lease on a claimed job by ``lease_seconds`` from now.

        The renewal is conditional on ``claimed_by == worker_id`` so a worker
        that has been preempted (its lease already reclaimed by another pod)
        cannot accidentally keep mutating the row.

        Args:
            optimization_id: ID of the job whose lease to extend.
            worker_id: Worker identity that originally claimed the job.
            lease_seconds: New lease duration measured from now.

        Returns:
            ``True`` when the row was updated, ``False`` when the worker no
            longer owns the lease (caller should abort processing).
        """
        ...

    def release_job(self, optimization_id: str, worker_id: str) -> bool:
        """Clear claim metadata for a job once processing has terminated.

        Idempotent and safe to call repeatedly; only clears fields when
        ``claimed_by == worker_id``.

        Args:
            optimization_id: ID of the job to release.
            worker_id: Worker identity that claimed it.

        Returns:
            ``True`` when claim metadata was cleared.
        """
        ...

    def stage_dataset(self, username: str, dataset_filename: str, rows: list[dict[str, Any]]) -> str:
        """Persist wizard-parsed dataset rows for an agent-driven submit.

        The frontend stages each parsed dataset right after upload so the
        generalist agent can call ``submit_job_run_post`` with a tiny
        ``staged_dataset_id`` instead of inlining the rows into its tool
        arguments.

        Args:
            username: Submitter who owns the staged copy.
            dataset_filename: Original filename for diagnostics.
            rows: Parsed dataset rows; must be non-empty.

        Returns:
            The opaque staged-dataset id used by ``/run``.

        Raises:
            ValueError: When ``rows`` is empty.
        """
        ...

    def get_staged_dataset(self, staged_dataset_id: str, username: str) -> list[dict[str, Any]] | None:
        """Fetch staged rows by id, scoped to the calling user.

        Args:
            staged_dataset_id: Id previously returned by ``stage_dataset``.
            username: Authenticated caller — scope guards against cross-user reads.

        Returns:
            The persisted rows, or ``None`` when the row is missing or owned
            by another user.
        """
        ...

    def delete_staged_dataset(self, staged_dataset_id: str, username: str) -> bool:
        """Drop a staged dataset once it has been consumed.

        Args:
            staged_dataset_id: Id to evict.
            username: Authenticated caller — scope guards against cross-user deletes.

        Returns:
            ``True`` when a row was deleted, ``False`` when none matched.
        """
        ...
