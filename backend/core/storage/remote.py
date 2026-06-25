"""PostgreSQL database backend for job storage.

Provides RemoteDBJobStore for persisting job state to a PostgreSQL database.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import Engine, case, create_engine, func, or_, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, defer, sessionmaker

from ..config import settings
from ..constants import STRUCTURAL_PROGRESS_EVENTS, TQDM_KEY_PREFIX
from .base import JobRecord, LogEntryRecord, ProgressEventRecord
from .checkpoint_store import GepaCheckpoint, PostgresCheckpointBlobStore, PostgresGridPairResultStore
from .models import (
    EMBEDDING_DIM,
    AgentStagedDatasetModel,
    Base,
    ConversationEmbeddingModel,
    GepaCheckpointModel,
    GridPairResultModel,
    JobEmbeddingModel,
    JobModel,
    LogEntryModel,
    OptimizationShareGrantModel,
    ProgressEventModel,
    UserQuotaAuditModel,
    UserQuotaOverrideModel,
    UserStorageQuotaOverrideModel,
)
from .schema_lock import schema_bootstrap_lock
from .usage import (
    StorageItem,
    StorageUsage,
    compute_user_storage,
    compute_user_storage_category_items,
    compute_user_storage_items,
    json_byte_size,
)

logger = logging.getLogger(__name__)

MAX_PROGRESS_EVENTS = 5000
MAX_LOG_ENTRIES = 5000
PROGRESS_TRIM_SAMPLE_RATE = 100
_IMMUTABLE_JOB_COLUMNS = frozenset({"optimization_id", "notified_at", "idempotency_key"})
# The JSON columns whose serialized size dominates a job's storage footprint and
# therefore make up ``jobs.stored_bytes``. ``latest_metrics`` / ``message`` are
# tiny and intentionally excluded to keep the recompute read narrow.
_STORED_BYTES_JSON_COLUMNS = ("payload", "result", "payload_overview")


def _build_connect_args(db_url: str) -> dict[str, Any]:
    """Build DBAPI connection args for the configured SQLAlchemy driver.

    Args:
        db_url: SQLAlchemy database URL used to infer the selected driver.

    Returns:
        Keyword arguments passed through SQLAlchemy to the DBAPI connect call.
    """
    connect_args: dict[str, Any] = {"options": "-c timezone=UTC"}
    driver = make_url(db_url).drivername
    # libpq-based drivers honour TCP keepalive params so a connection silently
    # dropped by a load balancer / NAT idle timeout is detected and recycled
    # instead of surfacing as a stall on the next checkout from the pool.
    if driver in {"postgresql", "postgresql+psycopg2", "postgresql+psycopg", "postgresql+psycopg3"}:
        connect_args.update(
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )

    if not settings.db_pgbouncer_transaction_mode:
        return connect_args

    if driver in {"postgresql+psycopg", "postgresql+psycopg3"}:
        connect_args["prepare_threshold"] = None
    elif driver == "postgresql+asyncpg":
        connect_args["prepared_statement_cache_size"] = 0
    return connect_args


def _build_engine_kwargs(db_url: str) -> dict[str, Any]:
    """Return SQLAlchemy engine options sourced from settings.

    Args:
        db_url: SQLAlchemy database URL used for driver-specific connect args.

    Returns:
        Engine keyword arguments for :func:`sqlalchemy.create_engine`.
    """
    return {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_pool_max_overflow,
        "pool_recycle": settings.db_pool_recycle_seconds,
        "pool_timeout": settings.db_pool_timeout_seconds,
        "connect_args": _build_connect_args(db_url),
    }


class RemoteDBJobStore:
    """PostgreSQL-backed job storage using SQLAlchemy.

    No threading lock needed — PostgreSQL handles concurrent access natively.
    """

    def __init__(self, db_url: str) -> None:
        """Build the SQLAlchemy engine and create tables.

        Pool sizing is controlled by ``DB_POOL_SIZE`` and
        ``DB_POOL_MAX_OVERFLOW`` so Kubernetes deployments can keep total
        Postgres connection budgets below the server cap.

        pgvector bootstrap and the embedding tables are created only when
        ``settings.embeddings_enabled`` is true *and* the pgvector extension is
        available. If embeddings are disabled or the extension can't be created,
        the rest of the schema is bootstrapped without the ``Vector`` columns so
        a plain PostgreSQL without pgvector still boots (vector search stays off).

        Args:
            db_url: PostgreSQL DSN to connect to.
        """
        self.vector_search_enabled = False
        # Set after the schema exists: BM25 lexical ranking via pg_search when
        # available, else explore search uses ILIKE substring matching.
        self.bm25_search_enabled = False
        self._max_progress_events = settings.progress_events_per_job_cap
        self._max_log_entries = settings.log_entries_per_job_cap
        self._code_version = settings.code_version
        self._progress_event_counters: defaultdict[str, int] = defaultdict(int)
        self._progress_counter_lock = threading.Lock()
        # Force every connection's session timezone to UTC so TZ-aware writes
        # into TIMESTAMPTZ columns round-trip without offset rotation, and any
        # naive value that slipped into legacy rows is interpreted as UTC.
        self._engine = create_engine(
            db_url,
            **_build_engine_kwargs(db_url),
        )
        if settings.embeddings_enabled and settings.embeddings_dim != EMBEDDING_DIM:
            # The embedding columns are a fixed-width vector(EMBEDDING_DIM). A
            # mismatched EMBEDDINGS_DIM boots fine but pgvector then rejects every
            # mismatched-length insert on the daemon embed thread (warn-and-drop),
            # so explore search silently degrades to lexical. Fail loudly instead.
            raise RuntimeError(
                f"EMBEDDINGS_DIM={settings.embeddings_dim} does not match the "
                f"vector({EMBEDDING_DIM}) schema column; embedding writes would be "
                f"silently rejected. Set EMBEDDINGS_DIM={EMBEDDING_DIM}, disable "
                "embeddings, or run a migration to change the column width."
            )
        if settings.embeddings_enabled and self._bootstrap_pgvector():
            with schema_bootstrap_lock(self._engine) as conn:
                Base.metadata.create_all(conn if conn is not None else self._engine)
            self.vector_search_enabled = self._bootstrap_vector_indexes()
        else:
            # Embeddings disabled, or pgvector is unavailable on this database
            # (managed/plain Postgres without CREATE EXTENSION privilege). Create
            # the full schema minus the Vector(512) tables so the app still boots;
            # otherwise CREATE TABLE job_embeddings raises UndefinedObject and
            # propagates unguarded out of create_app(). Vector search stays off.
            embedding_tables = {
                JobEmbeddingModel.__table__,
                ConversationEmbeddingModel.__table__,
            }
            non_embedding_tables = [table for table in Base.metadata.sorted_tables if table not in embedding_tables]
            with schema_bootstrap_lock(self._engine) as conn:
                Base.metadata.create_all(
                    conn if conn is not None else self._engine,
                    tables=non_embedding_tables,
                )
        # Lexical search ranking. Independent of pgvector/embeddings: BM25
        # serves the default (embeddings-off) explore search when pg_search is
        # installed, otherwise the ILIKE fallback handles it.
        self.bm25_search_enabled = settings.search_bm25_enabled and self._bootstrap_bm25()
        self._session_factory = sessionmaker(bind=self._engine)
        self._checkpoints = PostgresCheckpointBlobStore(self._engine)
        self._grid_pair_results = PostgresGridPairResultStore(self._engine)
        parsed_url = urlparse(db_url)
        db_location = parsed_url.hostname or "<masked>"
        if parsed_url.port:
            db_location = f"{db_location}:{parsed_url.port}"
        logger.info("Initialized PostgreSQL database at %s", db_location)

    @property
    def _progress_events_cap(self) -> int:
        """Return the per-job progress-event retention cap."""
        return getattr(self, "_max_progress_events", MAX_PROGRESS_EVENTS)

    @property
    def _log_entries_cap(self) -> int:
        """Return the per-job log-entry retention cap."""
        return getattr(self, "_max_log_entries", MAX_LOG_ENTRIES)

    @property
    def _current_code_version(self) -> str:
        """Return the cached worker code version for this store."""
        return getattr(self, "_code_version", settings.code_version)

    def _bootstrap_pgvector(self) -> bool:
        """Ensure the pgvector extension exists before creating Vector columns.

        Safe to call repeatedly. If the database role lacks the privilege
        to install the extension, the log line below is the operator's
        hint to run ``CREATE EXTENSION vector`` once out-of-band.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            return True
        except SQLAlchemyError as exc:
            logger.warning(
                "pgvector extension bootstrap failed (%s). "
                "Recommendation service will be unavailable until an admin runs "
                "'CREATE EXTENSION vector' on the database.",
                exc,
            )
            return False

    def _bootstrap_vector_indexes(self) -> bool:
        """Create HNSW cosine indexes on the job_embeddings vector columns.

        SQLAlchemy's create_all can't express HNSW, and we don't want to
        pay for a reindex on every restart — the IF NOT EXISTS guard
        makes the call free on warm databases.
        """
        statements = [
            (
                "CREATE INDEX IF NOT EXISTS idx_job_embeddings_summary_hnsw "
                "ON job_embeddings USING hnsw (embedding_summary vector_cosine_ops)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_job_embeddings_code_hnsw "
                "ON job_embeddings USING hnsw (embedding_code vector_cosine_ops)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_job_embeddings_schema_hnsw "
                "ON job_embeddings USING hnsw (embedding_schema vector_cosine_ops)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_conversation_embeddings_summary_hnsw "
                "ON conversation_embeddings USING hnsw (embedding_summary vector_cosine_ops)"
            ),
        ]
        try:
            with self._engine.connect() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
                conn.commit()
            return True
        except SQLAlchemyError as exc:
            logger.warning("HNSW index bootstrap skipped: %s", exc)
            return False

    def _bootstrap_bm25(self) -> bool:
        """Best-effort: enable BM25 lexical ranking via the pg_search extension.

        Creates the ``pg_search`` extension and a BM25 index over the ``jobs``
        ``payload_overview`` corpus (task name/description/optimizer/model/module
        — the text that exists when embeddings are off) so explore search ranks
        lexically with real relevance scores. Entirely optional: when pg_search
        is absent (the common case on a plain/managed Postgres without it) or the
        role can't create it, this logs and returns False and search falls back
        to ILIKE substring matching. Safe to call repeatedly; the
        ``IF NOT EXISTS`` guards make it free on warm databases.

        Returns:
            True when pg_search and the BM25 index are in place, else False.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_jobs_bm25 ON jobs "
                        "USING bm25 (optimization_id, payload_overview) "
                        "WITH (key_field='optimization_id')"
                    )
                )
                conn.commit()
            logger.info("BM25 lexical search enabled (pg_search).")
            return True
        except SQLAlchemyError as exc:
            logger.info(
                "BM25 search unavailable (%s); explore search uses ILIKE lexical "
                "matching. Install the pg_search extension to enable BM25 ranking.",
                exc,
            )
            return False

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy engine backing this store.

        Exposed so callers can run shared table operations (direct
        DDL, joined queries) outside the ORM session factory.

        Returns:
            The configured SQLAlchemy engine.
        """
        return self._engine

    def _get_session(self) -> Session:
        """Create and return a new SQLAlchemy session.

        Returns:
            A new session; the caller is responsible for closing it.
        """
        return self._session_factory()

    def _job_to_dict(self, job: JobModel, *, include_payload: bool = True) -> JobRecord:
        """Convert a JobModel ORM instance to its TypedDict representation.

        Args:
            job: SQLAlchemy ORM row to serialize.
            include_payload: When ``False``, the (potentially multi-MB) ``payload``
                JSONB is reported as ``None`` instead of being read off the row.
                List/SSE paths pass ``False`` so the column is never materialized;
                callers that defer it on the query (see :meth:`_rows_with_counts`)
                avoid the DB read entirely.

        Returns:
            A ``JobRecord`` with ISO-formatted timestamps and JSON columns
            normalized to plain dicts.
        """
        return cast(
            JobRecord,
            {
                "optimization_id": job.optimization_id,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "estimated_remaining_seconds": job.estimated_remaining_seconds,
                "message": job.message,
                "latest_metrics": job.latest_metrics or {},
                "result": job.result,
                "payload_overview": job.payload_overview or {},
                "payload": job.payload if include_payload else None,
                "username": job.username,
                "optimization_type": job.optimization_type,
                "attempts": job.attempts,
                "code_version": job.code_version,
                "stored_bytes": job.stored_bytes or 0,
                "accumulated_runtime_seconds": job.accumulated_runtime_seconds or 0.0,
            },
        )

    def create_job(
        self,
        optimization_id: str,
        estimated_remaining_seconds: float | None = None,
        *,
        username: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Create a new job record in the database.

        Args:
            optimization_id: Unique identifier for the new job.
            estimated_remaining_seconds: Initial ETA, or ``None`` if unknown.
            username: Submitter recorded on the row up-front so the partial
                unique index on ``(username, idempotency_key)`` can guard
                against duplicate POSTs before ``set_payload_overview`` runs.
            idempotency_key: Optional client-supplied dedup key; pairs with
                ``username`` for the uniqueness check.

        Returns:
            The newly inserted row as a ``JobRecord``.
        """
        now = datetime.now(UTC)
        job = JobModel(
            optimization_id=optimization_id,
            status="pending",
            created_at=now,
            estimated_remaining_seconds=estimated_remaining_seconds,
            latest_metrics={},
            payload_overview={},
            attempts=0,
            code_version=self._current_code_version,
            username=username,
            idempotency_key=idempotency_key,
        )
        session = self._get_session()
        try:
            session.add(job)
            session.commit()
            session.refresh(job)
            return self._job_to_dict(job)
        finally:
            session.close()

    def find_job_by_idempotency_key(self, username: str, idempotency_key: str) -> str | None:
        """Return the ``optimization_id`` previously submitted under this key.

        Args:
            username: Submitter to scope the lookup to.
            idempotency_key: Client-supplied dedup key from the request header.

        Returns:
            The matching job id, or ``None`` when no prior submission used the key.
        """
        if not username or not idempotency_key:
            return None
        session = self._get_session()
        try:
            row = (
                session.query(JobModel.optimization_id)
                .filter(JobModel.username == username)
                .filter(JobModel.idempotency_key == idempotency_key)
                .first()
            )
            return row[0] if row else None
        finally:
            session.close()

    def claim_completion_notification(self, optimization_id: str) -> bool:
        """Atomically claim the right to send the completion notification.

        Multiple paths can converge on a single job's completion: the worker
        that finished it, a peer that orphan-recovered it after a lease
        expiry, and the cancellation handler. Each one calls
        ``notify_job_completed`` which would otherwise re-send the message
        once per attempt. This method does a single ``UPDATE ... WHERE
        notified_at IS NULL`` and reports whether the caller won the CAS;
        only the winner should send the notification.

        Args:
            optimization_id: ID of the job whose notification is being sent.

        Returns:
            ``True`` if the caller won the race and should send the message,
            ``False`` when a prior attempt already notified (or the job is
            missing — there is nothing to notify about).
        """
        now = datetime.now(UTC)
        session = self._get_session()
        try:
            rows = (
                session.query(JobModel)
                .filter(JobModel.optimization_id == optimization_id)
                .filter(JobModel.notified_at.is_(None))
                .update({JobModel.notified_at: now}, synchronize_session=False)
            )
            session.commit()
            return rows > 0
        finally:
            session.close()

    def update_job(self, optimization_id: str, **kwargs: Any) -> None:
        """Update fields on an existing job.

        Datetime string values are automatically parsed from ISO format;
        ``latest_metrics`` is merged into the existing mapping rather
        than replacing it. The write is a non-locking ``UPDATE`` —
        ``SELECT ... FOR UPDATE`` would serialise concurrent writers on
        this row, which becomes a contention hotspot at scale and is
        unnecessary because each optimization has a single worker writer
        in flight (lifecycle cancellation is the only realistic peer and
        last-writer-wins is acceptable for ``status``/``message``).

        Args:
            optimization_id: ID of the job to update.
            **kwargs: Column values to overwrite.

        Raises:
            KeyError: When the job does not exist.
            ValueError: When ``kwargs`` contains a column name absent from ``JobModel``.
        """
        datetime_fields = {"created_at", "started_at", "completed_at"}
        mutable_columns = set(JobModel.__table__.columns.keys()) - _IMMUTABLE_JOB_COLUMNS
        invalid_fields = sorted(set(kwargs) - mutable_columns)
        if invalid_fields:
            raise ValueError(f"Unknown field '{invalid_fields[0]}' on JobModel")

        merging_metrics = "latest_metrics" in kwargs
        recompute_stored = bool(set(_STORED_BYTES_JSON_COLUMNS) & set(kwargs))
        update_values: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in datetime_fields and isinstance(value, str):
                value = datetime.fromisoformat(value)
            update_values[key] = value

        session = self._get_session()
        try:
            if merging_metrics:
                existing = (
                    session.query(JobModel.latest_metrics).filter(JobModel.optimization_id == optimization_id).first()
                )
                if existing is None:
                    raise KeyError(f"Job '{optimization_id}' not found")
                current_metrics = existing[0] or {}
                merged = dict(current_metrics)
                merged.update(update_values["latest_metrics"])
                update_values["latest_metrics"] = merged

            if recompute_stored:
                stored_row = (
                    session.query(
                        JobModel.payload,
                        JobModel.result,
                        JobModel.payload_overview,
                    )
                    .filter(JobModel.optimization_id == optimization_id)
                    .first()
                )
                if stored_row is None:
                    raise KeyError(f"Job '{optimization_id}' not found")
                update_values["stored_bytes"] = sum(
                    json_byte_size(update_values[col] if col in update_values else stored_row[i])
                    for i, col in enumerate(_STORED_BYTES_JSON_COLUMNS)
                )

            rows = (
                session.query(JobModel)
                .filter(JobModel.optimization_id == optimization_id)
                .update(update_values, synchronize_session=False)
            )
            if rows == 0:
                raise KeyError(f"Job '{optimization_id}' not found")
            session.commit()
        finally:
            session.close()

    def update_job_if_status(self, optimization_id: str, expected: tuple[str, ...], **kwargs: Any) -> bool:
        """Update a job only while its status is one of ``expected`` (compare-and-set).

        The conditional ``WHERE status IN (...)`` closes the last-writer-wins
        race on the worker-completion and pause/cancel paths: a terminal write
        can't clobber a status another writer already moved the row to. Datetime
        strings are parsed and ``stored_bytes`` is recomputed exactly as
        :meth:`update_job` does; ``latest_metrics`` merging is not supported.

        Args:
            optimization_id: ID of the job to update.
            expected: Status values the row must currently hold for the write to
                apply.
            **kwargs: Column values to overwrite.

        Returns:
            ``True`` when the row matched and was updated; ``False`` when the
            status no longer matched (the caller should treat the write as lost).

        Raises:
            ValueError: When ``kwargs`` names a column absent from ``JobModel``.
        """
        mutable_columns = set(JobModel.__table__.columns.keys()) - _IMMUTABLE_JOB_COLUMNS
        invalid_fields = sorted(set(kwargs) - mutable_columns)
        if invalid_fields:
            raise ValueError(f"Unknown field '{invalid_fields[0]}' on JobModel")
        datetime_fields = {"created_at", "started_at", "completed_at"}
        update_values: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in datetime_fields and isinstance(value, str):
                value = datetime.fromisoformat(value)
            update_values[key] = value

        session = self._get_session()
        try:
            if set(_STORED_BYTES_JSON_COLUMNS) & set(update_values):
                stored_row = (
                    session.query(JobModel.payload, JobModel.result, JobModel.payload_overview)
                    .filter(JobModel.optimization_id == optimization_id)
                    .first()
                )
                if stored_row is not None:
                    update_values["stored_bytes"] = sum(
                        json_byte_size(update_values[col] if col in update_values else stored_row[i])
                        for i, col in enumerate(_STORED_BYTES_JSON_COLUMNS)
                    )
            rows = (
                session.query(JobModel)
                .filter(JobModel.optimization_id == optimization_id)
                .filter(JobModel.status.in_(expected))
                .update(update_values, synchronize_session=False)
            )
            session.commit()
            return rows > 0
        finally:
            session.close()

    def get_job(self, optimization_id: str) -> JobRecord:
        """Retrieve a job by its ID.

        Args:
            optimization_id: ID of the job to fetch.

        Returns:
            The matching ``JobRecord``.

        Raises:
            KeyError: When the job does not exist.
        """
        session = self._get_session()
        try:
            job = session.query(JobModel).filter(JobModel.optimization_id == optimization_id).first()
            if not job:
                raise KeyError(f"Job '{optimization_id}' not found")
            return self._job_to_dict(job)
        finally:
            session.close()

    def get_job_status_fields(self, optimization_id: str) -> JobRecord:
        """Retrieve only the live-polling fields for a job.

        Selects ``status`` / ``message`` / ``latest_metrics`` directly so the
        per-job SSE loop can poll every few seconds without re-reading the
        ``payload`` JSONB that :meth:`get_job` materializes.

        Args:
            optimization_id: ID of the job to read.

        Returns:
            A partial ``JobRecord`` with ``status``, ``message`` and
            ``latest_metrics``.

        Raises:
            KeyError: When the job does not exist.
        """
        session = self._get_session()
        try:
            row = (
                session.query(JobModel.status, JobModel.message, JobModel.latest_metrics)
                .filter(JobModel.optimization_id == optimization_id)
                .first()
            )
            if row is None:
                raise KeyError(f"Job '{optimization_id}' not found")
            return cast(
                JobRecord,
                {"status": row[0], "message": row[1], "latest_metrics": row[2] or {}},
            )
        finally:
            session.close()

    def job_exists(self, optimization_id: str) -> bool:
        """Return ``True`` if the job exists in the database.

        Args:
            optimization_id: ID to check.

        Returns:
            Whether the row is present.
        """
        session = self._get_session()
        try:
            return session.query(JobModel).filter(JobModel.optimization_id == optimization_id).first() is not None
        finally:
            session.close()

    def delete_job(self, optimization_id: str) -> None:
        """Delete a job and all its associated logs and progress events.

        Missing IDs are a silent no-op.

        Args:
            optimization_id: ID of the job to remove.
        """
        session = self._get_session()
        try:
            session.query(LogEntryModel).filter(LogEntryModel.optimization_id == optimization_id).delete()
            session.query(ProgressEventModel).filter(ProgressEventModel.optimization_id == optimization_id).delete()
            session.query(JobEmbeddingModel).filter(JobEmbeddingModel.optimization_id == optimization_id).delete()
            session.query(GepaCheckpointModel).filter(GepaCheckpointModel.optimization_id == optimization_id).delete()
            session.query(GridPairResultModel).filter(GridPairResultModel.optimization_id == optimization_id).delete()
            session.query(JobModel).filter(JobModel.optimization_id == optimization_id).delete()
            session.commit()
        finally:
            session.close()

    def get_jobs_status_by_ids(self, optimization_ids: list[str]) -> dict[str, str]:
        """Return a ``{id: status}`` map for the requested IDs.

        Runs a single ``SELECT ... WHERE optimization_id IN (...)``
        so batch existence + status checks cost one round trip
        regardless of how many IDs are supplied.

        Args:
            optimization_ids: IDs to look up.

        Returns:
            Mapping of present IDs to their status strings; missing IDs
            are simply absent from the result.
        """
        if not optimization_ids:
            return {}
        session = self._get_session()
        try:
            rows = (
                session.query(JobModel.optimization_id, JobModel.status)
                .filter(JobModel.optimization_id.in_(optimization_ids))
                .all()
            )
            return {r[0]: r[1] for r in rows}
        finally:
            session.close()

    def delete_jobs(self, optimization_ids: list[str]) -> int:
        """Hard-delete a batch of jobs in a single transaction.

        Drops the associated log, progress-event, embedding and checkpoint rows
        first then commits once, so the round-trip cost is bounded regardless of
        batch size.

        Args:
            optimization_ids: IDs to remove. Duplicates and missing IDs are tolerated.

        Returns:
            The number of job rows actually deleted.
        """
        if not optimization_ids:
            return 0
        session = self._get_session()
        try:
            session.query(LogEntryModel).filter(LogEntryModel.optimization_id.in_(optimization_ids)).delete(
                synchronize_session=False
            )
            session.query(ProgressEventModel).filter(ProgressEventModel.optimization_id.in_(optimization_ids)).delete(
                synchronize_session=False
            )
            session.query(JobEmbeddingModel).filter(JobEmbeddingModel.optimization_id.in_(optimization_ids)).delete(
                synchronize_session=False
            )
            session.query(GepaCheckpointModel).filter(GepaCheckpointModel.optimization_id.in_(optimization_ids)).delete(
                synchronize_session=False
            )
            session.query(GridPairResultModel).filter(GridPairResultModel.optimization_id.in_(optimization_ids)).delete(
                synchronize_session=False
            )
            deleted = (
                session.query(JobModel)
                .filter(JobModel.optimization_id.in_(optimization_ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(deleted or 0)
        finally:
            session.close()

    def save_gepa_checkpoint(self, optimization_id: str, data: bytes, iteration: int, pair_index: int = -1) -> None:
        """Persist (or replace) the latest GEPA state blob for one run or grid pair.

        Args:
            optimization_id: Owning job id.
            data: Raw ``gepa_state.bin`` bytes for the latest iteration.
            iteration: The iteration index the state was saved at.
            pair_index: Grid pair index, or ``-1`` for a single run.
        """
        self._checkpoints.put(optimization_id, data=data, iteration=iteration, pair_index=pair_index)

    def get_gepa_checkpoint(self, optimization_id: str, pair_index: int = -1) -> GepaCheckpoint | None:
        """Return the saved GEPA checkpoint for one run/pair, or ``None``.

        Args:
            optimization_id: Job whose checkpoint is read.
            pair_index: Grid pair index, or ``-1`` for a single run.

        Returns:
            The :class:`GepaCheckpoint` (state bytes plus iteration), or ``None``.
        """
        return self._checkpoints.get(optimization_id, pair_index)

    def list_gepa_checkpoints(self, optimization_id: str) -> list[GepaCheckpoint]:
        """Return every saved checkpoint for a job (all grid pairs, or the single run).

        Args:
            optimization_id: Job whose checkpoints are read.

        Returns:
            The job's :class:`GepaCheckpoint` rows (possibly empty).
        """
        return self._checkpoints.list_for_optimization(optimization_id)

    def delete_gepa_checkpoint(self, optimization_id: str, pair_index: int = -1) -> None:
        """Drop one run/pair's GEPA checkpoint (no-op when absent).

        Args:
            optimization_id: Owning job id.
            pair_index: Grid pair index, or ``-1`` for a single run.
        """
        self._checkpoints.delete(optimization_id, pair_index)

    def delete_all_gepa_checkpoints(self, optimization_id: str) -> None:
        """Drop every checkpoint for a job (e.g. once a grid succeeds).

        Args:
            optimization_id: Job whose checkpoints are freed.
        """
        self._checkpoints.delete_all(optimization_id)

    def has_gepa_checkpoint(self, optimization_id: str) -> bool:
        """Return whether any resumable GEPA checkpoint exists for ``optimization_id``.

        Cheap key-only existence check (single run or any grid pair) used to gate
        the per-job ``resumable`` flag on the list/detail read paths.

        Args:
            optimization_id: Job to test.

        Returns:
            ``True`` when at least one checkpoint row exists.
        """
        return self._checkpoints.has_any(optimization_id)

    def save_grid_pair_result(self, optimization_id: str, pair_index: int, result: dict[str, Any]) -> None:
        """Persist (or replace) one completed grid pair's result so resume can skip it.

        Args:
            optimization_id: Owning grid job id.
            pair_index: The completed pair's index.
            result: The pair's serialized ``PairResult``.
        """
        self._grid_pair_results.put(optimization_id, pair_index, result)

    def get_grid_pair_results(self, optimization_id: str) -> dict[int, dict[str, Any]]:
        """Return ``{pair_index: result}`` for every completed pair of a grid.

        Args:
            optimization_id: Grid job whose finished pairs are read.

        Returns:
            Mapping of completed pair index to its serialized result (possibly empty).
        """
        return self._grid_pair_results.get_all(optimization_id)

    def delete_grid_pair_results(self, optimization_id: str) -> None:
        """Drop every stored pair result for a grid (e.g. once it succeeds).

        Args:
            optimization_id: Grid job whose pair results are freed.
        """
        self._grid_pair_results.delete_all(optimization_id)

    def has_grid_pair_results(self, optimization_id: str) -> bool:
        """Return whether the grid has any completed-pair result stored.

        Args:
            optimization_id: Grid job to test.

        Returns:
            ``True`` when at least one pair result exists.
        """
        return self._grid_pair_results.has_any(optimization_id)

    def requeue_for_resume(self, optimization_id: str, *, bump_attempts: bool = True) -> int | None:
        """Re-queue a terminal job in place so a worker resumes it from its checkpoint.

        Flips the existing row back to ``pending`` — same id, payload, seed and
        budget — and clears the prior claim/lease, mirroring
        :meth:`recover_orphaned_jobs`. A whole-job resume increments ``attempts``
        so it shares the ``job_max_attempts`` cap with pod-failure recovery; a
        targeted per-pair grid re-run passes ``bump_attempts=False`` so retrying
        individual pairs is not bounded by that cap. The caller owns the
        resumability preconditions.

        The finished leg's wall-clock duration is folded into
        ``accumulated_runtime_seconds`` before ``started_at``/``completed_at`` are
        cleared, so the resumed run's elapsed timer measures net active compute
        across all legs and excludes the paused gap between them.

        Args:
            optimization_id: The job to resume.
            bump_attempts: Whether to count this re-queue against the attempt cap.

        Returns:
            The new attempt count, or ``None`` when the job row is missing.
        """
        session = self._get_session()
        try:
            job = session.get(JobModel, optimization_id)
            if job is None:
                return None
            current = int(job.attempts or 0)
            next_attempt = current + 1 if bump_attempts else current
            job.attempts = next_attempt  # type: ignore[assignment]
            job.status = "pending"  # type: ignore[assignment]
            job.claimed_by = None  # type: ignore[assignment]
            job.claimed_at = None  # type: ignore[assignment]
            job.lease_expires_at = None  # type: ignore[assignment]
            # Fold the just-finished leg's exact duration into the running total
            # before the timestamps are cleared, so the resumed run's elapsed timer
            # reflects net active compute across legs and never the paused gap.
            if job.started_at is not None:
                leg_start = job.started_at if job.started_at.tzinfo else job.started_at.replace(tzinfo=UTC)
                leg_end_raw = job.completed_at or datetime.now(UTC)
                leg_end = leg_end_raw if leg_end_raw.tzinfo else leg_end_raw.replace(tzinfo=UTC)
                job.accumulated_runtime_seconds = float(  # type: ignore[assignment]
                    job.accumulated_runtime_seconds or 0.0
                ) + max(0.0, (leg_end - leg_start).total_seconds())
            job.completed_at = None  # type: ignore[assignment]
            job.started_at = None  # type: ignore[assignment]
            job.message = "Resuming" if bump_attempts else "Re-running grid pair"  # type: ignore[assignment]
            session.commit()
            return next_attempt
        finally:
            session.close()

    def requeue_for_rerun(self, optimization_id: str) -> bool:
        """Reset a terminal job in place for a clean from-scratch re-run.

        Unlike :meth:`requeue_for_resume`, which keeps the checkpoint and
        continues GEPA where it stopped, this discards every artefact of the
        previous attempt — logs, progress events, the search embedding, GEPA
        checkpoints and grid-pair results — and zeroes the row's runtime,
        result and attempt bookkeeping. Only the immutable identity (id, name,
        owner, payload) survives, so a worker re-runs the same configuration
        from iteration zero under the same id. The child-row deletes mirror
        :meth:`delete_job`, minus the ``JobModel`` row itself.

        Args:
            optimization_id: The job to reset and re-queue.

        Returns:
            ``True`` when the row existed and was reset, ``False`` when missing.
        """
        session = self._get_session()
        try:
            job = session.get(JobModel, optimization_id)
            if job is None:
                return False
            session.query(LogEntryModel).filter(LogEntryModel.optimization_id == optimization_id).delete()
            session.query(ProgressEventModel).filter(ProgressEventModel.optimization_id == optimization_id).delete()
            session.query(JobEmbeddingModel).filter(JobEmbeddingModel.optimization_id == optimization_id).delete()
            session.query(GepaCheckpointModel).filter(GepaCheckpointModel.optimization_id == optimization_id).delete()
            session.query(GridPairResultModel).filter(GridPairResultModel.optimization_id == optimization_id).delete()
            job.status = "pending"  # type: ignore[assignment]
            job.started_at = None  # type: ignore[assignment]
            job.completed_at = None  # type: ignore[assignment]
            job.estimated_remaining_seconds = None  # type: ignore[assignment]
            job.message = None  # type: ignore[assignment]
            job.latest_metrics = {}  # type: ignore[assignment]
            job.result = None  # type: ignore[assignment]
            job.attempts = 0  # type: ignore[assignment]
            job.claimed_by = None  # type: ignore[assignment]
            job.claimed_at = None  # type: ignore[assignment]
            job.lease_expires_at = None  # type: ignore[assignment]
            # Cleared so the fresh run is free to emit its own completion
            # notification — the flag is a single-shot guard set per finished run.
            job.notified_at = None  # type: ignore[assignment]
            job.accumulated_runtime_seconds = 0.0  # type: ignore[assignment]
            # The result is gone, so the footprint shrinks to payload + overview.
            job.stored_bytes = sum(  # type: ignore[assignment]
                json_byte_size(getattr(job, col)) for col in _STORED_BYTES_JSON_COLUMNS
            )
            session.commit()
            return True
        finally:
            session.close()

    def get_user_quota_override(self, username: str) -> tuple[bool, int | None]:
        """Return the live quota override row for ``username`` if present.

        Args:
            username: User identifier to resolve case-insensitively.

        Returns:
            ``(False, None)`` when no row exists, otherwise ``(True, quota)``.
            A present ``None`` quota means unlimited.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            return False, None
        session = self._get_session()
        try:
            row = session.get(UserQuotaOverrideModel, normalized_username)
            if row is None:
                return False, None
            return True, row.quota
        finally:
            session.close()

    def get_effective_user_quota(self, username: str) -> int | None:
        """Resolve a user's quota from live DB override, then static config.

        Args:
            username: User identifier to resolve.

        Returns:
            The numeric quota, or ``None`` for unlimited.
        """
        has_override, quota = self.get_user_quota_override(username)
        if has_override:
            return quota
        return settings.get_user_quota(username)

    def get_effective_user_storage_quota(self, username: str) -> int:
        """Return the unified storage budget in bytes the user is held to.

        Resolves an admin per-user override first (a live DB row that replaces
        the default ceiling for that user), falling back to the static
        ``settings.user_storage_quota_bytes`` default. Staying a method keeps the
        save/run gate, the usage meter, and the quota modal resolving the budget
        through one seam that tests can stub.

        Args:
            username: Owner whose effective byte budget is resolved.

        Returns:
            The byte budget the user's total storage is checked against.
        """
        override = self.get_user_storage_quota_override(username)
        if override is not None:
            return override
        return settings.user_storage_quota_bytes

    def compute_user_storage(self, username: str) -> StorageUsage:
        """Return the user's unified storage usage across every owned table.

        Args:
            username: Owner whose footprint is summed.

        Returns:
            The :class:`StorageUsage` total and per-category breakdown.
        """
        return compute_user_storage(self._engine, username)

    def compute_user_storage_items(self, username: str, limit: int = 20) -> list[StorageItem]:
        """Return the user's largest individual items for the cleanup list.

        Args:
            username: Owner whose items are ranked.
            limit: Maximum number of items to return.

        Returns:
            Up to ``limit`` :class:`StorageItem` rows ordered by descending size.
        """
        return compute_user_storage_items(self._engine, username, limit)

    def compute_user_storage_category_items(self, username: str, category: str, limit: int = 1000) -> list[StorageItem]:
        """List every deletable item the user owns in one storage category.

        Args:
            username: Owner whose items are listed.
            category: One of the deletable storage categories; any other value
                yields an empty list.
            limit: Defensive upper bound on rows returned for the category.

        Returns:
            The category's :class:`StorageItem` rows ordered by descending size.
        """
        return compute_user_storage_category_items(self._engine, username, category, limit)

    def get_user_storage_quota_override(self, username: str) -> int | None:
        """Return the per-user storage-budget override in bytes, if present.

        Args:
            username: User identifier to resolve case-insensitively.

        Returns:
            The override byte budget, or ``None`` when no override row exists.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            return None
        session = self._get_session()
        try:
            row = session.get(UserStorageQuotaOverrideModel, normalized_username)
            return row.quota_bytes if row is not None else None
        finally:
            session.close()

    def set_user_storage_quota_override(self, username: str, quota_bytes: int, updated_by: str | None = None) -> None:
        """Create or update a per-user storage-budget override.

        Args:
            username: User identifier to store case-insensitively.
            quota_bytes: Byte ceiling that replaces the default for this user.
            updated_by: Optional operator identifier for accountability.

        Raises:
            ValueError: When ``username`` is blank or ``quota_bytes`` is below one.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            raise ValueError("username must not be blank")
        if quota_bytes < 1:
            raise ValueError("quota_bytes must be at least 1")
        session = self._get_session()
        try:
            row = session.get(UserStorageQuotaOverrideModel, normalized_username)
            if row is None:
                row = UserStorageQuotaOverrideModel(username=normalized_username)
                session.add(row)
            row.quota_bytes = quota_bytes
            row.updated_at = datetime.now(UTC)
            row.updated_by = updated_by
            session.commit()
        finally:
            session.close()

    def delete_user_storage_quota_override(self, username: str) -> bool:
        """Delete a storage-budget override so the default budget applies again.

        Args:
            username: User identifier to clear case-insensitively.

        Returns:
            Whether a row was deleted.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            return False
        session = self._get_session()
        try:
            deleted = (
                session.query(UserStorageQuotaOverrideModel)
                .filter(UserStorageQuotaOverrideModel.username == normalized_username)
                .delete()
            )
            session.commit()
            return bool(deleted)
        finally:
            session.close()

    def list_user_storage_quota_overrides(self) -> list[dict[str, Any]]:
        """Return all storage-budget overrides ordered by username.

        Returns:
            Override rows with ISO-formatted ``updated_at`` values.
        """
        session = self._get_session()
        try:
            rows = (
                session.query(UserStorageQuotaOverrideModel)
                .order_by(UserStorageQuotaOverrideModel.username.asc())
                .all()
            )
            return [
                {
                    "username": row.username,
                    "quota_bytes": row.quota_bytes,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "updated_by": row.updated_by,
                }
                for row in rows
            ]
        finally:
            session.close()

    def set_user_quota_override(self, username: str, quota: int | None, updated_by: str | None = None) -> None:
        """Create or update the live quota override for a user.

        Args:
            username: User identifier to store case-insensitively.
            quota: Numeric quota, or ``None`` for unlimited.
            updated_by: Optional operator identifier for audit context.

        Raises:
            ValueError: When ``username`` is blank or ``quota`` is below one.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            raise ValueError("username must not be blank")
        if quota is not None and quota < 1:
            raise ValueError("quota must be at least 1")
        session = self._get_session()
        try:
            row = session.get(UserQuotaOverrideModel, normalized_username)
            if row is None:
                row = UserQuotaOverrideModel(username=normalized_username)
                session.add(row)
            row.quota = quota
            row.updated_at = datetime.now(UTC)
            row.updated_by = updated_by
            session.commit()
        finally:
            session.close()

    def delete_user_quota_override(self, username: str) -> bool:
        """Delete a live quota override so config fallback applies again.

        Args:
            username: User identifier to clear case-insensitively.

        Returns:
            Whether a row was deleted.
        """
        normalized_username = username.strip().lower()
        if not normalized_username:
            return False
        session = self._get_session()
        try:
            deleted = (
                session.query(UserQuotaOverrideModel)
                .filter(UserQuotaOverrideModel.username == normalized_username)
                .delete()
            )
            session.commit()
            return bool(deleted)
        finally:
            session.close()

    def list_user_quota_overrides(self) -> list[dict[str, Any]]:
        """Return all live quota overrides ordered by username.

        Returns:
            A list of override rows with ISO-formatted ``updated_at`` values.
        """
        session = self._get_session()
        try:
            rows = session.query(UserQuotaOverrideModel).order_by(UserQuotaOverrideModel.username.asc()).all()
            return [
                {
                    "username": row.username,
                    "quota": row.quota,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "updated_by": row.updated_by,
                }
                for row in rows
            ]
        finally:
            session.close()

    def search_usernames(self, query: str, *, limit: int = 10) -> list[str]:
        """Return distinct usernames known to the DB matching ``query``.

        Searches both job submissions (``jobs.username``) and existing quota
        overrides so admins can autocomplete previously-seen users without
        an external directory lookup.

        Args:
            query: Case-insensitive substring to match.
            limit: Maximum number of distinct usernames to return.

        Returns:
            Distinct lowercased usernames sorted alphabetically.
        """
        normalized = query.strip().lower()
        if not normalized:
            return []
        pattern = f"%{normalized}%"
        session = self._get_session()
        try:
            job_rows = (
                session.query(JobModel.username)
                .filter(JobModel.username.isnot(None))
                .filter(func.lower(JobModel.username).like(pattern))
                .distinct()
                .all()
            )
            override_rows = (
                session.query(UserQuotaOverrideModel.username)
                .filter(func.lower(UserQuotaOverrideModel.username).like(pattern))
                .all()
            )
            seen: set[str] = set()
            for (username,) in (*job_rows, *override_rows):
                if username:
                    seen.add(username.strip().lower())
            return sorted(seen)[: max(0, limit)]
        finally:
            session.close()

    def record_user_quota_audit(
        self,
        *,
        actor: str,
        target_username: str,
        action: str,
        old_quota: int | None,
        new_quota: int | None,
    ) -> None:
        """Record a quota administration audit event.

        Args:
            actor: Admin user who made the change.
            target_username: User whose quota changed.
            action: Operation name such as ``set`` or ``delete``.
            old_quota: Previous live quota, or ``None`` for unlimited/no row.
            new_quota: New live quota, or ``None`` for unlimited/default fallback.
        """
        session = self._get_session()
        try:
            session.add(
                UserQuotaAuditModel(
                    actor=actor.strip().lower(),
                    target_username=target_username.strip().lower(),
                    action=action,
                    old_quota=old_quota,
                    new_quota=new_quota,
                    created_at=datetime.now(UTC),
                )
            )
            session.commit()
        finally:
            session.close()

    def list_user_quota_audit_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent quota administration audit events.

        Args:
            limit: Maximum number of recent events to return.

        Returns:
            Recent audit events ordered newest-first.
        """
        session = self._get_session()
        try:
            rows = (
                session.query(UserQuotaAuditModel)
                .order_by(UserQuotaAuditModel.created_at.desc(), UserQuotaAuditModel.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": row.id,
                    "actor": row.actor,
                    "target_username": row.target_username,
                    "action": row.action,
                    "old_quota": row.old_quota,
                    "new_quota": row.new_quota,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        finally:
            session.close()

    def recover_orphaned_jobs(self) -> int:
        """Re-queue or fail jobs whose worker lease has expired.

        With the DB-backed claim queue, a "stuck" job is one whose
        ``lease_expires_at`` is in the past — the previous worker is presumed
        dead. Such jobs are moved back to ``pending`` until they reach the
        configured retry cap, letting a healthy peer pod claim the work.

        Rows that have *no* claim at all (``claimed_by IS NULL`` while still
        somehow in ``running``/``validating``) are also recovered, covering
        the bootstrapping case of a fleet that just upgraded from the legacy
        in-memory queue.

        Returns:
            The number of orphaned jobs handled.
        """
        session = self._get_session()
        try:
            now = datetime.now(UTC)
            orphaned = (
                session.query(JobModel)
                .filter(JobModel.status.in_(["running", "validating"]))
                .filter((JobModel.lease_expires_at.is_(None)) | (JobModel.lease_expires_at < now))
                .all()
            )
            for job in orphaned:
                next_attempt = int(job.attempts or 0) + 1
                job.attempts = next_attempt  # type: ignore[assignment]
                job.claimed_by = None  # type: ignore[assignment]
                job.claimed_at = None  # type: ignore[assignment]
                job.lease_expires_at = None  # type: ignore[assignment]
                if next_attempt >= settings.job_max_attempts:
                    job.status = "failed"  # type: ignore[assignment]
                    job.completed_at = now  # type: ignore[assignment]
                    if job.code_version and job.code_version != self._current_code_version:
                        job.message = (  # type: ignore[assignment]
                            "No compatible worker version available "
                            f"(job={job.code_version}, fleet={self._current_code_version})"
                        )
                    else:
                        job.message = f"Job failed after pod failure (attempt {next_attempt})"  # type: ignore[assignment]
                else:
                    job.status = "pending"  # type: ignore[assignment]
                    job.completed_at = None  # type: ignore[assignment]
                    job.message = f"Re-queued after pod failure (attempt {next_attempt})"  # type: ignore[assignment]
            session.commit()
            count = len(orphaned)
            if count:
                logger.warning("Handled %d orphaned jobs (expired lease)", count)
            return count
        finally:
            session.close()

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: float,
    ) -> JobRecord | None:
        """Atomically claim the oldest pending job using FOR UPDATE SKIP LOCKED.

        Two pods running this method concurrently are guaranteed to see
        disjoint result sets — Postgres' ``SKIP LOCKED`` causes each session
        to silently jump over rows the other has already row-locked. The
        outer ``UPDATE`` then writes the lease metadata, completing the claim
        in one round trip.

        On non-PostgreSQL dialects (eg. SQLite in tests) the query falls back
        to a non-locking SELECT-then-UPDATE; that is racy but tests run
        single-threaded so it is sufficient.

        Args:
            worker_id: Identifier of the calling worker (typically pod name).
            lease_seconds: Lease duration; the worker must call
                :meth:`extend_lease` before it expires.

        Returns:
            The claimed ``JobRecord`` or ``None`` if no job was available.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            return self._claim_next_job_postgres(worker_id, lease_seconds)
        return self._claim_next_job_fallback(worker_id, lease_seconds)

    def _claim_next_job_postgres(self, worker_id: str, lease_seconds: float) -> JobRecord | None:
        """PostgreSQL fast path using ``FOR UPDATE SKIP LOCKED``."""
        sql = text(
            """
            UPDATE jobs
            SET status = 'validating',
                claimed_by = :worker_id,
                claimed_at = :now,
                lease_expires_at = :lease_until
            WHERE optimization_id = (
                SELECT optimization_id FROM jobs
                WHERE status = 'pending'
                  AND (code_version IS NULL OR code_version = :code_version)
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING optimization_id
            """
        )
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._engine.begin() as conn:
            result = conn.execute(
                sql,
                {
                    "worker_id": worker_id,
                    "now": now,
                    "lease_until": lease_until,
                    "code_version": self._current_code_version,
                },
            )
            row = result.fetchone()
            if row is None:
                return None
            optimization_id = row[0]

        # Re-load the full row through the ORM so the returned dict matches
        # the shape of the rest of the API.
        return self.get_job(optimization_id)

    def _claim_next_job_fallback(self, worker_id: str, lease_seconds: float) -> JobRecord | None:
        """Best-effort claim for non-Postgres backends (tests).

        Holds an exclusive transaction so concurrent claims are serialized at
        the engine level. Not race-safe across processes but adequate for
        single-process test runs.
        """
        session = self._get_session()
        try:
            now = datetime.now(UTC)
            lease_until = now + timedelta(seconds=lease_seconds)
            job = (
                session.query(JobModel)
                .filter(JobModel.status == "pending")
                .filter(or_(JobModel.code_version.is_(None), JobModel.code_version == self._current_code_version))
                .order_by(JobModel.created_at.asc())
                .first()
            )
            if job is None:
                return None
            job.status = "validating"  # type: ignore[assignment]
            job.claimed_by = worker_id  # type: ignore[assignment]
            job.claimed_at = now  # type: ignore[assignment]
            job.lease_expires_at = lease_until  # type: ignore[assignment]
            session.commit()
            session.refresh(job)
            return self._job_to_dict(job)
        finally:
            session.close()

    def extend_lease(
        self,
        optimization_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> bool:
        """Extend the lease iff this worker still owns the claim.

        Args:
            optimization_id: ID of the job whose lease to extend.
            worker_id: Worker identity that originally claimed the job.
            lease_seconds: New lease duration measured from now.

        Returns:
            ``True`` when the lease was extended; ``False`` when the row no
            longer belongs to this worker (caller should abort processing).
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        sql = text(
            """
            UPDATE jobs
            SET lease_expires_at = :lease_until
            WHERE optimization_id = :oid
              AND claimed_by = :worker_id
            """
        )
        lease_until = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        with self._engine.begin() as conn:
            result = conn.execute(
                sql,
                {"oid": optimization_id, "worker_id": worker_id, "lease_until": lease_until},
            )
            return (result.rowcount or 0) > 0

    def release_job(self, optimization_id: str, worker_id: str) -> bool:
        """Clear claim metadata for a job, only if this worker owns it.

        Args:
            optimization_id: ID of the job to release.
            worker_id: Worker identity that claimed it.

        Returns:
            Whether claim metadata was actually cleared.
        """
        sql = text(
            """
            UPDATE jobs
            SET claimed_by = NULL,
                claimed_at = NULL,
                lease_expires_at = NULL
            WHERE optimization_id = :oid
              AND claimed_by = :worker_id
            """
        )
        with self._engine.begin() as conn:
            result = conn.execute(sql, {"oid": optimization_id, "worker_id": worker_id})
            return (result.rowcount or 0) > 0

    def recover_pending_jobs(self) -> list[str]:
        """Return IDs of still-pending jobs ordered oldest first.

        Returns:
            Pending job IDs in FIFO order so the scheduler can
            re-enqueue them on boot.
        """
        session = self._get_session()
        try:
            jobs = (
                session.query(JobModel)
                .filter(JobModel.status == "pending")
                .filter(or_(JobModel.code_version.is_(None), JobModel.code_version == self._current_code_version))
                .order_by(JobModel.created_at.asc())
                .all()
            )
            return [str(j.optimization_id) for j in jobs]
        finally:
            session.close()

    def set_payload_overview(self, optimization_id: str, overview: dict[str, Any]) -> None:
        """Store a summary overview of the job payload.

        The ``username`` field, if present in ``overview``, is hoisted
        to the job row so list queries don't need to parse JSON.
        Missing IDs are a silent no-op.

        Args:
            optimization_id: ID of the job to update.
            overview: Summary fields to persist.
        """
        session = self._get_session()
        try:
            job = session.query(JobModel).filter(JobModel.optimization_id == optimization_id).first()
            if job:
                job.payload_overview = overview or {}  # type: ignore[assignment]
                if "username" in (overview or {}):
                    job.username = (overview or {}).get("username")  # type: ignore[assignment]
                if "optimization_type" in (overview or {}):
                    job.optimization_type = (overview or {}).get("optimization_type")  # type: ignore[assignment]
                job.stored_bytes = (  # type: ignore[assignment]
                    json_byte_size(job.payload) + json_byte_size(job.result) + json_byte_size(overview or {})
                )
                session.commit()
        finally:
            session.close()

    def record_progress(self, optimization_id: str, message: str | None, metrics: dict[str, Any]) -> None:
        """Record a progress event and refresh the job's latest metrics.

        Event insertion and ``latest_metrics`` update are separate transactions
        so a transient insert failure does not roll back the UI's latest metric
        snapshot. Retention trimming runs on a sampled path to avoid counting
        rows for every high-frequency progress event.

        Args:
            optimization_id: ID of the job emitting the event.
            message: Human-readable event marker, or ``None``.
            metrics: Metric snapshot to store as ``latest_metrics``.
        """
        now = datetime.now(UTC)
        inserted = False
        try:
            inserted = self._insert_progress_event(optimization_id, message, metrics or {}, now)
        except SQLAlchemyError:
            logger.warning("Failed to insert progress event for %s", optimization_id, exc_info=True)

        if inserted and self._should_trim_progress_events(optimization_id):
            try:
                self._trim_progress_events(optimization_id)
            except SQLAlchemyError:
                logger.warning("Failed to trim progress events for %s", optimization_id, exc_info=True)

        if metrics:
            self._replace_latest_metrics(
                optimization_id,
                self._latest_metrics_with_sticky_tqdm(optimization_id, metrics),
            )

    def _insert_progress_event(
        self,
        optimization_id: str,
        message: str | None,
        metrics: dict[str, Any],
        timestamp: datetime,
    ) -> bool:
        """Insert one progress event if the parent job still exists.

        Args:
            optimization_id: ID of the job emitting the event.
            message: Human-readable event marker, or ``None``.
            metrics: Metric payload to persist with the event.
            timestamp: Timestamp assigned to the event row.

        Returns:
            Whether an event row was inserted.
        """
        session = self._get_session()
        try:
            exists = session.query(JobModel.optimization_id).filter(JobModel.optimization_id == optimization_id).first()
            if exists is None:
                return False
            session.add(
                ProgressEventModel(
                    optimization_id=optimization_id,
                    timestamp=timestamp,
                    event=message,
                    metrics=metrics,
                )
            )
            session.commit()
            return True
        finally:
            session.close()

    def _replace_latest_metrics(self, optimization_id: str, metrics: dict[str, Any]) -> None:
        """Replace the job's latest metrics snapshot without taking a row lock.

        Args:
            optimization_id: ID of the job to update.
            metrics: Latest metric snapshot from the subprocess.
        """
        session = self._get_session()
        try:
            session.query(JobModel).filter(JobModel.optimization_id == optimization_id).update(
                {JobModel.latest_metrics: metrics},
                synchronize_session=False,
            )
            session.commit()
        finally:
            session.close()

    def _latest_metrics_with_sticky_tqdm(self, optimization_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Splice the last-seen optimizer tqdm progress into a metric snapshot.

        ``latest_metrics`` is replaced wholesale on every progress event, but
        tqdm-driven optimizers (e.g. GEPA) tick their ``tqdm_*`` rollout bar only
        on ``optimizer_progress`` events while emitting many interleaved
        ``minibatch_feedback``/candidate events that carry no ``tqdm_*`` keys.
        Without carry-forward those interleaved events erase the progress bar and
        its stat cards between ticks. Remember the last-seen ``tqdm_*`` family per
        job and splice it back in so the bar persists for the life of the run.

        Args:
            optimization_id: ID of the job whose tqdm state is tracked.
            metrics: The incoming event's metric snapshot.

        Returns:
            ``metrics`` unchanged when it already carries ``tqdm_*`` keys (the
            sticky cache is refreshed in that case) or when no tqdm state has been
            seen yet; otherwise a new dict with the last-seen ``tqdm_*`` keys
            spliced beneath the incoming fields.
        """
        if not hasattr(self, "_tqdm_sticky"):
            self._tqdm_sticky = {}
            self._tqdm_sticky_lock = threading.Lock()
        incoming_tqdm = {key: value for key, value in metrics.items() if key.startswith(TQDM_KEY_PREFIX)}
        with self._tqdm_sticky_lock:
            if incoming_tqdm:
                self._tqdm_sticky[optimization_id] = incoming_tqdm
                return metrics
            carried = self._tqdm_sticky.get(optimization_id)
        if not carried:
            return metrics
        return {**carried, **metrics}

    def _should_trim_progress_events(self, optimization_id: str) -> bool:
        """Return whether this event should trigger a sampled retention trim.

        Args:
            optimization_id: ID of the job whose in-memory event counter is advanced.

        Returns:
            ``True`` every ``PROGRESS_TRIM_SAMPLE_RATE`` events for a job.
        """
        if not hasattr(self, "_progress_event_counters"):
            self._progress_event_counters = defaultdict(int)
            self._progress_counter_lock = threading.Lock()
        with self._progress_counter_lock:
            self._progress_event_counters[optimization_id] += 1
            return self._progress_event_counters[optimization_id] % PROGRESS_TRIM_SAMPLE_RATE == 0

    def _trim_progress_events(self, optimization_id: str) -> None:
        """Delete the oldest excess progress events for a job in one batch.

        Args:
            optimization_id: ID of the job whose retained progress rows should
                be brought back down to the configured cap.
        """
        session = self._get_session()
        try:
            event_count = (
                session.query(ProgressEventModel).filter(ProgressEventModel.optimization_id == optimization_id).count()
            )
            excess = event_count - self._progress_events_cap
            if excess <= 0:
                return

            # One ordered DELETE replaces the prior count→select→top-up→delete
            # round trips. Structural events sort last (flag 1) so they are
            # evicted only when non-structural rows alone cannot cover the
            # excess — identical preference to the old two-phase selection.
            structural_last = case(
                (ProgressEventModel.event.in_(STRUCTURAL_PROGRESS_EVENTS), 1),
                else_=0,
            )
            old_ids = (
                session.query(ProgressEventModel.id)
                .filter(ProgressEventModel.optimization_id == optimization_id)
                .order_by(structural_last.asc(), ProgressEventModel.timestamp.asc(), ProgressEventModel.id.asc())
                .limit(excess)
            )
            session.query(ProgressEventModel).filter(ProgressEventModel.id.in_(old_ids.scalar_subquery())).delete(
                synchronize_session=False
            )
            session.commit()
        finally:
            session.close()

    def get_progress_events(self, optimization_id: str, *, since: int = 0) -> list[ProgressEventRecord]:
        """Retrieve progress events for a job in chronological order.

        Args:
            optimization_id: ID of the job to inspect.
            since: Number of leading events to skip, for tail (delta) fetches;
                ``0`` returns the full history.

        Returns:
            Events ordered oldest-first, starting at offset ``since``.
        """
        session = self._get_session()
        try:
            query = (
                session.query(ProgressEventModel)
                .filter(ProgressEventModel.optimization_id == optimization_id)
                .order_by(ProgressEventModel.timestamp.asc(), ProgressEventModel.id.asc())
            )
            if since > 0:
                query = query.offset(since)
            events = query.all()
            return [
                cast(
                    ProgressEventRecord,
                    {
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        "event": e.event,
                        "metrics": e.metrics or {},
                    },
                )
                for e in events
            ]
        finally:
            session.close()

    def get_progress_count(self, optimization_id: str) -> int:
        """Return the number of progress events recorded for a job.

        Args:
            optimization_id: ID of the job to inspect.

        Returns:
            Number of stored events.
        """
        session = self._get_session()
        try:
            return (
                session.query(ProgressEventModel).filter(ProgressEventModel.optimization_id == optimization_id).count()
            )
        finally:
            session.close()

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
        """Append a log entry for a job.

        Silently discards the entry if the job no longer exists (a
        late log from a cleaned-up run is not an error). When the
        per-job log count reaches the configured retention cap, the oldest
        entry is evicted before the new one is inserted.

        Args:
            optimization_id: ID of the job emitting the log.
            level: Log level string (``INFO``, ``ERROR``, ...).
            logger_name: Originating logger name.
            message: Log line content.
            timestamp: Optional override for the entry timestamp; defaults to ``now``.
            pair_index: Optional grid-pair index when emitted from a sweep.
        """
        ts = timestamp or datetime.now(UTC)
        session = self._get_session()
        try:
            # An existence probe, not a critical section: appends are independent
            # inserts, so the per-job row lock only serialized writers and starved
            # the connection pool under concurrent log bursts. The cap eviction
            # below tolerates a transient over-count without correctness loss.
            exists = session.query(JobModel.optimization_id).filter(JobModel.optimization_id == optimization_id).first()
            if exists is None:
                logger.warning("Discarding log entry for missing job %s", optimization_id)
                return

            log_count = session.query(LogEntryModel).filter(LogEntryModel.optimization_id == optimization_id).count()
            if log_count >= self._log_entries_cap:
                oldest = (
                    session.query(LogEntryModel)
                    .filter(LogEntryModel.optimization_id == optimization_id)
                    .order_by(LogEntryModel.timestamp.asc())
                    .first()
                )
                if oldest:
                    session.delete(oldest)

            entry = LogEntryModel(
                optimization_id=optimization_id,
                timestamp=ts,
                level=level,
                logger=logger_name,
                message=message,
                pair_index=pair_index,
            )
            session.add(entry)

            session.commit()
        finally:
            session.close()

    def get_logs(
        self,
        optimization_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        level: str | None = None,
    ) -> list[LogEntryRecord]:
        """Retrieve log entries for a job, ordered ascending.

        Args:
            optimization_id: ID of the job to inspect.
            limit: Maximum number of entries to return; ``None`` means no cap.
            offset: Number of entries to skip.
            level: When set, restricts results to the given level.

        Returns:
            Matching log entries in chronological order.
        """
        session = self._get_session()
        try:
            q = session.query(LogEntryModel).filter(LogEntryModel.optimization_id == optimization_id)
            if level:
                q = q.filter(LogEntryModel.level == level)
            q = q.order_by(LogEntryModel.timestamp.asc())
            if offset:
                q = q.offset(offset)
            if limit is not None:
                q = q.limit(limit)
            logs = q.all()
            return [
                cast(
                    LogEntryRecord,
                    {
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                        "level": log.level,
                        "logger": log.logger,
                        "message": log.message,
                        "pair_index": log.pair_index,
                    },
                )
                for log in logs
            ]
        finally:
            session.close()

    def get_log_count(self, optimization_id: str, *, level: str | None = None) -> int:
        """Return the number of log entries for a job, optionally filtered by level.

        Args:
            optimization_id: ID of the job to inspect.
            level: When set, counts only entries at this level.

        Returns:
            Number of matching log entries.
        """
        session = self._get_session()
        try:
            q = session.query(LogEntryModel).filter(LogEntryModel.optimization_id == optimization_id)
            if level:
                q = q.filter(LogEntryModel.level == level)
            return q.count()
        finally:
            session.close()

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

        Progress and log counts are folded in via two aggregate
        queries so each returned row includes ``progress_count`` and
        ``log_count`` without N extra round trips.

        Args:
            status: Restrict to jobs with this status when set.
            username: Restrict to jobs owned by this user when set.
            optimization_type: Restrict to a particular run type when set.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip from the start.

        Returns:
            Matching ``JobRecord`` rows in newest-first order with
            ``progress_count`` and ``log_count`` populated.
        """
        session = self._get_session()
        try:
            q = session.query(JobModel).options(defer(JobModel.payload)).order_by(JobModel.created_at.desc())
            if status:
                q = q.filter(JobModel.status == status)
            if username:
                q = q.filter(JobModel.username == username)
            if optimization_type:
                q = q.filter(JobModel.optimization_type == optimization_type)
            jobs = q.offset(offset).limit(limit).all()
            return self._rows_with_counts(session, jobs)
        finally:
            session.close()

    def _rows_with_counts(self, session, jobs: list[JobModel]) -> list[JobRecord]:
        """Fold progress/log counts and summary text into job rows.

        Two aggregate queries (plus the embedding lookup) keyed on the page's
        ``optimization_ids`` so each row carries ``progress_count`` /
        ``log_count`` / ``summary_text`` without an N-per-row round trip. Shared
        by :meth:`list_jobs` and :meth:`list_jobs_shared_with`.

        Args:
            session: The open session the ``jobs`` were loaded on.
            jobs: The page of ``JobModel`` rows to enrich.

        Returns:
            The rows as ``JobRecord`` dicts with the folded counts.
        """
        optimization_ids = [j.optimization_id for j in jobs]
        progress_counts: dict[str, int] = (
            {
                row[0]: row[1]
                for row in session.query(ProgressEventModel.optimization_id, func.count())
                .filter(ProgressEventModel.optimization_id.in_(optimization_ids))
                .group_by(ProgressEventModel.optimization_id)
                .all()
            }
            if optimization_ids
            else {}
        )
        log_counts: dict[str, int] = (
            {
                row[0]: row[1]
                for row in session.query(LogEntryModel.optimization_id, func.count())
                .filter(LogEntryModel.optimization_id.in_(optimization_ids))
                .group_by(LogEntryModel.optimization_id)
                .all()
            }
            if optimization_ids
            else {}
        )
        summary_texts: dict[str, str | None] = (
            {
                row[0]: row[1]
                for row in session.query(
                    JobEmbeddingModel.optimization_id,
                    JobEmbeddingModel.summary_text,
                )
                .filter(JobEmbeddingModel.optimization_id.in_(optimization_ids))
                .all()
            }
            if optimization_ids
            else {}
        )
        result: list[JobRecord] = []
        for j in jobs:
            d = self._job_to_dict(j, include_payload=False)
            oid = str(j.optimization_id)
            d["progress_count"] = progress_counts.get(oid, 0)
            d["log_count"] = log_counts.get(oid, 0)
            d["summary_text"] = summary_texts.get(oid)
            result.append(d)
        return result

    def list_jobs_shared_with(self, username: str, *, limit: int = 50, offset: int = 0) -> list[JobRecord]:
        """List jobs shared with ``username`` via a member grant, newest first.

        Joins ``optimization_share_grants`` on ``grantee_username`` so a member
        sees runs they were invited to but do not own. The same count-folding as
        :meth:`list_jobs` is applied; the grant role is not attached here (the
        caller resolves roles for the page via
        :func:`core.api.sharing_access.list_grants_for_user`).

        Args:
            username: Grantee username (compared case-insensitively).
            limit: Maximum number of rows to return.
            offset: Number of rows to skip from the start.

        Returns:
            Matching ``JobRecord`` rows in newest-first order.
        """
        normalized = username.strip().lower()
        session = self._get_session()
        try:
            jobs = (
                session.query(JobModel)
                .options(defer(JobModel.payload))
                .join(
                    OptimizationShareGrantModel,
                    OptimizationShareGrantModel.optimization_id == JobModel.optimization_id,
                )
                .filter(OptimizationShareGrantModel.grantee_username == normalized)
                .order_by(JobModel.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return self._rows_with_counts(session, jobs)
        finally:
            session.close()

    def count_jobs_shared_with(self, username: str) -> int:
        """Count jobs shared with ``username`` via a member grant.

        Args:
            username: Grantee username (compared case-insensitively).

        Returns:
            Number of optimizations the user holds a grant on.
        """
        normalized = username.strip().lower()
        session = self._get_session()
        try:
            return (
                session.query(func.count(OptimizationShareGrantModel.optimization_id))
                .join(
                    JobModel,
                    OptimizationShareGrantModel.optimization_id == JobModel.optimization_id,
                )
                .filter(OptimizationShareGrantModel.grantee_username == normalized)
                .scalar()
                or 0
            )
        finally:
            session.close()

    def list_jobs_visible_to(
        self,
        username: str,
        *,
        status: str | None = None,
        optimization_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List jobs the caller owns or was granted access to, newest first.

        Unions the caller's own jobs with jobs shared to them via a member
        grant (Drive-style sharing), de-duplicated, ordered newest-first and
        paginated as a single set. Powers the unified control panel, where a
        collaborator sees their own runs and runs shared with them together.

        Args:
            username: The caller. Owned rows match exactly (mirroring
                :meth:`list_jobs`); grant rows match case-insensitively
                because grants store a lowercased grantee.
            status: Restrict to jobs with this status when set.
            optimization_type: Restrict to a particular run type when set.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip from the start.

        Returns:
            Matching ``JobRecord`` rows in newest-first order with
            ``progress_count`` / ``log_count`` / ``summary_text`` folded in.
        """
        normalized = username.strip().lower()
        session = self._get_session()
        try:
            grant_ids = session.query(OptimizationShareGrantModel.optimization_id).filter(
                OptimizationShareGrantModel.grantee_username == normalized
            )
            q = (
                session.query(JobModel)
                .options(defer(JobModel.payload))
                .filter(or_(JobModel.username == username, JobModel.optimization_id.in_(grant_ids)))
            )
            if status:
                q = q.filter(JobModel.status == status)
            if optimization_type:
                q = q.filter(JobModel.optimization_type == optimization_type)
            jobs = q.order_by(JobModel.created_at.desc()).offset(offset).limit(limit).all()
            return self._rows_with_counts(session, jobs)
        finally:
            session.close()

    def count_jobs_visible_to(
        self, username: str, *, status: str | None = None, optimization_type: str | None = None
    ) -> int:
        """Count jobs the caller owns or was granted access to.

        The union counterpart of :meth:`list_jobs_visible_to`; matching rules
        are identical.

        Args:
            username: The caller (owned exact, grants case-insensitive).
            status: Restrict count to this status when set.
            optimization_type: Restrict count to this run type when set.

        Returns:
            Number of optimizations visible to the caller under the filters.
        """
        normalized = username.strip().lower()
        session = self._get_session()
        try:
            grant_ids = session.query(OptimizationShareGrantModel.optimization_id).filter(
                OptimizationShareGrantModel.grantee_username == normalized
            )
            q = session.query(func.count(JobModel.optimization_id)).filter(
                or_(JobModel.username == username, JobModel.optimization_id.in_(grant_ids))
            )
            if status:
                q = q.filter(JobModel.status == status)
            if optimization_type:
                q = q.filter(JobModel.optimization_type == optimization_type)
            return q.scalar() or 0
        finally:
            session.close()

    def count_jobs(
        self, *, status: str | None = None, username: str | None = None, optimization_type: str | None = None
    ) -> int:
        """Count jobs matching the given filters.

        Args:
            status: Restrict count to this status when set.
            username: Restrict count to this owner when set.
            optimization_type: Restrict count to this run type when set.

        Returns:
            Number of matching rows.
        """
        session = self._get_session()
        try:
            q = session.query(func.count(JobModel.optimization_id))
            if status:
                q = q.filter(JobModel.status == status)
            if username:
                q = q.filter(JobModel.username == username)
            if optimization_type:
                q = q.filter(JobModel.optimization_type == optimization_type)
            return q.scalar() or 0
        finally:
            session.close()

    def get_queue_metrics(self) -> tuple[int, float]:
        """Return pending-job depth and oldest pending-job age in seconds.

        Returns:
            ``(pending_count, queue_age_seconds)`` with age set to ``0.0`` when
            no jobs are pending.
        """
        if self._engine.dialect.name == "postgresql":
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT
                            count(*)::int AS pending_count,
                            COALESCE(EXTRACT(EPOCH FROM (now() - min(created_at))), 0) AS queue_age_seconds
                        FROM jobs
                        WHERE status = 'pending'
                        """
                    )
                ).one()
            return int(row[0] or 0), float(row[1] or 0.0)

        session = self._get_session()
        try:
            pending_count, oldest_created_at = (
                session.query(func.count(JobModel.optimization_id), func.min(JobModel.created_at))
                .filter(JobModel.status == "pending")
                .one()
            )
            if oldest_created_at is None:
                return int(pending_count or 0), 0.0
            if oldest_created_at.tzinfo is None:
                oldest_created_at = oldest_created_at.replace(tzinfo=UTC)
            age_seconds = max(0.0, (datetime.now(UTC) - oldest_created_at).total_seconds())
            return int(pending_count or 0), age_seconds
        finally:
            session.close()

    def stage_dataset(self, username: str, dataset_filename: str, rows: list[dict[str, Any]]) -> str:
        """Persist wizard-parsed dataset rows for an agent-driven submit.

        Args:
            username: Submitter who owns the staged copy.
            dataset_filename: Original filename for diagnostics.
            rows: Parsed dataset rows; must be non-empty.

        Returns:
            The opaque staged-dataset id used by ``/run``.

        Raises:
            ValueError: When ``rows`` is empty.
        """
        if not rows:
            raise ValueError("staged dataset rows must be non-empty")
        staged_id = uuid4().hex
        session = self._get_session()
        try:
            session.add(
                AgentStagedDatasetModel(
                    id=staged_id,
                    username=username,
                    dataset_filename=dataset_filename,
                    rows=rows,
                    row_count=len(rows),
                )
            )
            session.commit()
            return staged_id
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def get_staged_dataset(self, staged_dataset_id: str, username: str) -> list[dict[str, Any]] | None:
        """Fetch staged rows by id, scoped to the calling user.

        Args:
            staged_dataset_id: Id previously returned by ``stage_dataset``.
            username: Authenticated caller — scope guards against cross-user reads.

        Returns:
            The persisted rows, or ``None`` when the row is missing or owned
            by another user.
        """
        session = self._get_session()
        try:
            row = (
                session.query(AgentStagedDatasetModel)
                .filter(
                    AgentStagedDatasetModel.id == staged_dataset_id,
                    AgentStagedDatasetModel.username == username,
                )
                .one_or_none()
            )
            if row is None:
                return None
            return list(row.rows)
        finally:
            session.close()

    def delete_staged_dataset(self, staged_dataset_id: str, username: str) -> bool:
        """Drop a staged dataset once it has been consumed.

        Args:
            staged_dataset_id: Id to evict.
            username: Authenticated caller — scope guards against cross-user deletes.

        Returns:
            ``True`` when a row was deleted, ``False`` when none matched.
        """
        session = self._get_session()
        try:
            deleted = (
                session.query(AgentStagedDatasetModel)
                .filter(
                    AgentStagedDatasetModel.id == staged_dataset_id,
                    AgentStagedDatasetModel.username == username,
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            return bool(deleted)
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()
