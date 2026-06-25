"""Unified per-user storage accounting for the Skynet storage budget.

A single number — the sum of every table a user's data lands in — backs the
storage budget that is enforced at submit/save time and surfaced in the usage
meter. Bytes are attributed to the standalone artifact that owns them: a job's
``payload`` + ``result`` plus the logs, progress events and embeddings it spawns
all count as the optimization's footprint, and a conversation's messages plus
its embeddings count as the chat's. The two dominant contributors (a job's
columns and the dataset blobs) are read from precomputed indexed columns
(``jobs.stored_bytes`` / ``datasets.byte_size``); the byproducts that fold into
them are sized live with a dialect-aware byte expression so the meter and the
gate agree to the byte.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, Text, cast, func, literal, select
from sqlalchemy.orm import Session

from ..config import settings
from ..constants import PAYLOAD_OVERVIEW_NAME
from .models import (
    EMBEDDING_DIM,
    AgentConversationModel,
    AgentMessageModel,
    AgentStagedDatasetModel,
    ConversationEmbeddingModel,
    DatasetModel,
    GepaCheckpointModel,
    GridPairResultModel,
    JobEmbeddingModel,
    JobModel,
    LogEntryModel,
    ProgressEventModel,
)

logger = logging.getLogger(__name__)

# A stored pgvector is dim×4 bytes (float32). ``job_embeddings`` holds three
# such vectors per row; ``conversation_embeddings`` holds one. Counted by row
# rather than measured because ``octet_length`` does not apply to the vector
# type and casting it to text would size the decimal repr, not the storage.
_VECTOR_BYTES = EMBEDDING_DIM * 4
_JOB_EMBEDDING_ROW_BYTES = _VECTOR_BYTES * 3
_CONVERSATION_EMBEDDING_ROW_BYTES = _VECTOR_BYTES

# The categories shown in the storage breakdown. Each is a standalone artifact
# the user can open in a cleanup drawer and delete directly. Bytes that exist
# only because of a parent — logs, progress events and embeddings — are folded
# into that parent's footprint (optimization or chat) rather than listed on
# their own, so every category here is independently deletable.
STORAGE_CATEGORIES = (
    "optimizations",
    "datasets",
    "agent_chats",
    "staged_uploads",
)


@dataclass(frozen=True)
class StorageUsage:
    """A user's total stored bytes and the per-category breakdown behind it.

    ``breakdown`` is keyed by :data:`STORAGE_CATEGORIES`; ``total`` is their sum.
    """

    total: int
    breakdown: dict[str, int]


@dataclass(frozen=True)
class StorageItem:
    """One owned object ranked by its individual storage footprint.

    ``type`` is one of ``"optimization"``, ``"dataset"``, ``"chat"`` or
    ``"staged_upload"`` and ``id`` is that object's primary key, so the cleanup
    UI can deep-link to it and route its delete. ``bytes`` is the object's full
    footprint — for an optimization or chat it includes the byproducts folded
    into it (logs, progress events, embeddings) — so the per-item sizes sum back
    to the category total the meter reports, and deleting the item frees exactly
    that many bytes.
    """

    id: str
    type: str
    name: str
    bytes: int


def json_byte_size(value: object) -> int:
    """Return the UTF-8 byte length of ``value`` serialized as compact JSON.

    Args:
        value: A JSON-serializable mapping or list, or ``None``.

    Returns:
        ``0`` when ``value`` is ``None``; otherwise the encoded length. Used to
        size a job's JSON columns at write time and to size an incoming
        submission before it is persisted.
    """
    if value is None:
        return 0
    return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))


def _byte_size(column: object, dialect_name: str):
    """Return a SQL expression for the byte length of ``column`` rendered as text.

    Args:
        column: The model column to size.
        dialect_name: ``engine.dialect.name`` — selects ``octet_length``
            (Postgres) versus ``length`` (SQLite and others, where there is no
            ``octet_length``).

    Returns:
        A SQLAlchemy expression yielding the per-row byte length, ``0`` on NULL.
    """
    as_text = cast(column, Text)
    length_fn = func.octet_length if dialect_name == "postgresql" else func.length
    return func.coalesce(length_fn(as_text), 0)


def compute_user_storage(engine: Engine, username: str) -> StorageUsage:
    """Sum every byte the user owns across the database into one budget figure.

    Args:
        engine: The shared SQLAlchemy engine all storage tables live on.
        username: The owner whose footprint is measured; matched
            case-insensitively against the lowercased owner columns.

    Returns:
        A :class:`StorageUsage` with the per-category breakdown and total. An
        empty/blank username yields an all-zero usage rather than scanning.
    """
    normalized = (username or "").strip().lower()
    empty = StorageUsage(total=0, breakdown=dict.fromkeys(STORAGE_CATEGORIES, 0))
    if not normalized:
        return empty

    dialect = engine.dialect.name
    with Session(engine) as session:
        user_job_ids = select(JobModel.optimization_id).where(JobModel.username == normalized)
        user_conversation_ids = select(AgentConversationModel.id).where(
            AgentConversationModel.username == normalized
        )

        def scalar(statement) -> int:
            """Execute a single-aggregate select and coerce the result to int."""
            return int(session.execute(statement).scalar() or 0)

        optimizations = scalar(
            select(func.coalesce(func.sum(JobModel.stored_bytes), 0)).where(JobModel.username == normalized)
        )
        datasets = scalar(
            select(func.coalesce(func.sum(DatasetModel.byte_size), 0)).where(
                DatasetModel.owner_username == normalized
            )
        )
        logs = scalar(
            select(func.coalesce(func.sum(_byte_size(LogEntryModel.message, dialect)), 0)).where(
                LogEntryModel.optimization_id.in_(user_job_ids)
            )
        )
        progress_events = scalar(
            select(func.coalesce(func.sum(_byte_size(ProgressEventModel.metrics, dialect)), 0)).where(
                ProgressEventModel.optimization_id.in_(user_job_ids)
            )
        )
        checkpoints = scalar(
            select(func.coalesce(func.sum(GepaCheckpointModel.stored_bytes), 0)).where(
                GepaCheckpointModel.optimization_id.in_(user_job_ids)
            )
        )
        grid_pairs = scalar(
            select(func.coalesce(func.sum(GridPairResultModel.stored_bytes), 0)).where(
                GridPairResultModel.optimization_id.in_(user_job_ids)
            )
        )
        agent_chats = scalar(
            select(
                func.coalesce(
                    func.sum(
                        _byte_size(AgentMessageModel.content, dialect)
                        + _byte_size(AgentMessageModel.tool_calls, dialect)
                    ),
                    0,
                )
            ).where(AgentMessageModel.conversation_id.in_(user_conversation_ids))
        )
        staged_uploads = scalar(
            select(func.coalesce(func.sum(_byte_size(AgentStagedDatasetModel.rows, dialect)), 0)).where(
                AgentStagedDatasetModel.username == normalized
            )
        )
        # job_embeddings / conversation_embeddings only exist on the semantic
        # backend; on SEARCH_BACKEND=lexical the tables are never created, so
        # there is nothing to count and the query would hit a missing relation.
        job_embedding_rows = (
            scalar(
                select(func.count(JobEmbeddingModel.optimization_id)).where(JobEmbeddingModel.user_id == normalized)
            )
            if settings.embeddings_enabled
            else 0
        )
        conversation_embedding_rows = (
            scalar(
                select(func.count(ConversationEmbeddingModel.conversation_id)).where(
                    ConversationEmbeddingModel.username == normalized
                )
            )
            if settings.embeddings_enabled
            else 0
        )
        optimization_embeddings = job_embedding_rows * _JOB_EMBEDDING_ROW_BYTES
        chat_embeddings = conversation_embedding_rows * _CONVERSATION_EMBEDDING_ROW_BYTES

    breakdown = {
        "optimizations": optimizations
        + logs
        + progress_events
        + checkpoints
        + grid_pairs
        + optimization_embeddings,
        "datasets": datasets,
        "agent_chats": agent_chats + chat_embeddings,
        "staged_uploads": staged_uploads,
    }
    return StorageUsage(total=sum(breakdown.values()), breakdown=breakdown)


def purge_expired_staged_datasets(engine: Engine, *, max_age_seconds: int) -> int:
    """Delete abandoned wizard staged datasets older than ``max_age_seconds``.

    A staged row is a transient parking spot for a wizard upload: a successful
    submit evicts it, but a wizard abandoned before submit leaves the row
    orphaned forever. The human submit path never reads a staged row back — it
    posts the rows inline — and an agent submit-by-id consumes one within
    seconds of staging, so a short TTL reclaims abandoned rows without any risk
    to an in-flight submit.

    Args:
        engine: SQLAlchemy engine the ``agent_staged_datasets`` table is on.
        max_age_seconds: Age past ``created_at`` after which a staged row is
            eligible for deletion.

    Returns:
        The number of staged rows deleted in this sweep.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    with Session(engine) as session:
        deleted = (
            session.query(AgentStagedDatasetModel)
            .filter(AgentStagedDatasetModel.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        session.commit()
    if deleted:
        logger.info("Purged %d expired staged dataset(s) older than %ds", deleted, max_age_seconds)
    return int(deleted)


def compute_user_storage_items(engine: Engine, username: str, limit: int = 20) -> list[StorageItem]:
    """Rank a user's largest individual items across optimizations, datasets and chats.

    Backs the "biggest items" cleanup list on the storage page: each category is
    queried for its top ``limit`` rows by footprint, the three lists are merged,
    and the overall largest ``limit`` are returned. Per-row byte figures mirror
    exactly what :func:`compute_user_storage` counts for the same category, so a
    listed item never reports a size the meter does not also attribute to it.

    Args:
        engine: The shared SQLAlchemy engine all storage tables live on.
        username: The owner whose items are ranked; matched case-insensitively.
        limit: Maximum number of items to return, and the per-category fetch cap.

    Returns:
        Up to ``limit`` :class:`StorageItem` rows ordered by descending size. An
        empty/blank username yields an empty list rather than scanning.
    """
    normalized = (username or "").strip().lower()
    if not normalized:
        return []

    dialect = engine.dialect.name
    items: list[StorageItem] = []
    with Session(engine) as session:
        optimization_rows = session.execute(
            select(JobModel.optimization_id, JobModel.payload_overview, JobModel.stored_bytes)
            .where(JobModel.username == normalized, JobModel.stored_bytes > 0)
            .order_by(JobModel.stored_bytes.desc())
            .limit(limit)
        ).all()
        for optimization_id, payload_overview, stored_bytes in optimization_rows:
            name = (payload_overview or {}).get(PAYLOAD_OVERVIEW_NAME) or optimization_id
            items.append(
                StorageItem(
                    id=optimization_id,
                    type="optimization",
                    name=str(name),
                    bytes=int(stored_bytes or 0),
                )
            )

        dataset_rows = session.execute(
            select(DatasetModel.id, DatasetModel.name, DatasetModel.byte_size)
            .where(DatasetModel.owner_username == normalized, DatasetModel.byte_size > 0)
            .order_by(DatasetModel.byte_size.desc())
            .limit(limit)
        ).all()
        for dataset_id, name, byte_size in dataset_rows:
            items.append(StorageItem(id=dataset_id, type="dataset", name=name, bytes=int(byte_size or 0)))

        chat_bytes = func.sum(
            _byte_size(AgentMessageModel.content, dialect) + _byte_size(AgentMessageModel.tool_calls, dialect)
        )
        conversation_rows = session.execute(
            select(AgentConversationModel.id, AgentConversationModel.title, chat_bytes.label("chat_size"))
            .join(AgentMessageModel, AgentMessageModel.conversation_id == AgentConversationModel.id)
            .where(AgentConversationModel.username == normalized)
            .group_by(AgentConversationModel.id, AgentConversationModel.title)
            .order_by(func.coalesce(chat_bytes, 0).desc())
            .limit(limit)
        ).all()
        for conversation_id, title, chat_size in conversation_rows:
            size = int(chat_size or 0)
            if size <= 0:
                continue
            items.append(StorageItem(id=conversation_id, type="chat", name=title or conversation_id, bytes=size))

    items.sort(key=lambda item: item.bytes, reverse=True)
    return items[:limit]


def compute_user_storage_category_items(
    engine: Engine, username: str, category: str, limit: int = 1000
) -> list[StorageItem]:
    """List every deletable item in one storage category, largest first.

    Backs the per-category cleanup drawer on the storage page. Unlike
    :func:`compute_user_storage_items` — which merges a capped top-N across
    categories — this returns the full set for a single category so the drawer
    shows all of the user's data, not just the biggest. An optimization's or
    chat's size folds in its byproducts (logs, progress events, embeddings)
    exactly as :func:`compute_user_storage` attributes them, so the per-item
    sizes sum back to that category's meter total.

    Args:
        engine: The shared SQLAlchemy engine all storage tables live on.
        username: The owner whose items are listed; matched case-insensitively.
        category: One of :data:`STORAGE_CATEGORIES`. Any other value — including
            a folded-away byproduct name such as ``embeddings`` — yields an empty
            list rather than scanning.
        limit: Defensive upper bound on rows returned for the category.

    Returns:
        The category's :class:`StorageItem` rows ordered by descending size. An
        empty/blank username or an unknown category yields an empty list.
    """
    normalized = (username or "").strip().lower()
    if not normalized or category not in STORAGE_CATEGORIES:
        return []

    dialect = engine.dialect.name
    items: list[StorageItem] = []
    with Session(engine) as session:
        if category == "optimizations":
            logs_bytes = (
                select(func.coalesce(func.sum(_byte_size(LogEntryModel.message, dialect)), 0))
                .where(LogEntryModel.optimization_id == JobModel.optimization_id)
                .scalar_subquery()
            )
            progress_bytes = (
                select(func.coalesce(func.sum(_byte_size(ProgressEventModel.metrics, dialect)), 0))
                .where(ProgressEventModel.optimization_id == JobModel.optimization_id)
                .scalar_subquery()
            )
            embedding_bytes = (
                select(func.count(JobEmbeddingModel.optimization_id) * _JOB_EMBEDDING_ROW_BYTES)
                .where(JobEmbeddingModel.optimization_id == JobModel.optimization_id)
                .scalar_subquery()
                if settings.embeddings_enabled
                else literal(0)
            )
            checkpoint_bytes = (
                select(func.coalesce(func.sum(GepaCheckpointModel.stored_bytes), 0))
                .where(GepaCheckpointModel.optimization_id == JobModel.optimization_id)
                .scalar_subquery()
            )
            grid_pair_bytes = (
                select(func.coalesce(func.sum(GridPairResultModel.stored_bytes), 0))
                .where(GridPairResultModel.optimization_id == JobModel.optimization_id)
                .scalar_subquery()
            )
            footprint = (
                JobModel.stored_bytes
                + logs_bytes
                + progress_bytes
                + checkpoint_bytes
                + grid_pair_bytes
                + embedding_bytes
            ).label("footprint")
            optimization_rows = session.execute(
                select(JobModel.optimization_id, JobModel.payload_overview, footprint)
                .where(JobModel.username == normalized, JobModel.stored_bytes > 0)
                .order_by(footprint.desc())
                .limit(limit)
            ).all()
            for optimization_id, payload_overview, footprint_bytes in optimization_rows:
                name = (payload_overview or {}).get(PAYLOAD_OVERVIEW_NAME) or optimization_id
                items.append(
                    StorageItem(
                        id=optimization_id,
                        type="optimization",
                        name=str(name),
                        bytes=int(footprint_bytes or 0),
                    )
                )
        elif category == "datasets":
            dataset_rows = session.execute(
                select(DatasetModel.id, DatasetModel.name, DatasetModel.byte_size)
                .where(DatasetModel.owner_username == normalized, DatasetModel.byte_size > 0)
                .order_by(DatasetModel.byte_size.desc())
                .limit(limit)
            ).all()
            for dataset_id, name, byte_size in dataset_rows:
                items.append(StorageItem(id=dataset_id, type="dataset", name=name, bytes=int(byte_size or 0)))
        elif category == "agent_chats":
            chat_bytes = func.sum(
                _byte_size(AgentMessageModel.content, dialect) + _byte_size(AgentMessageModel.tool_calls, dialect)
            )
            embedding_bytes = (
                select(func.count(ConversationEmbeddingModel.conversation_id) * _CONVERSATION_EMBEDDING_ROW_BYTES)
                .where(ConversationEmbeddingModel.conversation_id == AgentConversationModel.id)
                .scalar_subquery()
                if settings.embeddings_enabled
                else literal(0)
            )
            footprint = (func.coalesce(chat_bytes, 0) + embedding_bytes).label("chat_size")
            conversation_rows = session.execute(
                select(AgentConversationModel.id, AgentConversationModel.title, footprint)
                .join(AgentMessageModel, AgentMessageModel.conversation_id == AgentConversationModel.id)
                .where(AgentConversationModel.username == normalized)
                .group_by(AgentConversationModel.id, AgentConversationModel.title)
                .order_by(footprint.desc())
                .limit(limit)
            ).all()
            for conversation_id, title, chat_size in conversation_rows:
                size = int(chat_size or 0)
                if size <= 0:
                    continue
                items.append(StorageItem(id=conversation_id, type="chat", name=title or conversation_id, bytes=size))
        else:  # staged_uploads
            staged_bytes = _byte_size(AgentStagedDatasetModel.rows, dialect)
            staged_rows = session.execute(
                select(
                    AgentStagedDatasetModel.id,
                    AgentStagedDatasetModel.dataset_filename,
                    staged_bytes.label("staged_size"),
                )
                .where(AgentStagedDatasetModel.username == normalized)
                .order_by(staged_bytes.desc())
                .limit(limit)
            ).all()
            for staged_id, filename, staged_size in staged_rows:
                size = int(staged_size or 0)
                if size <= 0:
                    continue
                items.append(StorageItem(id=staged_id, type="staged_upload", name=filename or staged_id, bytes=size))

    return items
