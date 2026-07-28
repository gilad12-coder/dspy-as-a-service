"""Persistence for the personal dataset library — metadata plus a blob seam.

A saved dataset is a single file: lean metadata in the ``datasets`` table and
the row bytes one-to-one in ``dataset_blobs``. The bytes are serialized to
compact JSON and gzip-compressed, so the metadata table stays narrow and list
queries never drag the rows along. Blob read/write goes through the
:class:`DatasetBlobStore` seam so the bytes can later move to S3/MinIO without
touching the metadata schema, the router, or the UI.

Quota and dedupe are accounted on the compressed ``stored_bytes``: a re-save of
byte-identical rows returns the existing dataset instead of storing a second
copy, and the per-user quota sums ``stored_bytes`` across the owner's library.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import DatasetBlobModel, DatasetModel

_CONTENT_TYPE_JSON = "json"
_COMPRESSION_GZIP = "gzip"
# JSON separators with no whitespace — the canonical on-disk form whose SHA-256
# is the dedupe key, so byte-identical rows always hash identically.
_JSON_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class DatasetBlob:
    """The stored bytes for one dataset plus how they were encoded."""

    content_type: str
    compression: str
    data: bytes


@dataclass(frozen=True)
class StagedDataset:
    """Serialized-and-compressed rows plus the metadata derived from them.

    Produced once by :meth:`DatasetLibraryStore.stage` so the caller can gate on
    size/dedupe/quota before paying to persist, then committed with the same
    bytes — avoiding a re-serialization per gate.
    """

    digest: str
    byte_size: int
    stored_bytes: int
    row_count: int
    column_count: int
    column_schema: dict[str, Any]
    compressed: bytes


@dataclass(frozen=True)
class DatasetRecord:
    """A library dataset's metadata, decoupled from the ORM row."""

    id: str
    owner_username: str
    name: str
    source: str
    row_count: int
    column_count: int
    byte_size: int
    stored_bytes: int
    content_hash: str
    column_schema: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    """Encode dataset rows as canonical compact UTF-8 JSON bytes.

    Args:
        rows: The dataset rows to serialize.

    Returns:
        The uncompressed canonical JSON encoding whose SHA-256 is the dedupe key.
    """
    return json.dumps(rows, ensure_ascii=False, separators=_JSON_SEPARATORS).encode("utf-8")


def deserialize_rows(raw: bytes) -> list[dict[str, Any]]:
    """Decode rows previously produced by :func:`serialize_rows`.

    Args:
        raw: Uncompressed JSON bytes.

    Returns:
        The decoded dataset rows.
    """
    return json.loads(raw.decode("utf-8"))


def content_hash(raw: bytes) -> str:
    """Return the lowercase hex SHA-256 of the uncompressed canonical bytes.

    Args:
        raw: Uncompressed serialized rows.

    Returns:
        The 64-char hex digest used as the per-owner dedupe key.
    """
    return hashlib.sha256(raw).hexdigest()


class DatasetBlobStore(Protocol):
    """Swappable seam for reading and writing the compressed dataset bytes."""

    def put(self, dataset_id: str, blob: DatasetBlob) -> None:
        """Persist (or replace) the bytes for ``dataset_id``."""
        ...

    def get(self, dataset_id: str) -> DatasetBlob | None:
        """Return the stored bytes for ``dataset_id``, or ``None`` when absent."""
        ...

    def delete(self, dataset_id: str) -> None:
        """Remove the bytes for ``dataset_id`` (no-op when absent)."""
        ...


class PostgresDatasetBlobStore:
    """:class:`DatasetBlobStore` backed by the ``dataset_blobs`` table."""

    def __init__(self, engine: Engine) -> None:
        """Bind the blob store to a SQLAlchemy engine.

        Args:
            engine: Engine whose schema carries ``dataset_blobs``.
        """
        self._engine = engine

    def put(self, dataset_id: str, blob: DatasetBlob) -> None:
        """Insert or replace the compressed bytes for one dataset.

        Args:
            dataset_id: Owning dataset id.
            blob: Encoded bytes plus their content-type and compression.
        """
        with Session(self._engine) as session:
            existing = session.get(DatasetBlobModel, dataset_id)
            if existing is None:
                session.add(
                    DatasetBlobModel(
                        dataset_id=dataset_id,
                        content_type=blob.content_type,
                        compression=blob.compression,
                        data=blob.data,
                    )
                )
            else:
                existing.content_type = blob.content_type
                existing.compression = blob.compression
                existing.data = blob.data
            session.commit()

    def get(self, dataset_id: str) -> DatasetBlob | None:
        """Return the stored bytes for ``dataset_id``.

        Args:
            dataset_id: Dataset whose bytes are read.

        Returns:
            The :class:`DatasetBlob`, or ``None`` when no row exists.
        """
        with Session(self._engine) as session:
            row = session.get(DatasetBlobModel, dataset_id)
            if row is None:
                return None
            return DatasetBlob(content_type=row.content_type, compression=row.compression, data=row.data)

    def delete(self, dataset_id: str) -> None:
        """Remove the bytes for ``dataset_id`` if present.

        Args:
            dataset_id: Dataset whose bytes are dropped.
        """
        with Session(self._engine) as session:
            session.execute(delete(DatasetBlobModel).where(DatasetBlobModel.dataset_id == dataset_id))
            session.commit()


def _to_record(row: DatasetModel) -> DatasetRecord:
    """Project a ``DatasetModel`` ORM row onto an immutable :class:`DatasetRecord`.

    Args:
        row: The loaded ORM row.

    Returns:
        The detached metadata record.
    """
    return DatasetRecord(
        id=row.id,
        owner_username=row.owner_username,
        name=row.name,
        source=row.source,
        row_count=row.row_count,
        column_count=row.column_count,
        byte_size=row.byte_size,
        stored_bytes=row.stored_bytes,
        content_hash=row.content_hash,
        column_schema=dict(row.column_schema or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _column_count(rows: list[dict[str, Any]], column_schema: dict[str, Any]) -> int:
    """Derive the dataset's column count from its schema or first row.

    Args:
        rows: The dataset rows.
        column_schema: Saved column schema; its ``column_order`` wins when present.

    Returns:
        The number of columns.
    """
    order = column_schema.get("column_order")
    if isinstance(order, list) and order:
        return len(order)
    return len(rows[0]) if rows else 0


class DatasetLibraryStore:
    """Metadata CRUD plus blob lifecycle for the personal dataset library."""

    def __init__(self, engine: Engine, blob_store: DatasetBlobStore) -> None:
        """Bind the library store to an engine and a blob seam.

        Args:
            engine: Engine whose schema carries ``datasets``.
            blob_store: Seam used to persist the compressed row bytes.
        """
        self._engine = engine
        self._blobs = blob_store

    def used_bytes(self, owner_username: str) -> int:
        """Return the owner's total compressed library size for quota checks.

        Args:
            owner_username: Lowercased dataset owner.

        Returns:
            The sum of ``stored_bytes`` across the owner's datasets (0 when none).
        """
        with Session(self._engine) as session:
            total = session.scalar(
                select(func.coalesce(func.sum(DatasetModel.stored_bytes), 0)).where(
                    DatasetModel.owner_username == owner_username
                )
            )
        return int(total or 0)

    def find_by_hash(self, owner_username: str, digest: str) -> DatasetRecord | None:
        """Return the owner's dataset with this content hash, if one exists.

        Args:
            owner_username: Lowercased dataset owner.
            digest: Content-hash dedupe key.

        Returns:
            The matching :class:`DatasetRecord`, or ``None``.
        """
        with Session(self._engine) as session:
            row = session.scalars(
                select(DatasetModel)
                .where(DatasetModel.owner_username == owner_username, DatasetModel.content_hash == digest)
                .limit(1)
            ).one_or_none()
            return _to_record(row) if row is not None else None

    def stage(self, rows: list[dict[str, Any]], column_schema: dict[str, Any]) -> StagedDataset:
        """Serialize, hash, and compress rows once for size/dedupe/quota gating.

        Args:
            rows: The dataset rows to stage.
            column_schema: Saved column roles/kinds/order for wizard pre-fill.

        Returns:
            A :class:`StagedDataset` carrying the digest, sizes, and the
            compressed bytes ready for :meth:`commit_staged`.
        """
        raw = serialize_rows(rows)
        compressed = gzip.compress(raw)
        return StagedDataset(
            digest=content_hash(raw),
            byte_size=len(raw),
            stored_bytes=len(compressed),
            row_count=len(rows),
            column_count=_column_count(rows, column_schema),
            column_schema=column_schema,
            compressed=compressed,
        )

    def commit_staged(
        self, *, owner_username: str, name: str, source: str, staged: StagedDataset
    ) -> DatasetRecord:
        """Persist staged bytes as a new dataset (caller has gated size/quota).

        Args:
            owner_username: Lowercased dataset owner.
            name: Display name for the library entry.
            source: Producing surface (``'upload'`` / ``'tagger'`` / ``'optimization'``).
            staged: The output of :meth:`stage`.

        Returns:
            The persisted :class:`DatasetRecord`.
        """
        dataset_id = uuid4().hex
        now = datetime.now(UTC)
        record = DatasetModel(
            id=dataset_id,
            owner_username=owner_username,
            name=name,
            source=source,
            row_count=staged.row_count,
            column_count=staged.column_count,
            byte_size=staged.byte_size,
            stored_bytes=staged.stored_bytes,
            content_hash=staged.digest,
            column_schema=staged.column_schema,
            created_at=now,
            updated_at=now,
        )
        with Session(self._engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            result = _to_record(record)
        self._blobs.put(
            dataset_id,
            DatasetBlob(content_type=_CONTENT_TYPE_JSON, compression=_COMPRESSION_GZIP, data=staged.compressed),
        )
        return result

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        """Return one dataset's metadata, or ``None`` when unknown.

        Args:
            dataset_id: Dataset id.

        Returns:
            The :class:`DatasetRecord`, or ``None``.
        """
        with Session(self._engine) as session:
            row = session.get(DatasetModel, dataset_id)
            return _to_record(row) if row is not None else None

    def list_datasets(self, owner_username: str) -> list[DatasetRecord]:
        """Return the owner's datasets, newest first.

        Args:
            owner_username: Lowercased dataset owner.

        Returns:
            The owner's :class:`DatasetRecord` list ordered by ``updated_at`` desc.
        """
        with Session(self._engine) as session:
            rows = session.scalars(
                select(DatasetModel)
                .where(DatasetModel.owner_username == owner_username)
                .order_by(DatasetModel.updated_at.desc())
            ).all()
            return [_to_record(row) for row in rows]

    def list_datasets_by_ids(self, dataset_ids: list[str]) -> list[DatasetRecord]:
        """Return datasets for the given ids, newest first (for shared listing).

        Args:
            dataset_ids: Dataset ids to load (empty -> ``[]``).

        Returns:
            The matching :class:`DatasetRecord` list ordered by ``updated_at``
            desc; unknown ids are simply absent.
        """
        if not dataset_ids:
            return []
        with Session(self._engine) as session:
            rows = session.scalars(
                select(DatasetModel)
                .where(DatasetModel.id.in_(list(dataset_ids)))
                .order_by(DatasetModel.updated_at.desc())
            ).all()
            return [_to_record(row) for row in rows]

    def get_rows(self, dataset_id: str) -> list[dict[str, Any]] | None:
        """Return the decompressed rows for ``dataset_id``.

        Args:
            dataset_id: Dataset whose rows are read.

        Returns:
            The dataset rows, or ``None`` when the dataset or its blob is absent.
        """
        blob = self._blobs.get(dataset_id)
        if blob is None:
            return None
        raw = gzip.decompress(blob.data) if blob.compression == _COMPRESSION_GZIP else blob.data
        return deserialize_rows(raw)

    def update_rows(
        self,
        dataset_id: str,
        *,
        rows: list[dict[str, Any]],
        column_schema: dict[str, Any] | None = None,
    ) -> DatasetRecord | None:
        """Replace a dataset's rows (editor edit), re-hashing and recompressing.

        Args:
            dataset_id: Dataset to update.
            rows: The new rows.
            column_schema: Optional replacement column schema; left unchanged when
                ``None``.

        Returns:
            The updated :class:`DatasetRecord`, or ``None`` when the dataset is
            unknown.
        """
        raw = serialize_rows(rows)
        compressed = gzip.compress(raw)
        with Session(self._engine) as session:
            row = session.get(DatasetModel, dataset_id)
            if row is None:
                return None
            schema = column_schema if column_schema is not None else dict(row.column_schema or {})
            row.row_count = len(rows)
            row.column_count = _column_count(rows, schema)
            row.byte_size = len(raw)
            row.stored_bytes = len(compressed)
            row.content_hash = content_hash(raw)
            row.column_schema = schema
            row.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            result = _to_record(row)
        self._blobs.put(
            dataset_id,
            DatasetBlob(content_type=_CONTENT_TYPE_JSON, compression=_COMPRESSION_GZIP, data=compressed),
        )
        return result

    def rename_dataset(self, dataset_id: str, name: str) -> DatasetRecord | None:
        """Rename a dataset's library entry.

        Args:
            dataset_id: Dataset to rename.
            name: New display name.

        Returns:
            The updated :class:`DatasetRecord`, or ``None`` when unknown.
        """
        with Session(self._engine) as session:
            row = session.get(DatasetModel, dataset_id)
            if row is None:
                return None
            row.name = name
            row.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            return _to_record(row)

    def reassign_owner(self, dataset_id: str, owner_username: str) -> DatasetRecord | None:
        """Reassign a dataset's owner (sharing transfer).

        Args:
            dataset_id: Dataset whose owner moves.
            owner_username: The new owner's lowercased username.

        Returns:
            The updated :class:`DatasetRecord`, or ``None`` when unknown.
        """
        with Session(self._engine) as session:
            row = session.get(DatasetModel, dataset_id)
            if row is None:
                return None
            row.owner_username = owner_username
            row.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(row)
            return _to_record(row)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its bytes in a single transaction.

        The blob is removed in the same session as the metadata (not relying on FK
        cascade) so the delete is atomic — a crash can never leave a metadata row
        whose bytes are gone — and behaves identically on SQLite test stores,
        where ``PRAGMA foreign_keys`` is off, and Postgres.

        Args:
            dataset_id: Dataset to delete.

        Returns:
            ``True`` when a metadata row was deleted, ``False`` when none matched.
        """
        with Session(self._engine) as session:
            session.execute(delete(DatasetBlobModel).where(DatasetBlobModel.dataset_id == dataset_id))
            deleted = session.execute(
                delete(DatasetModel).where(DatasetModel.id == dataset_id)
            ).rowcount
            session.commit()
        return bool(deleted)
