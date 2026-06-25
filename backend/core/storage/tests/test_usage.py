"""Tests for unified per-user storage accounting (``core.storage.usage``).

Runs against an in-memory SQLite engine (the sibling job-store tests' pattern: a
``RemoteDBJobStore`` subclass that skips the pgvector bootstrap so
``Base.metadata.create_all`` stands up every table). Covers the compact-JSON
sizer, the empty-user fast path, and the ``jobs.stored_bytes`` write path that
the optimizations category sums.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.config import settings
from core.storage.models import (
    EMBEDDING_DIM,
    AgentStagedDatasetModel,
    Base,
    DatasetModel,
    JobEmbeddingModel,
    LogEntryModel,
)
from core.storage.remote import RemoteDBJobStore
from core.storage.usage import (
    STORAGE_CATEGORIES,
    compute_user_storage,
    compute_user_storage_category_items,
    compute_user_storage_items,
    json_byte_size,
    purge_expired_staged_datasets,
)


class _SQLiteJobStore(RemoteDBJobStore):
    """RemoteDBJobStore on in-memory SQLite (skips the pgvector bootstrap)."""

    def __init__(self) -> None:
        """Build an in-memory SQLite engine and create the ORM tables."""
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)


@pytest.fixture
def store() -> Iterator[_SQLiteJobStore]:
    """Yield a fresh SQLite-backed store and drop its tables afterwards."""
    s = _SQLiteJobStore()
    yield s
    Base.metadata.drop_all(s.engine)


def test_json_byte_size_none_is_zero() -> None:
    """``json_byte_size(None)`` is 0 — an absent JSON column costs nothing."""
    assert json_byte_size(None) == 0


def test_json_byte_size_matches_compact_utf8_length() -> None:
    """``json_byte_size`` measures the compact-JSON UTF-8 byte length."""
    value = {"q": "שלום", "a": 4}
    expected = len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))
    assert json_byte_size(value) == expected


def test_compute_user_storage_empty_user_is_all_zero(store: _SQLiteJobStore) -> None:
    """A user with nothing stored totals zero with every category present."""
    usage = compute_user_storage(store.engine, "ghost")
    assert usage.total == 0
    assert set(usage.breakdown) == set(STORAGE_CATEGORIES)
    assert all(value == 0 for value in usage.breakdown.values())


def test_compute_user_storage_blank_username_short_circuits(store: _SQLiteJobStore) -> None:
    """A blank username returns the all-zero usage without scanning tables."""
    usage = compute_user_storage(store.engine, "   ")
    assert usage.total == 0


def test_stored_bytes_tracks_payload_and_result(store: _SQLiteJobStore) -> None:
    """Writing payload/result sets ``stored_bytes`` and feeds the optimizations total."""
    store.create_job("job-1", username="alice")
    payload = {"signature_code": "x", "rows": [{"q": "1", "a": "1"}]}
    result = {"score": 0.9}
    store.update_job("job-1", payload=payload, result=result)

    expected = json_byte_size(payload) + json_byte_size(result) + json_byte_size({})
    usage = compute_user_storage(store.engine, "alice")
    assert usage.breakdown["optimizations"] == expected
    assert usage.total == expected


def test_set_payload_overview_folds_overview_into_stored_bytes(store: _SQLiteJobStore) -> None:
    """``set_payload_overview`` recomputes ``stored_bytes`` to include the overview."""
    store.create_job("job-2", username="alice")
    store.update_job("job-2", payload={"a": 1})
    overview = {"name": "My run"}
    store.set_payload_overview("job-2", overview)

    expected = json_byte_size({"a": 1}) + json_byte_size(None) + json_byte_size(overview)
    assert compute_user_storage(store.engine, "alice").breakdown["optimizations"] == expected


def test_compute_user_storage_is_owner_scoped(store: _SQLiteJobStore) -> None:
    """One user's job bytes never leak into another user's total."""
    store.create_job("job-a", username="alice")
    store.update_job("job-a", payload={"owner": "alice"})
    store.create_job("job-b", username="bob")
    store.update_job("job-b", payload={"owner": "bob"})

    alice = compute_user_storage(store.engine, "alice")
    expected = json_byte_size({"owner": "alice"}) + json_byte_size({})
    assert alice.breakdown["optimizations"] == expected
    assert alice.total == alice.breakdown["optimizations"]


def test_byproducts_fold_into_optimization_footprint(
    store: _SQLiteJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Logs and embeddings count toward the owning optimization, not a category of their own."""
    # job_embeddings exists (and thus has rows to count) only on the semantic
    # backend; usage accounting now skips it when embeddings are disabled, so
    # pin the flag on for the embedding-bytes assertion below. The SQLite test
    # store creates every table regardless, which is why this must be explicit.
    monkeypatch.setattr(settings, "embeddings_enabled", True)
    store.create_job("job-1", username="alice")
    store.update_job("job-1", payload={"a": 1})

    base = compute_user_storage(store.engine, "alice").breakdown["optimizations"]
    base_item = compute_user_storage_category_items(store.engine, "alice", "optimizations")[0]
    assert base_item.bytes == base

    message = "x" * 100
    with store._session_factory() as session:
        session.add(LogEntryModel(optimization_id="job-1", level="INFO", logger="t", message=message))
        session.add(JobEmbeddingModel(optimization_id="job-1", user_id="alice"))
        session.commit()

    embedding_bytes = EMBEDDING_DIM * 4 * 3
    usage = compute_user_storage(store.engine, "alice")
    assert set(usage.breakdown) == set(STORAGE_CATEGORIES)
    assert usage.breakdown["optimizations"] == base + len(message) + embedding_bytes
    assert usage.total == usage.breakdown["optimizations"]

    item = compute_user_storage_category_items(store.engine, "alice", "optimizations")[0]
    assert item.bytes == usage.breakdown["optimizations"]


def _insert_dataset(store: _SQLiteJobStore, *, dataset_id: str, owner: str, name: str, byte_size: int) -> None:
    """Insert a minimal dataset row directly for storage-items ranking tests."""
    with store._session_factory() as session:
        session.add(
            DatasetModel(
                id=dataset_id,
                owner_username=owner,
                name=name,
                source="upload",
                row_count=1,
                column_count=1,
                byte_size=byte_size,
                stored_bytes=byte_size,
                content_hash=dataset_id,
                column_schema={},
            )
        )
        session.commit()


def test_storage_items_empty_user_is_empty(store: _SQLiteJobStore) -> None:
    """A user with nothing stored ranks to an empty list."""
    assert compute_user_storage_items(store.engine, "ghost") == []


def test_storage_items_blank_username_short_circuits(store: _SQLiteJobStore) -> None:
    """A blank username returns no items without scanning tables."""
    assert compute_user_storage_items(store.engine, "   ") == []


def test_storage_items_rank_across_types_by_size(store: _SQLiteJobStore) -> None:
    """Optimizations and datasets merge into one list ordered by descending size."""
    store.create_job("job-small", username="alice")
    store.update_job("job-small", payload={"a": 1})
    store.set_payload_overview("job-small", {"name": "Small run"})
    store.create_job("job-big", username="alice")
    store.update_job("job-big", payload={"rows": list(range(50))})
    store.set_payload_overview("job-big", {"name": "Big run"})
    _insert_dataset(store, dataset_id="ds-1", owner="alice", name="Mid dataset", byte_size=200)

    items = compute_user_storage_items(store.engine, "alice")
    assert [item.bytes for item in items] == sorted((item.bytes for item in items), reverse=True)
    by_id = {item.id: item for item in items}
    assert by_id["job-big"].type == "optimization"
    assert by_id["job-big"].name == "Big run"
    assert by_id["ds-1"].type == "dataset"
    assert by_id["ds-1"].bytes == 200


def test_storage_items_are_owner_scoped(store: _SQLiteJobStore) -> None:
    """One user's items never surface in another user's ranking."""
    _insert_dataset(store, dataset_id="ds-alice", owner="alice", name="Alice set", byte_size=100)
    _insert_dataset(store, dataset_id="ds-bob", owner="bob", name="Bob set", byte_size=999)

    alice_ids = {item.id for item in compute_user_storage_items(store.engine, "alice")}
    assert alice_ids == {"ds-alice"}


def test_storage_items_honours_limit(store: _SQLiteJobStore) -> None:
    """The merged ranking is capped at ``limit`` items."""
    for index in range(5):
        _insert_dataset(
            store, dataset_id=f"ds-{index}", owner="alice", name=f"Set {index}", byte_size=(index + 1) * 10
        )

    items = compute_user_storage_items(store.engine, "alice", limit=2)
    assert len(items) == 2
    assert [item.bytes for item in items] == [50, 40]


def _insert_staged(store: _SQLiteJobStore, *, staged_id: str, username: str, filename: str, rows: list) -> None:
    """Insert a staged (pending) upload row directly for category-listing tests."""
    with store._session_factory() as session:
        session.add(
            AgentStagedDatasetModel(
                id=staged_id,
                username=username,
                dataset_filename=filename,
                rows=rows,
                row_count=len(rows),
            )
        )
        session.commit()


def test_purge_expired_staged_datasets_drops_only_rows_past_ttl(store: _SQLiteJobStore) -> None:
    """The TTL sweep deletes staged rows older than the cutoff and spares fresh ones."""
    now = datetime.now(UTC)
    with store._session_factory() as session:
        session.add(
            AgentStagedDatasetModel(
                id="stale",
                username="alice",
                dataset_filename="old.csv",
                rows=[{"q": "a"}],
                row_count=1,
                created_at=now - timedelta(minutes=20),
            )
        )
        session.add(
            AgentStagedDatasetModel(
                id="fresh",
                username="alice",
                dataset_filename="new.csv",
                rows=[{"q": "b"}],
                row_count=1,
                created_at=now - timedelta(minutes=2),
            )
        )
        session.commit()

    deleted = purge_expired_staged_datasets(store.engine, max_age_seconds=600)

    assert deleted == 1
    with store._session_factory() as session:
        surviving = {row.id for row in session.query(AgentStagedDatasetModel).all()}
    assert surviving == {"fresh"}


def test_category_items_blank_username_is_empty(store: _SQLiteJobStore) -> None:
    """A blank username returns no items without scanning tables."""
    assert compute_user_storage_category_items(store.engine, "  ", "datasets") == []


def test_category_items_non_deletable_category_is_empty(store: _SQLiteJobStore) -> None:
    """A byproduct category (no standalone artifact) yields an empty list."""
    store.create_job("job-x", username="alice")
    store.update_job("job-x", payload={"a": 1})
    assert compute_user_storage_category_items(store.engine, "alice", "embeddings") == []
    assert compute_user_storage_category_items(store.engine, "alice", "bogus") == []


def test_category_items_lists_all_datasets_largest_first(store: _SQLiteJobStore) -> None:
    """The datasets category returns every dataset, ordered by descending size."""
    _insert_dataset(store, dataset_id="ds-small", owner="alice", name="Small", byte_size=10)
    _insert_dataset(store, dataset_id="ds-big", owner="alice", name="Big", byte_size=900)
    _insert_dataset(store, dataset_id="ds-mid", owner="alice", name="Mid", byte_size=100)

    items = compute_user_storage_category_items(store.engine, "alice", "datasets")
    assert [item.id for item in items] == ["ds-big", "ds-mid", "ds-small"]
    assert all(item.type == "dataset" for item in items)


def test_category_items_lists_optimizations(store: _SQLiteJobStore) -> None:
    """The optimizations category returns the user's jobs by stored bytes."""
    store.create_job("job-1", username="alice")
    store.update_job("job-1", payload={"rows": list(range(20))})
    store.set_payload_overview("job-1", {"name": "My run"})

    items = compute_user_storage_category_items(store.engine, "alice", "optimizations")
    assert len(items) == 1
    assert items[0].id == "job-1"
    assert items[0].type == "optimization"
    assert items[0].name == "My run"
    assert items[0].bytes > 0


def test_category_items_lists_staged_uploads(store: _SQLiteJobStore) -> None:
    """The staged_uploads category returns pending uploads named by filename."""
    _insert_staged(store, staged_id="st-1", username="alice", filename="data.csv", rows=[{"q": "1"}])

    items = compute_user_storage_category_items(store.engine, "alice", "staged_uploads")
    assert len(items) == 1
    assert items[0].id == "st-1"
    assert items[0].type == "staged_upload"
    assert items[0].name == "data.csv"
    assert items[0].bytes > 0


def test_category_items_are_owner_scoped(store: _SQLiteJobStore) -> None:
    """One user's category items never surface in another user's listing."""
    _insert_dataset(store, dataset_id="ds-alice", owner="alice", name="Alice set", byte_size=100)
    _insert_dataset(store, dataset_id="ds-bob", owner="bob", name="Bob set", byte_size=999)

    alice_ids = {item.id for item in compute_user_storage_category_items(store.engine, "alice", "datasets")}
    assert alice_ids == {"ds-alice"}


def test_storage_quota_override_absent_returns_none(store: _SQLiteJobStore) -> None:
    """A user with no override row resolves to ``None``."""
    assert store.get_user_storage_quota_override("alice") is None


def test_effective_storage_quota_falls_back_to_default(store: _SQLiteJobStore) -> None:
    """Without an override the effective budget is the global default."""
    assert store.get_effective_user_storage_quota("alice") == settings.user_storage_quota_bytes


def test_storage_quota_override_replaces_default(store: _SQLiteJobStore) -> None:
    """An override becomes the user's effective budget, case-insensitively."""
    five_gb = 5 * 1024 * 1024 * 1024
    store.set_user_storage_quota_override("Alice", five_gb, updated_by="admin")

    assert store.get_user_storage_quota_override("alice") == five_gb
    assert store.get_effective_user_storage_quota("alice") == five_gb

    rows = store.list_user_storage_quota_overrides()
    assert rows == [{"username": "alice", "quota_bytes": five_gb, "updated_at": rows[0]["updated_at"], "updated_by": "admin"}]
    assert rows[0]["updated_at"] is not None


def test_storage_quota_override_can_go_below_default(store: _SQLiteJobStore) -> None:
    """An override may set a ceiling below the global default."""
    store.set_user_storage_quota_override("alice", 1024)
    assert store.get_effective_user_storage_quota("alice") == 1024


def test_storage_quota_override_rejects_non_positive(store: _SQLiteJobStore) -> None:
    """A byte budget below one is rejected at the store seam."""
    with pytest.raises(ValueError):
        store.set_user_storage_quota_override("alice", 0)


def test_delete_storage_quota_override_restores_default(store: _SQLiteJobStore) -> None:
    """Deleting an override restores the global default budget."""
    store.set_user_storage_quota_override("alice", 4096)
    assert store.delete_user_storage_quota_override("alice") is True
    assert store.get_user_storage_quota_override("alice") is None
    assert store.get_effective_user_storage_quota("alice") == settings.user_storage_quota_bytes
    assert store.delete_user_storage_quota_override("alice") is False
