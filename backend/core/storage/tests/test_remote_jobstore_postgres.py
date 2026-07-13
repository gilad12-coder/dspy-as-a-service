"""Live-Postgres concurrency tests for ``RemoteDBJobStore.claim_next_job``.

These tests verify that ``FOR UPDATE SKIP LOCKED`` actually serializes
concurrent claimers — something SQLite cannot exercise because it lacks
row-level locking. They are gated on ``SKYNET_TEST_DB_URL`` — a dedicated
opt-in, **never** ``REMOTE_DB_URL``, because setup can drop a legacy schema
and the tests mutate job rows, while ``tests/conftest.py`` loads
``backend/.env`` into the environment: a developer's ordinary config must
not point these tests at their local database. Use a throwaway::

    SKYNET_TEST_DB_URL=postgresql://skynet:skynet@localhost:5432/testdb \
        pytest backend/core/storage/tests/test_remote_jobstore_postgres.py

Without that env var the entire module is skipped, so unit-test CI is
unaffected.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text

from alembic import command
from core.storage.remote import RemoteDBJobStore

TEST_DB_URL = os.environ.get("SKYNET_TEST_DB_URL")
BACKEND_DIR = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL or not TEST_DB_URL.startswith("postgresql"),
    reason="SKYNET_TEST_DB_URL not set to a postgresql:// URL — skipping live-DB tests.",
)


def _reset_legacy_schema(db_url: str) -> None:
    """Drop ``public`` if the target DB holds pre-Alembic legacy tables.

    The baseline migration uses ``CREATE TABLE IF NOT EXISTS`` for idempotency
    on properly-versioned databases, but a target that was bootstrapped by
    hand for an earlier release leaves stale, schema-incompatible tables
    (e.g. an old ``job_logs`` keyed on ``job_id`` instead of
    ``optimization_id``) that the guard skips over. Detect that legacy state
    by the absence of ``alembic_version`` and a non-empty ``public`` schema,
    and reset it so the upgrade runs against a clean slate. Properly-migrated
    and empty databases are left untouched.

    Args:
        db_url: SQLAlchemy URL of the live Postgres test target.
    """
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            already_versioned = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'alembic_version'"
                )
            ).scalar()
            if already_versioned:
                return
            has_legacy_tables = conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1")
            ).scalar()
            if not has_legacy_tables:
                return
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def store() -> Iterator[RemoteDBJobStore]:
    """Yield a single store backed by the real Postgres for the whole module.

    The live database is upgraded to Alembic head first so model changes
    such as new indexed columns are present before rows are inserted.
    The store is reused across tests to avoid repeating that setup work.
    """
    _reset_legacy_schema(TEST_DB_URL or "")

    alembic_cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DB_URL or "")
    command.upgrade(alembic_cfg, "head")

    s = RemoteDBJobStore(TEST_DB_URL)
    try:
        yield s
    finally:
        s.engine.dispose()


@pytest.fixture
def seeded_jobs(store: RemoteDBJobStore) -> Iterator[list[str]]:
    """Seed a uniquely-prefixed batch of pending jobs and clean them up after.

    Using a per-test UUID prefix keeps these tests safe to run against a
    shared Postgres without colliding with real data or with parallel
    test runs.
    """
    prefix = f"concurrency-test-{uuid.uuid4().hex[:8]}"
    ids = [f"{prefix}-{i:03d}" for i in range(20)]
    for oid in ids:
        store.create_job(oid)
        # A pending row is only claimable once its payload is written (see
        # claim_next_job's payload-presence guard), so seed one here.
        store.update_job(oid, payload={"ok": True})
    try:
        yield ids
    finally:
        store.delete_jobs(ids)


def test_concurrent_claimers_never_collide(store: RemoteDBJobStore, seeded_jobs: list[str]) -> None:
    """Twenty pending rows + ten concurrent claimers → every claim is unique.

    Each worker drains aggressively (stops only after several consecutive
    empty responses) because ``FOR UPDATE SKIP LOCKED`` can transiently
    return ``None`` while a peer transaction holds the row lock during
    commit — a single ``None`` does not mean the queue is empty. We assert
    (a) no two workers received the same ``optimization_id`` and (b) every
    seeded job was claimed exactly once.
    """
    expected_ids = set(seeded_jobs)
    worker_count = 10
    empty_streak_to_stop = 5

    def worker(worker_idx: int) -> list[str]:
        """Drain the pending queue from this worker until the streak threshold."""
        worker_id = f"pod-{worker_idx}"
        out: list[str] = []
        empty_streak = 0
        while empty_streak < empty_streak_to_stop:
            row = store.claim_next_job(worker_id, lease_seconds=60.0)
            if row is None:
                empty_streak += 1
                time.sleep(0.01)
                continue
            empty_streak = 0
            out.append(row["optimization_id"])
        return out

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(pool.map(worker, range(worker_count)))

    claimed: list[str] = [oid for batch in results for oid in batch]

    assert len(claimed) == len(set(claimed)), (
        f"Duplicate claim detected — FOR UPDATE SKIP LOCKED failed. "
        f"Got {len(claimed)} claims but only {len(set(claimed))} unique IDs."
    )
    assert set(claimed).issubset(expected_ids), "Claim returned an unexpected ID — test bled into other rows."
    assert len(claimed) == len(expected_ids), (
        f"Expected all {len(expected_ids)} seeded jobs to be claimed, got {len(claimed)}."
    )


def test_claim_records_owner_and_lease(store: RemoteDBJobStore, seeded_jobs: list[str]) -> None:
    """A successful claim must persist ``claimed_by`` and ``lease_expires_at``.

    This is the read-back path the orphan-recovery loop relies on; if
    the UPDATE silently dropped these columns the lease would never
    expire and stuck workers would never be reclaimed. The columns
    aren't in the public TypedDict, so we query them directly.
    """
    row = store.claim_next_job("pod-solo", lease_seconds=42.0)
    assert row is not None
    assert row["status"] == "validating"

    with store.engine.connect() as conn:
        result = conn.execute(
            text("SELECT claimed_by, claimed_at, lease_expires_at FROM jobs WHERE optimization_id = :oid"),
            {"oid": row["optimization_id"]},
        ).fetchone()

    assert result is not None
    claimed_by, claimed_at, lease_expires_at = result
    assert claimed_by == "pod-solo"
    assert claimed_at is not None
    assert lease_expires_at is not None
    assert lease_expires_at > claimed_at


def test_parallel_progress_writers_finish_quickly(store: RemoteDBJobStore) -> None:
    """Parallel progress writers for distinct jobs avoid parent-row lock serialization."""
    prefix = f"progress-test-{uuid.uuid4().hex[:8]}"
    ids = [f"{prefix}-a", f"{prefix}-b"]
    for oid in ids:
        store.create_job(oid)

    def write_events(optimization_id: str) -> None:
        """Write a fixed batch of progress events for a job."""
        for i in range(100):
            store.record_progress(optimization_id, "optimizer_progress", {"i": i})

    try:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(write_events, ids))
        elapsed = time.perf_counter() - started

        assert elapsed < 2.0
        assert store.get_progress_count(ids[0]) == 100
        assert store.get_progress_count(ids[1]) == 100
    finally:
        store.delete_jobs(ids)
