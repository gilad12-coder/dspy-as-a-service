"""Tests for boot-time Alembic version sync (:func:`sync_migration_head`).

The non-Postgres no-op guard runs in ordinary (SQLite) CI. The adopt-vs-upgrade
behaviour needs a real Postgres and is gated on ``REMOTE_DB_URL`` — point it at a
throwaway ``postgresql://`` database to exercise it::

    REMOTE_DB_URL=postgresql://postgres:test@localhost:5432/testdb \
        pytest backend/core/storage/tests/test_migrate.py

Without that env var the live-DB cases are skipped, so unit-test CI is unaffected.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from alembic import command
from core.storage.migrate import sync_migration_head
from core.storage.models import (
    Base,
    BillingCustomerModel,
    ConversationEmbeddingModel,
    JobEmbeddingModel,
)

_HEAD = "b3c4d5e6f7a8"
# ``_HEAD``'s down_revision — the schema state just before the one-time-500 grant.
_PRE_500 = "f2b3c4d5e6a7"
_BACKEND_DIR = Path(__file__).resolve().parents[3]
REMOTE_DB_URL = os.environ.get("REMOTE_DB_URL")

_needs_pg = pytest.mark.skipif(
    not REMOTE_DB_URL or not REMOTE_DB_URL.startswith("postgresql"),
    reason="REMOTE_DB_URL not set to a postgresql:// URL — skipping live-DB migration tests.",
)


def test_sync_migration_head_is_noop_off_postgres() -> None:
    """On SQLite the lock helper yields ``None`` and Alembic is left untouched."""
    engine = create_engine("sqlite://")
    sync_migration_head(engine)
    assert not inspect(engine).has_table("alembic_version")


def _build_schema_like_prod(engine: Engine) -> None:
    """Create the full ORM schema minus the pgvector tables, as prod boot does.

    Args:
        engine: Engine on the wiped live-Postgres target.
    """
    embedding = {JobEmbeddingModel.__table__, ConversationEmbeddingModel.__table__}
    tables = [table for table in Base.metadata.sorted_tables if table not in embedding]
    Base.metadata.create_all(engine, tables=tables)


def _version(engine: Engine) -> str | None:
    """Return the stamped Alembic revision, or ``None`` when the DB is unadopted."""
    with engine.connect() as conn:
        if not inspect(conn).has_table("alembic_version"):
            return None
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def _grant(engine: Engine) -> int:
    """Return the seeded account's remaining free-grant credits."""
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT grant_remaining FROM billing_customers WHERE username = 'a@x.com'")
        ).scalar()


def _seed(engine: Engine, grant: int) -> None:
    """Insert one non-Premium billing row with the given remaining grant.

    Args:
        engine: Engine on the live-Postgres target.
        grant: Remaining free-grant credits to seed.
    """
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username="a@x.com",
                stripe_customer_id="local:test",
                credit_balance=0,
                grant_remaining=grant,
            )
        )
        session.commit()


@pytest.fixture
def fresh_pg() -> Iterator[Engine]:
    """Yield an engine over a freshly wiped ``public`` schema of the live target.

    Guards against catastrophe: this fixture ``DROP SCHEMA public CASCADE``s, so it
    refuses any non-local host — a stray ``REMOTE_DB_URL`` pointing at a managed
    Postgres (e.g. a proxy domain) skips instead of wiping real data.
    """
    host = make_url(REMOTE_DB_URL or "").host
    if host not in ("localhost", "127.0.0.1"):
        pytest.skip(f"refusing to wipe a non-local database (host={host!r})")
    engine = create_engine(REMOTE_DB_URL or "")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        engine.dispose()


@_needs_pg
def test_unadopted_database_is_stamped_not_replayed(fresh_pg: Engine) -> None:
    """A create_all-built, unstamped DB is stamped at head without running migrations."""
    _build_schema_like_prod(fresh_pg)
    _seed(fresh_pg, 200)
    assert _version(fresh_pg) is None
    sync_migration_head(fresh_pg)
    assert _version(fresh_pg) == _HEAD
    # Stamping must adopt in place, never replay the 500-grant migration onto rows.
    assert _grant(fresh_pg) == 200


@_needs_pg
def test_adopted_database_applies_pending_migrations(fresh_pg: Engine) -> None:
    """A DB stamped one revision back is upgraded, actually running the migration."""
    _build_schema_like_prod(fresh_pg)
    _seed(fresh_pg, 200)
    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    with fresh_pg.begin() as conn:
        cfg.attributes["connection"] = conn
        command.stamp(cfg, _PRE_500)
    assert _version(fresh_pg) == _PRE_500
    sync_migration_head(fresh_pg)
    assert _version(fresh_pg) == _HEAD
    # The one-time-500 migration tops the pre-existing 200 grant up to 500.
    assert _grant(fresh_pg) == 500


@_needs_pg
def test_sync_at_head_is_idempotent(fresh_pg: Engine) -> None:
    """Running the sync twice against a head DB leaves it at head."""
    _build_schema_like_prod(fresh_pg)
    sync_migration_head(fresh_pg)
    sync_migration_head(fresh_pg)
    assert _version(fresh_pg) == _HEAD
