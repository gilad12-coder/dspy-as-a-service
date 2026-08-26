"""Tests for boot-time synchronization to the on-premises schema baseline."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url

from core.storage.migrate import sync_migration_head

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_HEAD = ScriptDirectory.from_config(Config(str(_BACKEND_DIR / "alembic.ini"))).get_current_head()
TEST_DB_URL = os.environ.get("SKYNET_TEST_DB_URL")

_needs_pg = pytest.mark.skipif(
    not TEST_DB_URL or not TEST_DB_URL.startswith("postgresql"),
    reason="SKYNET_TEST_DB_URL is not a PostgreSQL URL.",
)


def test_sync_migration_head_is_noop_off_postgres() -> None:
    """Leave SQLite test stores under direct ORM schema management."""
    engine = create_engine("sqlite://")

    sync_migration_head(engine)

    assert not inspect(engine).has_table("alembic_version")


def test_sync_migration_head_uses_absolute_runtime_paths() -> None:
    """Resolve Alembic assets independently of the process working directory."""
    engine = create_engine("sqlite://")
    connection = object()

    with (
        patch("core.storage.migrate.schema_bootstrap_lock") as lock,
        patch("core.storage.migrate.command.upgrade") as upgrade,
    ):
        lock.return_value.__enter__.return_value = connection
        sync_migration_head(engine)

    config = upgrade.call_args.args[0]
    assert Path(config.get_main_option("script_location")) == _BACKEND_DIR / "alembic"
    assert Path(config.get_main_option("prepend_sys_path")) == _BACKEND_DIR


def _version(engine: Engine) -> str | None:
    """Return the database's current Alembic revision."""
    with engine.connect() as conn:
        if not inspect(conn).has_table("alembic_version"):
            return None
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


@pytest.fixture
def fresh_pg() -> Iterator[Engine]:
    """Yield a freshly cleared local PostgreSQL schema."""
    host = make_url(TEST_DB_URL or "").host
    if host not in ("localhost", "127.0.0.1"):
        pytest.skip(f"refusing to clear a non-local database (host={host!r})")
    engine = create_engine(TEST_DB_URL or "")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    try:
        yield engine
    finally:
        engine.dispose()


@_needs_pg
def test_fresh_database_applies_onprem_baseline(fresh_pg: Engine) -> None:
    """Create product tables and omit every hosted billing table."""
    sync_migration_head(fresh_pg)

    inspector = inspect(fresh_pg)
    assert _version(fresh_pg) == _HEAD
    assert inspector.has_table("jobs")
    assert inspector.has_table("users")
    assert inspector.has_table("byok_provider_keys")
    assert not inspector.has_table("billing_customers")
    assert not inspector.has_table("credit_ledger")


@_needs_pg
def test_sync_at_head_is_idempotent(fresh_pg: Engine) -> None:
    """Leave an already-current schema unchanged."""
    sync_migration_head(fresh_pg)
    sync_migration_head(fresh_pg)

    assert _version(fresh_pg) == _HEAD


@_needs_pg
def test_sync_preserves_root_logging_config(fresh_pg: Engine) -> None:
    """Keep application log handlers and levels intact during migration."""
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    level_before = root.level
    try:
        sync_migration_head(fresh_pg)
        assert sentinel in root.handlers
        assert root.level == level_before
    finally:
        root.removeHandler(sentinel)
