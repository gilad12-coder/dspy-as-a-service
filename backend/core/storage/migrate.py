"""Bring a fresh on-premises PostgreSQL schema to the Alembic head at boot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic.config import Config

from alembic import command

from .schema_lock import schema_bootstrap_lock

# backend/core/storage/migrate.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _BACKEND_DIR / "alembic.ini"
_ALEMBIC_SCRIPTS = _BACKEND_DIR / "alembic"


def sync_migration_head(engine: Any) -> None:
    """Upgrade a PostgreSQL database to the fresh on-premises schema head.

    Held under the schema-bootstrap advisory lock so exactly one replica migrates
    while peers wait, then proceeds through a no-op upgrade. Runs migrations on
    the lock-holding connection so they share its transaction rather than opening
    a second, unserialized session.

    Args:
        engine: The store's SQLAlchemy engine. On non-PostgreSQL dialects the lock
            helper yields ``None`` and this returns without touching Alembic.
    """
    with schema_bootstrap_lock(engine) as conn:
        if conn is None:
            return
        config = Config(str(_ALEMBIC_INI))
        config.set_main_option("script_location", str(_ALEMBIC_SCRIPTS))
        config.set_main_option("prepend_sys_path", str(_BACKEND_DIR))
        config.attributes["connection"] = conn
        # Keep env.py from running fileConfig(alembic.ini): that replaces the
        # root handlers and raises the root level to WARN, silencing every app
        # INFO log (JSON format included) for the rest of the process lifetime.
        config.attributes["configure_logger"] = False
        command.upgrade(config, "head")
