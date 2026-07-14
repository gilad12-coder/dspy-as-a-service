"""Keep an adopted database's Alembic version in step at boot.

Production builds its schema with ``Base.metadata.create_all`` and then tracks it
with Alembic, because ``create_all`` never ALTERs an existing table: a migration
that adds a column to a table an earlier boot already created would otherwise
never land — exactly the drift that silently stranded ``billing_provider_keys``.

On a database Alembic has never stamped, the ``create_all`` schema already *is*
head, so we stamp it rather than replay history (the pgvector baseline migration
would fail where the extension is absent). On an adopted database we upgrade,
applying whatever was added since. Serialized under the same advisory lock as
``create_all`` so concurrent replicas don't race, and Postgres-only — SQLite test
stores get ``None`` from the lock helper and skip Alembic, since their schema
comes straight from the ORM models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic.config import Config
from sqlalchemy import inspect, text

from alembic import command

from .schema_lock import schema_bootstrap_lock

# backend/core/storage/migrate.py -> backend/alembic.ini
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def sync_migration_head(engine: Any) -> None:
    """Stamp head on an unadopted database, else upgrade to head.

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
        config.attributes["connection"] = conn
        # Keep env.py from running fileConfig(alembic.ini): that replaces the
        # root handlers and raises the root level to WARN, silencing every app
        # INFO log (JSON format included) for the rest of the process lifetime.
        config.attributes["configure_logger"] = False
        if _is_adopted(conn):
            command.upgrade(config, "head")
        else:
            command.stamp(config, "head")


def _is_adopted(conn: Any) -> bool:
    """Return whether ``alembic_version`` exists and already holds a revision.

    Args:
        conn: A live connection bound to the bootstrap transaction.

    Returns:
        ``True`` once Alembic has stamped this database — so pending migrations
        should be applied rather than the freshly built schema stamped at head.
    """
    if not inspect(conn).has_table("alembic_version"):
        return False
    return conn.execute(text("SELECT 1 FROM alembic_version LIMIT 1")).first() is not None
