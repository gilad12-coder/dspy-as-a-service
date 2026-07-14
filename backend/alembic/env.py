"""Alembic environment — wires REMOTE_DB_URL into the migration runner.

Reads the database URL from the ``REMOTE_DB_URL`` env var (the same env
the application uses) so the same secret feeds both the migrate Job and
the running pods. Falls back to the value in ``alembic.ini`` if the env
var is unset (useful for local ``alembic revision --autogenerate``).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from core.storage.models import Base

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    # ``disable_existing_loggers=False`` keeps app loggers (e.g.
    # ``core.worker.engine``) usable after Alembic configures its own
    # logging, otherwise Python's ``logging.config.fileConfig`` marks
    # every pre-existing logger as ``disabled=True`` and silently drops
    # their records — breaking ``caplog`` capture in the test suite.
    # Even so, ``fileConfig`` replaces the root handlers and sets root level
    # to WARN, which silently killed every app INFO log after boot-time
    # migration sync — so the programmatic path (core/storage/migrate.py)
    # opts out via the ``configure_logger`` attribute; only the bare
    # ``alembic`` CLI still configures logging from the ini file.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

env_url = os.environ.get("REMOTE_DB_URL")
if env_url:
    config.set_main_option("sqlalchemy.url", env_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode (emit SQL to stdout, no connection).

    Useful for pre-rendering the migration SQL into a review-able artifact
    before running it against a managed database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database.

    The application boot path shares its advisory-locked connection through
    ``config.attributes['connection']`` so migrations run inside the same
    transaction as ``create_all`` and never open a second, unserialized session;
    that connection already owns the transaction, so no ``begin_transaction`` is
    started here. A bare ``alembic`` CLI invocation has no such attribute and
    builds its own engine from ``REMOTE_DB_URL`` instead.
    """
    shared = config.attributes.get("connection")
    if shared is not None:
        context.configure(connection=shared, target_metadata=target_metadata)
        context.run_migrations()
        return
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
