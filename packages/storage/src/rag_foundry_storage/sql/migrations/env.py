"""Alembic environment: async engine, URL from `config.attributes` only."""

import asyncio
import sys

from alembic import context
from rag_foundry_storage.sql.models import Base
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
target_metadata = Base.metadata


def _do_run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    url = config.attributes["database_url"].get_secret_value()
    engine = create_async_engine(url, poolclass=pool.NullPool, hide_parameters=True)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_do_run)
    finally:
        await engine.dispose()


def _run() -> None:
    # psycopg async cannot use the Windows default ProactorEventLoop.
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(_run_async())
    else:
        asyncio.run(_run_async())


if context.is_offline_mode():
    raise RuntimeError("offline migrations are not supported")
_run()
