"""Database engine and session factory lifecycle helpers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


@dataclass(frozen=True)
class Database:
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]

    def uow(self, *, read_only: bool = False):
        from rag_foundry_storage.sql.uow import SqlUnitOfWork

        return SqlUnitOfWork(self.sessionmaker, read_only=read_only)

    async def dispose(self) -> None:
        await self.engine.dispose()


def create_database(url: SecretStr) -> Database:
    engine = create_async_engine(url.get_secret_value(), pool_pre_ping=True, hide_parameters=True)
    return Database(engine, async_sessionmaker(engine, expire_on_commit=False))


@asynccontextmanager
async def open_database(url: SecretStr) -> AsyncIterator[Database]:
    db = create_database(url)
    try:
        yield db
    finally:
        await db.dispose()


def run_db_main[T](coro: Coroutine[object, object, T]) -> T:
    """Run a database coroutine; psycopg async cannot use the Windows Proactor loop. Never call from a running loop."""
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)
