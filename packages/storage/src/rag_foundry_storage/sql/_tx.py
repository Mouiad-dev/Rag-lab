"""Transaction helpers for session-bound SQL repositories."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def read(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    yield session


@asynccontextmanager
async def write(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    if session.info.get("read_only"):
        raise RuntimeError("read-only unit of work")
    nested = await session.begin_nested()
    try:
        yield session
    except BaseException:
        with suppress(Exception):
            await nested.rollback()
        raise
    else:
        await nested.commit()
