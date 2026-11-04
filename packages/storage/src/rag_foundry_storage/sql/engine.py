"""Async engine and session factory. The URL is a SecretStr and is never logged."""

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_sessionmaker(url: SecretStr) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(url.get_secret_value(), pool_pre_ping=True, hide_parameters=True)
    return async_sessionmaker(engine, expire_on_commit=False)
