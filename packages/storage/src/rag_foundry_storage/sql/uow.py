"""SQL-backed task-owned Unit of Work."""

import asyncio
from contextlib import suppress
from typing import Any

from rag_foundry_domain import Scope
from rag_foundry_domain.errors import StoreError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_foundry_storage.sql._tx import write
from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
from rag_foundry_storage.sql.index_repository import SqlIndexRepository
from rag_foundry_storage.sql.models import Project, Tenant
from rag_foundry_storage.sql.prompt_store import SqlPromptTemplateStore
from rag_foundry_storage.sql.repository import SqlDocumentRepository
from rag_foundry_storage.sql.settings_repository import SqlSettingsOverrideRepository


class SqlUnitOfWork:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], *, read_only: bool) -> None:
        self._sessionmaker, self.read_only = sessionmaker, read_only
        self._session: AsyncSession | None = None
        self._owner: asyncio.Task[Any] | None = None
        self._committed = False
        self._repos: dict[str, Any] = {}

    @property
    def session(self) -> AsyncSession:
        self._check_owner()
        if self._session is None:
            raise RuntimeError("unit of work is not open")
        return self._session

    def _check_owner(self) -> None:
        if self._owner is not None and asyncio.current_task() is not self._owner:
            raise RuntimeError("unit of work belongs to another task")

    async def __aenter__(self):
        if self._session is not None:
            raise RuntimeError("unit of work is already open")
        self._owner = asyncio.current_task()
        self._session = self._sessionmaker()
        self._session.info["read_only"] = self.read_only
        self._repos = {
            "documents": SqlDocumentRepository(self._session),
            "extractions": SqlExtractionRepository(self._session),
            "indexes": SqlIndexRepository(self._session),
            "prompts": SqlPromptTemplateStore(self._session),
            "settings": SqlSettingsOverrideRepository(self._session),
        }
        return self

    def _repo(self, name: str):
        self._check_owner()
        if self._session is None:
            raise RuntimeError("unit of work is not open")
        return self._repos[name]

    @property
    def documents(self):
        return self._repo("documents")

    @property
    def extractions(self):
        return self._repo("extractions")

    @property
    def indexes(self):
        return self._repo("indexes")

    @property
    def prompts(self):
        return self._repo("prompts")

    @property
    def settings(self):
        return self._repo("settings")

    async def commit(self) -> None:
        self._check_owner()
        if self.read_only:
            raise RuntimeError("read-only unit of work")
        try:
            await self.session.commit()
        except (OperationalError, InterfaceError):
            raise StoreError("store_unavailable") from None
        except DBAPIError:
            raise StoreError("commit_failed") from None
        self._committed = True

    async def rollback(self) -> None:
        self._check_owner()
        if self._session is not None:
            await self._session.rollback()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._check_owner()
        session, self._session = self._session, None
        try:
            if session is not None and not self._committed:
                with suppress(Exception):
                    await session.rollback()
        finally:
            if session is not None:
                with suppress(Exception):
                    await session.close()


async def ensure_local_scope(uow_factory, scope: Scope) -> None:
    """Upsert tenant and project for development and tests."""
    async with uow_factory(read_only=False) as uow:
        async with write(uow.session):
            await uow.session.execute(pg_insert(Tenant).values(id=scope.tenant_id).on_conflict_do_nothing())
            await uow.session.execute(
                pg_insert(Project).values(id=scope.project_id, tenant_id=scope.tenant_id).on_conflict_do_nothing()
            )
            if not await SqlDocumentRepository._project_exists(uow.session, scope):
                raise ValueError("project ID already belongs to another tenant")
        await uow.commit()
