"""Rollback-capable in-memory Unit of Work for domain tests."""

import asyncio
import copy
from typing import Any

from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.prompts import InMemoryPromptTemplateStore
from rag_foundry_domain.settings_store import InMemorySettingsOverrideRepository
from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository
from rag_foundry_domain.uow import WRITE_METHODS

_READ_ONLY = "read-only unit of work"


class _Guard:
    def __init__(
        self, repo: object, writes: frozenset[str], factory: "InMemoryUnitOfWorkFactory", read_only: bool
    ) -> None:
        self._repo, self._writes, self._read_only = repo, writes, read_only
        self._task = None

    def __getattr__(self, name: str) -> Any:
        if name in self._writes:

            async def reject(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError(_READ_ONLY)

            if self._read_only:
                return reject
        method = getattr(self._repo, name)
        if not callable(method):
            return method

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            if self._task is not asyncio.current_task():
                raise RuntimeError("unit of work belongs to another task")
            return await method(*args, **kwargs)

        return invoke


class _UnitOfWork:
    def __init__(self, factory: "InMemoryUnitOfWorkFactory", read_only: bool) -> None:
        self._factory, self.read_only = factory, read_only
        for family in WRITE_METHODS:
            repo = getattr(factory, family)
            setattr(self, family, _Guard(repo, WRITE_METHODS[family], factory, read_only))
        self._snapshot = None
        self._committed = False
        self._entered = False
        self._task = None

    async def __aenter__(self):
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("unit of work requires an asyncio task")
        if task in self._factory._active:
            raise RuntimeError("nested unit of work")
        if not self.read_only:
            await self._factory._lock.acquire()
            self._factory._owner = task
            self._snapshot = self._factory._take_snapshot()
        self._entered = True
        self._task = task
        self._factory._active.add(task)
        for family in WRITE_METHODS:
            getattr(self, family)._task = task
        self._factory.opened += 1
        self._factory.open_now += 1
        return self

    async def commit(self) -> None:
        if self._task is not asyncio.current_task():
            raise RuntimeError("unit of work belongs to another task")
        if self.read_only:
            raise RuntimeError(_READ_ONLY)
        if not self._entered:
            raise RuntimeError("unit of work is not open")
        self._committed = True
        self._factory.committed += 1

    async def rollback(self) -> None:
        if self._task is not asyncio.current_task():
            raise RuntimeError("unit of work belongs to another task")
        if not self.read_only and self._snapshot is not None:
            self._factory._restore(self._snapshot)
        self._factory.rolled_back += 1

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._task is not asyncio.current_task():
            raise RuntimeError("unit of work belongs to another task")
        try:
            if not self.read_only and (exc_type is not None or not self._committed):
                await self.rollback()
        finally:
            if self._entered:
                self._factory.open_now -= 1
                self._factory._active.discard(self._task)
            if not self.read_only and self._factory._owner is asyncio.current_task():
                self._factory._owner = None
                self._factory._lock.release()


class InMemoryUnitOfWorkFactory:
    def __init__(self, *, documents=None, extractions=None, indexes=None, prompts=None, settings=None) -> None:
        self.documents = documents or InMemoryDocumentRepository(())
        self.extractions = extractions or InMemoryExtractionRepository(self.documents)
        self.indexes = indexes or InMemoryIndexRepository(self.extractions)
        self.prompts = prompts or InMemoryPromptTemplateStore()
        self.settings = settings or InMemorySettingsOverrideRepository()
        self._lock = asyncio.Lock()
        self._owner = None
        self._active = set()
        self.opened = self.committed = self.rolled_back = self.open_now = 0

    def __call__(self, *, read_only: bool = False) -> _UnitOfWork:
        return _UnitOfWork(self, read_only)

    def _take_snapshot(self):
        snapshots = {}
        repos = tuple(getattr(self, family) for family in WRITE_METHODS)
        for repo in repos:
            state = {}
            for key, value in vars(repo).items():
                if callable(value) or isinstance(
                    value, (asyncio.Lock, asyncio.Event, asyncio.Condition, asyncio.Semaphore)
                ):
                    continue
                if any(value is other for other in repos):
                    continue
                state[key] = copy.deepcopy(value)
            snapshots[id(repo)] = (repo, state)
        return snapshots

    @staticmethod
    def _restore(snapshots) -> None:
        for repo, state in snapshots.values():
            for key in tuple(vars(repo)):
                value = vars(repo)[key]
                if callable(value) or isinstance(
                    value, (asyncio.Lock, asyncio.Event, asyncio.Condition, asyncio.Semaphore)
                ):
                    continue
                if any(value is other for other in (entry[0] for entry in snapshots.values())):
                    continue
                if key not in state:
                    delattr(repo, key)
            for key, value in state.items():
                setattr(repo, key, value)
