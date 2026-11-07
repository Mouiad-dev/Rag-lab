"""Adapters that expose existing repository ports through per-call units of work."""

from typing import Any

from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository, IndexRepository
from rag_foundry_domain.prompts import PromptTemplateStore
from rag_foundry_domain.settings_store import SettingsOverrideRepository
from rag_foundry_domain.uow import WRITE_METHODS, UnitOfWorkFactory

_PORTS = {
    "documents": DocumentRepository,
    "extractions": ExtractionRepository,
    "indexes": IndexRepository,
    "prompts": PromptTemplateStore,
    "settings": SettingsOverrideRepository,
}


class PerCall:
    def __init__(self, factory: UnitOfWorkFactory, family: str) -> None:
        if family not in WRITE_METHODS:
            raise ValueError("unknown repository family")
        self._factory, self._family = factory, family

    def __getattr__(self, name: str) -> Any:
        writes = WRITE_METHODS[self._family]

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            read_only = name not in writes
            async with self._factory(read_only=read_only) as uow:
                result = await getattr(getattr(uow, self._family), name)(*args, **kwargs)
                if not read_only:
                    await uow.commit()
                return result

        if name.startswith("_"):
            raise AttributeError(name)
        if not hasattr(_PORTS[self._family], name):
            raise AttributeError(name)
        return invoke


def per_call_documents(factory: UnitOfWorkFactory) -> PerCall:
    return PerCall(factory, "documents")


def per_call_extractions(factory: UnitOfWorkFactory) -> PerCall:
    return PerCall(factory, "extractions")


def per_call_indexes(factory: UnitOfWorkFactory) -> PerCall:
    return PerCall(factory, "indexes")


def per_call_prompts(factory: UnitOfWorkFactory) -> PerCall:
    return PerCall(factory, "prompts")


def per_call_settings(factory: UnitOfWorkFactory) -> PerCall:
    return PerCall(factory, "settings")
