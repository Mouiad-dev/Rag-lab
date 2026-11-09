"""Task-owned transaction boundary. One UoW per task, never shared or nested; exit without commit rolls back.

In a read-only UoW, commit() and every write method raise RuntimeError("read-only unit of work").
No model call or object-store I/O may happen while a UoW is open.
"""

from collections.abc import Mapping
from typing import Protocol

from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository, IndexRepository
from rag_foundry_domain.prompts import PromptTemplateStore
from rag_foundry_domain.settings_store import SettingsOverrideRepository

WRITE_METHODS: Mapping[str, frozenset[str]] = {
    "documents": frozenset({"create_document_or_get", "add_version_or_get"}),
    "extractions": frozenset({"create_view_or_get"}),
    "indexes": frozenset(
        {"create_index_version", "add_chunks", "add_sparse", "add_representations", "mark_ready", "mark_failed"}
    ),
    "prompts": frozenset({"create", "activate"}),
    "settings": frozenset({"append"}),
}


class UnitOfWork(Protocol):
    documents: DocumentRepository
    extractions: ExtractionRepository
    indexes: IndexRepository
    prompts: PromptTemplateStore
    settings: SettingsOverrideRepository
    read_only: bool

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def __aenter__(self) -> "UnitOfWork": ...

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self, *, read_only: bool = False) -> UnitOfWork: ...
