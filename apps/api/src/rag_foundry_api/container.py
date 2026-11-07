"""Application resource construction and database lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag_foundry_answering.runner import LocalRunner, ollama_model_factory
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, DocumentRepository
from rag_foundry_domain.runtime_settings import RuntimeSettings
from rag_foundry_domain.settings_store import RuntimeSettingsService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.intake import IntakeService


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    intake: IntakeService | None
    documents: DocumentRepository | None
    anchors: AnchorService | None
    runs: LocalRunner | None


@asynccontextmanager
async def open_container(
    settings: Settings, runtime_settings: tuple[RuntimeSettings, frozenset[str]]
) -> AsyncIterator[AppContainer]:
    base, file_keys = runtime_settings
    if settings.database_url is None:
        yield AppContainer(settings, None, None, None, None)
        return

    from rag_foundry_storage.local_store import LocalObjectStore
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
    from rag_foundry_storage.sql.index_repository import SqlIndexRepository
    from rag_foundry_storage.sql.prompt_store import SqlPromptTemplateStore
    from rag_foundry_storage.sql.repository import SqlDocumentRepository
    from rag_foundry_storage.sql.settings_repository import SqlSettingsOverrideRepository

    database = create_database(settings.database_url)
    try:
        maker = database.sessionmaker
        runtime = RuntimeSettingsService(base, file_keys=file_keys, repo=SqlSettingsOverrideRepository(maker))
        documents = SqlDocumentRepository(maker)
        store = LocalObjectStore(settings.object_store_root)
        intake = IntakeService(store, documents, max_bytes=settings.max_upload_bytes)
        views = SqlExtractionRepository(maker)
        anchors = AnchorService(store, views, provider_namespace=settings.provider_namespace)
        runs = LocalRunner(
            SqlIndexRepository(maker),
            anchors,
            model_factory=ollama_model_factory(settings.ollama_base_url),
            settings=runtime,
            prompts=SqlPromptTemplateStore(maker),
        )
        if settings.environment in ("dev", "test"):
            await documents.ensure_local_scope(LOCAL_DEV_SCOPE)
        yield AppContainer(settings, intake, documents, anchors, runs)
    finally:
        await database.dispose()
