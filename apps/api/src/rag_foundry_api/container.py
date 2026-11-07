"""Application resource construction and database lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag_foundry_answering.runner import LocalRunner, ollama_model_factory
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE
from rag_foundry_domain.runtime_settings import RuntimeSettings
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.intake import IntakeService


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    intake: IntakeService | None
    anchors: AnchorService | None
    runs: LocalRunner | None


@asynccontextmanager
async def open_container(
    settings: Settings, runtime_settings: tuple[RuntimeSettings, frozenset[str]]
) -> AsyncIterator[AppContainer]:
    base, file_keys = runtime_settings
    if settings.database_url is None:
        yield AppContainer(settings, None, None, None)
        return

    from rag_foundry_storage.local_store import LocalObjectStore
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.uow import ensure_local_scope

    database = create_database(settings.database_url)
    try:
        factory = database.uow
        store = LocalObjectStore(settings.object_store_root)
        intake = IntakeService(store, factory, max_bytes=settings.max_upload_bytes)
        anchors = AnchorService(store, factory, provider_namespace=settings.provider_namespace)
        runs = LocalRunner(
            factory,
            anchors,
            model_factory=ollama_model_factory(settings.ollama_base_url),
            base=base,
            file_keys=file_keys,
        )
        if settings.environment in ("dev", "test"):
            await ensure_local_scope(factory, LOCAL_DEV_SCOPE)
        yield AppContainer(settings, intake, anchors, runs)
    finally:
        await database.dispose()
