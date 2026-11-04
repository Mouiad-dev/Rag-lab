"""API entrypoint: app factory and ``run()``."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from rag_foundry_core import __version__
from rag_foundry_core.settings import Settings, load_settings
from rag_foundry_domain import LOCAL_DEV_SCOPE, DocumentRepository
from rag_foundry_ingestion.intake import IntakeService

from rag_foundry_api.documents import register_document_routes
from rag_foundry_api.errors import register_error_handlers
from rag_foundry_api.limits import UploadSizeLimitMiddleware


def create_app(
    settings: Settings | None = None,
    *,
    intake_service: IntakeService | None = None,
    repository: DocumentRepository | None = None,
) -> FastAPI:
    """`intake_service` and `repository` are injected together (tests); otherwise built from settings."""
    resolved = settings if settings is not None else load_settings()
    sql_repo = None
    if intake_service is None and resolved.database_url is not None:
        from rag_foundry_storage.local_store import LocalObjectStore
        from rag_foundry_storage.sql.engine import create_sessionmaker
        from rag_foundry_storage.sql.repository import SqlDocumentRepository

        sql_repo = SqlDocumentRepository(create_sessionmaker(resolved.database_url))
        intake_service = IntakeService(
            LocalObjectStore(resolved.object_store_root), sql_repo, max_bytes=resolved.max_upload_bytes
        )
        repository = sql_repo

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if sql_repo is not None and resolved.environment in ("dev", "test"):
            await sql_repo.ensure_local_scope(LOCAL_DEV_SCOPE)
        yield

    app = FastAPI(title="RAG Foundry API", version=__version__, lifespan=lifespan)
    app.state.settings = resolved
    app.state.intake_service = intake_service
    app.state.repository = repository
    register_error_handlers(app)
    app.add_middleware(UploadSizeLimitMiddleware, max_upload_bytes=resolved.max_upload_bytes)
    register_document_routes(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


def run() -> None:
    s = load_settings()
    # psycopg async cannot run on the Windows ProactorEventLoop; uvicorn accepts an import string as loop factory.
    loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(create_app(s), host=s.api_host, port=s.api_port, log_level=s.log_level.lower(), loop=loop)
