"""API entrypoint: app factory and ``run()``."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from rag_foundry_core import __version__
from rag_foundry_core.settings import Settings, load_settings
from rag_foundry_domain.runtime_settings import RuntimeSettings, load_runtime_settings_file

from rag_foundry_api.anchors import register_anchor_routes
from rag_foundry_api.container import open_container
from rag_foundry_api.documents import register_document_routes
from rag_foundry_api.errors import register_error_handlers
from rag_foundry_api.limits import UploadSizeLimitMiddleware
from rag_foundry_api.runs import register_run_routes


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else load_settings()
    runtime = (RuntimeSettings(), frozenset())
    if resolved.runtime_config_path is not None:
        runtime = load_runtime_settings_file(resolved.runtime_config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[dict[str, object]]:
        async with open_container(resolved, runtime) as container:
            yield {"container": container}

    app = FastAPI(title="RAG Foundry API", version=__version__, lifespan=lifespan)
    app.state.settings = resolved
    register_error_handlers(app)
    app.add_middleware(UploadSizeLimitMiddleware, max_upload_bytes=resolved.max_upload_bytes)
    register_document_routes(app)
    register_anchor_routes(app)
    register_run_routes(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


def run() -> None:
    s = load_settings()
    # psycopg async cannot run on the Windows ProactorEventLoop; uvicorn accepts an import string as loop factory.
    loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(create_app(s), host=s.api_host, port=s.api_port, log_level=s.log_level.lower(), loop=loop)
