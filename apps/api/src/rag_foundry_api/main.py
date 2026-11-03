"""API entrypoint: app factory and ``run()``."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from rag_foundry_core import __version__
from rag_foundry_core.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="RAG Foundry API", version=__version__)
    app.state.settings = settings if settings is not None else load_settings()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


def run() -> None:
    s = load_settings()
    uvicorn.run(create_app(s), host=s.api_host, port=s.api_port, log_level=s.log_level.lower())
