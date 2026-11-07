"""Synchronous run route."""

from fastapi import FastAPI
from rag_foundry_contracts.runs import RunRequest, RunResponse

from rag_foundry_api.auth import Auth
from rag_foundry_api.deps import RunsDep
from rag_foundry_api.errors import api_error


async def create_run(project_id: str, body: RunRequest, auth: Auth, service: RunsDep) -> RunResponse:
    scope = auth.scope_for(project_id)
    if service is None:
        raise api_error(503, "storage_unavailable", "storage is not configured")
    return await service.run(scope, body)


def register_run_routes(app: FastAPI) -> None:
    app.add_api_route("/api/v1/projects/{project_id}/runs", create_run, methods=["POST"], response_model=RunResponse)
