"""Synchronous run route."""

from fastapi import FastAPI, Request
from rag_foundry_contracts.runs import RunRequest, RunResponse

from rag_foundry_api.auth import Auth
from rag_foundry_api.errors import api_error


async def create_run(project_id: str, body: RunRequest, request: Request, auth: Auth) -> RunResponse:
    scope = auth.scope_for(project_id)
    service = getattr(request.app.state, "run_service", None)
    if service is None:
        raise api_error(503, "storage_unavailable", "storage is not configured")
    return await service.run(scope, body)


def register_run_routes(app: FastAPI) -> None:
    app.add_api_route("/api/v1/projects/{project_id}/runs", create_run, methods=["POST"], response_model=RunResponse)
