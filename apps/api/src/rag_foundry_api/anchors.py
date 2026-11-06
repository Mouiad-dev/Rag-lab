"""Source anchor resolution route."""

from __future__ import annotations

from fastapi import FastAPI, Request
from rag_foundry_contracts.resolve import AnchorResolveRequest, AnchorResolveResponse

from rag_foundry_api.auth import Auth
from rag_foundry_api.errors import api_error


async def resolve_anchors(
    project_id: str, body: AnchorResolveRequest, request: Request, auth: Auth
) -> AnchorResolveResponse:
    scope = auth.scope_for(project_id)
    service = getattr(request.app.state, "anchor_service", None)
    if service is None:
        raise api_error(503, "storage_unavailable", "storage is not configured")
    results = await service.resolve(scope, body.anchors)
    return AnchorResolveResponse(results=results)


def register_anchor_routes(app: FastAPI) -> None:
    app.add_api_route(
        "/api/v1/projects/{project_id}/anchors:resolve",
        resolve_anchors,
        methods=["POST"],
        response_model=AnchorResolveResponse,
    )
