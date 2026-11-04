"""Upload and version routes under /api/v1. Handlers stay thin; hashing and file I/O live in the service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from rag_foundry_domain import DocumentRepository, DocumentVersion, IntakeOutcome, NotFoundError
from rag_foundry_ingestion.intake import IntakeService
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from rag_foundry_api.auth import Auth
from rag_foundry_api.errors import api_error

_READ_SIZE = 1024 * 1024


class VersionResponse(BaseModel):
    document_id: str
    version_id: str
    version_number: int
    sha256: str
    size_bytes: int
    media_type: str
    text_encoding: str | None
    original_filename: str
    created_at: datetime
    deduplicated: bool


def _to_response(document_id: str, v: DocumentVersion, deduplicated: bool) -> VersionResponse:
    return VersionResponse(
        document_id=document_id,
        version_id=v.id,
        version_number=v.version_number,
        sha256=v.sha256,
        size_bytes=v.size_bytes,
        media_type=str(v.media_type),
        text_encoding=v.text_encoding,
        original_filename=v.original_filename,
        created_at=v.created_at,
        deduplicated=deduplicated,
    )


def _service(request: Request) -> IntakeService:
    service = getattr(request.app.state, "intake_service", None)
    if service is None:
        raise api_error(503, "storage_unavailable", "storage is not configured")
    return service


def _repository(request: Request) -> DocumentRepository:
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        raise api_error(503, "storage_unavailable", "storage is not configured")
    return repo


async def _chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(_READ_SIZE):
        yield chunk


async def _ingest(request: Request, project_id: str, auth: Auth, document_id: str | None) -> IntakeOutcome:
    service = _service(request)
    scope = auth.scope_for(project_id)
    try:
        async with request.form(max_files=1, max_fields=0) as form:
            items = form.multi_items()
            if len(items) != 1 or items[0][0] != "file" or not isinstance(items[0][1], UploadFile):
                raise api_error(422, "validation_error", "exactly one file part named 'file' is required")
            upload = items[0][1]
            return await service.ingest(
                scope,
                _chunks(upload),
                media_type=upload.content_type,
                filename=upload.filename or "",
                document_id=document_id,
            )
    except HTTPException as exc:
        if exc.status_code == 400:  # Starlette: too many files or fields, or a malformed body
            raise api_error(422, "validation_error", "malformed multipart body") from None
        raise


def _status(response: Response, outcome: IntakeOutcome) -> VersionResponse:
    response.status_code = 200 if outcome.deduplicated else 201
    return _to_response(outcome.document.id, outcome.version, outcome.deduplicated)


async def upload_document(project_id: str, request: Request, response: Response, auth: Auth) -> VersionResponse:
    return _status(response, await _ingest(request, project_id, auth, None))


async def upload_version(
    project_id: str, document_id: str, request: Request, response: Response, auth: Auth
) -> VersionResponse:
    return _status(response, await _ingest(request, project_id, auth, document_id))


async def get_version(document_id: str, version_id: str, request: Request, auth: Auth) -> VersionResponse:
    repo = _repository(request)
    for scope in auth.scopes():
        try:
            version = await repo.get_version(scope, document_id, version_id)
        except NotFoundError:
            continue
        return _to_response(document_id, version, False)
    raise NotFoundError()


def register_document_routes(app: FastAPI) -> None:
    """Added with add_api_route (not include_router) so app.routes keeps plain route objects with a path."""
    prefix = "/api/v1"
    app.add_api_route(f"{prefix}/projects/{{project_id}}/documents", upload_document, methods=["POST"], status_code=201)
    app.add_api_route(
        f"{prefix}/projects/{{project_id}}/documents/{{document_id}}/versions",
        upload_version,
        methods=["POST"],
        status_code=201,
    )
    app.add_api_route(f"{prefix}/documents/{{document_id}}/versions/{{version_id}}", get_version, methods=["GET"])
