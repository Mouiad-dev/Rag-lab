"""One error envelope for every API error. Messages are fixed text: no input echo, filenames or paths."""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from rag_foundry_domain import (
    AnchorError,
    EmptyUploadError,
    IntakeError,
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaTypeMismatchError,
    NotFoundError,
    ObjectStoreError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from rag_foundry_domain.errors import IndexingError, RunError
from rag_foundry_providers.errors import BudgetError, ProviderError
from starlette.exceptions import HTTPException

_RETRYABLE = frozenset({429, 503})

_INTAKE_STATUS: tuple[tuple[type[IntakeError], int], ...] = (
    (UploadTooLargeError, 413),
    (UnsupportedMediaTypeError, 415),
    (NotFoundError, 404),
    (EmptyUploadError, 422),
    (MediaTypeMismatchError, 422),
    (InvalidTextEncodingError, 422),
    (InvalidFilenameError, 422),
)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": str(uuid.uuid4()),
            "retryable": status in _RETRYABLE,
        }
    }
    return JSONResponse(body, status_code=status)


def api_error(status: int, code: str, message: str) -> HTTPException:
    """An HTTPException whose detail carries the stable code; the handler builds the envelope."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _http_exception(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return error_response(exc.status_code, str(detail["code"]), str(detail["message"]))
    return error_response(exc.status_code, f"http_{exc.status_code}", "request failed")


async def _validation_error(_: Request, __: Exception) -> JSONResponse:
    return error_response(422, "validation_error", "request validation failed")


async def _intake_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, IntakeError)
    if isinstance(exc, ObjectStoreError):
        return error_response(500, "object_store_error", "storage failure")
    for kind, status in _INTAKE_STATUS:
        if isinstance(exc, kind):
            return error_response(status, exc.code, exc.message)
    return error_response(422, exc.code, exc.message)


async def _anchor_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AnchorError)
    return error_response(422, exc.code, exc.code.replace("_", " "))


async def _run_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RunError)
    if exc.code == "too_many_runs":
        return error_response(429, exc.code, "too many runs")
    if exc.code == "run_timeout":
        return error_response(503, exc.code, "run timed out")
    return error_response(422, exc.code, exc.code.replace("_", " "))


async def _indexing_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, IndexingError)
    if exc.code in ("not_ready", "incompatible_embedding"):
        return error_response(409, exc.code, exc.code.replace("_", " "))
    return error_response(422, exc.code, exc.code.replace("_", " "))


async def _budget_error(_: Request, __: Exception) -> JSONResponse:
    return error_response(409, "budget_exceeded", "run budget exceeded")


async def _provider_error(_: Request, __: Exception) -> JSONResponse:
    return error_response(503, "provider_unavailable", "model provider unavailable")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(IntakeError, _intake_error)
    app.add_exception_handler(AnchorError, _anchor_error)
    app.add_exception_handler(RunError, _run_error)
    app.add_exception_handler(IndexingError, _indexing_error)
    app.add_exception_handler(BudgetError, _budget_error)
    app.add_exception_handler(ProviderError, _provider_error)
