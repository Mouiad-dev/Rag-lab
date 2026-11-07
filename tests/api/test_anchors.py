"""HTTP boundary and stored-payload checks for anchor resolution."""

import asyncio
import uuid

from fastapi.testclient import TestClient
from rag_foundry_api.deps import get_anchor_service, get_intake_service
from rag_foundry_api.main import create_app
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE
from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_ingestion.intake import IntakeService
from rag_foundry_storage import LocalObjectStore

PROJECT = LOCAL_DEV_SCOPE.project_id
RESOLVE = f"/api/v1/projects/{PROJECT}/anchors:resolve"
MIB = 1024 * 1024


def _error(response, status: int) -> dict:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id", "retryable"}
    assert uuid.UUID(error["request_id"]).version == 4
    return error


def _env(tmp_path, *, max_upload_bytes=3 * MIB):
    store = LocalObjectStore(tmp_path / "objects")
    docs = InMemoryDocumentRepository([LOCAL_DEV_SCOPE])
    views = InMemoryExtractionRepository(docs)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    intake = IntakeService(store, uow, max_bytes=max_upload_bytes)
    extraction = ExtractionService(store, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    anchors = AnchorService(store, uow, provider_namespace="rag-foundry-local")
    app = create_app(Settings(environment="test", max_upload_bytes=max_upload_bytes))
    app.dependency_overrides[get_intake_service] = lambda: intake
    app.dependency_overrides[get_anchor_service] = lambda: anchors
    return TestClient(app), store, docs, views, intake, extraction, anchors


def _run(coro):
    return asyncio.run(coro)


class _Stream:
    def __init__(self, data: bytes):
        self.data = data

    def __aiter__(self):
        async def generate():
            yield self.data

        return generate()


async def _stored_anchor(intake, extraction, anchors, text: str, *, name="a.md"):
    data = text.encode("utf-8")
    uploaded = await intake.ingest(LOCAL_DEV_SCOPE, _Stream(data), media_type="text/markdown", filename=name)
    doc_id, version_id = uploaded.document.id, uploaded.version.id
    view = (await extraction.extract(LOCAL_DEV_SCOPE, doc_id, version_id)).view
    payload = await extraction.load_payload(LOCAL_DEV_SCOPE, view)
    block = payload.blocks[0]
    return await anchors.anchor(LOCAL_DEV_SCOPE, view.id, block.block_id, block.start, block.end)


def test_exact_english_and_arabic_round_trip_from_stored_views(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    en, ar = (
        _run(_stored_anchor(intake, extraction, anchors, "Anchor English")),
        _run(_stored_anchor(intake, extraction, anchors, "\u0645\u0631\u062d\u0628\u0627", name="ar.md")),
    )
    response = client.post(RESOLVE, json={"anchors": [en.model_dump(mode="json"), ar.model_dump(mode="json")]})
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [item["status"] for item in results] == ["exact", "exact"]
    assert [item["exact"] for item in results] == ["Anchor English", "\u0645\u0631\u062d\u0628\u0627"]
    assert [ord(ch) for ch in results[1]["exact"]] == [0x645, 0x631, 0x62D, 0x628, 0x627]
    assert ":resolve" in RESOLVE
    fallback = client.post(RESOLVE.replace(":resolve", "/resolve"), json={"anchors": [en.model_dump(mode="json")]})
    assert fallback.status_code == 404


def test_tampered_anchor_is_200_quote_mismatch(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    anchor = _run(_stored_anchor(intake, extraction, anchors, "Anchor English"))
    tampered = anchor.model_copy(update={"exact": "X" + anchor.exact[1:]})
    response = client.post(RESOLVE, json={"anchors": [tampered.model_dump(mode="json")]})
    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["status"] == "mismatch" and item["reason"] == "quote_mismatch"


def test_unknown_project_is_404_and_prod_is_401(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    body = {"anchors": [_run(_stored_anchor(intake, extraction, anchors, "Anchor English")).model_dump(mode="json")]}
    assert (
        _error(client.post(f"/api/v1/projects/{uuid.uuid4()}/anchors:resolve", json=body), 404)["code"] == "not_found"
    )
    prod = TestClient(create_app(Settings(environment="prod")))
    assert _error(prod.post(RESOLVE, json=body), 401)["code"] == "auth_not_configured"


def test_anchor_count_version_and_view_limit_envelopes(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    anchor = _run(_stored_anchor(intake, extraction, anchors, "Anchor English"))
    raw = anchor.model_dump(mode="json")
    assert _error(client.post(RESOLVE, json={"anchors": [raw] * 101}), 422)["code"] == "validation_error"
    assert (
        _error(client.post(RESOLVE, json={"contract_version": "2.0", "anchors": [raw]}), 422)["code"]
        == "validation_error"
    )
    batch = [
        _run(_stored_anchor(intake, extraction, anchors, f"Unique anchor {i}", name=f"{i}.md")).model_dump(mode="json")
        for i in range(11)
    ]
    error = _error(client.post(RESOLVE, json={"anchors": batch}), 422)
    assert error["code"] == "too_many_views"


def test_service_missing_and_response_has_no_storage_details(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    anchor = _run(_stored_anchor(intake, extraction, anchors, "Anchor English"))
    missing = TestClient(create_app(Settings(environment="test")))
    body = {"anchors": [anchor.model_dump(mode="json")]}
    assert _error(missing.post(RESOLVE, json=body), 503)["code"] == "storage_unavailable"
    response = client.post(RESOLVE, json={"anchors": [anchor.model_dump(mode="json")]})
    assert response.status_code == 200
    assert "object_key" not in response.text and "filesystem" not in response.text
    assert str(tmp_path) not in response.text


def test_unknown_project_and_malformed_body_without_storage_keep_order():
    client = TestClient(create_app(Settings(environment="test", database_url=None)))
    anchor = {
        "anchor_id": "a1",
        "source": {
            "provider_namespace": "test",
            "document_id": "d1",
            "document_version_id": "v1",
            "extraction_view_id": "e1",
        },
        "block_id": "b00000",
        "precision": "exact_text",
        "exact": "hello",
        "text_range": {"start": 0, "end": 5},
    }
    unknown = client.post(f"/api/v1/projects/{uuid.uuid4()}/anchors:resolve", json={"anchors": [anchor]})
    malformed = client.post(RESOLVE, json={})
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (404, "not_found")
    assert (malformed.status_code, malformed.json()["error"]["code"]) == (422, "validation_error")


def test_non_upload_post_body_over_one_mib_is_rejected_before_service():
    class Boom:
        async def resolve(self, *_args):
            raise AssertionError("anchor service was called")

    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_anchor_service] = lambda: Boom()
    client = TestClient(app)
    response = client.post(RESOLVE, content=b"x" * (MIB + 1))
    assert _error(response, 413)["code"] == "request_too_large"


def test_chunked_non_upload_post_body_over_one_mib_is_rejected_before_service():
    class Boom:
        async def resolve(self, *_args):
            raise AssertionError("anchor service was called")

    observed = {}
    app = create_app(Settings(environment="test"))
    app.dependency_overrides[get_anchor_service] = lambda: Boom()

    class RecordTransferEncoding:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                observed["transfer-encoding"] = dict(scope["headers"]).get(b"transfer-encoding")
            await self.wrapped(scope, receive, send)

    client = TestClient(RecordTransferEncoding(app))
    response = client.post(
        RESOLVE,
        content=(b"x" * (64 * 1024) for _ in range(20)),
        headers={"content-type": "application/json"},
    )
    assert observed["transfer-encoding"] == b"chunked"
    assert _error(response, 413)["code"] == "request_too_large"


def test_other_non_upload_post_path_has_one_mib_cap():
    client = TestClient(create_app(Settings(environment="test")))
    response = client.post("/api/v1/not-a-real-post-route", content=b"x" * (MIB + 1))
    assert _error(response, 413)["code"] == "request_too_large"


def test_upload_over_one_mib_remains_accepted_within_upload_limit(tmp_path):
    client, *_ = _env(tmp_path)
    body = b"a" * (MIB + 1)
    response = client.post(f"/api/v1/projects/{PROJECT}/documents", files={"file": ("large.txt", body, "text/plain")})
    assert response.status_code == 201, response.text


def test_anchor_id_over_256_is_request_validation(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    raw = _run(_stored_anchor(intake, extraction, anchors, "Anchor English")).model_dump(mode="json")
    raw["anchor_id"] = "x" * 257
    response = client.post(RESOLVE, json={"anchors": [raw]})
    assert _error(response, 422)["code"] == "validation_error"


def test_missing_and_foreign_claims_are_indistinguishable(tmp_path):
    client, _store, _docs, _views, intake, extraction, anchors = _env(tmp_path)
    anchor = _run(_stored_anchor(intake, extraction, anchors, "Anchor English"))
    unknown_view = anchor.model_dump(mode="json")
    unknown_view["source"]["extraction_view_id"] = str(uuid.uuid4())
    foreign_claim = anchor.model_dump(mode="json")
    foreign_claim["source"]["document_id"] = str(uuid.uuid4())
    foreign_claim["source"]["document_version_id"] = str(uuid.uuid4())
    foreign_claim["source"]["extraction_view_id"] = str(uuid.uuid4())
    response = client.post(RESOLVE, json={"anchors": [unknown_view, foreign_claim]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == results[1]["status"] == "unavailable"
    assert results[0]["reason"] == results[1]["reason"] == "not_found"
    assert foreign_claim["source"]["document_id"] not in response.text
    assert foreign_claim["source"]["document_version_id"] not in response.text
