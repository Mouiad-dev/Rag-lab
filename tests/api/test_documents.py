"""Upload and version API tests (ADR-0009). Injected in-memory repository and a temp object store."""

import asyncio
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from rag_foundry_api.deps import get_intake_service
from rag_foundry_api.main import create_app
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE
from rag_foundry_domain.testing import InMemoryDocumentRepository
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion.intake import IntakeService
from rag_foundry_storage.local_store import LocalObjectStore

MAX = 1000
PROJECT = LOCAL_DEV_SCOPE.project_id
DOCS = f"/api/v1/projects/{PROJECT}/documents"
OTHER_PROJECT = "11111111-2222-4333-8444-555555555555"


def _client(tmp_path, *, environment="test", **kwargs) -> TestClient:
    repo = InMemoryDocumentRepository([LOCAL_DEV_SCOPE])
    service = IntakeService(
        LocalObjectStore(tmp_path / "objects"), InMemoryUnitOfWorkFactory(documents=repo), max_bytes=MAX
    )
    settings = Settings(environment=environment, max_upload_bytes=MAX, object_store_root=tmp_path / "objects")
    app = create_app(settings)
    app.dependency_overrides[get_intake_service] = lambda: service
    return TestClient(app, **kwargs)


@pytest.fixture
def client(tmp_path):
    return _client(tmp_path)


def _file(data: bytes, name: str = "a.txt", ctype: str = "text/plain"):
    return {"file": (name, data, ctype)}


def _assert_envelope(response, status: int) -> dict:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id", "retryable"}
    assert uuid.UUID(error["request_id"]).version == 4
    return error


def test_create_dedup_new_version_and_get(client):
    first = client.post(DOCS, files=_file(b"hello"))
    assert first.status_code == 201
    body = first.json()
    assert body["version_number"] == 1 and body["deduplicated"] is False
    assert body["size_bytes"] == 5 and body["media_type"] == "text/plain" and body["text_encoding"] == "utf-8"
    assert set(body) == {
        "document_id",
        "version_id",
        "version_number",
        "sha256",
        "size_bytes",
        "media_type",
        "text_encoding",
        "original_filename",
        "created_at",
        "deduplicated",
    }

    again = client.post(DOCS, files=_file(b"hello"))
    assert again.status_code == 200
    assert again.json()["deduplicated"] is True
    assert again.json()["version_id"] == body["version_id"]

    versions = f"{DOCS}/{body['document_id']}/versions"
    v2 = client.post(versions, files=_file(b"hello again"))
    assert v2.status_code == 201
    assert v2.json()["version_number"] == 2 and v2.json()["document_id"] == body["document_id"]
    back = client.post(versions, files=_file(b"hello"))
    assert back.status_code == 200 and back.json()["version_number"] == 1

    got = client.get(f"/api/v1/documents/{body['document_id']}/versions/{v2.json()['version_id']}")
    assert got.status_code == 200
    assert got.json()["version_number"] == 2 and got.json()["deduplicated"] is False
    assert "object_key" not in got.text


def test_no_object_key_in_responses(client):
    assert "object_key" not in client.post(DOCS, files=_file(b"x")).text


def test_arabic_filename_round_trip(client):
    name = "\u0645\u0644\u0641 \u0639\u0631\u0628\u064a.txt"
    r = client.post(DOCS, files=_file("\u0645\u0631\u062d\u0628\u0627".encode(), name))
    assert r.status_code == 201
    assert r.json()["original_filename"] == name
    got = client.get(f"/api/v1/documents/{r.json()['document_id']}/versions/{r.json()['version_id']}")
    assert got.json()["original_filename"] == name


def test_oversize_by_content_length_is_413(client):
    r = client.post(DOCS, files=_file(b"a" * (MAX + 70 * 1024)))
    assert "content-length" in r.request.headers
    assert _assert_envelope(r, 413)["code"] == "upload_too_large"


def test_early_content_length_413_never_touches_store_or_repo(tmp_path):
    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"touched {name}")

    settings = Settings(environment="test", max_upload_bytes=MAX)
    app = create_app(settings)
    app.dependency_overrides[get_intake_service] = lambda: Boom()
    c = TestClient(app)
    headers = {"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(MAX + 70 * 1024)}
    r = c.post(DOCS, content=b"short", headers=headers)
    assert _assert_envelope(r, 413)["code"] == "upload_too_large"


def test_empty_file_is_422(client):
    assert _assert_envelope(client.post(DOCS, files=_file(b"")), 422)["code"] == "empty_upload"


def test_oversize_chunked_body_is_413(client):
    def body():
        for _ in range(80):
            yield b"a" * 1024

    r = client.post(DOCS, content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert "content-length" not in r.request.headers
    assert r.request.headers["transfer-encoding"] == "chunked"
    assert _assert_envelope(r, 413)["code"] == "upload_too_large"


def test_file_over_limit_but_within_overhead_is_413_from_service(client):
    r = client.post(DOCS, files=_file(b"a" * (MAX + 1)))
    assert _assert_envelope(r, 413)["code"] == "upload_too_large"


def test_two_file_parts_is_422(client):
    r = client.post(DOCS, files=[("file", ("a.txt", b"a", "text/plain")), ("file", ("b.txt", b"b", "text/plain"))])
    _assert_envelope(r, 422)


def test_extra_text_field_is_422(client):
    r = client.post(DOCS, files=_file(b"a"), data={"note": "x"})
    _assert_envelope(r, 422)


def test_wrongly_named_file_is_422(client):
    r = client.post(DOCS, files={"upload": ("a.txt", b"a", "text/plain")})
    _assert_envelope(r, 422)


def test_non_multipart_is_422(client):
    _assert_envelope(client.post(DOCS, content=b"hello", headers={"Content-Type": "text/plain"}), 422)
    _assert_envelope(client.post(DOCS, data={"a": "b"}), 422)
    _assert_envelope(client.post(DOCS), 422)


def test_unsupported_media_type_is_415(client):
    r = client.post(DOCS, files=_file(b"{}", "a.json", "application/json"))
    assert _assert_envelope(r, 415)["code"] == "unsupported_media_type"


def test_bad_filename_is_422(client):
    r = client.post(DOCS, files=_file(b"a", "a/b.txt"))
    assert _assert_envelope(r, 422)["code"] == "invalid_filename"


def test_other_project_is_404_same_as_unknown_document(client):
    other = client.post(f"/api/v1/projects/{OTHER_PROJECT}/documents", files=_file(b"a"))
    assert _assert_envelope(other, 404)["code"] == "not_found"
    unknown_doc = client.post(f"{DOCS}/{uuid.uuid4()}/versions", files=_file(b"a"))
    assert _assert_envelope(unknown_doc, 404)["message"] == other.json()["error"]["message"]
    get = client.get(f"/api/v1/documents/{uuid.uuid4()}/versions/{uuid.uuid4()}")
    assert _assert_envelope(get, 404)["message"] == other.json()["error"]["message"]
    bad = client.get("/api/v1/documents/not-a-uuid/versions/also-not")
    _assert_envelope(bad, 404)


def test_prod_is_401(tmp_path):
    c = _client(tmp_path, environment="prod")
    assert _assert_envelope(c.post(DOCS, files=_file(b"a")), 401)["code"] == "auth_not_configured"
    _assert_envelope(c.get(f"/api/v1/documents/{uuid.uuid4()}/versions/{uuid.uuid4()}"), 401)


def test_no_database_and_no_injection_is_503_and_health_ok():
    c = TestClient(create_app(Settings(environment="test")))
    error = _assert_envelope(c.post(DOCS, files=_file(b"a")), 503)
    assert error["code"] == "storage_unavailable" and error["retryable"] is True
    _assert_envelope(c.get(f"/api/v1/documents/{uuid.uuid4()}/versions/{uuid.uuid4()}"), 503)
    assert c.get("/health").status_code == 200


def test_unknown_project_and_malformed_body_without_storage_keep_order():
    client = TestClient(create_app(Settings(environment="test", database_url=None)))
    unknown = client.post("/api/v1/projects/unknown/documents", files=_file(b"x"))
    malformed = client.post(DOCS, data="broken", headers={"content-type": "multipart/form-data; boundary=bad"})
    unknown_version = client.post("/api/v1/projects/unknown/documents/document/versions", files=_file(b"x"))
    malformed_version = client.post(
        f"{DOCS}/document/versions", data="broken", headers={"content-type": "multipart/form-data; boundary=bad"}
    )
    for response in (unknown, malformed, unknown_version, malformed_version):
        assert (response.status_code, response.json()["error"]["code"]) == (503, "storage_unavailable")


def test_unknown_route_has_envelope(client):
    _assert_envelope(client.get("/nope"), 404)


# --- end to end with the SQL repository (needs a real PostgreSQL) ---

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")


@pytest.mark.db
@pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")
def test_end_to_end_with_sql_repository(tmp_path):
    from rag_foundry_storage.sql.migrate import run_migrations

    assert _URL
    run_migrations(SecretStr(_URL), "upgrade", "head")
    settings = Settings(
        environment="test", database_url=SecretStr(_URL), object_store_root=tmp_path / "objects", max_upload_bytes=MAX
    )
    backend_options = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
    unique = uuid.uuid4().hex.encode()
    with TestClient(create_app(settings), backend_options=backend_options) as c:
        first = c.post(DOCS, files=_file(b"db " + unique))
        assert first.status_code == 201, first.text
        again = c.post(DOCS, files=_file(b"db " + unique))
        assert again.status_code == 200 and again.json()["version_id"] == first.json()["version_id"]
        doc = first.json()["document_id"]
        v2 = c.post(f"{DOCS}/{doc}/versions", files=_file(b"db two " + unique))
        assert v2.status_code == 201 and v2.json()["version_number"] == 2
        got = c.get(f"/api/v1/documents/{doc}/versions/{v2.json()['version_id']}")
        assert got.status_code == 200 and got.json()["sha256"] == v2.json()["sha256"]
        assert "object_key" not in got.text
        _assert_envelope(c.get(f"/api/v1/documents/{uuid.uuid4()}/versions/{uuid.uuid4()}"), 404)
