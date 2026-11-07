"""Pin route validation and storage-unavailable ordering with offline services."""

from fastapi.testclient import TestClient
from rag_foundry_api.main import Settings, create_app


def _result(response):
    body = response.json()
    return response.status_code, body["error"]["code"]


def test_upload_error_order_without_storage():
    app = create_app(Settings(database_url=None))
    with TestClient(app) as client:
        valid = client.post("/api/v1/projects/project/documents", files={"file": ("x.txt", b"hello", "text/plain")})
        other_project = client.post(
            "/api/v1/projects/unknown/documents", files={"file": ("x.txt", b"hello", "text/plain")}
        )
        malformed = client.post(
            "/api/v1/projects/project/documents",
            data="broken",
            headers={"content-type": "multipart/form-data; boundary=bad"},
        )
    assert _result(valid) == (503, "storage_unavailable")
    assert _result(other_project) == (503, "storage_unavailable")
    assert _result(malformed) == (503, "storage_unavailable")


def test_new_version_error_order_without_storage():
    app = create_app(Settings(database_url=None))
    with TestClient(app) as client:
        valid = client.post(
            "/api/v1/projects/project/documents/document/versions", files={"file": ("x.txt", b"hello", "text/plain")}
        )
        other_project = client.post(
            "/api/v1/projects/unknown/documents/document/versions", files={"file": ("x.txt", b"hello", "text/plain")}
        )
        malformed = client.post(
            "/api/v1/projects/project/documents/document/versions",
            data="broken",
            headers={"content-type": "multipart/form-data; boundary=bad"},
        )
    assert _result(valid) == (503, "storage_unavailable")
    assert _result(other_project) == (503, "storage_unavailable")
    assert _result(malformed) == (503, "storage_unavailable")


def test_get_version_error_order_without_storage():
    app = create_app(Settings(database_url=None))
    with TestClient(app) as client:
        valid = client.get("/api/v1/documents/document/versions/version")
        other_tenant = client.get("/api/v1/documents/other-document/versions/version")
    # The get-version route takes no body, so there is no malformed-body case to pin here.
    assert _result(valid) == (503, "storage_unavailable")
    assert _result(other_tenant) == (503, "storage_unavailable")


def test_anchor_resolve_error_order_without_storage():
    app = create_app(Settings(database_url=None))
    malformed_body = {"anchors": []}
    with TestClient(app) as client:
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
        valid = client.post("/api/v1/projects/project/anchors:resolve", json={"anchors": [anchor]})
        other_project = client.post("/api/v1/projects/unknown/anchors:resolve", json={"anchors": [anchor]})
        malformed = client.post("/api/v1/projects/project/anchors:resolve", json=malformed_body)
    assert _result(valid) == (404, "not_found")
    assert _result(other_project) == (404, "not_found")
    assert _result(malformed) == (422, "validation_error")


def test_run_error_order_without_storage():
    app = create_app(Settings(database_url=None))
    with TestClient(app) as client:
        valid = client.post("/api/v1/projects/project/runs", json={"question": "hello"})
        other_project = client.post("/api/v1/projects/unknown/runs", json={"question": "hello"})
        malformed = client.post("/api/v1/projects/project/runs", json={"question": ""})
    assert _result(valid) == (404, "not_found")
    assert _result(other_project) == (404, "not_found")
    assert _result(malformed) == (422, "validation_error")
