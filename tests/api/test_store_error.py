from fastapi import FastAPI
from fastapi.testclient import TestClient
from rag_foundry_api.errors import register_error_handlers
from rag_foundry_domain.errors import StoreError


def test_store_unavailable_error_envelope():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/")
    async def endpoint():
        raise StoreError("store_unavailable")

    response = TestClient(app).get("/")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"
    assert response.json()["error"]["message"] == "storage unavailable"


def test_commit_failed_error_envelope():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/")
    async def endpoint():
        raise StoreError("commit_failed")

    response = TestClient(app).get("/")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "storage_error"
    assert response.json()["error"]["message"] == "storage error"
