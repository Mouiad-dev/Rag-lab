"""Anchor resolution through SQL repositories (requires the test PostgreSQL)."""

import asyncio
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from rag_foundry_api.deps import get_anchor_service, get_intake_service
from rag_foundry_api.main import create_app
from rag_foundry_core.settings import Settings
from rag_foundry_domain import LOCAL_DEV_SCOPE
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_documents, per_call_extractions
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_ingestion.intake import IntakeService
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope
from sqlalchemy import text

PROJECT = LOCAL_DEV_SCOPE.project_id
DOCS = f"/api/v1/projects/{PROJECT}/documents"
RESOLVE = f"/api/v1/projects/{PROJECT}/anchors:resolve"
_URL = os.environ.get("RAGF_TEST_DATABASE_URL")

pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]


class _Stream:
    def __init__(self, data: bytes):
        self.data = data

    def __aiter__(self):
        async def generate():
            yield self.data

        return generate()


@pytest.fixture
def sql_repositories():
    assert _URL
    url = SecretStr(_URL)
    run_migrations(url, "upgrade", "head")
    db = create_database(url)
    maker = db.sessionmaker

    async def setup():
        async with maker() as session, session.begin():
            await session.execute(
                text(
                    "TRUNCATE runtime_setting_overrides, tree_edges, tree_nodes, summary_trees, "
                    "graph_spans, graph_relations, "
                    "graph_entities, entity_graphs, chunk_representations, representation_sets, "
                    "chunk_sparse_vectors, chunks, index_version_views, "
                    "index_versions, extraction_views, "
                    "document_versions, documents, projects, tenants"
                )
            )
        docs = per_call_documents(db.uow)
        await ensure_local_scope(db.uow, LOCAL_DEV_SCOPE)
        return docs, per_call_extractions(db.uow), db

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        docs, views, _db = runner.run(setup())
    try:
        yield docs, views, db
    finally:

        async def close():
            await db.dispose()

        with asyncio.Runner(loop_factory=factory) as runner:
            runner.run(close())


def _run(coro):
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)


def _services(tmp_path, docs, views, db):
    store = LocalObjectStore(tmp_path / "objects")
    intake = IntakeService(store, db.uow, max_bytes=1000)
    extraction = ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    anchors = AnchorService(store, db.uow, provider_namespace="rag-foundry-local")
    app = create_app(Settings(environment="test", max_upload_bytes=1000))
    app.dependency_overrides[get_intake_service] = lambda: intake
    app.dependency_overrides[get_anchor_service] = lambda: anchors
    return store, intake, extraction, anchors, app


def test_sql_upload_extract_and_resolve_exact(tmp_path, sql_repositories):
    docs, views, db = sql_repositories
    _store, _intake, extraction, anchors, app = _services(tmp_path, docs, views, db)
    backend_options = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
    with TestClient(app, backend_options=backend_options) as client:
        unique = uuid.uuid4().hex
        expected = "Canonical \u00c5 text " + unique
        body = ("Canonical A\u030a text " + unique).encode("utf-8")
        uploaded = client.post(DOCS, files={"file": ("source.md", body, "text/markdown")})
        assert uploaded.status_code == 201, uploaded.text
        result = uploaded.json()
        version = _run(docs.get_version(LOCAL_DEV_SCOPE, result["document_id"], result["version_id"]))
        view = _run(extraction.extract(LOCAL_DEV_SCOPE, result["document_id"], result["version_id"])).view
        payload = _run(extraction.load_payload(LOCAL_DEV_SCOPE, view))
        block = payload.blocks[0]
        assert block.block_id == "b00000"
        assert payload.canonical_text[block.start : block.end] == expected
        anchor = _run(anchors.anchor(LOCAL_DEV_SCOPE, view.id, block.block_id, 0, block.end - block.start))

        resolved = client.post(RESOLVE, json={"anchors": [anchor.model_dump(mode="json")]})
        assert resolved.status_code == 200, resolved.text
        item = resolved.json()["results"][0]
        assert item["status"] == "exact"
        assert item["exact"] == expected
        assert version.id == result["version_id"]
        assert "object_key" not in resolved.text


def test_sql_foreign_scope_anchor_is_unavailable(tmp_path, sql_repositories, monkeypatch):
    docs, views, db = sql_repositories
    _store, intake, extraction, anchors, app = _services(tmp_path, docs, views, db)
    foreign = make_scope()
    _run(ensure_local_scope(db.uow, foreign))
    unique = uuid.uuid4().hex
    uploaded = _run(
        intake.ingest(
            foreign,
            _Stream(("Foreign payload " + unique).encode("utf-8")),
            media_type="text/markdown",
            filename="foreign.md",
        )
    )
    view = _run(extraction.extract(foreign, uploaded.document.id, uploaded.version.id)).view
    payload = _run(extraction.load_payload(foreign, view))
    block = payload.blocks[0]
    anchor = _run(anchors.anchor(foreign, view.id, block.block_id, 0, block.end - block.start))
    assert _run(anchors.resolve(foreign, [anchor]))[0].status == "exact"

    from rag_foundry_ingestion import anchor_service

    original_read_payload = anchor_service.read_payload
    calls = []

    async def spy_read_payload(*args, **kwargs):
        calls.append((args, kwargs))
        return await original_read_payload(*args, **kwargs)

    monkeypatch.setattr(anchor_service, "read_payload", spy_read_payload)

    backend_options = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
    with TestClient(app, backend_options=backend_options) as client:
        response = client.post(RESOLVE, json={"anchors": [anchor.model_dump(mode="json")]})
    assert calls == []
    assert response.status_code == 200, response.text
    item = response.json()["results"][0]
    assert item["status"] == "unavailable" and item["reason"] == "not_found"
    assert unique not in response.text
    assert str(foreign.tenant_id) not in response.text
    assert str(foreign.project_id) not in response.text


def test_sql_malformed_extraction_view_id_is_unavailable(tmp_path, sql_repositories, monkeypatch):
    docs, views, db = sql_repositories
    _store, intake, extraction, anchors, app = _services(tmp_path, docs, views, db)
    # Upload bytes are unique so the SQL version is never a deduplicated fixture artifact.
    unique = uuid.uuid4().hex
    uploaded = _run(
        intake.ingest(
            LOCAL_DEV_SCOPE,
            _Stream(("Malformed payload " + unique).encode("utf-8")),
            media_type="text/markdown",
            filename="malformed.md",
        )
    )
    view = _run(extraction.extract(LOCAL_DEV_SCOPE, uploaded.document.id, uploaded.version.id)).view
    payload = _run(extraction.load_payload(LOCAL_DEV_SCOPE, view))
    block = payload.blocks[0]
    anchor = _run(anchors.anchor(LOCAL_DEV_SCOPE, view.id, block.block_id, 0, block.end - block.start))
    assert _run(anchors.resolve(LOCAL_DEV_SCOPE, [anchor]))[0].status == "exact"

    from rag_foundry_ingestion import anchor_service

    original_read_payload = anchor_service.read_payload
    calls = []

    async def spy_read_payload(*args, **kwargs):
        calls.append((args, kwargs))
        return await original_read_payload(*args, **kwargs)

    monkeypatch.setattr(anchor_service, "read_payload", spy_read_payload)
    raw = anchor.model_dump(mode="json")
    raw["source"]["extraction_view_id"] = "not-a-uuid"

    backend_options = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
    with TestClient(app, backend_options=backend_options) as client:
        response = client.post(RESOLVE, json={"anchors": [raw]})
    assert calls == []
    assert response.status_code == 200, response.text
    item = response.json()["results"][0]
    assert item["status"] == "unavailable" and item["reason"] == "not_found"
    assert "not-a-uuid" not in response.text
    assert "Traceback" not in response.text and "psycopg" not in response.text
