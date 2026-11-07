"""Characterize current service and SQL transaction boundaries."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.index_testing import make_chunk, make_new_index_version
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_documents, per_call_extractions, per_call_indexes
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion import IntakeService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]


def _run(coro):
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)


async def _build(tmp_path):
    assert _URL
    await asyncio.to_thread(run_migrations, SecretStr(_URL), "upgrade", "head")
    db = create_database(SecretStr(_URL))
    scope = make_scope()
    docs = per_call_documents(db.uow)
    await ensure_local_scope(db.uow, scope)
    views, indexes = per_call_extractions(db.uow), per_call_indexes(db.uow)
    store = LocalObjectStore(tmp_path / "objects")
    intake = IntakeService(store, docs, max_bytes=100_000)
    payload = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 20).encode()
    outcome = await intake.ingest(scope, _bytes(payload), media_type="text/plain", filename="source.txt")
    extraction = ExtractionService(store, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(scope, outcome.document.id, outcome.version.id)).view
    anchors = AnchorService(store, views, provider_namespace="test")
    chunker = RecursiveChunker(RecursiveConfig(chunk_size=32, chunk_overlap=0))

    def index_service(index, batch_size):
        return IndexService(store, views, index, anchors, batch_size=batch_size)

    return db, scope, docs, views, indexes, store, view, anchors, chunker, payload, intake, index_service


async def _bytes(data):
    yield data


class _Embedder:
    def __init__(self, *, fail_at=None):
        self.profile = SimpleNamespace(
            id="test-profile",
            provider="test",
            model="test",
            capabilities=SimpleNamespace(embedding_dim=4),
        )
        self.calls = 0
        self.fail_at = fail_at

    async def embed(self, texts, *, input_type):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("embed failed")
        return SimpleNamespace(vectors=tuple((1.0, 0.0, 0.0, 0.0) for _ in texts))


class _TrackingIndex:
    def __init__(self, repository):
        self.repository = repository
        self.created = None

    async def create_index_version(self, *args):
        self.created = await self.repository.create_index_version(*args)
        return self.created

    def __getattr__(self, name):
        return getattr(self.repository, name)


def test_index_build_failure_keeps_committed_batches_and_marks_failed(tmp_path):
    async def run():
        db, scope, _, _, indexes, _, view, _, chunker, _, _, index_service = await _build(tmp_path)
        try:
            tracking = _TrackingIndex(indexes)
            service = index_service(tracking, 2)
            embedder = _Embedder(fail_at=2)
            with pytest.raises(RuntimeError, match="embed failed"):
                await service.build(scope, (view.id,), chunker, embedder)
            failed = await indexes.get_index_version(scope, tracking.created.id)
            assert failed.state.value == "failed"
            chunks = await indexes.list_chunks(scope, tracking.created.id)
            # Batch 1 (ordinals 0 and 1) was committed; the failure happened in batch 2.
            assert [c.ordinal for c in chunks] == [0, 1]
            assert embedder.calls == 2
        finally:
            await db.dispose()

    _run(run())


def test_add_chunks_atomic_failure_and_repository_reuse(tmp_path):
    async def run():
        db, scope, _, _, indexes, _, view, _, _, _, _, _ = await _build(tmp_path)
        try:
            version = await indexes.create_index_version(scope, make_new_index_version((view.id,)))
            first = (make_chunk(view.id, 0, (1, 0, 0, 0), "first"),)
            await indexes.add_chunks(scope, version.id, first)
            from rag_foundry_domain.errors import IndexingError

            with pytest.raises(IndexingError, match="duplicate_ordinal"):
                await indexes.add_chunks(
                    scope,
                    version.id,
                    (make_chunk(view.id, 1, (1, 0, 0, 0), "temporary"), make_chunk(view.id, 0, (0, 1, 0, 0), "dup")),
                )
            assert [c.ordinal for c in await indexes.list_chunks(scope, version.id)] == [0]
            successful = (make_chunk(view.id, 1, (1, 0, 0, 0), "second"), make_chunk(view.id, 2, (0, 1, 0, 0), "third"))
            await indexes.add_chunks(scope, version.id, successful)
            ready = await indexes.mark_ready(scope, version.id)
            assert ready.chunk_count == 3
        finally:
            await db.dispose()

    _run(run())


def test_intake_deduplicates_sequential_and_concurrent_identical_uploads(tmp_path):
    async def run():
        db, scope, _, _, _, _, _, _, _, _, intake, _ = await _build(tmp_path)
        try:
            data = b"concurrent identical upload contents"

            async def ingest(content=data):
                return await intake.ingest(scope, _bytes(content), media_type="text/plain", filename="same.txt")

            one, two = await ingest(), await ingest()
            assert one.document.id == two.document.id
            assert one.version.id == two.version.id
            assert two.deduplicated
            # The race needs content not stored yet: exactly one concurrent upload creates it.
            fresh = b"concurrent race contents, not uploaded before"
            outcomes = await asyncio.gather(*(ingest(fresh) for _ in range(5)))
            assert len({o.document.id for o in outcomes}) == 1
            assert sum(not o.deduplicated for o in outcomes) == 1
        finally:
            await db.dispose()

    _run(run())


def test_intake_rejects_unknown_project_before_reading_upload(tmp_path):
    async def run():
        db, scope, _, _, _, _, _, _, _, _, intake, _ = await _build(tmp_path)
        try:
            from rag_foundry_domain import Scope, new_id
            from rag_foundry_domain.errors import NotFoundError

            started = False

            async def chunks():
                nonlocal started
                started = True
                yield b"data"

            with pytest.raises(NotFoundError):
                await intake.ingest(
                    Scope(tenant_id=scope.tenant_id, project_id=new_id()),
                    chunks(),
                    media_type="text/plain",
                    filename="x.txt",
                )
            assert not started
        finally:
            await db.dispose()

    _run(run())
