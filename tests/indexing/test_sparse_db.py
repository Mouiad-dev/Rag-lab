"""Sparse SQL storage, scope isolation and immutable rows (requires test PostgreSQL)."""

import asyncio
import os
import sys

import pytest
from _fixture import FACT_EN, MARKDOWN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from pydantic import SecretStr
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_indexes
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_providers.messages import SparseVector
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from test_extraction_service import ingest

URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not URL, reason="RAGF_TEST_DATABASE_URL is not set")]


def test_sparse_persistence_scope_and_immutability(tmp_path):
    async def run():
        url = SecretStr(URL)
        await asyncio.to_thread(run_migrations, url, "upgrade", "head")
        db = create_database(url)
        try:
            async with db.sessionmaker() as session, session.begin():
                await session.execute(
                    text(
                        "TRUNCATE runtime_setting_overrides, kg_spans, kg_facts, kg_entities, knowledge_graphs, "
                        "tree_edges, tree_nodes, summary_trees, "
                        "graph_spans, graph_relations, "
                        "graph_entities, entity_graphs, chunk_representations, representation_sets, "
                        "chunk_sparse_vectors, chunks, index_version_views, "
                        "index_versions, "
                        "extraction_views, document_versions, documents, projects, tenants"
                    )
                )
            scope = make_scope()
            await ensure_local_scope(db.uow, scope)
            store = LocalObjectStore(tmp_path / "objects")
            from rag_foundry_ingestion import IntakeService

            intake = IntakeService(store, db.uow, max_bytes=100_000)
            doc_id, doc_version = await ingest(
                intake, scope, (MARKDOWN + "\n\n" + FACT_EN).encode(), name="sparse.md", media="text/markdown"
            )
            view = (
                await ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits()).extract(
                    scope, doc_id, doc_version
                )
            ).view
            anchors = AnchorService(store, db.uow, provider_namespace="test")
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            version = await IndexService(store, db.uow, anchors).build(
                scope, (view.id,), chunker, LexicalTestEmbedder()
            )
            repo = per_call_indexes(db.uow)
            chunks = await repo.list_chunks(scope, version.id)
            vector = SparseVector(terms=("inventory", "identifier"), weights=(1.0, 0.5))
            with pytest.raises(IndexingError) as unavailable:
                await repo.sparse_search(scope, version.id, "test-encoder", (vector.terms, vector.weights), 5)
            assert unavailable.value.code == "sparse_unavailable"
            await repo.add_sparse(
                scope, version.id, "test-encoder", tuple((c.id, (vector.terms, vector.weights)) for c in chunks)
            )
            hits = await repo.sparse_search(scope, version.id, "test-encoder", (vector.terms, vector.weights), 5)
            assert hits and hits[0].score > 0 and len(hits[0].terms) <= 8
            with pytest.raises(IndexingError) as duplicate:
                await repo.add_sparse(
                    scope, version.id, "test-encoder", ((chunks[0].id, (vector.terms, vector.weights)),)
                )
            assert duplicate.value.code == "duplicate_sparse_vector"
            with pytest.raises(IndexingError) as mismatch:
                await repo.sparse_search(scope, version.id, "other-encoder", (vector.terms, vector.weights), 5)
            assert mismatch.value.code == "incompatible_sparse_encoder"
            with pytest.raises(NotFoundError):
                await repo.sparse_search(make_scope(), version.id, "test-encoder", (vector.terms, vector.weights), 5)
            async with db.sessionmaker() as session, session.begin():
                with pytest.raises(IntegrityError):
                    await session.execute(
                        text("UPDATE chunk_sparse_vectors SET encoder_key='changed' WHERE chunk_id=:id"),
                        {"id": chunks[0].id},
                    )
        finally:
            await db.dispose()

    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(run())
