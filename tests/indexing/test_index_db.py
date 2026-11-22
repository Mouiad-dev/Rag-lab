"""SQL-backed indexing and retrieval checks (requires the test PostgreSQL)."""

import asyncio
import os
import sys

import pytest
from _fixture import FACT_AR, FACT_EN, MARKDOWN, QUERY_AR, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from pydantic import SecretStr
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_indexes
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope
from sqlalchemy import text
from test_extraction_service import ingest

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]


def test_sql_index_retrieves_fixture_facts_and_scopes_search(tmp_path):
    async def run():
        assert _URL
        url = SecretStr(_URL)
        await asyncio.to_thread(run_migrations, url, "upgrade", "head")
        db = create_database(url)
        maker = db.sessionmaker
        try:
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
            scope = make_scope()
            await ensure_local_scope(db.uow, scope)
            store = LocalObjectStore(tmp_path / "objects")
            from rag_foundry_ingestion import IntakeService

            intake = IntakeService(store, db.uow, max_bytes=100_000)
            doc_id, version_id = await ingest(intake, scope, MARKDOWN.encode(), name="index.md", media="text/markdown")
            extraction = ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits())
            view = (await extraction.extract(scope, doc_id, version_id)).view
            embedder = LexicalTestEmbedder()
            anchors = AnchorService(store, db.uow, provider_namespace="test")
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            service = IndexService(store, db.uow, anchors)
            version = await service.build(scope, (view.id,), chunker, embedder)
            assert version.state.value == "ready"
            assert version.chunk_count >= 15
            inspected = await service.inspect_chunks(scope, version.id)
            assert inspected and all(row.anchors for row in inspected)
            resolutions = await anchors.resolve(scope, tuple(a for row in inspected for a in row.anchors))
            assert resolutions and all(item.status.value == "exact" for item in resolutions)
            retriever = DenseRetriever(
                DenseConfig(k=5), scope=scope, index_version=version, index=per_call_indexes(db.uow), embedder=embedder
            )
            for query, fact in ((QUERY_EN, FACT_EN), (QUERY_AR, FACT_AR)):
                result = await retriever.retrieve(query)
                hit = next(hit for hit in result.hits if fact in hit.text)
                assert hit.rank <= 2
            with pytest.raises(NotFoundError):
                await per_call_indexes(db.uow).search(make_scope(), version.id, (1.0,) * 256, 5)
        finally:
            await db.dispose()

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(run())
