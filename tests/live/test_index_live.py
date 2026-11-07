"""Opt-in end-to-end dense retrieval using local Ollama and test PostgreSQL."""

import asyncio
import os
import sys

import pytest
from _fixture import FACT_AR, FACT_EN, MARKDOWN, QUERY_AR, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from pydantic import SecretStr
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_indexes
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService, NoRetrieval, NoRetrievalConfig
from rag_foundry_ingestion import IntakeService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope
from sqlalchemy import text
from test_extraction_service import ingest

pytestmark = [
    pytest.mark.live,
    pytest.mark.db,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
    pytest.mark.skipif(not os.environ.get("RAGF_TEST_DATABASE_URL"), reason="RAGF_TEST_DATABASE_URL is not set"),
]


def test_live_index_retrieves_known_facts(tmp_path):
    async def run():
        settings = load_settings()
        url = SecretStr(os.environ["RAGF_TEST_DATABASE_URL"])
        await asyncio.to_thread(run_migrations, url, "upgrade", "head")
        db = create_database(url)
        maker = db.sessionmaker
        try:
            async with maker() as session, session.begin():
                await session.execute(
                    text(
                        "TRUNCATE runtime_setting_overrides, chunks, index_version_views, "
                        "index_versions, extraction_views, "
                        "document_versions, documents, projects, tenants"
                    )
                )
            scope = make_scope()
            await ensure_local_scope(db.uow, scope)
            store = LocalObjectStore(tmp_path / "objects")
            intake = IntakeService(store, db.uow, max_bytes=100_000)
            doc_id, version_id = await ingest(
                intake, scope, MARKDOWN.encode(), name="live-index.md", media="text/markdown"
            )
            extraction = ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits())
            view = (await extraction.extract(scope, doc_id, version_id)).view
            embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url)
            anchors = AnchorService(store, db.uow, provider_namespace="test")
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            service = IndexService(store, db.uow, anchors)
            version = await service.build(scope, (view.id,), chunker, embedder)
            assert version.state.value == "ready"
            assert version.chunk_count >= 15
            inspected = await service.inspect_chunks(scope, version.id)
            assert inspected and all(row.anchors for row in inspected)
            all_anchors = tuple(anchor for row in inspected for anchor in row.anchors)
            resolved = await anchors.resolve(scope, all_anchors)
            assert resolved and all(item.status.value == "exact" for item in resolved)
            retriever = DenseRetriever(
                DenseConfig(k=5), scope=scope, index_version=version, index=per_call_indexes(db.uow), embedder=embedder
            )
            found = []
            for query, fact, other in ((QUERY_EN, FACT_EN, FACT_AR), (QUERY_AR, FACT_AR, FACT_EN)):
                result = await retriever.retrieve(query)
                fact_hit = next(hit for hit in result.hits if fact in hit.text)
                distractor = next(hit for hit in result.hits if fact not in hit.text)
                assert other not in fact_hit.text
                assert fact_hit.rank <= 5
                fact_chunk = next(row for row in inspected if row.id == fact_hit.chunk_id)
                assert fact_chunk.anchors
                exact = await anchors.resolve(scope, fact_chunk.anchors)
                assert all(item.status.value == "exact" for item in exact)
                found.append(fact_hit.chunk_id)
                print(
                    f"[live] {ascii(query)}: fact rank={fact_hit.rank} score={fact_hit.score:.6f}; "
                    f"distractor rank={distractor.rank} score={distractor.score:.6f}"
                )
            assert len(set(found)) == 2
            control = await NoRetrieval(NoRetrievalConfig()).retrieve(QUERY_EN)
            assert not control.hits and control.grounding == "none"
            missing = await retriever.retrieve("What is the population of Port Sallow?")
            print(
                f"[live] losing case population: top_score={missing.hits[0].score:.6f}"
                if missing.hits
                else "[live] losing case population: no hits"
            )
        finally:
            await db.dispose()

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(run())
