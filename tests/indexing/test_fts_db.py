"""PostgreSQL full-text retrieval checks (requires the test PostgreSQL)."""

import asyncio
import os
import sys

import pytest
from _fixture import FACT_EN, MARKDOWN, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from pydantic import SecretStr
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_adapters import per_call_indexes
from rag_foundry_indexing import BM25Config, BM25Retriever, FtsConfig, FtsRetriever, IndexService
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


def test_fts_simple_scope_and_r03_comparison(tmp_path):
    async def run():
        assert _URL
        url = SecretStr(_URL)
        await asyncio.to_thread(run_migrations, url, "upgrade", "head")
        db = create_database(url)
        try:
            async with db.sessionmaker() as session, session.begin():
                await session.execute(
                    text(
                        "TRUNCATE runtime_setting_overrides, graph_spans, graph_relations, "
                        "graph_entities, entity_graphs, chunk_representations, representation_sets, "
                        "chunk_sparse_vectors, chunks, index_version_views, "
                        "index_versions, extraction_views, document_versions, documents, projects, tenants"
                    )
                )
            scope = make_scope()
            await ensure_local_scope(db.uow, scope)
            store = LocalObjectStore(tmp_path / "objects")
            from rag_foundry_ingestion import IntakeService

            intake = IntakeService(store, db.uow, max_bytes=100_000)
            id_text = (
                "\u0631\u0642\u0645 \u0627\u0644\u0642\u0637\u0639\u0629 AB-1234 "
                "\u0641\u064a \u0627\u0644\u0645\u062e\u0632\u0646."
            )
            alt_text = id_text.replace("AB-1234", "AB-1243")
            content = MARKDOWN + "\n\n" + id_text + "\n\n" + alt_text
            doc_id, version_id = await ingest(intake, scope, content.encode(), name="fts.md", media="text/markdown")
            extraction = ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits())
            view = (await extraction.extract(scope, doc_id, version_id)).view
            embedder = LexicalTestEmbedder()
            anchors = AnchorService(store, db.uow, provider_namespace="test")
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            version = await IndexService(store, db.uow, anchors).build(scope, (view.id,), chunker, embedder)
            indexes = per_call_indexes(db.uow)
            retriever = FtsRetriever(FtsConfig(k=50), scope=scope, index_version=version, index=indexes)
            id_hits = (await retriever.retrieve("AB-1234")).hits
            assert any("AB-1234" in hit.text for hit in id_hits)
            assert not any("AB-1243" in hit.text and "AB-1234" not in hit.text for hit in id_hits)
            arabic_query = "\u0627\u0644\u0645\u062e\u0632\u0646"
            assert any(
                "AB-1234" in hit.text or "AB-1243" in hit.text for hit in (await retriever.retrieve(arabic_query)).hits
            )
            assert not (await retriever.retrieve(QUERY_EN)).hits
            or_retriever = FtsRetriever(
                FtsConfig(k=50, operator="or"), scope=scope, index_version=version, index=indexes
            )
            assert any(FACT_EN in hit.text for hit in (await or_retriever.retrieve(QUERY_EN)).hits)
            with pytest.raises(NotFoundError):
                await retriever.__class__(
                    FtsConfig(), scope=make_scope(), index_version=version, index=indexes
                ).retrieve("AB-1234")
            malicious = '"); DROP TABLE chunks; --'
            async with db.sessionmaker() as session:
                before = (await session.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
            assert isinstance((await retriever.retrieve(malicious)).hits, tuple)
            async with db.sessionmaker() as session:
                assert (await session.execute(text("SELECT to_regclass('chunks')"))).scalar_one() == "chunks"
                assert (await session.execute(text("SELECT count(*) FROM chunks"))).scalar_one() == before
            bm25 = BM25Retriever(BM25Config(k=5), scope=scope, index_version=version, index=indexes)
            # Losing case for 'simple': a diacritized query does not match the plain word in AND mode, and
            # OR mode splits it into single letters (diacritics are not word characters), which match nothing.
            # R03 folds diacritics first, so it finds the chunk.
            diacritized = "\u0627\u0644\u0645\u064e\u062e\u0652\u0632\u064e\u0646"
            assert not (await retriever.retrieve(diacritized)).hits
            assert not any("AB-12" in hit.text for hit in (await or_retriever.retrieve(diacritized)).hits)
            assert any("AB-1234" in hit.text for hit in (await bm25.retrieve(diacritized)).hits)
            print("diacritized query: R04 and/or no hits; R03 hits")
            for query, operator in (("AB-1234", "and"), (QUERY_EN, "or")):
                r03 = await bm25.retrieve(query)
                r04 = await FtsRetriever(
                    FtsConfig(k=5, operator=operator), scope=scope, index_version=version, index=indexes
                ).retrieve(query)
                for label, result in (("R03", r03), ("R04", r04)):
                    for hit in result.hits[:3]:
                        print(label, hit.rank, hit.score, hit.score_components, hit.text[:40])
                assert r03.scoring != r04.scoring
        finally:
            await db.dispose()

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(run())
