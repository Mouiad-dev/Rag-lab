import asyncio
import hashlib
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from _fixture import FACT_AR, FACT_EN, MARKDOWN, QUERY_AR, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.indexing import IndexState, corpus_sha256_for, embedding_key_for
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing import (
    DenseConfig,
    DenseRetriever,
    IndexService,
    NoRetrieval,
    NoRetrievalConfig,
    register_builtin_plugins,
)
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.fake import fake_embedding_profile
from rag_foundry_techniques import TechniqueRegistry, load_catalogue
from test_extraction_service import ingest, setup


@dataclass
class Env:
    scope: object
    extraction: object
    store: object
    views: object
    view: object
    index: InMemoryIndexRepository
    anchors: AnchorService
    chunker: RecursiveChunker

    def service(self, index=None, **kwargs) -> IndexService:
        return IndexService(self.store, self.views, index or self.index, self.anchors, **kwargs)


class RecordingIndex:
    def __init__(self, inner):
        self.inner = inner
        self.version = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def create_index_version(self, scope, new):
        self.version = await self.inner.create_index_version(scope, new)
        return self.version


async def _env(tmp_path) -> Env:
    scope = make_scope()
    extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
    doc_id, version_id = await ingest(intake, scope, MARKDOWN.encode("utf-8"), name="notes.md", media="text/markdown")
    view = (await extraction.extract(scope, doc_id, version_id)).view
    return Env(
        scope=scope,
        extraction=extraction,
        store=store,
        views=views,
        view=view,
        index=InMemoryIndexRepository(views),
        anchors=AnchorService(store, views, provider_namespace="test"),
        chunker=RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP)),
    )


async def _built(tmp_path):
    env = await _env(tmp_path)
    embedder = LexicalTestEmbedder()
    built = await env.service().build(env.scope, (env.view.id,), env.chunker, embedder)
    return env, embedder, built


def test_build_stores_exact_spans_and_pins_versions(tmp_path):
    async def run():
        env, embedder, built = await _built(tmp_path)
        chunks = await env.index.list_chunks(env.scope, built.id)
        payload = await env.extraction.load_payload(env.scope, env.view)
        assert built.chunk_count == len(chunks) >= 3 * 5
        assert all(row.text == payload.canonical_text[row.start : row.end] for row in chunks)
        assert built.unanchored_chunk_count == 0
        assert built.chunker_id == "C05"
        assert built.chunker_version
        assert built.chunker_config_sha256 == hashlib.sha256(built.chunker_config_json.encode("utf-8")).hexdigest()
        assert built.corpus_sha256 == corpus_sha256_for([(env.view.id, env.view.payload_sha256)])
        assert built.embedding_model == "lexical-test-v1"
        assert built.embedding_key == embedding_key_for(embedder.profile.provider, "lexical-test-v1", 256)
        museum_chunks = [row for row in chunks if "maritime museum" in row.text]
        assert museum_chunks and all(row.split_level >= 2 for row in museum_chunks)

    asyncio.run(run())


def test_documents_are_embedded_in_bounded_batches(tmp_path):
    async def run():
        env, embedder, built = await _built(tmp_path)
        documents = [texts for texts, input_type in embedder.calls if input_type == "document"]
        assert len(documents) == math.ceil(built.chunk_count / 32)
        assert all(len(texts) <= 32 for texts in documents)
        small = LexicalTestEmbedder()
        rebuilt = await env.service(batch_size=4).build(env.scope, (env.view.id,), env.chunker, small)
        assert [len(texts) for texts, _ in small.calls][:-1] == [4] * (len(small.calls) - 1)
        assert len(small.calls) == math.ceil(rebuilt.chunk_count / 4) >= 2
        assert all(input_type == "document" for _, input_type in small.calls)

    asyncio.run(run())


def test_inspected_chunk_anchors_resolve_exact(tmp_path):
    async def run():
        env, _embedder, built = await _built(tmp_path)
        inspected = await env.service().inspect_chunks(env.scope, built.id)
        assert inspected and all(item.anchors for item in inspected)
        resolutions = await env.anchors.resolve(env.scope, tuple(a for row in inspected for a in row.anchors))
        assert resolutions and all(item.status.value == "exact" for item in resolutions)

    asyncio.run(run())


def test_dense_retrieval_ranks_each_fact_in_its_own_chunk(tmp_path):
    async def run():
        env, embedder, built = await _built(tmp_path)
        retriever = DenseRetriever(
            DenseConfig(k=5), scope=env.scope, index_version=built, index=env.index, embedder=embedder
        )
        result_en = await retriever.retrieve(QUERY_EN)
        result_ar = await retriever.retrieve(QUERY_AR)
        assert embedder.calls[-1][1] == embedder.calls[-2][1] == "query"
        assert result_en.technique_id == result_ar.technique_id == "R02"
        assert result_en.grounding == result_ar.grounding == "corpus"
        en_fact_hit = next(hit for hit in result_en.hits if FACT_EN in hit.text)
        ar_fact_hit = next(hit for hit in result_ar.hits if FACT_AR in hit.text)
        assert en_fact_hit.rank <= 2 and ar_fact_hit.rank <= 2
        assert en_fact_hit.chunk_id != ar_fact_hit.chunk_id
        assert FACT_AR not in en_fact_hit.text and FACT_EN not in ar_fact_hit.text

    asyncio.run(run())


def test_query_bounds(tmp_path):
    async def run():
        env, embedder, built = await _built(tmp_path)
        retriever = DenseRetriever(
            DenseConfig(), scope=env.scope, index_version=built, index=env.index, embedder=embedder
        )
        for query, code in ((" \n ", "empty_query"), ("x" * 2001, "query_too_long")):
            with pytest.raises(IndexingError) as exc:
                await retriever.retrieve(query)
            assert exc.value.code == code

    asyncio.run(run())


def test_no_retrieval_control_makes_no_calls():
    async def run():
        embedder = LexicalTestEmbedder()
        none = await NoRetrieval(NoRetrievalConfig()).retrieve(QUERY_EN)
        assert none.technique_id == "R01"
        assert none.grounding == "none" and not none.hits and none.index_version_id is None
        assert embedder.calls == []

    asyncio.run(run())


def test_incompatible_embedder_is_refused_before_any_call(tmp_path):
    async def run():
        env, _embedder, built = await _built(tmp_path)

        class SpyEmbedder:
            profile = fake_embedding_profile(dim=256).model_copy(update={"model": "different-model"})

            def __init__(self):
                self.calls = 0

            async def embed(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("incompatible embedder must not be called")

        spy = SpyEmbedder()
        assert spy.profile.provider == built.embedding_provider
        assert spy.profile.capabilities.embedding_dim == 256
        wrong = DenseRetriever(DenseConfig(), scope=env.scope, index_version=built, index=env.index, embedder=spy)
        with pytest.raises(IndexingError) as exc:
            await wrong.retrieve(QUERY_EN)
        assert exc.value.code == "incompatible_embedding" and spy.calls == 0

    asyncio.run(run())


def test_build_refusals_create_no_version(tmp_path):
    async def run():
        env = await _env(tmp_path)
        embedder = LexicalTestEmbedder()
        recorded = RecordingIndex(env.index)
        with pytest.raises(IndexingError) as exc:
            await env.service(recorded, max_views=2).build(
                env.scope, tuple(str(uuid.uuid4()) for _ in range(3)), env.chunker, embedder
            )
        assert exc.value.code == "too_many_views" and recorded.version is None
        with pytest.raises(NotFoundError):
            await env.service(recorded).build(make_scope(), (env.view.id,), env.chunker, embedder)
        assert recorded.version is None
        missing = LexicalTestEmbedder()
        missing.profile = SimpleNamespace(
            id="missing-dim", provider="test", model="missing-dim", capabilities=SimpleNamespace(embedding_dim=None)
        )
        with pytest.raises(IndexingError) as exc:
            await env.service(recorded).build(env.scope, (env.view.id,), env.chunker, missing)
        assert exc.value.code == "missing_embedding_dim" and recorded.version is None
        assert embedder.calls == [] and missing.calls == []

    asyncio.run(run())


def test_failed_builds_are_marked_failed_and_not_searchable(tmp_path):
    async def run():
        env = await _env(tmp_path)
        embedder = LexicalTestEmbedder()

        class FailsSecondBatch:
            profile = embedder.profile

            def __init__(self):
                self.calls = 0

            async def embed(self, texts, *, input_type="document"):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("batch failed")
                return await embedder.embed(texts, input_type=input_type)

        recorded = RecordingIndex(env.index)
        with pytest.raises(RuntimeError, match="batch failed"):
            await env.service(recorded, batch_size=2).build(env.scope, (env.view.id,), env.chunker, FailsSecondBatch())
        failed = await env.index.get_index_version(env.scope, recorded.version.id)
        assert failed.state is IndexState.FAILED
        with pytest.raises(IndexingError) as exc:
            await env.index.search(env.scope, failed.id, (1.0,) + (0.0,) * 255, 5)
        assert exc.value.code == "not_ready"

        class DropsVector:
            profile = embedder.profile

            async def embed(self, texts, *, input_type="document"):
                return SimpleNamespace(vectors=())

        mismatch = RecordingIndex(env.index)
        with pytest.raises(IndexingError) as exc:
            await env.service(mismatch).build(env.scope, (env.view.id,), env.chunker, DropsVector())
        assert exc.value.code == "embedding_count_mismatch"
        assert (await env.index.get_index_version(env.scope, mismatch.version.id)).state is IndexState.FAILED

    asyncio.run(run())


def test_registry_validates_retriever_configs():
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R01", {}) == NoRetrievalConfig()
    assert registry.validate_config("R02", {"k": 5}) == DenseConfig(k=5)
