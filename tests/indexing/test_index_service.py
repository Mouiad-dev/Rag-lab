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
from rag_foundry_chunking import ParentChildChunker, ParentChildConfig, RecursiveChunker, RecursiveConfig
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.indexing import IndexState, corpus_sha256_for, embedding_key_for
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
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
        factory = InMemoryUnitOfWorkFactory(extractions=self.views, indexes=index or self.index)
        return IndexService(self.store, factory, self.anchors, **kwargs)


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
        anchors=AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test"),
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
        assert all(row.parent_start_cp is None and row.parent_end_cp is None for row in chunks)
        inspected = await env.service().inspect_chunks(env.scope, built.id)
        assert all(row.parent_anchors == () for row in inspected)
        assert built.chunker_id == "C05"
        assert built.chunker_version
        assert built.chunker_config_sha256 == hashlib.sha256(built.chunker_config_json.encode("utf-8")).hexdigest()
        assert built.corpus_sha256 == corpus_sha256_for([(env.view.id, env.view.payload_sha256)])
        assert built.embedding_model == "lexical-test-v1"
        assert built.embedding_key == embedding_key_for(embedder.profile.provider, "lexical-test-v1", 256)
        museum_chunks = [row for row in chunks if "maritime museum" in row.text]
        assert museum_chunks and all(row.split_level >= 2 for row in museum_chunks)

    asyncio.run(run())


def test_build_passes_extraction_block_structure_to_chunker(tmp_path):
    class RecordingChunker:
        manifest = RecursiveChunker.manifest

        def __init__(self, inner):
            self.inner = inner
            self.config = inner.config
            self.seen = None

        def chunk(self, doc):
            self.seen = doc
            return self.inner.chunk(doc)

    async def run():
        env = await _env(tmp_path)
        wrapper = RecordingChunker(env.chunker)
        await env.service().build(env.scope, (env.view.id,), wrapper, LexicalTestEmbedder())
        payload = await env.extraction.load_payload(env.scope, env.view)
        assert [(b.kind, b.section_path, b.heading_level) for b in wrapper.seen.blocks] == [
            (b.kind.value, b.section_path, b.heading_level) for b in payload.blocks
        ]

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


def test_c18_build_stores_parent_spans_and_parent_anchors(tmp_path):
    async def run():
        env = await _env(tmp_path)
        chunker = ParentChildChunker(ParentChildConfig(parent_size=500, child_size=100))
        built = await env.service().build(env.scope, (env.view.id,), chunker, LexicalTestEmbedder())
        rows = await env.index.list_chunks(env.scope, built.id)
        assert rows and all(row.parent_start_cp is not None and row.parent_end_cp is not None for row in rows)
        inspected = await env.service().inspect_chunks(env.scope, built.id)
        assert all(row.parent_anchors for row in inspected)
        resolutions = await env.anchors.resolve(
            env.scope, tuple(anchor for row in inspected for anchor in row.parent_anchors)
        )
        assert resolutions and all(item.status.value == "exact" for item in resolutions)
        payload = await env.extraction.load_payload(env.scope, env.view)
        block_start = {b.block_id: b.start for b in payload.blocks}
        for item in inspected:  # ChunkInspection is a ChunkRecord plus anchors
            row = item
            starts = [block_start[a.block_id] + a.text_range.start for a in item.parent_anchors]
            ends = [block_start[a.block_id] + a.text_range.end for a in item.parent_anchors]
            # The parent anchors cover the stored parent span exactly, not the child span.
            assert (min(starts), max(ends)) == (row.parent_start_cp, row.parent_end_cp)
        assert any(
            (row.parent_start_cp, row.parent_end_cp) != (row.start, row.end)
            and row.parent_start_cp <= row.start
            and row.end <= row.parent_end_cp
            for row in rows
        )

    asyncio.run(run())


def test_parent_span_past_the_text_fails_before_any_index_version(tmp_path):
    class OverreachingChunker(ParentChildChunker):
        def chunk_with_parents(self, doc):
            spans = super().chunk_with_parents(doc)
            last = spans[-1]
            return (*spans[:-1], last.model_copy(update={"parent_end": len(doc.text) + 1}))

    async def run():
        env = await _env(tmp_path)
        recording = RecordingIndex(env.index)
        chunker = OverreachingChunker(ParentChildConfig(parent_size=500, child_size=100))
        with pytest.raises(IndexingError) as caught:
            await env.service(index=recording).build(env.scope, (env.view.id,), chunker, LexicalTestEmbedder())
        assert caught.value.code == "parent_span_out_of_bounds"
        assert recording.version is None

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


def test_index_build_has_no_uow_during_embed_or_payload_reads(tmp_path, monkeypatch):
    async def run():
        env = await _env(tmp_path)
        factory = InMemoryUnitOfWorkFactory(extractions=env.views, indexes=env.index)
        service = IndexService(env.store, factory, env.anchors, batch_size=2)
        from rag_foundry_indexing import service as service_module

        original = service_module.read_payload

        async def checked_read_payload(*args, **kwargs):
            assert factory.open_now == 0
            return await original(*args, **kwargs)

        monkeypatch.setattr(service_module, "read_payload", checked_read_payload)

        class CheckedEmbedder:
            profile = LexicalTestEmbedder().profile

            async def embed(self, texts, *, input_type="document"):
                assert factory.open_now == 0
                return await LexicalTestEmbedder().embed(texts, input_type=input_type)

        await service.build(env.scope, (env.view.id,), env.chunker, CheckedEmbedder())

    asyncio.run(run())


def test_second_batch_embed_failure_commits_only_create_and_first_batch(tmp_path):
    async def run():
        env = await _env(tmp_path)
        embedder = LexicalTestEmbedder()

        class FailSecond:
            profile = embedder.profile

            def __init__(self):
                self.calls = 0

            async def embed(self, texts, *, input_type="document"):
                self.calls += 1
                assert factory.open_now == 0
                if self.calls == 2:
                    raise RuntimeError("batch failed")
                return await embedder.embed(texts, input_type=input_type)

        recorded = RecordingIndex(env.index)
        factory = InMemoryUnitOfWorkFactory(extractions=env.views, indexes=recorded)
        service = IndexService(
            env.store,
            factory,
            env.anchors,
            batch_size=2,
        )
        with pytest.raises(RuntimeError, match="batch failed"):
            await service.build(env.scope, (env.view.id,), env.chunker, FailSecond())
        assert factory.committed == 3
        assert factory.rolled_back == 0
        assert factory.opened == 4
        assert recorded.version is not None
        assert (await env.index.get_index_version(env.scope, recorded.version.id)).state is IndexState.FAILED
        assert [row.ordinal for row in await env.index.list_chunks(env.scope, recorded.version.id)] == [0, 1]

    asyncio.run(run())


def test_second_batch_add_chunks_failure_exits_before_mark_failed(tmp_path):
    async def run():
        env = await _env(tmp_path)

        class FailSecondAdd:
            def __init__(self):
                self.calls = 0
                self.version = None
                self.failed_after_rollback = False

            def __getattr__(self, name):
                method = getattr(env.index, name)
                if name == "create_index_version":

                    async def create(*args, **kwargs):
                        self.version = await method(*args, **kwargs)
                        return self.version

                    return create
                if name == "add_chunks":

                    async def add(*args, **kwargs):
                        self.calls += 1
                        if self.calls == 2:
                            raise RuntimeError("write failed")
                        return await method(*args, **kwargs)

                    return add
                if name == "mark_failed":

                    async def mark(*args, **kwargs):
                        self.failed_after_rollback = factory.rolled_back == 1 and factory.open_now == 1
                        return await method(*args, **kwargs)

                    return mark
                return method

        indexes = FailSecondAdd()
        factory = InMemoryUnitOfWorkFactory(extractions=env.views, indexes=indexes)
        service = IndexService(env.store, factory, env.anchors, batch_size=2)
        with pytest.raises(RuntimeError, match="write failed"):
            await service.build(env.scope, (env.view.id,), env.chunker, LexicalTestEmbedder())
        assert indexes.failed_after_rollback
        assert factory.rolled_back == 1
        assert (await env.index.get_index_version(env.scope, indexes.version.id)).state is IndexState.FAILED
        assert [row.ordinal for row in await env.index.list_chunks(env.scope, indexes.version.id)] == [0, 1]

    asyncio.run(run())
