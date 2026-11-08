import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.index_testing import make_new_index_version
from rag_foundry_indexing import (
    DenseConfig,
    DenseRetriever,
    MetadataFilter,
    MetadataFilterConfig,
    MetadataFilteredRetriever,
    intersect,
    register_builtin_plugins,
)
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._multi_doc_index import multi_doc_index


@pytest.fixture
def corpus():
    embedder = LexicalTestEmbedder()
    data = asyncio.run(multi_doc_index(embedder))
    inner = DenseRetriever(
        DenseConfig(k=50), scope=data.owner, index_version=data.owner_version, index=data.index, embedder=embedder
    )
    wrapper = MetadataFilteredRetriever(
        MetadataFilterConfig(k=50),
        scope=data.owner,
        index_version=data.owner_version,
        index=data.index,
        extractions=data.extractions,
        documents=data.documents,
        inner=inner,
    )
    return data, embedder, inner, wrapper


async def views_of(data, retriever, *tags):
    by_document = {choice.document_id: choice.view_id for choice in await retriever.choices()}
    return {by_document[data.document_ids[tag]] for tag in tags}


def test_document_media_and_date_filters(corpus):
    async def check():
        data, _, _, retriever = corpus
        inner_top = await retriever._inner.retrieve("quarterly report")
        base = await retriever.retrieve("quarterly report")
        assert len(base.hits) == 6
        assert [hit.chunk_id for hit in base.hits] == [hit.chunk_id for hit in inner_top.hits]
        by_doc = await MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(document_ids=(data.document_ids["report-2025"],))),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        ).retrieve("quarterly report")
        assert len(by_doc.hits) == 2
        assert {hit.extraction_view_id for hit in by_doc.hits} == await views_of(data, retriever, "report-2025")
        markdown = MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(media_types=(MediaType.TEXT_MARKDOWN,))),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        markdown_hits = (await markdown.retrieve("quarterly report")).hits
        assert len(markdown_hits) == 2
        assert {hit.extraction_view_id for hit in markdown_hits} == await views_of(data, retriever, "report-2024")
        recent = MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(created_from=datetime(2026, 2, 1, tzinfo=UTC))),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        earliest = (await retriever.choices())[0].view_id
        recent_result = await recent.retrieve("quarterly report")
        assert recent_result.hits
        assert all(hit.extraction_view_id != earliest for hit in recent_result.hits)
        assert {hit.extraction_view_id for hit in recent_result.hits} == await views_of(
            data, retriever, "report-2025", "notes"
        )

    asyncio.run(check())


def test_foreign_document_filter_short_circuits_inner(corpus):
    async def check():
        data, embedder, _, retriever = corpus
        before = len(embedder.calls)
        filtered = MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(document_ids=(data.document_ids["stranger"],))),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        assert (await filtered.retrieve("quarterly report")).hits == ()
        assert len(embedder.calls) == before

    asyncio.run(check())


def test_mixed_scope_filters_only_return_owner_subset(corpus):
    async def check():
        data, _, _, retriever = corpus
        unfiltered = await retriever.retrieve("quarterly report")
        filtered = MetadataFilteredRetriever(
            MetadataFilterConfig(
                filter=MetadataFilter(document_ids=(data.document_ids["report-2025"], data.document_ids["stranger"]))
            ),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        mixed_hits = (await filtered.retrieve("quarterly report")).hits
        ids = [h.chunk_id for h in mixed_hits]
        assert len(ids) == 2
        assert {h.extraction_view_id for h in mixed_hits} == await views_of(data, retriever, "report-2025")
        assert set(ids) <= {h.chunk_id for h in unfiltered.hits}

    asyncio.run(check())


def test_filter_rejects_authority_fields():
    for key in ("tenant_id", "project_id", "index_version_id"):
        with pytest.raises(ValidationError):
            MetadataFilter.model_validate({key: "7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f"})
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    with pytest.raises(ValidationError):
        registry.validate_config("R11", {"filter": {"tenant_id": "7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f"}})
    assert registry.validate_config("R11", {}) == MetadataFilterConfig()


def test_scope_not_found_and_intersections_are_subsets(corpus):
    async def check():
        data, _, _, retriever = corpus
        foreign = MetadataFilteredRetriever(
            MetadataFilterConfig(),
            scope=data.stranger,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        with pytest.raises(NotFoundError) as exc:
            await foreign.retrieve("quarterly report")
        unknown = MetadataFilteredRetriever(
            MetadataFilterConfig(),
            scope=data.owner,
            index_version=data.owner_version.model_copy(update={"id": "00000000-0000-4000-8000-000000000000"}),
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        with pytest.raises(NotFoundError) as unknown_exc:
            await unknown.retrieve("quarterly report")
        assert str(exc.value) == str(unknown_exc.value)  # same answer as "does not exist"
        own = MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(document_ids=(data.document_ids["report-2024"],))),
            scope=data.stranger,
            index_version=data.stranger_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=retriever._inner,
        )
        assert (await own.retrieve("quarterly report")).hits == ()
        filters = [
            MetadataFilter(),
            MetadataFilter(document_ids=(data.document_ids["report-2024"],)),
            MetadataFilter(document_ids=(data.document_ids["report-2025"], data.document_ids["stranger"])),
            MetadataFilter(media_types=(MediaType.TEXT_MARKDOWN,)),
            MetadataFilter(created_from=datetime(2026, 2, 1, tzinfo=UTC)),
            MetadataFilter(document_ids=()),  # matches nothing
        ]
        for f in filters:
            for g in filters:
                combined = MetadataFilteredRetriever(
                    MetadataFilterConfig(k=50, filter=intersect(f, g)),
                    scope=data.owner,
                    index_version=data.owner_version,
                    index=data.index,
                    extractions=data.extractions,
                    documents=data.documents,
                    inner=retriever._inner,
                )
                ids = {h.chunk_id for h in (await combined.retrieve("quarterly report")).hits}
                for item in (f, g):
                    separate = MetadataFilteredRetriever(
                        MetadataFilterConfig(k=50, filter=item),
                        scope=data.owner,
                        index_version=data.owner_version,
                        index=data.index,
                        extractions=data.extractions,
                        documents=data.documents,
                        inner=retriever._inner,
                    )
                    assert ids <= {h.chunk_id for h in (await separate.retrieve("quarterly report")).hits}

    asyncio.run(check())


def test_extra_filter_version_and_ready_errors(corpus):
    async def check():
        data, _, inner, retriever = corpus
        a = MetadataFilteredRetriever(
            MetadataFilterConfig(k=50, filter=MetadataFilter(document_ids=(data.document_ids["report-2024"],))),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=inner,
        )
        assert (
            await a.retrieve(
                "quarterly report", extra_filter=MetadataFilter(document_ids=(data.document_ids["report-2025"],))
            )
        ).hits == ()
        assert await a.retrieve("quarterly report", extra_filter=None) == await a.retrieve("quarterly report")

        class WrongVersion:
            async def retrieve(self, query):
                result = await inner.retrieve(query)
                return result.model_copy(update={"index_version_id": data.stranger_version.id})

        mismatched = MetadataFilteredRetriever(
            MetadataFilterConfig(),
            scope=data.owner,
            index_version=data.owner_version,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=WrongVersion(),
        )
        with pytest.raises(IndexingError) as exc:
            await mismatched.retrieve("quarterly report")
        assert exc.value.code == "selection_version_mismatch"
        building = await data.index.create_index_version(
            data.owner, make_new_index_version(data.owner_version.extraction_view_ids, dim=256)
        )
        not_ready = MetadataFilteredRetriever(
            MetadataFilterConfig(),
            scope=data.owner,
            index_version=building,
            index=data.index,
            extractions=data.extractions,
            documents=data.documents,
            inner=inner,
        )
        with pytest.raises(IndexingError) as exc:
            await not_ready.retrieve("quarterly report")
        assert exc.value.code == "not_ready"

    asyncio.run(check())


@pytest.mark.parametrize("field", ["document_ids", "document_version_ids"])
def test_id_lists_are_capped_at_100(field):
    ids = tuple(f"7f3c2b1e-5d4a-4c8b-9e6f-{i:012d}" for i in range(101))
    with pytest.raises(ValidationError):
        MetadataFilter.model_validate({field: ids})
    assert len(getattr(MetadataFilter.model_validate({field: ids[:100]}), field)) == 100
