import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import ChunkRecord, FtsScoredChunk, IndexState, IndexVersion
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing.fts import FtsConfig, FtsRetriever


def test_fts_config_and_registry():
    assert FtsConfig().k == 5
    with pytest.raises(ValidationError):
        FtsConfig(k=51)
    assert FtsConfig.model_validate({}) == FtsConfig()
    from rag_foundry_indexing import register_builtin_plugins
    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R04", {}) == FtsConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("R04", {"unknown": True})


def _chunk():
    return ChunkRecord(
        id="11111111-1111-4111-8111-111111111111",
        tenant_id="22222222-2222-4222-8222-222222222222",
        project_id="33333333-3333-4333-8333-333333333333",
        index_version_id="44444444-4444-4444-8444-444444444444",
        extraction_view_id="55555555-5555-4555-8555-555555555555",
        ordinal=0,
        start=0,
        end=4,
        split_level=0,
        block_ids=("b1",),
        text="fact",
        text_sha256="a" * 64,
        parent_start_cp=0,
        parent_end_cp=4,
    )


def _version():
    return IndexVersion.model_construct(id="44444444-4444-4444-8444-444444444444", state=IndexState.READY)


class Stub:
    def __init__(self):
        self.calls = []

    async def fts_search(self, scope, index_version_id, query, k, *, operator):
        self.calls.append((scope, index_version_id, query, k, operator))
        return [FtsScoredChunk(chunk=_chunk(), ts_rank=0.25, rank=1)]


def test_fts_maps_sql_scores_and_parent_spans():
    async def run():
        index = Stub()
        retriever = FtsRetriever(FtsConfig(), scope=make_scope(), index_version=_version(), index=index)
        result = await retriever.retrieve("fact")
        assert result.technique_id == "R04"
        assert result.scoring == "pg-fts-simple-ts_rank_cd"
        hit = result.hits[0]
        assert hit.score == 0.25 and hit.distance is None
        assert hit.score_components[0].name == "ts_rank_cd"
        assert (hit.parent_start, hit.parent_end) == (0, 4)
        assert index.calls[0][2:] == ("fact", 5, "and")

    asyncio.run(run())


def test_empty_query_rejected_without_index_call():
    async def run():
        index = Stub()
        retriever = FtsRetriever(FtsConfig(), scope=make_scope(), index_version=_version(), index=index)
        with pytest.raises(IndexingError, match="empty_query"):
            await retriever.retrieve("  ")
        assert not index.calls

    asyncio.run(run())


def test_in_memory_index_has_no_fake_fts():
    from rag_foundry_domain.index_testing import InMemoryIndexRepository

    repo = InMemoryIndexRepository(views=None)
    with pytest.raises(IndexingError) as caught:
        asyncio.run(repo.fts_search(make_scope(), "v", "fact", 5, operator="and"))
    assert caught.value.code == "fts_unavailable"
