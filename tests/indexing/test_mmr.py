import asyncio

import pytest
from pydantic import ValidationError
from rag_foundry_domain import NotFoundError
from rag_foundry_domain.ports import ChunkVectorReader
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing import DenseConfig, DenseRetriever, MmrConfig, MmrSelector
from rag_foundry_techniques import RetrievalResult, RetrievedChunk, ScoreComponent

from tests.indexing._inmemory_index import ready_index
from tests.indexing._lexical_embedder import LexicalTestEmbedder


def _result(version_id, texts, scores):
    hits = tuple(
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            extraction_view_id="view",
            start=0,
            end=len(text),
            block_ids=("b00000",),
            text=text,
            distance=1.0 - score,
            score=score,
            rank=i + 1,
            score_components=(ScoreComponent(name="score", value=score),),
        )
        for i, (text, score) in enumerate(zip(texts, scores, strict=True))
    )
    return RetrievalResult(
        technique_id="T01", grounding="corpus", index_version_id=version_id, hits=hits, scoring="test"
    )


async def _lexical_hand_case():
    owner, _, index, version = await ready_index([(0, 7, "fixture")], LexicalTestEmbedder())
    texts = ["tidal observatory sea levels", "tidal observatory sea levels", "chess club tuesday", "sea levels tuesday"]
    scores = [1.0, 0.9, 0.5, 0.0]
    source = _result(version.id, texts, scores)
    selected = await MmrSelector(
        MmrConfig(k=2, similarity="lexical"), scope=owner, index_version=version, index=index
    ).select(source)
    assert [hit.chunk_id for hit in selected.hits] == ["chunk-0", "chunk-2"]
    values = [{component.name: component.value for component in hit.score_components} for hit in selected.hits]
    assert values[0]["mmr_score"] == 0.5
    assert values[1]["mmr_score"] == 0.25
    ordered = await MmrSelector(
        MmrConfig(k=4, lambda_mult=1.0, similarity="lexical"), scope=owner, index_version=version, index=index
    ).select(source)
    assert [hit.chunk_id for hit in ordered.hits] == [hit.chunk_id for hit in source.hits]
    tied = await MmrSelector(
        MmrConfig(k=1, lambda_mult=0.0, similarity="lexical"), scope=owner, index_version=version, index=index
    ).select(source)
    assert tied.hits[0].chunk_id == "chunk-0"


async def _embedding_mode_and_negative_cases():
    embedder = LexicalTestEmbedder()
    owner, stranger, index, version = await ready_index(
        [(0, 17, "tidal observatory"), (18, 28, "sea levels")], embedder
    )
    dense = await DenseRetriever(
        DenseConfig(k=4), scope=owner, index_version=version, index=index, embedder=embedder
    ).retrieve("tidal sea")
    stored = await index.chunk_embeddings(owner, version.id, tuple(hit.chunk_id for hit in dense.hits))
    rows = {row.id: row for row in await index.list_chunks(owner, version.id)}
    expected = await embedder.embed([rows[hit.chunk_id].text for hit in dense.hits], input_type="document")
    assert all(
        all(abs(actual - target) <= 1e-6 for actual, target in zip(stored[hit.chunk_id], vector, strict=True))
        for hit, vector in zip(dense.hits, expected.vectors, strict=True)
    )
    assert isinstance(index, ChunkVectorReader)
    selected = await MmrSelector(MmrConfig(k=4), scope=owner, index_version=version, index=index, vectors=index).select(
        dense
    )
    assert set(hit.chunk_id for hit in selected.hits) <= set(hit.chunk_id for hit in dense.hits)
    assert all(
        {"mmr_relevance", "mmr_max_similarity", "mmr_score"} <= {c.name for c in hit.score_components}
        for hit in selected.hits
    )
    with pytest.raises(NotFoundError):
        await MmrSelector(MmrConfig(), scope=stranger, index_version=version, index=index, vectors=index).select(dense)
    other, _, other_index, other_version = await ready_index([(0, 7, "foreign")], LexicalTestEmbedder())
    foreign_chunk = (await other_index.list_chunks(other, other_version.id))[0]
    forged_hit = dense.hits[0].model_copy(update={"chunk_id": foreign_chunk.id})
    forged = dense.model_copy(update={"hits": (forged_hit,)})
    with pytest.raises(Exception) as missing:
        await MmrSelector(MmrConfig(), scope=owner, index_version=version, index=index, vectors=index).select(forged)
    assert getattr(missing.value, "code", None) == "selection_missing_vector"
    with pytest.raises(Exception) as mismatch:
        await MmrSelector(
            MmrConfig(similarity="lexical"), scope=owner, index_version=other_version, index=index
        ).select(dense)
    assert getattr(mismatch.value, "code", None) == "selection_version_mismatch"
    with pytest.raises(Exception) as unavailable:
        MmrSelector(MmrConfig(), scope=owner, index_version=version, index=index)
    assert getattr(unavailable.value, "code", None) == "selection_vectors_unavailable"


def test_registry_config():
    from pathlib import Path

    from rag_foundry_indexing import register_builtin_plugins
    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R09", {}) == MmrConfig()
    with pytest.raises(ValidationError):
        registry.validate_config("R09", {"lambda_mult": 1.5})


def test_chunk_embeddings_is_a_read_through_the_unit_of_work():
    from rag_foundry_domain.uow_adapters import per_call_indexes
    from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory

    inner = InMemoryUnitOfWorkFactory()
    modes = []

    def factory(*, read_only):
        modes.append(read_only)
        return inner(read_only=read_only)

    with pytest.raises(NotFoundError):
        asyncio.run(per_call_indexes(factory).chunk_embeddings(make_scope(), "missing-version", ("a",)))
    assert modes == [True]


def test_lexical_hand_case():
    asyncio.run(_lexical_hand_case())


def test_embedding_mode_and_negative_cases():
    asyncio.run(_embedding_mode_and_negative_cases())
