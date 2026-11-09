import asyncio
import json

import pytest
from _inmemory_index import ready_index
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_indexing.inspection import RetrievalInspection, inspect_retrieval
from rag_foundry_indexing.rerank import apply_order, rank_changes
from rag_foundry_techniques import RetrievalResult, RetrievedChunk


def _result(version="v1"):
    hits = tuple(
        RetrievedChunk(
            chunk_id=chunk_id,
            extraction_view_id="view",
            start=i * 5,
            end=i * 5 + 5,
            block_ids=("block",),
            text=chunk_id,
            score=score,
            rank=i + 1,
        )
        for i, (chunk_id, score) in enumerate((("a", 3.0), ("b", 2.0)))
    )
    return RetrievalResult(technique_id="R02", grounding="corpus", index_version_id=version, hits=hits)


def test_inspection_json_round_trip_and_rank_changes():
    async def run():
        owner, _, index, version = await ready_index([(0, 1, "a"), (1, 2, "b")], LexicalTestEmbedder())
        source = _result(version.id)
        reordered, _, _ = apply_order(source, [(1, 0.9, ()), (0, 0.5, ())], technique_id="R19", scoring="rerank")
        inspection = await inspect_retrieval(owner, index, "question", [("retrieve", source), ("rerank", reordered)])
        assert RetrievalInspection.model_validate(json.loads(inspection.model_dump_json())) == inspection
        assert inspection.stages[0].changes == ()
        assert inspection.stages[1].changes == rank_changes(source, reordered, allow_added=False)

    asyncio.run(run())


def test_inspection_rechecks_scope_and_rejects_forged_rerank_and_mixed_version():
    async def run():
        owner, stranger, index, version = await ready_index([(0, 1, "a"), (1, 2, "b")], LexicalTestEmbedder())
        source = _result(version.id)
        with pytest.raises(NotFoundError):
            await inspect_retrieval(stranger, index, "question", [("retrieve", source)])
        assert await inspect_retrieval(owner, index, "question", [("retrieve", source)])
        extra = RetrievedChunk(
            chunk_id="extra", extraction_view_id="view", start=10, end=15, block_ids=(), text="extra", score=1.0, rank=3
        )
        forged = source.model_copy(update={"hits": (*source.hits, extra)})
        with pytest.raises(IndexingError, match="rerank_added_candidate"):
            await inspect_retrieval(owner, index, "question", [("retrieve", source), ("rerank", forged)])
        with pytest.raises(IndexingError, match="inspection_version_mismatch"):
            await inspect_retrieval(owner, index, "question", [("retrieve", source), ("fuse", _result("v2"))])

    asyncio.run(run())


def test_non_rerank_stage_can_add_candidate_and_inspection_is_frozen():
    async def run():
        owner, _, index, version = await ready_index([(0, 1, "a")], LexicalTestEmbedder())
        source = _result(version.id).model_copy(update={"hits": _result(version.id).hits[:1]})
        extra = source.hits[0].model_copy(update={"chunk_id": "c", "rank": 2})
        added = source.model_copy(update={"technique_id": "F01", "hits": (*source.hits, extra)})
        inspection = await inspect_retrieval(owner, index, "question", [("retrieve", source), ("fuse", added)])
        assert inspection.stages[1].changes[-1].chunk_id == "c"
        with pytest.raises((TypeError, ValueError)):
            inspection.question = "changed"

    asyncio.run(run())
