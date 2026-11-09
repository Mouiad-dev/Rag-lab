import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing.rerank import RerankTrace, apply_order, check_rerank_input, rank_changes
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.usage import Operation, TokenSource, Usage
from rag_foundry_techniques import RetrievalResult, RetrievedChunk


def _hit(chunk_id, rank, score, *, start=0):
    return RetrievedChunk(
        chunk_id=chunk_id,
        extraction_view_id="view",
        start=start,
        end=start + 5,
        block_ids=("block",),
        text=chunk_id,
        distance=0.1,
        score=score,
        rank=rank,
        parent_start=0,
        parent_end=10,
    )


def _result(hits):
    return RetrievalResult(
        technique_id="R02", grounding="corpus", index_version_id="version", hits=tuple(hits), scoring="score"
    )


def test_apply_order_drops_and_reorders_without_changing_provenance():
    source = _result((_hit("a", 1, 3.0), _hit("b", 2, 2.0), _hit("c", 3, 1.0)))
    result, changes, dropped = apply_order(source, [(2, 0.9, ()), (0, 0.5, ())], technique_id="R19", scoring="rerank")
    assert [hit.chunk_id for hit in result.hits] == ["c", "a"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert [hit.score for hit in result.hits] == [0.9, 0.5]
    assert dropped == ("b",)
    assert [(row.chunk_id, row.before_rank, row.after_rank, row.moved) for row in changes] == [
        ("a", 1, 2, -1),
        ("b", 2, None, None),
        ("c", 3, 1, 2),
    ]
    assert [(c.value, c.source) for c in (result.hits[0].score_components[0], result.hits[1].score_components[0])] == [
        (3.0, "R02"),
        (1.0, "R02"),
    ]
    for output, original in zip(result.hits, (source.hits[2], source.hits[0]), strict=True):
        assert (
            output.chunk_id,
            output.extraction_view_id,
            output.start,
            output.end,
            output.block_ids,
            output.text,
            output.distance,
            output.parent_start,
            output.parent_end,
        ) == (
            original.chunk_id,
            original.extraction_view_id,
            original.start,
            original.end,
            original.block_ids,
            original.text,
            original.distance,
            original.parent_start,
            original.parent_end,
        )
    assert result.grounding == source.grounding
    assert result.index_version_id == source.index_version_id


@pytest.mark.parametrize("order", [[(3, 1.0, ())], [(0, 1.0, ()), (0, 0.5, ())]])
def test_invalid_order_raises(order):
    with pytest.raises(IndexingError, match="rerank_invalid_order"):
        apply_order(_result((_hit("a", 1, 3.0),)), order, technique_id="R19", scoring=None)


def test_rank_changes_rejects_added_or_modified_candidate():
    before = _result((_hit("a", 1, 1.0),))
    with pytest.raises(IndexingError, match="rerank_added_candidate"):
        rank_changes(before, _result((_hit("a", 1, 1.0), _hit("b", 2, 0.5))), allow_added=False)
    with pytest.raises(IndexingError, match="rerank_added_candidate"):
        rank_changes(before, _result((_hit("a", 1, 1.0, start=1),)), allow_added=False)


def test_check_rerank_input_maps_bad_input_and_enforces_candidate_limit():
    with pytest.raises(IndexingError, match="rerank_invalid_input"):
        check_rerank_input(_result(()).model_copy(update={"grounding": "query"}), 3)
    source = _result((_hit("a", 1, 1.0), _hit("b", 2, 0.5)))
    with pytest.raises(IndexingError, match="rerank_too_many_candidates"):
        check_rerank_input(source, 1)


def test_rerank_trace_usage_json_round_trip():
    profile = fake_chat_profile("rerank-test")
    usage = Usage(
        profile_id=profile.id,
        provider=profile.provider,
        model=profile.model,
        operation=Operation.CHAT,
        input_tokens=1,
        output_tokens=1,
        token_source=TokenSource.ESTIMATED,
        latency_ms=1,
        estimated_cost_usd=0,
    )
    trace = RerankTrace(
        technique_id="R19",
        input_technique_id="R02",
        status="applied",
        reason=None,
        profile_id=profile.id,
        model=profile.model,
        input_count=1,
        output_count=1,
        dropped=(),
        usage=(usage,),
    )
    loaded = RerankTrace.model_validate_json(trace.model_dump_json())
    assert loaded == trace
    assert isinstance(loaded.usage[0], Usage)
