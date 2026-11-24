import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing.packing import ContextPackConfig, ContextPacker
from rag_foundry_techniques import RetrievalResult, RetrievedChunk


def _hits(count=5):
    rows = [
        ("h1", "Alpha one.", 40, 50),
        ("h2", "Beta two three.", 0, 15),
        ("h3", "Gamma 12345.", 20, 32),
        ("h4", "Delta.", 60, 66),
        ("h5", "Eps five six seven.", 80, 99),
    ]
    return tuple(
        RetrievedChunk(
            chunk_id=chunk_id,
            extraction_view_id="view",
            start=start,
            end=end,
            block_ids=(),
            text=text,
            score=1.0,
            rank=index,
        )
        for index, (chunk_id, text, start, end) in enumerate(rows[:count], 1)
    )


def _result(hits=None, grounding="corpus", version="v1"):
    return RetrievalResult(
        technique_id="R02", grounding=grounding, index_version_id=version, hits=_hits() if hits is None else hits
    )


def _pack(order="rank", hits=None, **kwargs):
    config = ContextPackConfig(
        order=order,
        prompt_overhead_tokens=kwargs.pop("overhead", 30),
        per_item_overhead_tokens=kwargs.pop("per_item", 2),
        **kwargs.pop("config", {}),
    )
    return ContextPacker(config).pack(
        _result(hits),
        model_context_window=kwargs.pop("window", 100),
        reserve_output_tokens=kwargs.pop("reserve", 50),
        fixed_text=kwargs.pop("fixed_text", ""),
    )


def test_rank_budget_fixed_text_and_losing_case():
    packed = _pack()
    assert [item.rank for item in packed.items] == [1, 2, 3]
    assert [item.cost_tokens for item in packed.trace.placed] == [5, 6, 6]
    assert [item.position for item in packed.trace.placed] == [1, 2, 3]
    assert packed.trace.used_tokens == 17
    assert [(item.rank, item.reason, item.cost_tokens) for item in packed.trace.dropped] == [
        (4, "budget", 4),
        (5, "budget", 7),
    ]
    fixed = _pack(fixed_text="Question: x?")
    assert fixed.trace.fixed_tokens == 4
    assert fixed.trace.budget_tokens == 16
    assert [item.rank for item in fixed.items] == [1, 2]
    assert fixed.trace.used_tokens == 11
    assert [(item.rank, item.reason) for item in fixed.trace.dropped] == [(3, "budget"), (4, "budget"), (5, "budget")]
    losing = _pack(hits=(_hits()[-1].model_copy(update={"rank": 1}),), window=10, reserve=5, overhead=0)
    assert not losing.items and losing.trace.used_tokens == 0
    assert [(item.rank, item.reason, item.cost_tokens) for item in losing.trace.dropped] == [(1, "budget", 7)]


def test_ordering_variants():
    args = {"window": 1000, "reserve": 10, "overhead": 0}
    for count, expected in [(5, [1, 3, 5, 4, 2]), (4, [1, 3, 4, 2]), (3, [1, 3, 2]), (2, [1, 2]), (1, [1])]:
        packed = _pack("lost_in_middle", _hits(count), **args)
        assert [item.rank for item in packed.items] == expected
        assert [item.rank for item in packed.trace.placed] == expected
        assert [item.position for item in packed.trace.placed] == list(range(1, count + 1))
    packed = _pack("source_order", **args)
    assert [item.rank for item in packed.items] == [2, 3, 1, 4, 5]


def test_contained_kept_by_prior_hit():
    base = _hits(2)
    contained = RetrievedChunk(
        chunk_id="d", extraction_view_id="view", start=46, end=50, block_ids=(), text="one.", score=1.0, rank=2
    )
    second = base[1].model_copy(update={"rank": 3})
    packed = _pack(hits=(base[0], contained, second))
    assert [item.rank for item in packed.items] == [1, 3]
    dropped = next(item for item in packed.trace.dropped if item.chunk_id == "d")
    assert (dropped.reason, dropped.kept_chunk_id) == ("contained", base[0].chunk_id)


def test_window_and_input_errors_and_determinism():
    with pytest.raises(IndexingError, match="packing_no_budget"):
        _pack(reserve=70)
    with pytest.raises(IndexingError, match="packing_no_context_window"):
        ContextPacker(ContextPackConfig(context_window_tokens=None)).pack(
            _result(), model_context_window=None, reserve_output_tokens=1
        )
    with pytest.raises(IndexingError, match="packing_window_exceeds_model"):
        ContextPacker(ContextPackConfig(context_window_tokens=300)).pack(
            _result(), model_context_window=256, reserve_output_tokens=1
        )
    assert (
        ContextPacker(ContextPackConfig(context_window_tokens=300, prompt_overhead_tokens=0))
        .pack(_result(), model_context_window=None, reserve_output_tokens=1)
        .trace.context_window
        == 300
    )
    hits = _hits()
    duplicate_rank = (hits[0], hits[1].model_copy(update={"rank": 1}))
    with pytest.raises(IndexingError, match="packing_invalid_input"):
        _pack(hits=duplicate_rank)
    with pytest.raises(IndexingError, match="packing_invalid_input"):
        ContextPacker(ContextPackConfig()).pack(
            _result(hits=(), grounding="none", version=None), model_context_window=1000, reserve_output_tokens=1
        )
    gap = (hits[0], hits[2].model_copy(update={"rank": 3}))
    first = _pack(hits=gap)
    second = _pack(hits=gap)
    assert [item.rank for item in first.items] == [1, 3]
    assert first.items == gap
    assert first.model_dump_json() == second.model_dump_json()


def test_more_input_errors():
    hits = _hits()
    with pytest.raises(IndexingError, match="packing_invalid_input"):
        _pack(hits=(hits[0], hits[1].model_copy(update={"chunk_id": "h1"})))
    many = tuple(hits[0].model_copy(update={"chunk_id": f"c{i}", "rank": i}) for i in range(1, 52))
    with pytest.raises(IndexingError, match="packing_invalid_input"):
        _pack(hits=many)
    with pytest.raises(IndexingError, match="packing_invalid_input"):
        ContextPacker(ContextPackConfig()).pack(
            _result(grounding="corpus", version=None), model_context_window=1000, reserve_output_tokens=1
        )


def test_contained_after_budget_stop_and_trace_fields():
    hits = _hits()
    inner = hits[0].model_copy(update={"chunk_id": "d", "text": "one.", "start": 46, "rank": 5})
    packed = _pack(hits=(hits[0], hits[1], hits[2], hits[4].model_copy(update={"rank": 4}), inner))
    assert [item.rank for item in packed.items] == [1, 2, 3]
    assert [(d.rank, d.reason, d.kept_chunk_id) for d in packed.trace.dropped] == [
        (4, "budget", None),
        (5, "contained", "h1"),
    ]
    assert packed.trace.tokenizer == "lab-regex-v1"
    kept = _pack(hits=(hits[0], inner.model_copy(update={"rank": 2})), config={"drop_contained": False})
    assert [item.rank for item in kept.items] == [1, 2]
