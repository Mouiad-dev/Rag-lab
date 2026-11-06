from rag_foundry_answering.gate import apply_gate, default_min_score
from rag_foundry_techniques.kinds import RetrievedChunk


def hit(rank, score, text):
    return RetrievedChunk(
        chunk_id=f"c{rank}",
        extraction_view_id="v",
        start=0,
        end=1,
        block_ids=(),
        text=text,
        distance=1 - score,
        score=score,
        rank=rank,
    )


def test_gate_empty_below_equal_and_filtering():
    assert apply_gate([], min_score=0.5, max_evidence_chars=10).outcome == "no_evidence"
    assert apply_gate([hit(1, 0.49, "a")], min_score=0.5, max_evidence_chars=10).outcome == "low_score"
    result = apply_gate(
        [hit(1, 0.5, "a"), hit(2, 0.8, "b"), hit(3, 0.4, "c")],
        min_score=0.5,
        max_evidence_chars=10,
    )
    assert result.outcome == "pass"
    assert tuple(item.rank for item in result.used) == (1, 2)
    assert not result.truncated


def test_gate_character_cap_keeps_first_and_marks_truncated():
    result = apply_gate(
        [hit(1, 0.9, "abc"), hit(2, 0.8, "def"), hit(3, 0.7, "g")],
        min_score=0.5,
        max_evidence_chars=4,
    )
    assert tuple(item.rank for item in result.used) == (1,)
    assert result.truncated


def test_default_min_score():
    assert default_min_score("ollama", "bge-m3", 1024) == 0.45
    assert default_min_score("other", "model", 10) is None
