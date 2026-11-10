import asyncio
import hashlib
import json

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import (
    CompressingRetriever,
    ExtractiveCompressionConfig,
    ExtractiveCompressor,
    evidence_constraints,
    is_qualifier,
)
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

H = "The bridge opened on 3 March 2019. Tolls are 5 USD per car. This does not apply to bicycles. The cafe sells tea."
TA = (
    "\u0627\u0641\u062a\u062a\u062d \u0627\u0644\u062c\u0633\u0631 \u0639\u0627\u0645 "
    "\u0662\u0660\u0661\u0669. \u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631 "
    "\u0665 \u062f\u0648\u0644\u0627\u0631. \u0644\u0627 \u062a\u0646\u0637\u0628\u0642 "
    "\u0627\u0644\u0631\u0633\u0648\u0645 \u0639\u0644\u0649 \u0627\u0644\u062f\u0631\u0627\u062c\u0627\u062a. "
    "\u0627\u0644\u0645\u0642\u0647\u0649 \u064a\u0628\u064a\u0639 \u0627\u0644\u0634\u0627\u064a."
)
QTA = "\u0645\u0627 \u0631\u0633\u0648\u0645 \u0627\u0644\u062c\u0633\u0631\u061f"
Q = "What are the bridge tolls?"


def hit(text, start, rank, chunk_id=None, tid="R02"):
    return RetrievedChunk(
        chunk_id=chunk_id or f"c{rank}",
        extraction_view_id="v1",
        start=start,
        end=start + len(text),
        block_ids=("b1",),
        text=text,
        score=1.0 / rank,
        rank=rank,
    )


def result(*hits, tid="R02"):
    return RetrievalResult(technique_id=tid, grounding="corpus", index_version_id="iv1", hits=tuple(hits))


def compress(question, res, **config):
    return ExtractiveCompressor(ExtractiveCompressionConfig(**config)).compress(question, res)


def spans(out):
    return [(h.start, h.end) for h in out.result.hits]


def test_evidence_constraints():
    assert len(H) == 112
    assert evidence_constraints(H[0:34]) == ("3", "2019", "march")
    assert evidence_constraints(H[35:59]) == ("5", "5 usd")
    assert evidence_constraints(H[60:92]) == ("not",)
    assert evidence_constraints(H[93:112]) == ()
    assert evidence_constraints("Due 2024-03-01, not later.") == ("2024-03-01", "not")
    assert evidence_constraints(
        "\u0627\u0641\u062a\u062a\u062d \u0627\u0644\u062c\u0633\u0631 \u0641\u064a \u0645\u0627\u0631\u0633 "
        "\u0662\u0660\u0661\u0669."
    ) == (
        "2019",
        "\u0645\u0627\u0631\u0633",
    )
    assert evidence_constraints("\u0623\u0628\u0631\u064a\u0644") == ("\u0623\u0628\u0631\u064a\u0644",)
    assert evidence_constraints(TA) == ("2019", "5", "\u0644\u0627", "5 \u062f\u0648\u0644\u0627\u0631")


def test_qualifier_kept_and_dropped_sentence():
    out = compress(Q, result(hit(H, 100, 1)))
    assert out.trace.question_terms == ("bridge", "tolls")
    assert spans(out) == [(100, 192)]
    only = out.result.hits[0]
    assert only.text == H[0:92]
    assert only.chunk_id.startswith("cmp_") and len(only.chunk_id) == 28
    expected = (
        "cmp_" + hashlib.sha256(json.dumps(["R26", "c1", 100, 192], separators=(",", ":")).encode()).hexdigest()[:24]
    )
    assert only.chunk_id == expected
    reasons = {(d.start, d.end): d.reason for d in out.trace.kept + out.trace.dropped}
    assert reasons[(160, 192)] == "qualifier"
    assert reasons[(193, 212)] == "no_overlap"
    assert out.trace.lost_constraints == ()
    assert (out.trace.input_chars, out.trace.output_chars) == (112, 92)
    assert out.result.technique_id == "R26"
    assert only.parent_start is None


def test_without_qualifier_loses_negation():
    out = compress(Q, result(hit(H, 100, 1)), keep_qualifiers=False)
    assert spans(out) == [(100, 159)]
    assert out.trace.lost_constraints == ("not",)


def test_budget_drops_and_reports():
    out = compress(Q, result(hit(H, 100, 1)), max_total_chars=60)
    assert spans(out) == [(100, 159)]
    reasons = {(d.start, d.end): d.reason for d in out.trace.dropped}
    assert reasons[(160, 192)] == "budget"
    assert out.trace.lost_constraints == ("not",)


def test_two_pieces_and_lost_constraints():
    out = compress("bridge bicycles", result(hit(H, 100, 1)))
    assert spans(out) == [(100, 134), (160, 192)]
    assert [h.rank for h in out.result.hits] == [1, 2]
    assert out.trace.lost_constraints == ("5", "5 usd")
    for h in out.result.hits:
        assert h.text == H[h.start - 100 : h.end - 100]


def test_removed_hit_and_input_rank():
    cafe = hit("The cafe sells tea.", 0, 1, chunk_id="cafe")
    out = compress(Q, result(cafe, hit(H, 100, 2, chunk_id="main")))
    assert out.trace.removed_hits == ("cafe",)
    assert len(out.result.hits) == 1
    got = out.result.hits[0]
    assert got.rank == 1
    ranks = [c for c in got.score_components if c.name == "input_rank"]
    assert ranks[0].value == 2.0 and ranks[0].source == "R02" and ranks[0].source_rank == 2
    assert [c.value for c in got.score_components if c.name == "kept_sentences"] == [3.0]


def test_whole_hit_keeps_chunk_id():
    text = "Tolls are 5 USD per car."
    out = compress(Q, result(hit(text, 40, 1, chunk_id="whole")))
    got = out.result.hits[0]
    assert (got.chunk_id, got.start, got.end) == ("whole", 40, 40 + len(text))


def test_arabic():
    assert len(TA) == 90
    out = compress(QTA, result(hit(TA, 0, 1)))
    assert spans(out) == [(0, 71)]
    assert out.trace.lost_constraints == ()
    off = compress(QTA, result(hit(TA, 0, 1)), keep_qualifiers=False)
    assert spans(off) == [(0, 41)]
    assert off.trace.lost_constraints == ("\u0644\u0627",)


def test_errors_and_wrapper():
    with pytest.raises(IndexingError) as gap:
        compress(Q, result(hit(H, 100, 1), hit("Other tolls text.", 400, 3)))
    assert gap.value.code == "compression_invalid_input"
    bad = hit(H, 100, 1).model_copy(update={"end": 150})
    with pytest.raises(IndexingError) as length:
        compress(Q, result(bad))
    assert length.value.code == "compression_invalid_input"
    with pytest.raises(IndexingError) as empty:
        compress("  ", result(hit(H, 100, 1)))
    assert empty.value.code == "empty_query"
    with pytest.raises(IndexingError) as long:
        compress("a" * 2001, result(hit(H, 100, 1)))
    assert long.value.code == "query_too_long"

    class Base:
        async def retrieve(self, query):
            return result(hit(H, 100, 1))

    wrapper = CompressingRetriever(Base(), ExtractiveCompressor(ExtractiveCompressionConfig()))
    got = asyncio.run(wrapper.retrieve(Q))
    assert wrapper.last is not None and wrapper.last.result == got
    assert wrapper.last.input.technique_id == "R02"
    again = compress(Q, result(hit(H, 100, 1)))
    assert again.model_dump_json() == compress(Q, result(hit(H, 100, 1))).model_dump_json()


def test_condition_word_qualifiers_and_max_hits():
    assert is_qualifier("Only for residents.")
    assert is_qualifier("\u0625\u0630\u0627 \u0643\u0627\u0646.")
    assert not is_qualifier("The cafe sells tea.")
    with pytest.raises(IndexingError) as many:
        compress(Q, result(hit(H, 100, 1), hit("Other tolls text.", 300, 2)), max_hits=1)
    assert many.value.code == "compression_invalid_input"
