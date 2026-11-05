from dataclasses import replace

import pytest
from rag_foundry_contracts.resolve import ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import PageRegion, Precision, SourceAnchor, SourceRef, TextRange
from rag_foundry_domain.errors import AnchorError
from rag_foundry_domain.extraction import BlockKind, SourceBlock
from rag_foundry_domain.offsets import OffsetMap
from rag_foundry_ingestion.anchors import anchor_id_for, anchors_for_span, make_anchor, resolve_anchor
from rag_foundry_ingestion.payload import ExtractionPayload

REF = SourceRef(
    provider_namespace="local",
    document_id="doc-1",
    document_version_id="ver-1",
    extraction_view_id="view-1",
)


def payload(text="\u0645\u064e\U0001f600xyz"):
    blocks = (SourceBlock(block_id="b00000", ordinal=0, kind=BlockKind.PARAGRAPH, start=0, end=len(text)),)
    return ExtractionPayload(
        extractor="txt",
        extractor_version="1",
        profile_id="canonical-v1",
        unicode_version="15.0.0",
        source_space="decoded_text",
        page_starts=None,
        canonical_text=text,
        blocks=blocks,
        source_map=OffsetMap.identity(len(text)),
    )


def test_code_point_ranges_context_and_json_round_trip():
    p = payload()
    anchor = make_anchor(p, REF, "b00000", 2, 3, context=2)
    assert anchor.exact == "\U0001f600"
    assert anchor.prefix == "\u0645\u064e"
    assert anchor.suffix == "xy"
    restored = SourceAnchor.model_validate_json(anchor.model_dump_json())
    assert resolve_anchor(p, REF, restored).status is ResolutionStatus.EXACT


def test_anchor_ids_are_deterministic_and_validate_namespace():
    one = anchor_id_for(REF, "b00000", 1, 2)
    assert one == anchor_id_for(REF, "b00000", 1, 2)
    assert (
        len(
            {
                one,
                anchor_id_for(REF, "b00000", 2, 2),
                anchor_id_for(REF, "b00000", 1, 3),
                anchor_id_for(REF, "b00001", 1, 2),
                anchor_id_for(REF.model_copy(update={"provider_namespace": "other"}), "b00000", 1, 2),
                anchor_id_for(REF.model_copy(update={"document_id": "doc-2"}), "b00000", 1, 2),
                anchor_id_for(REF.model_copy(update={"document_version_id": "ver-2"}), "b00000", 1, 2),
                anchor_id_for(REF.model_copy(update={"extraction_view_id": "view-2"}), "b00000", 1, 2),
            }
        )
        == 8
    )
    with pytest.raises(AnchorError) as exc:
        anchor_id_for(REF.model_copy(update={"provider_namespace": "Bad"}), "b00000", 1, 2)
    assert exc.value.code == "invalid_namespace"
    assert "Bad" not in str(exc.value)


def test_span_emits_clipped_anchors_per_block_without_separator():
    text = "first\n\nsecond"
    blocks = (
        SourceBlock(block_id="b00000", ordinal=0, kind=BlockKind.PARAGRAPH, start=0, end=5),
        SourceBlock(block_id="b00001", ordinal=1, kind=BlockKind.PARAGRAPH, start=7, end=13),
    )
    p = ExtractionPayload(
        "txt", "1", "canonical-v1", "15.0.0", "decoded_text", None, text, blocks, OffsetMap.identity(len(text))
    )
    anchors = anchors_for_span(p, REF, 2, 10, context=2)
    assert [(a.block_id, a.exact) for a in anchors] == [("b00000", "rst"), ("b00001", "sec")]
    assert anchors[0].suffix is None
    assert anchors[1].prefix is None


@pytest.mark.parametrize(
    "start,end,code",
    [(1, 1, "empty_range"), (-1, 1, "range_out_of_bounds"), (0, 99, "range_out_of_bounds")],
)
def test_creation_rejects_empty_and_out_of_range(start, end, code):
    with pytest.raises(AnchorError) as exc:
        make_anchor(payload(), REF, "b00000", start, end)
    assert exc.value.code == code


def test_missing_block_and_span_without_block_raise_typed_errors():
    p = payload("first\n\nsecond")
    with pytest.raises(AnchorError) as missing:
        make_anchor(p, REF, "b00001", 0, 1)
    assert missing.value.code == "block_not_found"

    blocks = (
        SourceBlock(block_id="b00000", ordinal=0, kind=BlockKind.PARAGRAPH, start=0, end=5),
        SourceBlock(block_id="b00001", ordinal=1, kind=BlockKind.PARAGRAPH, start=7, end=13),
    )
    separated = replace(p, blocks=blocks)
    with pytest.raises(AnchorError) as no_block:
        anchors_for_span(separated, REF, 5, 7)
    assert no_block.value.code == "no_block_in_span"


@pytest.mark.parametrize(
    "anchor,reason,status",
    [
        ("foreign", ResolutionReason.FOREIGN_NAMESPACE, ResolutionStatus.UNAVAILABLE),
        ("source", ResolutionReason.SOURCE_MISMATCH, ResolutionStatus.MISMATCH),
        ("precision", ResolutionReason.UNSUPPORTED_PRECISION, ResolutionStatus.UNAVAILABLE),
        ("range_none", ResolutionReason.UNSUPPORTED_PRECISION, ResolutionStatus.UNAVAILABLE),
        ("exact_none", ResolutionReason.UNSUPPORTED_PRECISION, ResolutionStatus.UNAVAILABLE),
        ("regions", ResolutionReason.UNSUPPORTED_PRECISION, ResolutionStatus.UNAVAILABLE),
        ("block", ResolutionReason.BLOCK_NOT_FOUND, ResolutionStatus.MISMATCH),
        ("bounds", ResolutionReason.RANGE_OUT_OF_BOUNDS, ResolutionStatus.MISMATCH),
        ("empty", ResolutionReason.RANGE_OUT_OF_BOUNDS, ResolutionStatus.MISMATCH),
        ("quote", ResolutionReason.QUOTE_MISMATCH, ResolutionStatus.MISMATCH),
        ("prefix", ResolutionReason.PREFIX_MISMATCH, ResolutionStatus.MISMATCH),
        ("suffix", ResolutionReason.SUFFIX_MISMATCH, ResolutionStatus.MISMATCH),
        ("page", ResolutionReason.PAGE_MISMATCH, ResolutionStatus.MISMATCH),
    ],
)
def test_resolution_reasons(anchor, reason, status):
    p = payload()
    base = make_anchor(p, REF, "b00000", 2, 3)
    if anchor == "foreign":
        candidate = base
        view = REF.model_copy(update={"provider_namespace": "other"})
    else:
        view = REF
        if anchor == "source":
            candidate = base.model_copy(update={"source": REF.model_copy(update={"document_id": "other"})})
        elif anchor == "precision":
            candidate = base.model_copy(update={"precision": Precision.PAGE_ONLY, "page_index": 0})
        elif anchor == "range_none":
            candidate = base.model_copy(
                update={"precision": Precision.PAGE_ONLY, "text_range": None, "exact": None, "page_index": 0}
            )
        elif anchor == "exact_none":
            candidate = base.model_copy(update={"exact": None})
        elif anchor == "regions":
            region = PageRegion(page_index=0, x0=0, y0=0, x1=1, y1=1)
            candidate = base.model_copy(update={"regions": (region,), "page_index": 0})
        elif anchor == "block":
            candidate = base.model_copy(update={"block_id": "b00001"})
        elif anchor == "bounds":
            candidate = base.model_copy(update={"text_range": TextRange(start=1, end=99), "exact": "x" * 98})
        elif anchor == "empty":
            candidate = base.model_copy(update={"text_range": TextRange(start=2, end=2), "exact": ""})
        elif anchor == "quote":
            candidate = base.model_copy(update={"exact": "z"})
        elif anchor == "prefix":
            candidate = base.model_copy(update={"prefix": "\u0645X"})
        elif anchor == "suffix":
            candidate = base.model_copy(update={"suffix": "wrong"})
        else:
            candidate = base.model_copy(update={"page_index": 8})
    candidate = SourceAnchor.model_validate(candidate.model_dump())
    result = resolve_anchor(p, view, candidate)
    assert result.status is status
    assert result.reason is reason


def test_decomposed_quote_is_not_normalized_and_page_block_metadata_is_used():
    p = payload("caf\u00e9x")
    changed = make_anchor(p, REF, "b00000", 3, 5).model_copy(update={"exact": "e\u0301"})
    changed = SourceAnchor.model_validate(changed.model_dump())
    result = resolve_anchor(p, REF, changed)
    assert result.status is ResolutionStatus.MISMATCH
    assert result.reason is ResolutionReason.QUOTE_MISMATCH

    page_block = SourceBlock(block_id="b00000", ordinal=0, kind=BlockKind.PAGE, start=0, end=4, page_index=3)
    page_payload = replace(p, blocks=(page_block,))
    page_anchor = make_anchor(page_payload, REF, "b00000", 0, 1)
    assert page_anchor.page_index == 3 and page_anchor.regions == ()
