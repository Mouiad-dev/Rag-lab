"""Pure source anchor construction and resolution over an extraction payload."""

import hashlib
import json
import re

from rag_foundry_contracts.resolve import AnchorResolution, ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import Precision, SourceAnchor, SourceRef, TextRange
from rag_foundry_domain.errors import AnchorError

from rag_foundry_ingestion.payload import ExtractionPayload

_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def anchor_id_for(ref: SourceRef, block_id: str, start: int, end: int) -> str:
    ns = ref.provider_namespace
    if not _NAMESPACE.fullmatch(ns):
        raise AnchorError("invalid_namespace")
    values = [
        "anchor-id-v1",
        ns,
        ref.document_id,
        ref.document_version_id,
        ref.extraction_view_id,
        block_id,
        start,
        end,
    ]
    encoded = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "anc_" + hashlib.sha256(encoded).hexdigest()[:32]


def _block(payload: ExtractionPayload, block_id: str):
    for block in payload.blocks:
        if block.block_id == block_id:
            return block
    raise AnchorError("block_not_found")


def make_anchor(
    payload: ExtractionPayload,
    ref: SourceRef,
    block_id: str,
    start: int,
    end: int,
    *,
    context: int = 32,
) -> SourceAnchor:
    block = _block(payload, block_id)
    block_text = payload.canonical_text[block.start : block.end]
    if start == end:
        raise AnchorError("empty_range")
    if start < 0 or end < 0 or start > end or end > len(block_text):
        raise AnchorError("range_out_of_bounds")
    if context < 0:
        raise ValueError("context must be non-negative")
    exact = block_text[start:end]
    prefix = block_text[max(0, start - context) : start] or None
    suffix = block_text[end : end + context] or None
    return SourceAnchor(
        anchor_id=anchor_id_for(ref, block_id, start, end),
        source=ref,
        block_id=block_id,
        section_path=block.section_path,
        text_range=TextRange(start=start, end=end),
        exact=exact,
        prefix=prefix,
        suffix=suffix,
        page_index=block.page_index,
        page_label=None,
        regions=(),
        precision=Precision.EXACT_TEXT,
    )


def anchors_for_span(
    payload: ExtractionPayload,
    ref: SourceRef,
    start: int,
    end: int,
    *,
    context: int = 32,
) -> tuple[SourceAnchor, ...]:
    if start == end:
        raise AnchorError("empty_range")
    if start < 0 or end < 0 or start > end or end > len(payload.canonical_text):
        raise AnchorError("range_out_of_bounds")
    anchors = []
    for block in payload.blocks:
        left, right = max(start, block.start), min(end, block.end)
        if left < right:
            anchors.append(
                make_anchor(
                    payload,
                    ref,
                    block.block_id,
                    left - block.start,
                    right - block.start,
                    context=context,
                )
            )
    if not anchors:
        raise AnchorError("no_block_in_span")
    return tuple(anchors)


def resolve_anchor(payload: ExtractionPayload, view_ref: SourceRef, anchor: SourceAnchor) -> AnchorResolution:
    def failure(status: ResolutionStatus, reason: ResolutionReason) -> AnchorResolution:
        return AnchorResolution(anchor_id=anchor.anchor_id, status=status, reason=reason)

    if anchor.source.provider_namespace != view_ref.provider_namespace:
        return failure(ResolutionStatus.UNAVAILABLE, ResolutionReason.FOREIGN_NAMESPACE)
    if (
        anchor.source.document_id != view_ref.document_id
        or anchor.source.document_version_id != view_ref.document_version_id
        or anchor.source.extraction_view_id != view_ref.extraction_view_id
    ):
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.SOURCE_MISMATCH)
    if (
        anchor.precision is not Precision.EXACT_TEXT
        or anchor.text_range is None
        or anchor.exact is None
        or bool(anchor.regions)
    ):
        return failure(ResolutionStatus.UNAVAILABLE, ResolutionReason.UNSUPPORTED_PRECISION)
    block = next((b for b in payload.blocks if b.block_id == anchor.block_id), None)
    if block is None:
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.BLOCK_NOT_FOUND)
    block_text = payload.canonical_text[block.start : block.end]
    start, end = anchor.text_range.start, anchor.text_range.end
    if start == end or end > len(block_text):
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.RANGE_OUT_OF_BOUNDS)
    if block_text[start:end] != anchor.exact:
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.QUOTE_MISMATCH)
    if anchor.prefix is not None and (
        len(anchor.prefix) > start or block_text[start - len(anchor.prefix) : start] != anchor.prefix
    ):
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.PREFIX_MISMATCH)
    if anchor.suffix is not None and block_text[end : end + len(anchor.suffix)] != anchor.suffix:
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.SUFFIX_MISMATCH)
    if anchor.page_index is not None and anchor.page_index != block.page_index:
        return failure(ResolutionStatus.MISMATCH, ResolutionReason.PAGE_MISMATCH)
    return AnchorResolution(
        anchor_id=anchor.anchor_id,
        status=ResolutionStatus.EXACT,
        exact=anchor.exact,
        block_id=block.block_id,
        section_path=block.section_path,
        page_index=block.page_index,
    )
