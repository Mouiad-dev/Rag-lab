import asyncio
import functools
import json

import pytest
from _pdf_fixtures import build_pdf
from rag_foundry_contracts.resolve import ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import SourceAnchor
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from test_extraction_service import Stream, ingest, setup


def sync(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


HEADING = "# Rechunk evidence"
ENGLISH = "The evidence stays pinned to its original source block."
ARABIC = (
    "\u0647\u064e\u0630\u064e\u0627 \u0646\u064e\u0635\u064c\u0651 "
    "\u0645\u0640\u0640\u062b\u0627\u0644\u0650\u064a\u064c\u0651."
)
MIXED = "A mixed quote \U0001f680 keeps its code point position."
REPEATED = "This repeated paragraph identifies its second block."
MARKDOWN = f"{HEADING}\n\n{ENGLISH}\n\n{ARABIC}\n\n{MIXED}\n\n{REPEATED}\n\n{REPEATED}\n"


async def extract_fixture(tmp_path, data=None, name="rechunk.md", media="text/markdown"):
    if data is None:
        data = MARKDOWN.encode("utf-8")
    _svc, intake, _spy, _store, docs, views, scope = setup(tmp_path)
    doc_id, version_id = await ingest(intake, scope, data, name=name, media=media)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    extraction = ExtractionService(_spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(scope, doc_id, version_id)).view
    payload = await extraction.load_payload(scope, view)
    return scope, _spy, docs, views, extraction, view, payload


def _spans(payload):
    return [(block.start, block.end) for block in payload.blocks]


def _chunks_a(text):
    return [(start, min(start + 40, len(text))) for start in range(0, len(text), 30)]


def _chunks_b(payload):
    return [(b.start, b.start + (b.end - b.start) // 2) for b in payload.blocks] + [
        (b.start + (b.end - b.start) // 2, b.end) for b in payload.blocks
    ]


@sync
async def test_anchors_resolve_across_synthetic_rechunkings(tmp_path):
    _setup_svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
    doc_id, version_id = await ingest(intake, scope, MARKDOWN.encode("utf-8"), name="rechunk.md", media="text/markdown")
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    view = (await extraction.extract(scope, doc_id, version_id)).view
    payload = await extraction.load_payload(scope, view)
    svc = AnchorService(spy, uow, provider_namespace="test")
    assert len(payload.blocks) == 6
    blocks = payload.blocks
    facts = [
        (blocks[2], ARABIC.index("\u0646\u064e\u0635\u064c\u0651"), "\u0646\u064e\u0635\u064c\u0651"),
        (blocks[3], MIXED.index("\U0001f680"), "\U0001f680"),
        (blocks[5], 0, REPEATED),
    ]
    fact_anchors = [
        await svc.anchor(scope, view.id, block.block_id, start, start + len(quote)) for block, start, quote in facts
    ]
    fact_json = [anchor.model_dump_json() for anchor in fact_anchors]
    text = payload.canonical_text
    a_spans = _chunks_a(text)
    b_spans = _chunks_b(payload)
    assert {v for span in a_spans for v in span} != {v for span in b_spans for v in span}

    async def run_chunks(spans):
        chunk_anchors = []
        for start, end in spans:
            anchors = await svc.anchors_for_span(scope, view.id, start, end)
            chunk_anchors.append(anchors)
        serialized = [[anchor.model_dump_json() for anchor in group] for group in chunk_anchors]
        reloaded = [[SourceAnchor.model_validate_json(raw) for raw in group] for group in serialized]
        fresh = AnchorService(spy, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
        for chunk_index, ((start, end), group) in enumerate(zip(spans, reloaded, strict=True)):
            assert all(set(json.loads(raw)) == set(SourceAnchor.model_fields) for raw in serialized[chunk_index])
            results = await fresh.resolve(scope, group)
            assert all(result.status is ResolutionStatus.EXACT for result in results)
            assert "".join(result.exact for result in results) == "".join(
                text[max(start, block.start) : min(end, block.end)]
                for block in payload.blocks
                if max(start, block.start) < min(end, block.end)
            )
        for block, start, quote in facts:
            relevant = [a for group in reloaded for a in group if a.block_id == block.block_id]
            intervals = sorted(
                (max(a.text_range.start, start), min(a.text_range.end, start + len(quote)))
                for a in relevant
                if a.text_range.end > start and a.text_range.start < start + len(quote)
            )
            cursor = start
            for left, right in intervals:
                assert left <= cursor
                cursor = max(cursor, right)
            assert cursor >= start + len(quote)
        return reloaded

    anchors_a = await run_chunks(a_spans)
    anchors_b = await run_chunks(b_spans)
    assert fact_json == [
        (await svc.anchor(scope, view.id, block.block_id, start, start + len(quote))).model_dump_json()
        for block, start, quote in facts
    ]
    for anchor, (block, _start, quote) in zip(fact_anchors, facts, strict=True):
        result = (await svc.resolve(scope, [anchor]))[0]
        assert result.status is ResolutionStatus.EXACT
        assert result.exact == quote
        assert result.block_id == block.block_id
    assert fact_anchors[2].block_id == blocks[5].block_id
    assert blocks[4].block_id != blocks[5].block_id
    _rs, reverse_intake, reverse_spy, _rst, reverse_docs, reverse_views, reverse_scope = setup(tmp_path / "reverse")
    reverse_doc_id, reverse_version_id = await ingest(
        reverse_intake, reverse_scope, MARKDOWN.encode("utf-8"), name="rechunk.md", media="text/markdown"
    )
    reverse_uow = InMemoryUnitOfWorkFactory(documents=reverse_docs, extractions=reverse_views)
    reverse_extraction = ExtractionService(reverse_spy, reverse_uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    reverse_view = (await reverse_extraction.extract(reverse_scope, reverse_doc_id, reverse_version_id)).view
    reverse_payload = await reverse_extraction.load_payload(reverse_scope, reverse_view)
    reverse_svc = AnchorService(reverse_spy, reverse_uow, provider_namespace="test")
    reverse_facts = [
        await reverse_svc.anchor(
            reverse_scope, reverse_view.id, reverse_payload.blocks[index].block_id, start, start + len(quote)
        )
        for index, (_block, start, quote) in zip((2, 3, 5), facts, strict=True)
    ]
    reverse_fact_json = [a.model_dump_json() for a in reverse_facts]
    for spans in (b_spans, a_spans):
        for start, end in spans:
            await reverse_svc.anchors_for_span(reverse_scope, reverse_view.id, start, end)
    assert [a.model_dump_json() for a in reverse_facts] == reverse_fact_json
    for index, (anchor, (_block, _start, quote)) in enumerate(zip(reverse_facts, facts, strict=True)):
        resolved = (await reverse_svc.resolve(reverse_scope, [anchor]))[0]
        original = (await svc.resolve(scope, [fact_anchors[index]]))[0]
        assert resolved.exact == original.exact == quote
        assert resolved.block_id == original.block_id
    changed_quote = fact_anchors[0].model_copy(update={"exact": "X" + fact_anchors[0].exact[1:]})
    changed_result = (await svc.resolve(scope, [changed_quote]))[0]
    assert changed_result.status is ResolutionStatus.MISMATCH
    assert changed_result.reason is ResolutionReason.QUOTE_MISMATCH
    assert anchors_a and anchors_b

    v2 = MARKDOWN.replace(ARABIC, ARABIC[:6] + "\u0632" + ARABIC[7:])
    v2_outcome = await intake.ingest(
        scope, Stream(v2.encode("utf-8")), media_type="text/markdown", filename="v2.md", document_id=view.document_id
    )
    view2 = (await extraction.extract(scope, view.document_id, v2_outcome.version.id)).view
    payload2 = await extraction.load_payload(scope, view2)
    assert (await svc.resolve(scope, [fact_anchors[0]]))[0].status is ResolutionStatus.EXACT
    assert len(payload2.blocks) == len(blocks) == 6
    v2_arabic_block = payload2.blocks[2]
    v2_arabic_text = payload2.canonical_text[v2_arabic_block.start : v2_arabic_block.end]
    assert v2_arabic_text.startswith(ARABIC[:6] + "\u0632")
    assert len(payload2.canonical_text[payload2.blocks[2].start : payload2.blocks[2].end]) == len(
        text[blocks[2].start : blocks[2].end]
    )
    switched = fact_anchors[0].model_copy(
        update={
            "source": fact_anchors[0].source.model_copy(
                update={
                    "document_id": view2.document_id,
                    "document_version_id": view2.document_version_id,
                    "extraction_view_id": view2.id,
                }
            )
        }
    )
    v2_result = (await svc.resolve(scope, [switched]))[0]
    assert v2_result.status is ResolutionStatus.MISMATCH
    assert v2_result.reason is ResolutionReason.QUOTE_MISMATCH
    assert v2_result.exact is None and v2_result.block_id is None  # never relocated to another span

    v3 = MARKDOWN.replace(ARABIC, "\u0628" + ARABIC[1:])
    v3_outcome = await intake.ingest(
        scope, Stream(v3.encode("utf-8")), media_type="text/markdown", filename="v3.md", document_id=view.document_id
    )
    view3 = (await extraction.extract(scope, view.document_id, v3_outcome.version.id)).view
    payload3 = await extraction.load_payload(scope, view3)
    assert len(payload3.blocks) == len(blocks) == 6
    v3_arabic_block = payload3.blocks[2]
    v3_arabic_text = payload3.canonical_text[v3_arabic_block.start : v3_arabic_block.end]
    assert v3_arabic_text.startswith("\u0628" + ARABIC[1:])
    assert len(payload3.canonical_text[payload3.blocks[2].start : payload3.blocks[2].end]) == len(
        text[blocks[2].start : blocks[2].end]
    )
    switched3 = fact_anchors[0].model_copy(
        update={
            "source": fact_anchors[0].source.model_copy(
                update={
                    "document_id": view3.document_id,
                    "document_version_id": view3.document_version_id,
                    "extraction_view_id": view3.id,
                }
            )
        }
    )
    v3_result = (await svc.resolve(scope, [switched3]))[0]
    assert v3_result.status is ResolutionStatus.MISMATCH
    assert v3_result.reason is ResolutionReason.PREFIX_MISMATCH
    bad_range = fact_anchors[0].model_copy(
        update={
            "text_range": fact_anchors[0].text_range.model_copy(
                update={
                    "start": len(payload.canonical_text[blocks[2].start : blocks[2].end]) + 1,
                    "end": len(payload.canonical_text[blocks[2].start : blocks[2].end])
                    + 1
                    + len(fact_anchors[0].exact),
                }
            )
        }
    )
    range_result = (await svc.resolve(scope, [bad_range]))[0]
    assert range_result.status is ResolutionStatus.MISMATCH
    assert range_result.reason is ResolutionReason.RANGE_OUT_OF_BOUNDS
    other = make_scope()
    unavailable = (await svc.resolve(other, [fact_anchors[0]]))[0]
    assert unavailable.status is ResolutionStatus.UNAVAILABLE
    assert unavailable.reason is ResolutionReason.NOT_FOUND


@sync
async def test_pdf_anchor_uses_page_text_without_regions(tmp_path):
    pytest.importorskip("pypdfium2")
    scope, spy, _docs, views, _extraction, view, payload = await extract_fixture(
        tmp_path, build_pdf(["First page", "Second page"]), name="two-pages.pdf", media="application/pdf"
    )
    block = next(b for b in payload.blocks if b.page_index == 1)
    page_text = payload.canonical_text[block.start : block.end]
    svc = AnchorService(spy, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
    anchor = await svc.anchor(scope, view.id, block.block_id, 0, len(page_text))
    assert anchor.page_index == 1
    assert anchor.regions == ()
    assert anchor.precision.value == "exact_text"
