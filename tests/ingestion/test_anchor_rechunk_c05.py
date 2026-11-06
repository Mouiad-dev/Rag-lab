import asyncio
import json

from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_contracts.resolve import ResolutionStatus
from rag_foundry_contracts.source import SourceAnchor
from rag_foundry_domain.errors import AnchorError
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_techniques import ChunkInput, TextBlock
from test_anchor_rechunk import ARABIC, MARKDOWN, MIXED, REPEATED
from test_extraction_service import ingest, setup


def test_c05_rechunk_preserves_fact_anchors(tmp_path):
    async def run():
        _svc, intake, spy, _store, docs, views, scope = setup(tmp_path)
        doc_id, version_id = await ingest(intake, scope, MARKDOWN.encode(), name="rechunk.md", media="text/markdown")
        extraction = ExtractionService(spy, docs, views, extractors=EXTRACTORS, limits=ExtractionLimits())
        view = (await extraction.extract(scope, doc_id, version_id)).view
        payload = await extraction.load_payload(scope, view)
        blocks = payload.blocks
        assert len(blocks) == 6
        svc = AnchorService(spy, views, provider_namespace="test")
        facts = [
            (blocks[2], ARABIC.index("\u0646\u064e\u0635\u064c\u0651"), "\u0646\u064e\u0635\u064c\u0651"),
            (blocks[3], MIXED.index("\U0001f680 keeps"), "\U0001f680 keeps"),
            (blocks[5], 0, REPEATED),
        ]
        fact_anchors = [
            await svc.anchor(scope, view.id, block.block_id, start, start + len(quote)) for block, start, quote in facts
        ]
        originals = [anchor.model_dump_json() for anchor in fact_anchors]
        text = payload.canonical_text
        chunkings = [
            RecursiveChunker(RecursiveConfig(chunk_size=40, chunk_overlap=10)),
            RecursiveChunker(RecursiveConfig(chunk_size=120, chunk_overlap=0)),
        ]
        spans = [
            chunker.chunk(
                ChunkInput(
                    text=text, blocks=tuple(TextBlock(block_id=b.block_id, start=b.start, end=b.end) for b in blocks)
                )
            )
            for chunker in chunkings
        ]
        assert {(s.start, s.end) for s in spans[0]} != {(s.start, s.end) for s in spans[1]}
        assert any(span.split_level >= 2 for span in spans[0])

        for chunk_spans in spans:
            serialized = []
            for span in chunk_spans:
                try:
                    anchors = await svc.anchors_for_span(scope, view.id, span.start, span.end)
                except AnchorError as exc:
                    assert exc.code == "no_block_in_span"
                    assert not any(b.start < span.end and b.end > span.start for b in blocks)
                    continue
                serialized.append([anchor.model_dump_json() for anchor in anchors])
            loaded = [[SourceAnchor.model_validate_json(raw) for raw in group] for group in serialized]
            fresh = AnchorService(spy, views, provider_namespace="test")
            resolutions = await fresh.resolve(scope, tuple(anchor for group in loaded for anchor in group))
            assert all(item.status is ResolutionStatus.EXACT for item in resolutions)
            for block, start, quote in facts:
                positions = [i for i, char in enumerate(quote) if not char.isspace()]
                ranges = [
                    (a.text_range.start, a.text_range.end)
                    for group in loaded
                    for a in group
                    if a.block_id == block.block_id
                ]
                assert all(any(left <= start + i < right for left, right in ranges) for i in positions)
        assert originals == [
            (await svc.anchor(scope, view.id, block.block_id, start, start + len(quote))).model_dump_json()
            for block, start, quote in facts
        ]
        for anchor, (block, _start, quote) in zip(fact_anchors, facts, strict=True):
            result = (await svc.resolve(scope, [anchor]))[0]
            assert result.status is ResolutionStatus.EXACT and result.exact == quote
            assert result.block_id == block.block_id
        assert fact_anchors[2].block_id == "b00005"
        assert blocks[4].block_id == "b00004"
        assert all(set(json.loads(raw)) == set(SourceAnchor.model_fields) for group in serialized for raw in group)

    asyncio.run(run())
