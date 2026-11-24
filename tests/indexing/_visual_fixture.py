"""Shared setup for the R41 demo and the live run: chart_report.pdf indexed as one text chunk per PDF block.

Test-only helper. It builds the text index (R02 arm) and the page maps; the caller supplies the embedder and the
page-image encoder, so the demo uses fakes and the live test uses the local models.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_new_index_version
from rag_foundry_domain.indexing import NewChunk, embedding_key_for
from rag_foundry_domain.testing import (
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)
from rag_foundry_indexing import DenseConfig, DenseRetriever
from rag_foundry_indexing.visual import VisualConfig, VisualMultivectorRetriever, build_visual_index
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "multimodal"
PDF = (FIXTURES / "chart_report.pdf").read_bytes()
QUESTIONS = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))["questions"]
PAGES = 4


@dataclass
class Setup:
    scope: object
    documents: object
    views: object
    index: object
    version: object
    view: object


async def build_text_index(embedder) -> Setup:
    """Index every PDF block as one chunk (page 3 has no text, so no chunk)."""
    scope = make_scope()
    documents = InMemoryDocumentRepository([scope])
    views = InMemoryExtractionRepository(documents)
    index = InMemoryIndexRepository(views)
    extracted = get_extractor("application/pdf").extract(PDF, text_encoding=None, limits=ExtractionLimits())
    new_version = make_new_version("visual", "chart.pdf").model_copy(
        update={
            "sha256": hashlib.sha256(PDF).hexdigest(),
            "size_bytes": len(PDF),
            "media_type": MediaType.APPLICATION_PDF,
            "text_encoding": None,
            "original_filename": "chart.pdf",
        }
    )
    document = await documents.create_document_or_get(scope, new_version, display_name="chart")
    new_view = make_new_extraction(document.version, tag="visual", extractor="pdf-pypdfium2").model_copy(
        update={
            "canonical_sha256": hashlib.sha256(extracted.canonical_text.encode("utf-8")).hexdigest(),
            "char_count": len(extracted.canonical_text),
            "block_count": len(extracted.blocks),
            "page_count": PAGES,
        }
    )
    view = (await views.create_view_or_get(scope, new_view)).view
    profile = embedder.profile
    dim = profile.capabilities.embedding_dim
    config = make_new_index_version((view.id,), dim=dim).model_copy(
        update={
            "embedding_profile_id": profile.id,
            "embedding_provider": profile.provider,
            "embedding_model": profile.model,
            "embedding_key": embedding_key_for(profile.provider, profile.model, dim),
        }
    )
    version = await index.create_index_version(scope, config)
    chunks = []
    for ordinal, block in enumerate(extracted.blocks):
        text = extracted.canonical_text[block.start : block.end]
        embedded = await embedder.embed([text], input_type="document")
        chunks.append(
            NewChunk(
                extraction_view_id=view.id,
                ordinal=ordinal,
                start=block.start,
                end=block.end,
                split_level=0,
                block_ids=(block.block_id,),
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                embedding=embedded.vectors[0],
            )
        )
    await index.add_chunks(scope, version.id, tuple(chunks))
    version = await index.mark_ready(scope, version.id)
    return Setup(scope, documents, views, index, version, view)


async def build_pages(setup: Setup, encoder):
    return await build_visual_index(
        setup.scope,
        setup.version,
        {setup.view.id: PDF},
        setup.index,
        setup.documents,
        encoder,
        extraction_repo=setup.views,
    )


def chunk_pages(built) -> dict[str, int]:
    """Chunk id to the page it sits on (from the visual index)."""
    return {cid: page.page_index for page in built.pages for cid in page.chunk_ids}


async def r41_page_ranking(setup: Setup, built, encoder, question: str):
    """Full page ranking [(page, score, anchored)] and the page of each hit chunk (k=3, the default)."""
    wide = VisualMultivectorRetriever(
        VisualConfig(k=50), scope=setup.scope, index_version=setup.version, index=built, encoder=encoder
    )
    report = await wide.search_with_report(question)
    rows = [(int(h.score_components[0].term.split(":")[1]), h.score, True) for h in report.result.hits]
    rows += [(u.page_index, u.score, False) for u in report.unanchored_pages]
    rows.sort(key=lambda r: (-r[1], r[0]))
    narrow = VisualMultivectorRetriever(
        VisualConfig(), scope=setup.scope, index_version=setup.version, index=built, encoder=encoder
    )
    hits = (await narrow.retrieve(question)).hits
    pages_of = chunk_pages(built)
    return rows, [pages_of[h.chunk_id] for h in hits]


async def r02_chunk_ranking(setup: Setup, built, embedder, question: str):
    """R02 chunk ranking [(chunk ordinal, page, score)] over every chunk (all have text, so all are on a page)."""
    retriever = DenseRetriever(
        DenseConfig(k=50), scope=setup.scope, index_version=setup.version, index=setup.index, embedder=embedder
    )
    result = await retriever.retrieve(question)
    pages_of = chunk_pages(built)
    return [(built.chunks[h.chunk_id].ordinal, pages_of[h.chunk_id], h.score) for h in result.hits]


def r02_page_rows(chunk_rows):
    """Best chunk per page, by cosine: [(page, score)] in rank order."""
    best: dict[int, float] = {}
    for _ordinal, page, score in chunk_rows:
        best[page] = max(best.get(page, score), score)
    return sorted(best.items(), key=lambda r: (-r[1], r[0]))


def _rank_of(expected, ordered_pages):
    if expected is None:
        return None
    return ordered_pages.index(expected) + 1 if expected in ordered_pages else None


def format_r41(question, rows) -> str:
    pages = [r[0] for r in rows]
    expected = question["expected_page_index"]
    rank = _rank_of(expected, pages)
    shown = rows[0] if rank is None else rows[rank - 1]
    head = f"top1=p{shown[0]} score={shown[1]:.3f}" if rank is None else f"rank={rank} score={shown[1]:.3f}"
    full = " ".join(f"p{p}:{s:.3f}{'' if a else '(no text)'}" for p, s, a in rows)
    return f"{head} top1_page={rows[0][0]} ranking(page:score)={full}"


def format_r02(question, chunk_rows) -> str:
    page_rows = r02_page_rows(chunk_rows)
    expected = question["expected_page_index"]
    rank = _rank_of(expected, [p for p, _ in page_rows])
    head = (
        f"top1=p{page_rows[0][0]} score={page_rows[0][1]:.3f}"
        if rank is None
        else f"rank={rank} score={page_rows[rank - 1][1]:.3f}"
    )
    pages = " ".join(f"p{p}:{s:.3f}" for p, s in page_rows)
    chunks = " ".join(f"c{o}@p{p}:{s:.3f}" for o, p, s in chunk_rows)
    return f"{head} top1_page={page_rows[0][0]} page_ranking(best chunk per page)={pages} chunk_ranking={chunks}"
