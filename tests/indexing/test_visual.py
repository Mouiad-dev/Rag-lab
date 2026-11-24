import asyncio
import hashlib
from pathlib import Path

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError, IndexingError, NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_new_index_version
from rag_foundry_domain.indexing import NewChunk, embedding_key_for
from rag_foundry_domain.testing import (
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)
from rag_foundry_indexing.visual import (
    VisualConfig,
    VisualMultivectorRetriever,
    build_visual_index,
)
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_techniques import Fidelity, PluginKind
from rag_foundry_techniques.errors import TechniqueUnavailableError

from tests.providers._fake_page_encoder import FakePageEncoder

pytest.importorskip("pypdfium2")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "multimodal"
PDF = (FIXTURES / "chart_report.pdf").read_bytes()
AXIS = {"text": 0, "chart": 1, "table": 2, "blank": 3}


def _axis_vector(axis: int, dim: int) -> tuple[float, ...]:
    return tuple(1.0 if i == axis else 0.0 for i in range(dim))


class AxisEncoder(FakePageEncoder):
    """Page k gets patch vectors on axis k (hand-checkable MaxSim); query words name an axis."""

    def __init__(self) -> None:
        super().__init__(dim=4, patches=2)

    async def encode_pages(self, pngs):
        result = await super().encode_pages(pngs)
        items = tuple(
            item.model_copy(update={"vectors": (_axis_vector(k, 4), _axis_vector(k, 4))})
            for k, item in enumerate(result.items)
        )
        return result.model_copy(update={"items": items})

    async def encode_queries(self, texts):
        result = await super().encode_queries(texts)
        items = tuple(
            item.model_copy(update={"vectors": tuple(_axis_vector(AXIS[w], 4) for w in text.split())})
            for text, item in zip(texts, result.items, strict=True)
        )
        return result.model_copy(update={"items": items})


async def _fixture(pdf=PDF, *, span=False, two=False):
    owner, stranger = make_scope(), make_scope()
    documents = InMemoryDocumentRepository([owner, stranger])
    views = InMemoryExtractionRepository(documents)
    index = InMemoryIndexRepository(views)
    extracted = get_extractor("application/pdf").extract(pdf, text_encoding=None, limits=ExtractionLimits())
    new_version = make_new_version("visual", "chart.pdf").model_copy(
        update={
            "sha256": hashlib.sha256(pdf).hexdigest(),
            "size_bytes": len(pdf),
            "media_type": MediaType.APPLICATION_PDF,
            "text_encoding": None,
            "original_filename": "chart.pdf",
        }
    )
    document = await documents.create_document_or_get(owner, new_version, display_name="chart")
    new_view = make_new_extraction(document.version, tag="visual", extractor="pdf-pypdfium2").model_copy(
        update={
            "canonical_sha256": hashlib.sha256(extracted.canonical_text.encode("utf-8")).hexdigest(),
            "char_count": len(extracted.canonical_text),
            "block_count": len(extracted.blocks),
            "page_count": 4,
        }
    )
    view = (await views.create_view_or_get(owner, new_view)).view
    view_ids = [view.id]
    pdf2 = None
    if two:
        pdf2 = bytearray(pdf)
        pdf2[7] = ord("5")  # a second, different document version with the same text
        pdf2 = bytes(pdf2)
        doc2 = await documents.create_document_or_get(
            owner,
            new_version.model_copy(update={"sha256": hashlib.sha256(pdf2).hexdigest(), "original_filename": "b.pdf"}),
            display_name="chart2",
        )
        new_view2 = new_view.model_copy(
            update={"document_id": doc2.version.document_id, "document_version_id": doc2.version.id}
        )
        view2 = (await views.create_view_or_get(owner, new_view2)).view
        view_ids.append(view2.id)
    embedder = LexicalTestEmbedder()
    profile = embedder.profile
    dim = profile.capabilities.embedding_dim
    config = make_new_index_version(tuple(sorted(view_ids)), dim=dim).model_copy(
        update={
            "embedding_profile_id": profile.id,
            "embedding_provider": profile.provider,
            "embedding_model": profile.model,
            "embedding_key": embedding_key_for(profile.provider, profile.model, dim),
        }
    )
    version = await index.create_index_version(owner, config)
    spans = [(b.start, b.end, (b.block_id,)) for b in extracted.blocks]
    if span:  # one chunk over page 0 and page 1, then the page 2 chunk
        spans = [(spans[0][0], spans[1][1], (spans[0][2][0], spans[1][2][0])), spans[2]]
    chunks = []
    for ordinal, (start, end, block_ids) in enumerate(spans):
        text = extracted.canonical_text[start:end]
        embedded = await embedder.embed([text], input_type="document")
        chunks.append(
            NewChunk(
                extraction_view_id=view.id,
                ordinal=ordinal,
                start=start,
                end=end,
                split_level=0,
                block_ids=block_ids,
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                embedding=embedded.vectors[0],
            )
        )
    await index.add_chunks(owner, version.id, tuple(chunks))
    version = await index.mark_ready(owner, version.id)
    base = (owner, stranger, documents, views, index, version, view, extracted)
    return (*base, view2, pdf2) if two else base


async def _built(encoder=None):
    owner, stranger, documents, views, index, version, view, extracted = await _fixture()
    encoder = encoder or AxisEncoder()
    built = await build_visual_index(owner, version, {view.id: PDF}, index, documents, encoder, extraction_repo=views)
    return owner, stranger, documents, views, index, version, view, extracted, built, encoder


def _retriever(scope, version, built, encoder, **config):
    return VisualMultivectorRetriever(
        VisualConfig(**config), scope=scope, index_version=version, index=built, encoder=encoder
    )


def _run(coro):
    return asyncio.run(coro)


def test_manifest_and_registration():
    manifest = VisualMultivectorRetriever.manifest
    assert manifest.technique_id == "R41"
    assert manifest.kind is PluginKind.RETRIEVER
    assert manifest.fidelity is Fidelity.ADAPTATION
    from rag_foundry_indexing import register_builtin_plugins
    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R41", {}) == VisualConfig()


async def _chart_first():
    owner, _, _, _, _, version, view, extracted, built, encoder = await _built()
    stats = built.stats()
    assert (stats.pages, stats.patches, stats.dim, stats.estimated_bytes) == (4, 8, 4, 8 * 4 * 4)
    assert (built.tenant_id, built.project_id) == (owner.tenant_id, owner.project_id)
    report = await _retriever(owner, version, built, encoder).search_with_report("chart")
    result = report.result
    assert result.technique_id == "R41" and result.scoring == "r41-page-maxsim"
    top = result.hits[0]
    assert top.rank == 1 and top.score == pytest.approx(1.0)
    assert top.score_components[0].name == "page_maxsim" and top.score_components[0].term == "page:1"
    return result, extracted


def test_chart_query_ranks_page_one_first():
    _run(_chart_first())


async def _orders():
    owner, _, _, _, _, version, _, _, built, encoder = await _built()
    retriever = _retriever(owner, version, built, encoder)
    first = [h.score_components[0].term for h in (await retriever.retrieve("chart")).hits]
    second = [h.score_components[0].term for h in (await retriever.retrieve("table text")).hits]
    assert first[0] == "page:1" and second[:2] in (["page:2", "page:0"], ["page:0", "page:2"])
    assert first != second


def test_two_queries_give_different_orders():
    _run(_orders())


async def _unanchored():
    owner, _, _, _, _, version, view, _, built, encoder = await _built()
    retriever = _retriever(owner, version, built, encoder, k=10)
    report = await retriever.search_with_report("blank")
    assert [p.page_index for p in report.unanchored_pages] == [3]
    assert report.unanchored_pages[0].score == pytest.approx(1.0)
    assert all(h.score_components[0].term != "page:3" for h in report.result.hits)
    assert len(report.result.hits) == 3  # pages 0, 1, 2 only
    plain = await retriever.retrieve("blank")
    assert plain == report.result and not hasattr(plain, "unanchored_pages")
    assert 3 not in {int(h.score_components[0].term.split(":")[1]) for h in plain.hits}


def test_text_less_page_is_reported_never_cited():
    _run(_unanchored())


async def _hits_on_scored_page():
    owner, _, _, _, _, version, _, extracted, built, encoder = await _built()
    result = await _retriever(owner, version, built, encoder, k=10).retrieve("text chart table")
    pages = {b.block_id: b.page_index for b in extracted.blocks}
    assert len(result.hits) == 3
    for hit in result.hits:
        page = int(hit.score_components[0].term.split(":")[1])
        assert all(pages[b] == page for b in hit.block_ids)
        assert hit.text == extracted.canonical_text[hit.start : hit.end]


def test_hits_carry_chunks_on_the_scored_page():
    _run(_hits_on_scored_page())


async def _sha_mismatch():
    owner, _, documents, views, index, version, view, _ = await _fixture()
    changed = bytearray(PDF)
    changed[7] = ord("5")  # header %PDF-1.4 -> %PDF-1.5: still a valid, identically rendering PDF
    assert bytes(changed) != PDF
    encoder = AxisEncoder()
    with pytest.raises(ExtractionError) as err:
        await build_visual_index(
            owner, version, {view.id: bytes(changed)}, index, documents, encoder, extraction_repo=views
        )
    assert err.value.code == "page_image_invalid"
    assert encoder.calls == 0  # nothing was encoded, so nothing was indexed


def test_one_changed_byte_is_page_image_invalid():
    _run(_sha_mismatch())


async def _cross_tenant_query():
    owner, stranger, _, _, _, version, _, _, built, encoder = await _built()
    calls = encoder.calls
    retriever = _retriever(stranger, version, built, encoder)
    with pytest.raises(NotFoundError):
        await retriever.retrieve("chart")
    assert encoder.calls == calls  # no query was encoded for the wrong tenant
    # the same scope still works (the failure is the scope, not the setup)
    assert (await _retriever(owner, version, built, encoder).retrieve("chart")).hits


def test_cross_tenant_query_is_not_found_with_zero_encoder_calls():
    _run(_cross_tenant_query())


async def _build_scope_checks():
    owner, stranger, documents, views, index, version, view, _ = await _fixture()
    encoder = AxisEncoder()
    with pytest.raises(NotFoundError):  # another tenant asks to index the owner's view
        await build_visual_index(stranger, version, {view.id: PDF}, index, documents, encoder, extraction_repo=views)
    other_view = "00000000-0000-4000-8000-000000000001"
    with pytest.raises(NotFoundError):  # a view that is not part of this index version
        await build_visual_index(owner, version, {other_view: PDF}, index, documents, encoder, extraction_repo=views)
    assert encoder.calls == 0


def test_build_rejects_other_tenant_and_foreign_view():
    _run(_build_scope_checks())


async def _other_version_rejected():
    owner, _, _, _, _, version, _, _, built, encoder = await _built()
    other = version.model_copy(update={"id": "00000000-0000-4000-8000-000000000002"})
    with pytest.raises(IndexingError) as err:
        await _retriever(owner, other, built, encoder).retrieve("chart")
    assert str(err.value.code) == "visual_index_missing"
    with pytest.raises(IndexingError):
        await _retriever(owner, version, None, encoder).retrieve("chart")


def test_index_for_another_version_is_rejected():
    _run(_other_version_rejected())


async def _unavailable_and_checks():
    owner, _, _, _, _, version, _, _, built, encoder = await _built()
    with pytest.raises(TechniqueUnavailableError):
        await _retriever(owner, version, built, None).retrieve("chart")

    class Missing(AxisEncoder):
        async def encode_queries(self, texts):
            raise AdapterDependencyError("missing")

    with pytest.raises(TechniqueUnavailableError):
        await _retriever(owner, version, built, Missing()).retrieve("chart")
    with pytest.raises(IndexingError):
        await _retriever(owner, version, built, encoder).retrieve("   ")


def test_unavailable_and_query_checks():
    _run(_unavailable_and_checks())


async def _dim_drift():
    owner, _, documents, views, index, version, view, _ = await _fixture()

    class Drift(AxisEncoder):
        async def encode_pages(self, pngs):
            result = await super().encode_pages(pngs)
            items = list(result.items)
            items[-1] = items[-1].model_copy(update={"vectors": ((1.0, 0.0, 0.0),)})
            return result.model_copy(update={"items": tuple(items)})

    with pytest.raises(IndexingError) as err:
        await build_visual_index(owner, version, {view.id: PDF}, index, documents, Drift(), extraction_repo=views)
    assert err.value.code == "multivector_dim_mismatch"


def test_dim_drift_is_rejected():
    _run(_dim_drift())


async def _two_pass_foreign():
    owner, _, documents, views, index, version, view, _, view2, pdf2 = await _fixture(two=True)
    first, second = sorted([view.id, view2.id])
    pdfs = {view.id: PDF, view2.id: pdf2}
    encoder = AxisEncoder()
    foreign = "ffffffff-ffff-4fff-8fff-ffffffffffff"  # sorts after both valid views
    with pytest.raises(NotFoundError):
        await build_visual_index(
            owner, version, {**pdfs, foreign: PDF}, index, documents, encoder, extraction_repo=views
        )
    assert encoder.calls == 0
    tampered = {**pdfs, second: pdfs[second] + b" "}
    with pytest.raises(ExtractionError) as err:
        await build_visual_index(owner, version, tampered, index, documents, encoder, extraction_repo=views)
    assert err.value.code == "page_image_invalid"
    assert encoder.calls == 0
    built = await build_visual_index(owner, version, pdfs, index, documents, encoder, extraction_repo=views)
    assert built.stats().pages == 8 and first != second


def test_bad_second_view_fails_before_any_encoder_call():
    _run(_two_pass_foreign())


async def _chunk_spanning_pages():
    owner, _, documents, views, index, version, view, extracted = await _fixture(span=True)
    encoder = AxisEncoder()
    built = await build_visual_index(owner, version, {view.id: PDF}, index, documents, encoder, extraction_repo=views)
    retriever = _retriever(owner, version, built, encoder, k=10)
    report = await retriever.search_with_report("text chart table")
    ids = [h.chunk_id for h in report.result.hits]
    assert len(ids) == len(set(ids)) == 2  # the two-page chunk once, plus the page 2 chunk
    assert [p.page_index for p in report.unanchored_pages] == [3]  # a covered page is not "unanchored"


def test_chunk_spanning_two_pages_is_cited_once():
    _run(_chunk_spanning_pages())


async def _bad_encoder_output():
    owner, _, documents, views, index, version, view, _ = await _fixture()

    class Short(AxisEncoder):
        async def encode_pages(self, pngs):
            result = await super().encode_pages(pngs)
            return result.model_copy(update={"items": result.items[:-1]})

    class Empty(AxisEncoder):
        async def encode_pages(self, pngs):
            result = await super().encode_pages(pngs)
            items = (result.items[0].model_copy(update={"vectors": ()}), *result.items[1:])
            return result.model_copy(update={"items": items})

    for encoder, code in ((Short(), "multivector_count_mismatch"), (Empty(), "multivector_page_empty")):
        with pytest.raises(IndexingError) as err:
            await build_visual_index(owner, version, {view.id: PDF}, index, documents, encoder, extraction_repo=views)
        assert err.value.code == code


def test_wrong_item_count_and_empty_page_are_coded_errors():
    _run(_bad_encoder_output())
