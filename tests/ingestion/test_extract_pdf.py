import asyncio

import pytest
from _pdf_fixtures import build_pdf
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor
from rag_foundry_ingestion.extract.pdf import EXTRACTOR, PdfTextLayerExtractor

pytest.importorskip("pypdfium2")

LIMITS = ExtractionLimits()


def run(data: bytes, limits: ExtractionLimits = LIMITS):
    return EXTRACTOR.extract(data, text_encoding=None, limits=limits)


def test_registered_for_pdf():
    assert get_extractor(MediaType.APPLICATION_PDF) is EXTRACTOR
    assert EXTRACTOR.name == "pdf-pypdfium2"
    assert EXTRACTOR.version.startswith("1+pypdfium2==")


def test_pages_blocks_warnings_and_map_invariants():
    data = build_pdf(["Hello first page", "", "Second page here"])
    r = run(data)
    assert r.source_space == "pdf_page_text"
    assert [b.kind for b in r.blocks] == [BlockKind.PAGE, BlockKind.PAGE]
    assert [b.page_index for b in r.blocks] == [0, 2]
    assert [r.canonical_text[b.start : b.end] for b in r.blocks] == ["Hello first page", "Second page here"]
    assert r.warnings == ("page_without_text:1",)
    assert r.page_starts == tuple(b.start for b in r.blocks)
    assert r.extractor == "pdf-pypdfium2"
    # map invariants: canonical -> raw page text, raw -> library page text with inserted separators
    assert r.source_map.n_len == len(r.canonical_text)
    assert r.source_map.o_len == r.raw_map.o_len
    raw_len = r.raw_map.n_len
    assert raw_len == len("Hello first page") + 2 + len("Second page here")
    for b in r.blocks:
        span = r.source_map.to_original(b.start, b.end)
        assert 0 <= span.start < span.end <= r.source_map.o_len
    assert run(data) == r


def test_user_password_is_encrypted_error():
    data = build_pdf(["secret page"], user=b"userpw", owner=b"ownerpw")
    with pytest.raises(ExtractionError) as err:
        run(data)
    assert err.value.code == "pdf_encrypted"
    assert "secret" not in str(err.value)


def test_owner_password_only_opens_normally():
    data = build_pdf(["open text"], user=b"", owner=b"ownerpw")
    r = run(data)
    assert r.canonical_text == "open text"


def test_garbage_is_malformed():
    with pytest.raises(ExtractionError) as err:
        run(b"this is not a pdf at all")
    assert err.value.code == "pdf_malformed"


def test_no_text_layer():
    with pytest.raises(ExtractionError) as err:
        run(build_pdf(["", ""]))
    assert err.value.code == "no_text_layer"


def test_max_pages():
    data = build_pdf(["a", "b", "c"])
    with pytest.raises(ExtractionError) as err:
        run(data, ExtractionLimits(max_pages=2))
    assert err.value.code == "too_many_pages"
    assert run(data, ExtractionLimits(max_pages=3)).canonical_text == "a\n\nb\n\nc"


def test_max_chars():
    data = build_pdf(["abcdef", "ghijkl"])
    with pytest.raises(ExtractionError) as err:
        run(data, ExtractionLimits(max_chars=10))
    assert err.value.code == "text_too_long"
    assert run(data, ExtractionLimits(max_chars=14)).canonical_text == "abcdef\n\nghijkl"


def test_lone_surrogate_replaced_one_for_one(monkeypatch):
    pages = {0: "ab\ud800cd", 1: "fine"}
    monkeypatch.setattr(PdfTextLayerExtractor, "_read_page", lambda self, doc, index: pages[index])
    r = run(build_pdf(["x", "y"]))
    assert r.canonical_text == "ab\ufffdcd\n\nfine"
    assert r.warnings == ("invalid_text:0",)
    assert r.raw_map.o_len == len("ab\ufffdcd") + len("fine")


def test_surrogatepass_is_what_the_library_receives(monkeypatch):
    seen = {}
    import pypdfium2 as pdfium

    original = pdfium.PdfTextPage.get_text_range

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfTextPage, "get_text_range", spy)
    run(build_pdf(["text"]))
    assert seen == {"errors": "surrogatepass"}


def test_concurrent_extractions_are_identical():
    data = build_pdf(["Hello first page", "Second page here", "Third"])

    async def go():
        return await asyncio.gather(*(asyncio.to_thread(run, data) for _ in range(4)))

    results = asyncio.run(go())
    assert all(r == results[0] for r in results)
    assert results[0].canonical_text == "Hello first page\n\nSecond page here\n\nThird"


@pytest.mark.skip("needs owner-supplied fixture (ADR-0010 D9)")
def test_arabic_pdf():
    raise AssertionError("unreachable")
