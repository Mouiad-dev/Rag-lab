import asyncio
import hashlib
import struct

import pytest
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor
from rag_foundry_ingestion.extract.pdf import EXTRACTOR, PdfTextLayerExtractor

pytest.importorskip("pypdfium2")

LIMITS = ExtractionLimits()

_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(byte ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def _pad(password: bytes) -> bytes:
    return (password + _PAD)[:32]


def _escape(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")


def build_pdf(pages: list[str], *, user: bytes | None = None, owner: bytes | None = None) -> bytes:
    """A minimal PDF (Standard-14 Helvetica, ASCII). With `user`/`owner`: security handler V1/R2, 40-bit RC4."""
    n = len(pages)
    file_id = hashlib.md5(b"rag-foundry-test-id").digest()
    key = b""
    enc_obj = b""
    if user is not None or owner is not None:
        user = user or b""
        owner = owner if owner is not None else user
        perms = -4
        # PDF 1.7 algorithm 3 (R2): /O
        o_value = rc4(hashlib.md5(_pad(owner)).digest()[:5], _pad(user))
        # algorithm 2 (R2): encryption key
        key = hashlib.md5(_pad(user) + o_value + struct.pack("<i", perms) + file_id).digest()[:5]
        # algorithm 4 (R2): /U
        u_value = rc4(key, _PAD)
        enc_obj = b"<< /Filter /Standard /V 1 /R 2 /Length 40 /P %d /O <%s> /U <%s> >>" % (
            perms,
            o_value.hex().encode(),
            u_value.hex().encode(),
        )
    objects: list[bytes] = []  # object i+1 at index i
    first_page_obj = 4
    kids = b" ".join(b"%d 0 R" % (first_page_obj + 2 * i) for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        page_no = first_page_obj + 2 * i
        stream = b"BT /F1 12 Tf 72 700 Td (%s) Tj ET" % _escape(text) if text else b""
        if key:
            obj_key = hashlib.md5(key + struct.pack("<I", page_no + 1)[:3] + b"\x00\x00").digest()[:10]
            stream = rc4(obj_key, stream)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
            b"/Resources << /Font << /F1 3 0 R >> >> >>" % (page_no + 1)
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    if enc_obj:
        objects.append(enc_obj)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i + 1, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    trailer = b"<< /Size %d /Root 1 0 R /ID [<%s> <%s>]" % (
        len(objects) + 1,
        file_id.hex().encode(),
        file_id.hex().encode(),
    )
    if enc_obj:
        trailer += b" /Encrypt %d 0 R" % len(objects)
    out += b"trailer\n" + trailer + b" >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)


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
