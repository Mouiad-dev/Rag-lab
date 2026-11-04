import pytest
from rag_foundry_domain import (
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaType,
    MediaTypeMismatchError,
    UnsupportedMediaTypeError,
)
from rag_foundry_ingestion import ContentInspector, parse_media_type, validate_filename


def inspect(media: MediaType, name: str, *chunks: bytes):
    insp = ContentInspector(media, name)
    for c in chunks:
        insp.feed(c)
    return insp.finish()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("text/plain", MediaType.TEXT_PLAIN),
        ("  TEXT/PLAIN; charset=UTF-8 ", MediaType.TEXT_PLAIN),
        ("text/markdown; charset=utf8", MediaType.TEXT_MARKDOWN),
        ("text/html; boundary=x", MediaType.TEXT_HTML),
        ("application/pdf; foo=bar", MediaType.APPLICATION_PDF),
    ],
)
def test_parse_accepts(raw, expected):
    assert parse_media_type(raw) is expected


@pytest.mark.parametrize("raw", [None, "", "image/png", "text/csv", "text/plain; charset=latin-1"])
def test_parse_rejects(raw):
    with pytest.raises(UnsupportedMediaTypeError):
        parse_media_type(raw)


def test_filename_accepts_and_normalizes():
    decomposed = "café ملف.txt"
    assert validate_filename(decomposed) == "café ملف.txt"
    zwnj = "ملف\u200cا.txt"
    assert validate_filename(zwnj) == zwnj
    assert validate_filename("a\u200eb\u200fc\u061cd\u200d.txt")
    assert validate_filename("a" * 255) == "a" * 255


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a" * 256,
        "a/b",
        "a\b",
        "a\x00b",
        "a\nb",
        " a",
        "a ",
        ".",
        "..",
        "a\u202ecod.txt",
        "a\u2066b",
        "a\u2028b",
        "a\u2029b",
        "a\ud800.txt",
    ],
)
def test_filename_rejects(name):
    with pytest.raises(InvalidFilenameError):
        validate_filename(name)


@pytest.mark.parametrize(
    ("media", "name"),
    [
        (MediaType.TEXT_PLAIN, "a.TXT"),
        (MediaType.TEXT_MARKDOWN, "a.md"),
        (MediaType.TEXT_MARKDOWN, "a.markdown"),
        (MediaType.TEXT_HTML, "a.htm"),
        (MediaType.TEXT_HTML, "a.HTML"),
        (MediaType.APPLICATION_PDF, "a.PDF"),
    ],
)
def test_extension_accepts(media, name):
    ContentInspector(media, name)


@pytest.mark.parametrize(
    ("media", "name"),
    [
        (MediaType.TEXT_PLAIN, "a.md"),
        (MediaType.TEXT_HTML, "a.txt"),
        (MediaType.APPLICATION_PDF, "a"),
        (MediaType.TEXT_MARKDOWN, "a.pdf"),
    ],
)
def test_extension_mismatch(media, name):
    with pytest.raises(MediaTypeMismatchError):
        ContentInspector(media, name)


def test_pdf_magic_ok_and_split():
    assert inspect(MediaType.APPLICATION_PDF, "a.pdf", b"%PDF-1.7 rest") is None
    assert inspect(MediaType.APPLICATION_PDF, "a.pdf", b"%P", b"DF", b"-1.4") is None


@pytest.mark.parametrize("chunks", [(b"%PDX-1",), (b"%P", b"X"), (b"%PDF",), (b"hello",)])
def test_pdf_magic_bad(chunks):
    with pytest.raises(MediaTypeMismatchError):
        inspect(MediaType.APPLICATION_PDF, "a.pdf", *chunks)


def test_text_utf8_and_bom():
    assert inspect(MediaType.TEXT_PLAIN, "a.txt", b"hello") == "utf-8"
    assert inspect(MediaType.TEXT_PLAIN, "a.txt", b"\xef\xbb\xbfhello") == "utf-8-sig"
    assert inspect(MediaType.TEXT_PLAIN, "a.txt", b"\xef", b"\xbb", b"\xbfhi") == "utf-8-sig"
    assert inspect(MediaType.TEXT_HTML, "a.html", b"<p>x</p>") == "utf-8"


def test_arabic_character_split_across_chunks():
    data = "مرحبا".encode()
    assert inspect(MediaType.TEXT_MARKDOWN, "a.md", data[:1], data[1:]) == "utf-8"


@pytest.mark.parametrize("chunks", [(b"\xff\xfe",), (b"ab\x00cd",), (b"\xd9",), (b"\xd9", b"a")])
def test_text_invalid(chunks):
    with pytest.raises(InvalidTextEncodingError):
        inspect(MediaType.TEXT_PLAIN, "a.txt", *chunks)
