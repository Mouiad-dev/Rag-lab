import importlib.util
import json
import struct
import zlib
from pathlib import Path

import pytest
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_ingestion.extract import ExtractionLimits
from rag_foundry_ingestion.extract.pdf import EXTRACTOR
from rag_foundry_ingestion.page_images import PageImage, encode_png_rgb, render_pdf_pages

pytest.importorskip("pypdfium2")
import pypdfium2 as pdfium  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "multimodal"
_spec = importlib.util.spec_from_file_location("make_chart_pdf", FIXTURES / "make_chart_pdf.py")
make_chart_pdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_chart_pdf)


def _decode_png(png: bytes) -> tuple[int, int, list[bytes]]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, data = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", png[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        if kind == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert (depth, ctype) == (8, 2)
        elif kind == b"IDAT":
            idat += data
        pos += 12 + length
    raw = zlib.decompress(idat)
    row_len = width * 3 + 1
    rows = []
    for y in range(height):
        row = raw[y * row_len : (y + 1) * row_len]
        assert row[0] == 0
        rows.append(row[1:])
    return width, height, rows


def _pixel(rows: list[bytes], x: int, y: int) -> tuple[int, int, int]:
    return tuple(rows[y][x * 3 : x * 3 + 3])


def test_generator_is_deterministic_and_matches_committed_fixture():
    assert make_chart_pdf.build_chart_pdf() == make_chart_pdf.build_chart_pdf()
    assert (FIXTURES / "chart_report.pdf").read_bytes() == make_chart_pdf.build_chart_pdf()


def test_questions_reference_valid_pages():
    data = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))
    kinds = {q["kind"] for q in data["questions"]}
    assert kinds == {"chart", "table", "text", "unanswerable"}
    for q in data["questions"]:
        assert q["expected_page_index"] in (0, 1, 2, None)
        assert (q["expected_page_index"] is None) == (q["kind"] == "unanswerable")


def test_extraction_has_text_on_pages_0_to_2_and_none_on_page_3():
    r = EXTRACTOR.extract(make_chart_pdf.build_chart_pdf(), text_encoding=None, limits=ExtractionLimits())
    assert [b.page_index for b in r.blocks] == [0, 1, 2]
    assert "page_without_text:3" in r.warnings
    assert "Regional report" in r.canonical_text
    assert "Opening hours" in r.canonical_text


def test_renders_four_valid_pngs_with_matching_page_indices():
    images = render_pdf_pages(make_chart_pdf.build_chart_pdf())
    assert [i.page_index for i in images] == [0, 1, 2, 3]
    for img in images:
        width, height, rows = _decode_png(img.png)
        assert (width, height) == (img.width, img.height) == (918, 1188)
        assert len(rows) == height


def test_red_rectangle_reads_as_red_not_blue():
    img = render_pdf_pages(make_chart_pdf.build_chart_pdf(), scale=1.0)[3]
    _, _, rows = _decode_png(img.png)
    # rectangle (156,296)-(456,496) pt, filled (0.8, 0.2, 0.2); centre (306,396) pt -> pixel y from top 396
    r, g, b = _pixel(rows, 306, 396)
    assert r > b + 100
    assert (r, g, b) == (204, 51, 51)
    assert _pixel(rows, 10, 10) == (255, 255, 255)


def test_page_count_limit_raises_before_rendering(monkeypatch):
    monkeypatch.setattr(pdfium.PdfPage, "render", lambda *a, **k: pytest.fail("rendered before limit check"))
    with pytest.raises(ExtractionError) as err:
        render_pdf_pages(make_chart_pdf.build_chart_pdf(), max_pages=3)
    assert err.value.code == "page_render_limit"


def test_pixel_budget_raises_before_rendering(monkeypatch):
    monkeypatch.setattr(pdfium.PdfPage, "render", lambda *a, **k: pytest.fail("rendered before budget check"))
    with pytest.raises(ExtractionError) as err:
        render_pdf_pages(make_chart_pdf.build_chart_pdf(), scale=2.0, max_pixels=1_000_000)
    assert err.value.code == "page_render_limit"
    monkeypatch.undo()
    # ceil(612*1.5) * ceil(792*1.5) = 918 * 1188 = 1_090_584: exactly at the limit is allowed
    assert len(render_pdf_pages(make_chart_pdf.build_chart_pdf(), max_pixels=918 * 1188)) == 4
    with pytest.raises(ExtractionError):
        render_pdf_pages(make_chart_pdf.build_chart_pdf(), max_pixels=918 * 1188 - 1)


@pytest.mark.parametrize("size", [(float("inf"), 792.0), (float("nan"), 792.0), (0.0, 792.0), (612.0, -1.0)])
def test_invalid_page_size_is_page_image_invalid_and_nothing_renders(monkeypatch, size):
    monkeypatch.setattr(pdfium.PdfPage, "get_size", lambda self: size)
    monkeypatch.setattr(pdfium.PdfPage, "render", lambda *a, **k: pytest.fail("rendered an invalid page"))
    with pytest.raises(ExtractionError) as err:
        render_pdf_pages(make_chart_pdf.build_chart_pdf())
    assert err.value.code == "page_image_invalid"


@pytest.mark.parametrize("scale", [float("inf"), float("nan"), 0.0, -1.0, 100.5, 1e300])
def test_non_finite_or_non_positive_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        render_pdf_pages(make_chart_pdf.build_chart_pdf(), scale=scale)


def test_invalid_pdf_is_page_image_invalid():
    with pytest.raises(ExtractionError) as err:
        render_pdf_pages(b"not a pdf at all")
    assert err.value.code == "page_image_invalid"


def test_encode_png_rejects_wrong_buffer_size():
    with pytest.raises(ExtractionError) as err:
        encode_png_rgb(2, 2, b"\x00" * 5)
    assert err.value.code == "page_image_invalid"


def test_repr_has_no_bytes():
    img = PageImage(0, 1, 1, b"\x89PNG-secret-bytes")
    assert "PNG" not in repr(img)
    assert "bytes" not in repr(img)
    assert repr(img) == "PageImage(page_index=0, width=1, height=1)"
