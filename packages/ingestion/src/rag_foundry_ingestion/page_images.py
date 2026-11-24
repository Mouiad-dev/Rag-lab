"""Bounded PDF page rendering to PNG bytes (R41 page-image retrieval input).

`page_index` matches the extraction `page_index` (zero-based pdfium page order). The pixel budget is computed from the
page size and scale before any page is rendered. PNGs are encoded with stdlib zlib (RGB, filter 0); there is no Pillow.
Errors carry codes only, never content. Rendering shares the PDFium lock with the text extractor.
"""

import math
import struct
import zlib
from dataclasses import dataclass, field

from rag_foundry_domain.errors import ExtractionError

from rag_foundry_ingestion.extract import pdf as _pdf

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PageImage:
    page_index: int
    width: int
    height: int
    png: bytes = field(repr=False)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png_rgb(width: int, height: int, rgb_rows: bytes) -> bytes:
    """Encode tightly packed RGB rows (width*3 bytes per row) as a PNG using filter type 0."""
    row_len = width * 3
    if width <= 0 or height <= 0 or len(rgb_rows) != row_len * height:
        raise ExtractionError("page_image_invalid", "the rendered page buffer has an unexpected size")
    raw = b"".join(b"\x00" + rgb_rows[y * row_len : (y + 1) * row_len] for y in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")


def _bitmap_to_rgb_rows(bitmap) -> tuple[int, int, bytes]:
    width, height, stride, channels = bitmap.width, bitmap.height, bitmap.stride, bitmap.n_channels
    if channels not in (3, 4) or stride < width * channels:
        raise ExtractionError("page_image_invalid", "the rendered page has an unsupported pixel layout")
    buf = bytes(bitmap.buffer)
    if len(buf) < stride * height:
        raise ExtractionError("page_image_invalid", "the rendered page buffer is too small")
    out = bytearray()
    for y in range(height):
        row = buf[y * stride : y * stride + width * channels]
        if channels == 3:
            out += row
        else:  # drop the 4th (alpha or padding) channel
            rgb = bytearray(width * 3)
            rgb[0::3] = row[0::4]
            rgb[1::3] = row[1::4]
            rgb[2::3] = row[2::4]
            out += rgb
    return width, height, bytes(out)


def render_pdf_pages(
    pdf: bytes, *, scale: float = 1.5, max_pages: int = 20, max_pixels: int = 4_000_000
) -> tuple[PageImage, ...]:
    if not (math.isfinite(scale) and 0 < scale <= 100) or max_pages < 1 or max_pixels < 1:
        raise ValueError("scale must be finite and in (0, 100]; max_pages and max_pixels must be positive")
    if _pdf.pdfium is None:
        raise ExtractionError("unsupported_extractor", "page rendering needs the 'pdf' extra (pypdfium2)")
    pdfium = _pdf.pdfium
    with _pdf._PDFIUM_LOCK:
        try:
            doc = pdfium.PdfDocument(bytes(pdf))
        except Exception:
            raise ExtractionError("page_image_invalid", "the PDF cannot be opened") from None
        try:
            n_pages = len(doc)
            if n_pages > max_pages:
                raise ExtractionError("page_render_limit", "the PDF exceeds the page render limit")
            # Budget check for every page first, so nothing is rendered when any page is too large.
            for index in range(n_pages):
                try:
                    page = doc[index]
                    try:
                        width_pt, height_pt = page.get_size()
                    finally:
                        page.close()
                except Exception:
                    raise ExtractionError("page_image_invalid", "a page cannot be read") from None
                # An untrusted MediaBox can yield inf, NaN or a non-positive size; reject it before math.ceil.
                if not (math.isfinite(width_pt) and math.isfinite(height_pt)) or width_pt <= 0 or height_pt <= 0:
                    raise ExtractionError("page_image_invalid", "a page has an invalid size")
                if math.ceil(width_pt * scale) * math.ceil(height_pt * scale) > max_pixels:
                    raise ExtractionError("page_render_limit", "a page exceeds the pixel budget")
            images: list[PageImage] = []
            for index in range(n_pages):
                try:
                    page = doc[index]
                    try:
                        bitmap = page.render(scale=scale, rev_byteorder=True)  # RGB(x) instead of BGR(x)
                        try:
                            width, height, rgb = _bitmap_to_rgb_rows(bitmap)
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                except ExtractionError:
                    raise
                except Exception:
                    raise ExtractionError("page_image_invalid", "a page cannot be rendered") from None
                images.append(PageImage(index, width, height, encode_png_rgb(width, height, rgb)))
            return tuple(images)
        finally:
            doc.close()
