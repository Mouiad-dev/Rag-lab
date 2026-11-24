"""PDF text-layer extractor (`pdf-pypdfium2`, version 1): one `page` block per page with text (ADR-0010 D1, D8, D11).

Raw text is the library's page text joined by a blank line. Pages without text are omitted from the raw text, blocks and
`page_starts` (so `page_starts[i]` belongs to block `i`); their `page_index` is never reused and each gets a warning.
No OCR, geometry or rendering. Parsing runs only in the worker/CLI path (ADR-0010 D4); there is no subprocess isolation
yet, so a native crash or hang can take down the worker (planned for P8.4). Text never appears in errors or warnings.
"""

import re
import threading

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_domain.offsets import OffsetMapBuilder

from rag_foundry_ingestion.extract.base import (
    EXTRACTORS,
    ExtractionLimits,
    ExtractionResult,
    RawBlock,
    finish,
    register,
)

try:
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_raw
except ImportError:  # the `pdf` extra is not installed
    pdfium = None  # type: ignore[assignment]

PYPDFIUM2_PIN = "5.14.0"
_SEPARATOR = "\n\n"
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
# PDFium is not thread-safe: one lock around every call into it.
_PDFIUM_LOCK = threading.Lock()


def _has_content(text: str) -> bool:
    return any(c.isprintable() and not c.isspace() for c in text)


class PdfTextLayerExtractor:
    name = "pdf-pypdfium2"
    version = f"1+pypdfium2=={PYPDFIUM2_PIN}"
    media_types = (MediaType.APPLICATION_PDF,)

    def _read_page(self, doc: "pdfium.PdfDocument", index: int) -> str:
        page = doc[index]
        try:
            textpage = page.get_textpage()
            try:
                # surrogatepass keeps lone surrogates so they can be replaced one-for-one instead of dropped
                return textpage.get_text_range(errors="surrogatepass")
            finally:
                textpage.close()
        finally:
            page.close()

    def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
        with _PDFIUM_LOCK:
            return self._extract_locked(bytes(data), limits)

    def _extract_locked(self, data: bytes, limits: ExtractionLimits) -> ExtractionResult:
        try:
            doc = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            if exc.err_code == pdfium_raw.FPDF_ERR_PASSWORD:
                raise ExtractionError("pdf_encrypted", "the PDF needs a password") from None
            raise ExtractionError("pdf_malformed", "the PDF cannot be opened") from None
        except Exception:
            raise ExtractionError("pdf_malformed", "the PDF cannot be opened") from None
        warnings: list[str] = []
        raw_parts: list[str] = []
        raw_blocks: list[RawBlock] = []
        page_starts: list[int] = []
        raw_len = 0
        try:
            n_pages = len(doc)
            if n_pages > limits.max_pages:
                raise ExtractionError("too_many_pages", "the PDF exceeds the page limit")
            for index in range(n_pages):
                try:
                    text = self._read_page(doc, index)
                except Exception:
                    raise ExtractionError("pdf_malformed", "a page cannot be read") from None
                if _LONE_SURROGATE.search(text):
                    text = _LONE_SURROGATE.sub("\ufffd", text)
                    warnings.append(f"invalid_text:{index}")
                if not _has_content(text):
                    warnings.append(f"page_without_text:{index}")
                    continue
                start = raw_len + (len(_SEPARATOR) if raw_parts else 0)
                raw_len = start + len(text)
                if raw_len > limits.max_chars:
                    raise ExtractionError("text_too_long", "text exceeds the character limit")
                raw_parts.append(text)
                page_starts.append(start)
                raw_blocks.append(RawBlock(BlockKind.PAGE, start, raw_len, page_index=index))
        finally:
            doc.close()
        if not raw_parts:
            raise ExtractionError("no_text_layer", "the PDF has no text layer")
        builder = OffsetMapBuilder()
        for i, text in enumerate(raw_parts):
            if i:
                builder.atomic(len(_SEPARATOR), 0)  # inserted separator
            builder.cp(len(text))
        return finish(
            _SEPARATOR.join(raw_parts),
            builder.build(),
            raw_blocks,
            extractor=self.name,
            extractor_version=self.version,
            source_space="pdf_page_text",
            page_starts=page_starts,
            warnings=warnings,
        )


EXTRACTOR = PdfTextLayerExtractor()
if pdfium is not None:
    register(EXTRACTOR)
else:
    # Fail explicitly and name the missing extra (never a silent fallback).
    class _MissingPdfExtra:
        name = EXTRACTOR.name
        version = EXTRACTOR.version
        media_types = EXTRACTOR.media_types

        def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult:
            raise ExtractionError("unsupported_extractor", "PDF extraction needs the 'pdf' extra (pypdfium2)")

    EXTRACTORS.setdefault(MediaType.APPLICATION_PDF, _MissingPdfExtra())
