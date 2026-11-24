"""Media type, filename and content checks for uploads (ADR-0009 D5, D8). HTML is never parsed."""

import codecs
import unicodedata
from typing import Literal

from rag_foundry_domain import (
    InvalidFilenameError,
    InvalidTextEncodingError,
    MediaType,
    MediaTypeMismatchError,
    UnsupportedMediaTypeError,
)

_EXTENSIONS: dict[MediaType, tuple[str, ...]] = {
    MediaType.TEXT_PLAIN: (".txt",),
    MediaType.TEXT_MARKDOWN: (".md", ".markdown"),
    MediaType.TEXT_HTML: (".html", ".htm"),
    MediaType.APPLICATION_PDF: (".pdf",),
}
_PDF_MAGIC = b"%PDF-"
_BOM = b"\xef\xbb\xbf"
# Control, surrogate, line/paragraph separators and bidi overrides/isolates. ZWNJ, ZWJ, LRM, RLM, ALM stay allowed.
_BAD_CATEGORIES = frozenset({"Cc", "Cs", "Zl", "Zp"})
_BIDI_CONTROLS = frozenset(chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)))


def parse_media_type(raw: str | None) -> MediaType:
    if raw is None:
        raise UnsupportedMediaTypeError()
    base, *params = raw.split(";")
    try:
        media_type = MediaType(base.strip().lower())
    except ValueError:
        raise UnsupportedMediaTypeError() from None
    for param in params:
        name, sep, value = param.partition("=")
        if sep and name.strip().lower() == "charset":
            if value.strip().strip('"').strip().lower() not in ("utf-8", "utf8"):
                raise UnsupportedMediaTypeError("charset must be utf-8")
    return media_type


def validate_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFC", name)
    if not 1 <= len(normalized) <= 255:
        raise InvalidFilenameError()
    if normalized.startswith(" ") or normalized.endswith(" ") or normalized in (".", ".."):
        raise InvalidFilenameError()
    for ch in normalized:
        if ch in "/\\" or ch in _BIDI_CONTROLS or unicodedata.category(ch) in _BAD_CATEGORIES:
            raise InvalidFilenameError()
    return normalized


class ContentInspector:
    """Feed chunks in order, then call `finish()`. Raises typed intake errors."""

    def __init__(self, declared: MediaType, filename: str) -> None:
        if not filename.lower().endswith(_EXTENSIONS[declared]):
            raise MediaTypeMismatchError()
        self._declared = declared
        self._head = b""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

    @property
    def _is_pdf(self) -> bool:
        return self._declared is MediaType.APPLICATION_PDF

    def feed(self, chunk: bytes) -> None:
        if self._is_pdf:
            if len(self._head) < len(_PDF_MAGIC):
                self._head = (self._head + chunk)[: len(_PDF_MAGIC)]
                if not _PDF_MAGIC.startswith(self._head):
                    raise MediaTypeMismatchError()
            return
        if len(self._head) < len(_BOM):
            self._head = (self._head + chunk)[: len(_BOM)]
        if b"\x00" in chunk:
            raise InvalidTextEncodingError("text contains NUL")
        try:
            self._decoder.decode(chunk)
        except UnicodeDecodeError:
            raise InvalidTextEncodingError() from None

    def finish(self) -> Literal["utf-8", "utf-8-sig"] | None:
        if self._is_pdf:
            if self._head != _PDF_MAGIC:
                raise MediaTypeMismatchError()
            return None
        try:
            self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise InvalidTextEncodingError() from None
        return "utf-8-sig" if self._head == _BOM else "utf-8"
