"""Extractor interface and shared helpers (ADR-0010). Pure functions over bytes: no I/O, no network.

Offset maps go from the later text to the earlier one: `raw_map` maps raw text -> decoded source, `source_map` maps
canonical text -> decoded source (or PDF raw page text).
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError, InvalidTextEncodingError
from rag_foundry_domain.extraction import BlockKind, SourceBlock
from rag_foundry_domain.offsets import OffsetMap

from rag_foundry_ingestion.normalize import CANONICAL_V1, apply

SourceSpace = Literal["decoded_text", "pdf_page_text"]


@dataclass(frozen=True)
class ExtractionLimits:
    max_chars: int = 5_000_000
    max_pages: int = 2_000


@dataclass(frozen=True)
class RawBlock:
    """A block on raw-text offsets (half-open code points), before canonicalization."""

    kind: BlockKind
    start: int
    end: int
    section_path: tuple[str, ...] = ()
    heading_level: int | None = None
    page_index: int | None = None


@dataclass(frozen=True)
class ExtractionResult:
    canonical_text: str
    blocks: tuple[SourceBlock, ...]
    source_map: OffsetMap
    raw_map: OffsetMap
    source_space: SourceSpace
    page_starts: tuple[int, ...] | None
    warnings: tuple[str, ...]
    extractor: str
    extractor_version: str
    profile_id: str
    unicode_version: str


class Extractor(Protocol):
    name: str
    version: str
    media_types: tuple[MediaType, ...]

    def extract(self, data: bytes, *, text_encoding: str | None, limits: ExtractionLimits) -> ExtractionResult: ...


EXTRACTORS: dict[MediaType, Extractor] = {}


def register(extractor: Extractor) -> None:
    for media_type in extractor.media_types:
        existing = EXTRACTORS.get(media_type)
        if existing is not None and existing is not extractor:
            raise ValueError("a different extractor is already registered for this media type")
    for media_type in extractor.media_types:
        EXTRACTORS[media_type] = extractor


def get_extractor(media_type: MediaType | str) -> Extractor:
    try:
        return EXTRACTORS[MediaType(media_type)]
    except (KeyError, ValueError):
        raise ExtractionError("unsupported_extractor", "no extractor is available for this media type") from None


def decode_text(data: bytes, text_encoding: str | None, limits: ExtractionLimits) -> str:
    """Decode stored bytes. `utf-8-sig` drops the BOM, so later offsets start after it."""
    try:
        text = data.decode(text_encoding or "utf-8")
    except (UnicodeDecodeError, LookupError):
        raise InvalidTextEncodingError("text cannot be decoded with the declared encoding") from None
    if len(text) > limits.max_chars:
        raise ExtractionError("text_too_long", "text exceeds the character limit")
    return text


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_lines(raw: str) -> list[tuple[int, int]]:
    """Spans of lines without their terminators. A final terminator does not start an extra line."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _LINE_BREAK.finditer(raw):
        spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(raw):
        spans.append((pos, len(raw)))
    return spans


def finish(
    raw: str,
    raw_map: OffsetMap,
    blocks_on_raw: Sequence[RawBlock],
    *,
    extractor: str,
    extractor_version: str,
    source_space: SourceSpace,
    page_starts: Sequence[int] | None = None,
    warnings: Sequence[str] = (),
) -> ExtractionResult:
    """Apply `CANONICAL_V1` to `raw`, compose the source map and move block ranges to canonical offsets."""
    if raw_map.n_len != len(raw):
        raise ValueError("raw_map must map the raw text")
    norm = apply(CANONICAL_V1, raw)
    canonical = norm.text
    if not canonical or not blocks_on_raw:
        raise ExtractionError("no_text_layer", "no extractable text")
    n_raw = len(raw)
    blocks: list[SourceBlock] = []
    prev_raw_end = prev_end = 0
    for ordinal, rb in enumerate(blocks_on_raw):
        if not (prev_raw_end <= rb.start <= rb.end <= n_raw):
            raise ExtractionError("integrity_mismatch", "blocks are out of order or out of range")
        prev_raw_end = rb.end
        r = norm.map.to_normalized(rb.start, rb.end)
        if r.end <= r.start or r.start < prev_end:
            raise ExtractionError("integrity_mismatch", "blocks are empty or overlap after normalization")
        prev_end = r.end
        blocks.append(
            SourceBlock(
                block_id=f"b{ordinal:05d}",
                ordinal=ordinal,
                kind=rb.kind,
                start=r.start,
                end=r.end,
                section_path=tuple(apply(CANONICAL_V1, t).text for t in rb.section_path),
                heading_level=rb.heading_level,
                page_index=rb.page_index,
            )
        )
    starts: tuple[int, ...] | None = None
    if page_starts is not None:
        raw_starts = tuple(page_starts)
        if any(not 0 <= p <= n_raw for p in raw_starts) or any(
            a > b for a, b in zip(raw_starts, raw_starts[1:], strict=False)
        ):
            raise ExtractionError("integrity_mismatch", "page starts are out of order or out of range")
        starts = tuple(norm.map.to_normalized(p, p).start for p in raw_starts)
        if any(a >= b for a, b in zip(starts, starts[1:], strict=False)):
            raise ExtractionError("integrity_mismatch", "page starts are not strictly increasing after normalization")
    return ExtractionResult(
        canonical_text=canonical,
        blocks=tuple(blocks),
        source_map=norm.map.compose(raw_map),
        raw_map=raw_map,
        source_space=source_space,
        page_starts=starts,
        warnings=tuple(warnings),
        extractor=extractor,
        extractor_version=extractor_version,
        profile_id=norm.profile_id,
        unicode_version=unicodedata.unidata_version,
    )
