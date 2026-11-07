"""Arabic, English, and mixed-script boundary chunking (C21)."""

from __future__ import annotations

import unicodedata
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import (
    DEFAULT_ABBREVIATIONS,
    _unsafe,
    pack,
    sentence_ranges,
    word_ranges,
)

_BIDI = frozenset("\u200e\u200f\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
_CLAUSE_MARKS = frozenset("\u060c\u061b,;:")
_ALEF_FOLD = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627", "\u0671": "\u0627"}
_OPENING = frozenset("\u202a\u202b\u202d\u202e\u2066\u2067\u2068")
_CLOSING = frozenset("\u202c\u2069\u200e\u200f\u061c")


def script_of(char: str) -> Literal["arabic", "latin", "other"]:
    code = ord(char)
    category = unicodedata.category(char)
    if not category.startswith("L") or char == "\u0640" or char in "\u200c\u200d":
        return "other"
    if any(
        low <= code <= high
        for low, high in ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
    ):
        return "arabic"
    if ("A" <= char <= "Z") or ("a" <= char <= "z") or 0x00C0 <= code <= 0x024F:
        return "latin"
    return "other"


class MultilingualConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=800, ge=32, le=8000)
    overlap_sentences: int = Field(default=0, ge=0, le=3)
    abbreviations: tuple[str, ...] = DEFAULT_ABBREVIATIONS

    @field_validator("abbreviations")
    @classmethod
    def valid_abbreviations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not value or value != value.lower() or any(c.isspace() for c in value) or value.endswith(".")
            for value in values
        ):
            raise ValueError("abbreviations must be lowercase tokens without whitespace or trailing dots")
        if len(set(values)) != len(values):
            raise ValueError("abbreviations must be unique")
        return values


def _clause_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result = []
    cursor = start
    i = start
    while i < end:
        if text[i] in _CLAUSE_MARKS and i + 1 < end and text[i + 1].isspace():
            result.append((cursor, i + 1))
            i += 1
            while i < end and text[i].isspace():
                i += 1
            cursor = i
        else:
            i += 1
    if cursor < end:
        result.append((cursor, end))
    return result


def _script_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    # Nonletters are skipped when detecting script changes. Opening bidi
    # controls move with the following run; closing controls with the prior run.
    letters = [i for i in range(start, end) if script_of(text[i]) != "other"]
    cuts = []
    for left, right in zip(letters, letters[1:], strict=False):
        if script_of(text[left]) != script_of(text[right]):
            cut = right
            while cut > left and text[cut - 1] in _OPENING:
                cut -= 1
            while cut < end and text[cut] in _CLOSING:
                cut += 1
            cuts.append(cut)
    result = []
    cursor = start
    for cut in cuts:
        if cursor < cut:
            result.append((cursor, cut))
        cursor = cut
    if cursor < end:
        result.append((cursor, end))
    return result


def _safe_bidi_cuts(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    # Same safety rules as boundaries.safe_cut_ranges, plus: never cut right after
    # an opening bidi control or right before a closing one.
    def unsafe(i: int) -> bool:
        return _unsafe(text, i, start) or text[i - 1] in _OPENING or text[i] in _CLOSING

    ranges = []
    cursor = start
    while cursor < end:
        cut = min(cursor + size, end)
        if cut < end:
            safe = cut
            while safe > cursor and unsafe(safe):
                safe -= 1
            cut = safe if safe > cursor else min(cursor + size, end)
        ranges.append((cursor, cut))
        cursor = cut
    return ranges


class MultilingualChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C21",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=MultilingualConfig,
    )

    def __init__(self, config: MultilingualConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        text = doc.text
        units: list[tuple[int, int, int]] = []
        for start, end in sentence_ranges(text, 0, len(text), self.config.abbreviations):
            if end - start <= self.config.chunk_size:
                units.append((start, end, 0))
                continue
            for clause_start, clause_end in _clause_ranges(text, start, end):
                if clause_end - clause_start <= self.config.chunk_size:
                    units.append((clause_start, clause_end, 1))
                    continue
                for word_start, word_end in word_ranges(text, clause_start, clause_end):
                    if word_end - word_start <= self.config.chunk_size:
                        units.append((word_start, word_end, 2))
                        continue
                    for run_start, run_end in _script_ranges(text, word_start, word_end):
                        if run_end - run_start <= self.config.chunk_size:
                            units.append((run_start, run_end, 3))
                        else:
                            units.extend(
                                (a, b, 4) for a, b in _safe_bidi_cuts(text, run_start, run_end, self.config.chunk_size)
                            )
        packed = pack(text, units, self.config.chunk_size, self.config.overlap_sentences)
        return tuple(ChunkSpan(start=a, end=b, split_level=level) for a, b, level in packed)


def normalize_with_map(text: str, start: int = 0, end: int | None = None) -> tuple[str, tuple[int, ...]]:
    stop = len(text) if end is None else end
    if start < 0 or stop < start or stop > len(text):
        raise ValueError("invalid text range")
    output = []
    mapping = []
    for offset in range(start, stop):
        char = text[offset]
        code = ord(char)
        if char in _BIDI or char == "\u0640" or 0x064B <= code <= 0x065F or code == 0x0670 or 0x06D6 <= code <= 0x06ED:
            continue
        output.append(_ALEF_FOLD.get(char, char))
        mapping.append(offset)
    return "".join(output), tuple(mapping)
