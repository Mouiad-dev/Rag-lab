"""Deterministic recursive character chunker (C05)."""

from __future__ import annotations

import unicodedata
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "\u061f",
    "\u06d4",
    "; ",
    "\u061b",
    ", ",
    "\u060c",
    " ",
    "",
)


class RecursiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=800, ge=32, le=8000)
    chunk_overlap: int = Field(default=100, ge=0)
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    @model_validator(mode="after")
    def validate_config(self) -> RecursiveConfig:
        if self.chunk_overlap > self.chunk_size // 4:
            raise ValueError("chunk_overlap must not exceed chunk_size // 4")
        if not self.separators or self.separators[-1] != "":
            raise ValueError("separators must be non-empty and end with an empty separator")
        if len(set(self.separators)) != len(self.separators):
            raise ValueError("separators must be unique")
        return self


class RecursiveChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C05",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=RecursiveConfig,
    )

    def __init__(self, config: RecursiveConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        text = doc.text
        pieces: list[tuple[int, int, int]] = []

        def split(start: int, end: int, level: int) -> None:
            if end - start <= self.config.chunk_size:
                pieces.append((start, end, level))
                return
            separator = "" if level >= 63 else self.config.separators[min(level, len(self.config.separators) - 1)]
            if not separator:
                cursor = start
                while cursor < end:
                    cut = min(cursor + self.config.chunk_size, end)
                    if cut < end:
                        while cut > cursor and _unsafe_boundary(text, cut):
                            cut -= 1
                        if cut == cursor:
                            cut = min(cursor + self.config.chunk_size, end)
                    pieces.append((cursor, cut, level))
                    cursor = cut
                return
            cursor = start
            segments: list[tuple[int, int]] = []
            while cursor < end:
                position = text.find(separator, cursor, end)
                if position < 0:
                    break
                piece_end = position + len(separator)
                segments.append((cursor, piece_end))
                cursor = piece_end
            if cursor < end:
                segments.append((cursor, end))
            if not segments:
                split(start, end, level + 1)
                return
            for piece_start, piece_end in segments:
                if piece_end - piece_start > self.config.chunk_size:
                    split(piece_start, piece_end, level + 1)
                else:
                    pieces.append((piece_start, piece_end, level))

        split(0, len(text), 0)

        merged: list[tuple[int, int, tuple[tuple[int, int, int], ...]]] = []
        for piece in pieces:
            start, end, _ = piece
            if merged and end - merged[-1][0] <= self.config.chunk_size:
                old_start, _, old_pieces = merged[-1]
                merged[-1] = (old_start, end, (*old_pieces, piece))
            else:
                merged.append((start, end, (piece,)))

        spans: list[ChunkSpan] = []
        previous_start = -1
        previous_end = -1
        for index, (start, end, chunk_pieces) in enumerate(merged):
            if index and self.config.chunk_overlap:
                prior_start, prior_end, prior_pieces = merged[index - 1]
                overlap_start = next(
                    (
                        piece_start
                        for piece_start, _, _ in prior_pieces
                        if prior_end - piece_start <= self.config.chunk_overlap and piece_start > previous_start
                    ),
                    start,
                )
                proposed_start = min(start, overlap_start)
                if end - proposed_start <= self.config.chunk_size:
                    start = proposed_start
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start == end or end <= previous_end:
                continue
            levels = [piece[2] for piece in chunk_pieces]
            if index and start < merged[index][0]:
                levels.extend(level for piece_start, _, level in merged[index - 1][2] if piece_start >= start)
            spans.append(ChunkSpan(start=start, end=end, split_level=max(levels)))
            previous_start = start
            previous_end = end
        spans.sort(key=lambda span: (span.start, span.end))
        return tuple(spans)


def _unsafe_boundary(text: str, cut: int) -> bool:
    char = text[cut]
    return (
        unicodedata.combining(char) != 0
        or char in ("\u200d", "\ufe0e", "\ufe0f")
        or (text[cut - 1] == "\r" and char == "\n")
    )
