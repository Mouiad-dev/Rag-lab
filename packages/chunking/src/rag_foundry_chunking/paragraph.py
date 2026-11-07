"""Paragraph packing chunker (C07)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import pack, safe_cut_ranges, sentence_ranges, trim, word_ranges


class ParagraphConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=1200, ge=32, le=8000)
    overlap_paragraphs: int = Field(default=0, ge=0, le=2)
    overflow: Literal["sentences", "hard_cut"] = "sentences"
    use_blocks: bool = True


def _blank_line_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    i = start
    while i < end:
        if text[i] == "\n":
            j = i + 1
            while j < end and text[j] != "\n" and text[j].isspace():
                j += 1
            if j < end and text[j] == "\n":
                item = trim(text, cursor, i + 1)
                if item is not None:
                    result.append(item)
                cursor = j + 1
                i = cursor
                continue
        i += 1
    item = trim(text, cursor, end)
    if item is not None:
        result.append(item)
    return result


def paragraph_ranges(doc: ChunkInput, use_blocks: bool) -> list[tuple[int, int]]:
    text = doc.text
    if not use_blocks or not doc.blocks:
        return _blank_line_ranges(text, 0, len(text))
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for block in doc.blocks:
        ranges.extend(_blank_line_ranges(text, cursor, block.start))
        ranges.extend(_blank_line_ranges(text, block.start, block.end))
        cursor = block.end
    ranges.extend(_blank_line_ranges(text, cursor, len(text)))
    return ranges


class ParagraphChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C07",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ParagraphConfig,
    )

    def __init__(self, config: ParagraphConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        text = doc.text
        units: list[tuple[int, int, int]] = []
        for start, end in paragraph_ranges(doc, self.config.use_blocks):
            if end - start <= self.config.chunk_size:
                units.append((start, end, 0))
            elif self.config.overflow == "hard_cut":
                units.extend(
                    (cut_start, cut_end, 3)
                    for cut_start, cut_end in safe_cut_ranges(text, start, end, self.config.chunk_size)
                )
            else:
                for sentence_start, sentence_end in sentence_ranges(text, start, end):
                    if sentence_end - sentence_start <= self.config.chunk_size:
                        units.append((sentence_start, sentence_end, 1))
                        continue
                    for word_start, word_end in word_ranges(text, sentence_start, sentence_end):
                        if word_end - word_start <= self.config.chunk_size:
                            units.append((word_start, word_end, 2))
                        else:
                            units.extend(
                                (cut_start, cut_end, 3)
                                for cut_start, cut_end in safe_cut_ranges(
                                    text, word_start, word_end, self.config.chunk_size
                                )
                            )
        packed = pack(text, units, self.config.chunk_size, self.config.overlap_paragraphs)
        return tuple(ChunkSpan(start=start, end=end, split_level=level) for start, end, level in packed)
