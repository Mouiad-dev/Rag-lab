"""Arabic-aware sentence chunker (C06)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking.boundaries import (
    DEFAULT_ABBREVIATIONS,
    pack,
    safe_cut_ranges,
    sentence_ranges,
    word_ranges,
)


class SentenceConfig(BaseModel):
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


class SentenceChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C06",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SentenceConfig,
    )

    def __init__(self, config: SentenceConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        text = doc.text
        units: list[tuple[int, int, int]] = []
        for start, end in sentence_ranges(text, 0, len(text), self.config.abbreviations):
            if end - start <= self.config.chunk_size:
                units.append((start, end, 0))
                continue
            for word_start, word_end in word_ranges(text, start, end):
                if word_end - word_start <= self.config.chunk_size:
                    units.append((word_start, word_end, 1))
                else:
                    units.extend(
                        (cut_start, cut_end, 2)
                        for cut_start, cut_end in safe_cut_ranges(text, word_start, word_end, self.config.chunk_size)
                    )
        packed = pack(text, units, self.config.chunk_size, self.config.overlap_sentences)
        return tuple(ChunkSpan(start=start, end=end, split_level=level) for start, end, level in packed)
