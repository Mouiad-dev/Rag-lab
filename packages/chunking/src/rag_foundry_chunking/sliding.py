"""Overlapping unit-window chunking and storage statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer, char_units, word_units
from rag_foundry_chunking._windows import pack_windows


class SlidingWindowConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: Literal["chars", "words", "tokens"] = "words"
    window_size: int = Field(default=200, ge=2, le=8000)
    overlap: int = Field(default=50, ge=1)
    tokenizer: Literal["lab-regex-v1"] = "lab-regex-v1"

    @model_validator(mode="after")
    def validate_window(self) -> SlidingWindowConfig:
        if self.overlap > self.window_size // 2:
            raise ValueError("overlap must be at most half the window_size")
        if self.unit == "chars" and self.window_size < 8:
            raise ValueError("character windows require window_size >= 8")
        return self


class SlidingWindowChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C04",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=SlidingWindowConfig,
    )

    def __init__(self, config: SlidingWindowConfig, tokenizer: Tokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else LabRegexTokenizer()

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        if self.config.unit == "chars":
            units = char_units(doc.text)
            weights = tuple(end - start for start, end in units)
        elif self.config.unit == "words":
            units = word_units(doc.text)
            weights = (1,) * len(units)
        else:
            units = self.tokenizer.token_spans(doc.text)
            weights = (1,) * len(units)
        return pack_windows(
            doc.text, units, weights, self.config.window_size, self.config.overlap, trim_whitespace=False
        )


@dataclass(frozen=True)
class WindowStats:
    chunks: int
    stored_code_points: int
    unique_code_points: int
    duplicated_code_points: int
    storage_ratio: float


def window_stats(spans: tuple[ChunkSpan, ...] | list[ChunkSpan]) -> WindowStats:
    intervals = sorted((span.start, span.end) for span in spans if span.end > span.start)
    stored = sum(end - start for start, end in intervals)
    unique = 0
    if intervals:
        left, right = intervals[0]
        for start, end in intervals[1:]:
            if start > right:
                unique += right - left
                left, right = start, end
            else:
                right = max(right, end)
        unique += right - left
    return WindowStats(len(spans), stored, unique, stored - unique, stored / unique if unique else 0.0)
