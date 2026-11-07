"""Fixed character, word and token chunkers."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest

from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer, char_units, word_units
from rag_foundry_chunking._windows import pack_windows


class FixedCharConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=800, ge=8, le=8000)


class FixedWordConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    words_per_chunk: int = Field(default=150, ge=1, le=2000)


class FixedTokenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens_per_chunk: int = Field(default=256, ge=8, le=8192)
    tokenizer: Literal["lab-regex-v1"] = "lab-regex-v1"


class FixedCharChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C01",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=FixedCharConfig,
    )

    def __init__(self, config: FixedCharConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        units = char_units(doc.text)
        return pack_windows(doc.text, units, tuple(end - start for start, end in units), self.config.chunk_size, 0)


class FixedWordChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C02",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=FixedWordConfig,
    )

    def __init__(self, config: FixedWordConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        units = word_units(doc.text)
        return pack_windows(doc.text, units, (1,) * len(units), self.config.words_per_chunk, 0)


class FixedTokenChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C03",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=FixedTokenConfig,
    )

    def __init__(self, config: FixedTokenConfig, tokenizer: Tokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else LabRegexTokenizer()

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        units = self.tokenizer.token_spans(doc.text)
        return pack_windows(doc.text, units, (1,) * len(units), self.config.tokens_per_chunk, 0)
