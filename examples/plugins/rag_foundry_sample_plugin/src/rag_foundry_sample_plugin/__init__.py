"""Sample external plugin: one span per group of page blocks (simulation, not a C11 implementation)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_techniques import (
    ChunkInput,
    ChunkSpan,
    Fidelity,
    PluginKind,
    PluginManifest,
    TechniqueUnavailableError,
)


class PageBlockConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_pages: int = Field(1, ge=1)


class PageBlockChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C11",
        kind=PluginKind.CHUNKER,
        version="0.1.0",
        fidelity=Fidelity.SIMULATION,
        config_model=PageBlockConfig,
    )

    def __init__(self, config: PageBlockConfig) -> None:
        self.config = config

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        pages = [b for b in doc.blocks if b.kind == "page"]
        if not pages:
            raise TechniqueUnavailableError("C11 sample needs page blocks")
        step = self.config.max_pages
        return tuple(
            ChunkSpan(start=group[0].start, end=group[-1].end, split_level=0)
            for group in (pages[i : i + step] for i in range(0, len(pages), step))
        )
