"""R21 sentence-window context expansion."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_chunking.boundaries import sentence_ranges
from rag_foundry_domain import Scope
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult

from rag_foundry_indexing.context_expansion import (
    ExpandedRetrieval,
    ExpansionSeed,
    SourceTextReader,
    assemble,
    load_sources,
    merge_overlapping,
)


class SentenceWindowConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    window: int = Field(default=1, ge=1, le=5)
    stay_in_paragraph: bool = True
    max_window_chars: int = Field(default=2000, ge=20, le=20000)
    max_total_chars: int = Field(default=8000, ge=20, le=100000)
    max_seeds: int = Field(default=20, ge=1, le=50)


class SentenceWindowExpander:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R21",
        kind=PluginKind.CONTEXT_BUILDER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SentenceWindowConfig,
    )

    def __init__(
        self,
        config: SentenceWindowConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        source: SourceTextReader,
    ):
        self.config, self.scope, self.index_version, self.index, self.source = (
            config,
            scope,
            index_version,
            index,
            source,
        )

    async def expand(self, result: RetrievalResult) -> ExpandedRetrieval:
        _, sources = await load_sources(
            result,
            scope=self.scope,
            index=self.index,
            index_version=self.index_version,
            source=self.source,
            max_seeds=self.config.max_seeds,
        )
        candidates = []
        for hit in result.hits:
            text = sources[hit.extraction_view_id].text
            if self.config.stay_in_paragraph:
                paragraphs = [
                    (m.start(), m.end()) for m in re.finditer(r"(?s)(?:^|\n[ \t]*\n)(.*?)(?=\n[ \t]*\n|\Z)", text)
                ]
                bounds = [p for p in paragraphs if p[0] < hit.end and p[1] > hit.start]
            else:
                bounds = [(0, len(text))]
            sentences = sorted({span for a, b in bounds for span in sentence_ranges(text, a, b)})
            covered = [i for i, (a, b) in enumerate(sentences) if a < hit.end and b > hit.start]
            if not covered:
                candidates.append(
                    (
                        hit.extraction_view_id,
                        hit.start,
                        hit.end,
                        (ExpansionSeed(chunk_id=hit.chunk_id, start=hit.start, end=hit.end, rank=hit.rank),),
                        ("seed_only",),
                    )
                )
                continue
            left, right = min(covered), max(covered)
            width = self.config.window
            flags = []
            while True:
                lo, hi = max(0, left - width), min(len(sentences) - 1, right + width)
                start, end = min(hit.start, sentences[lo][0]), max(hit.end, sentences[hi][1])
                if end - start <= self.config.max_window_chars:
                    break
                if width:
                    width -= 1
                    if "window_reduced" not in flags:
                        flags.append("window_reduced")
                    continue
                start, end = hit.start, hit.end
                flags.append("seed_only")
                break
            candidates.append(
                (
                    hit.extraction_view_id,
                    start,
                    end,
                    (ExpansionSeed(chunk_id=hit.chunk_id, start=hit.start, end=hit.end, rank=hit.rank),),
                    tuple(flags),
                )
            )
        merged = merge_overlapping(candidates)
        return assemble(
            result,
            merged,
            sources,
            technique_id="R21",
            rule="sentence-window",
            max_total_chars=self.config.max_total_chars,
            scoring="sentence-window",
        )
