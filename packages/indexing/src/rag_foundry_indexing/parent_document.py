"""R22 parent document retrieval: return the stored C18 parent span of each child hit."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
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


class ParentDocumentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_parent_chars: int = Field(default=4000, ge=20, le=20000)
    max_total_chars: int = Field(default=8000, ge=20, le=100000)
    max_seeds: int = Field(default=20, ge=1, le=50)
    on_missing_parent: Literal["error", "keep_child"] = "error"


class ParentDocumentExpander:
    """Replace each child hit by its stored parent span.

    Children of the same parent collapse into one output at the best child's rank. A parent longer
    than ``max_parent_chars`` keeps the child (flag ``parent_too_large``). Text always comes from the view.
    """

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R22",
        kind=PluginKind.CONTEXT_BUILDER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ParentDocumentConfig,
    )

    def __init__(
        self,
        config: ParentDocumentConfig,
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
            seed = (ExpansionSeed(chunk_id=hit.chunk_id, start=hit.start, end=hit.end, rank=hit.rank),)
            if hit.parent_start is None or hit.parent_end is None:
                if self.config.on_missing_parent == "error":
                    raise IndexingError("parent_unavailable")
                candidates.append((hit.extraction_view_id, hit.start, hit.end, seed, ("no_parent",)))
                continue
            if hit.parent_end > len(sources[hit.extraction_view_id].text):
                raise IndexingError("expansion_source_mismatch")
            if hit.parent_end - hit.parent_start > self.config.max_parent_chars:
                candidates.append((hit.extraction_view_id, hit.start, hit.end, seed, ("parent_too_large",)))
                continue
            candidates.append((hit.extraction_view_id, hit.parent_start, hit.parent_end, seed, ()))
        return assemble(
            result,
            merge_overlapping(candidates),
            sources,
            technique_id="R22",
            rule="parent-document",
            max_total_chars=self.config.max_total_chars,
            scoring="parent-document",
        )
