"""R25 hierarchical auto-merging over the two C18 levels (child chunk, parent span)."""

from __future__ import annotations

from typing import ClassVar

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

_PAGE = 500


class AutoMergeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    merge_ratio: float = Field(default=0.5, gt=0, le=1)
    min_children: int = Field(default=2, ge=1, le=10)
    max_parent_chars: int = Field(default=4000, ge=20, le=20000)
    max_total_chars: int = Field(default=8000, ge=20, le=100000)
    max_seeds: int = Field(default=20, ge=1, le=50)
    max_index_chunks: int = Field(default=5000, ge=1, le=20000)


class AutoMergingExpander:
    """Replace retrieved children by their parent only when a stated ratio rule holds.

    Two levels only (child chunk, C18 parent span). For a parent with m retrieved children out of
    n children in the index version, merge if and only if ``m / n >= merge_ratio`` and
    ``m >= min_children`` and the parent length is ``<= max_parent_chars``. Otherwise the children
    keep their own spans. This is a ratio rule over two levels, not a multi-level tree merge.
    """

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R25",
        kind=PluginKind.CONTEXT_BUILDER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=AutoMergeConfig,
    )

    def __init__(
        self,
        config: AutoMergeConfig,
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
        self._counts: dict[str, dict[tuple[str, int, int], int]] = {}

    async def _child_counts(self, version: IndexVersion) -> dict[tuple[str, int, int], int]:
        if version.id in self._counts:
            return self._counts[version.id]
        if version.chunk_count > self.config.max_index_chunks:
            raise IndexingError("auto_merge_index_too_large")
        counts: dict[tuple[str, int, int], int] = {}
        offset = 0
        while True:
            page = await self.index.list_chunks(self.scope, version.id, offset=offset, limit=_PAGE)
            for chunk in page:
                if chunk.parent_start_cp is not None and chunk.parent_end_cp is not None:
                    key = (chunk.extraction_view_id, chunk.parent_start_cp, chunk.parent_end_cp)
                    counts[key] = counts.get(key, 0) + 1
            if len(page) < _PAGE:
                break
            offset += _PAGE
        self._counts[version.id] = counts
        return counts

    async def expand(self, result: RetrievalResult) -> ExpandedRetrieval:
        version, sources = await load_sources(
            result,
            scope=self.scope,
            index=self.index,
            index_version=self.index_version,
            source=self.source,
            max_seeds=self.config.max_seeds,
        )
        counts = await self._child_counts(version)
        groups: dict[tuple[str, int, int], list] = {}
        for hit in result.hits:
            if hit.parent_start is not None and hit.parent_end is not None:
                if hit.parent_end > len(sources[hit.extraction_view_id].text):
                    raise IndexingError("expansion_source_mismatch")
                groups.setdefault((hit.extraction_view_id, hit.parent_start, hit.parent_end), []).append(hit)
        candidates = []
        merged_keys: set[tuple[str, int, int]] = set()
        for hit in result.hits:
            seed = (ExpansionSeed(chunk_id=hit.chunk_id, start=hit.start, end=hit.end, rank=hit.rank),)
            if hit.parent_start is None or hit.parent_end is None:
                candidates.append((hit.extraction_view_id, hit.start, hit.end, seed, ()))
                continue
            key = (hit.extraction_view_id, hit.parent_start, hit.parent_end)
            members = list({h.chunk_id: h for h in groups[key]}.values())
            m, n = len(members), counts.get(key, 0)
            if n < m:
                raise IndexingError("expansion_source_mismatch")
            merge = (
                m / n >= self.config.merge_ratio
                and m >= self.config.min_children
                and hit.parent_end - hit.parent_start <= self.config.max_parent_chars
            )
            if not merge:
                candidates.append((hit.extraction_view_id, hit.start, hit.end, seed, (f"not_merged:{m}/{n}",)))
            elif key not in merged_keys:
                merged_keys.add(key)
                seeds = tuple(
                    ExpansionSeed(chunk_id=h.chunk_id, start=h.start, end=h.end, rank=h.rank)
                    for h in sorted(members, key=lambda h: h.rank)
                )
                candidates.append(
                    (hit.extraction_view_id, hit.parent_start, hit.parent_end, seeds, (f"merged:{m}/{n}",))
                )
        return assemble(
            result,
            merge_overlapping(candidates),
            sources,
            technique_id="R25",
            rule="auto-merge",
            max_total_chars=self.config.max_total_chars,
            scoring="auto-merge",
        )
