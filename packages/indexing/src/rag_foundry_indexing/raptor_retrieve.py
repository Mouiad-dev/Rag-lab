"""R37 tree retrieval (RAPTOR adaptation): summary nodes route the search, hits are original leaf chunks.

Two modes follow the paper. Collapsed: every node of every level is ranked by query similarity and the best
``top_nodes`` each contribute leaves. Traversal: start at the roots and keep the ``beam`` best nodes per level
down to the leaves. A summary is synthetic routing text. It is never a hit, a citation or answer context; each
hit is the stored source chunk with its original text and span, reached through lineage.

Ties are broken by (extraction view, start, end) of the node's first lineage span, then the node ID,
so results never depend on random IDs.
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexVersion, embedding_key_for
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_domain.summary_tree import SummaryTree
from rag_foundry_providers.ports import EmbeddingModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

_PAGE = 100


class RaptorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["collapsed", "traversal"] = "collapsed"
    top_nodes: int = Field(default=5, ge=1, le=20)
    leaves_per_node: int = Field(default=1, ge=1, le=10)
    beam: int = Field(default=2, ge=1, le=10)
    max_hits: int = Field(default=10, ge=1, le=50)


class RaptorTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["collapsed", "traversal"]
    selected_nodes: tuple[tuple[str, int, float, bool], ...]  # (node_id, level, similarity, synthetic)
    hit_lineage: tuple[tuple[str, str], ...]  # (chunk_id, via_node_id)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class RaptorRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R37",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=RaptorConfig,
    )

    def __init__(
        self,
        config: RaptorConfig,
        *,
        scope: Scope,
        tree: SummaryTree,
        index: IndexRepository,
        index_version: IndexVersion,
        embedder: EmbeddingModel,
    ) -> None:
        tree.check_scope(scope, index_version.id)
        profile = embedder.profile
        dim = profile.capabilities.embedding_dim
        key = None if dim is None else embedding_key_for(profile.provider, profile.model, dim)
        if key is None or key != index_version.embedding_key or key != tree.embedding_key:
            raise IndexingError("incompatible_embedding")
        self.config, self._scope, self._tree = config, scope, tree
        self._index, self._version, self._embedder = index, index_version, embedder
        self._nodes = {n.node_id: n for n in tree.nodes}
        self.last_trace: RaptorTrace | None = None

    def _order(self, node_id: str) -> tuple:
        """Tie-break key: (extraction view, start, end) of the first lineage span, then node ID."""
        first = self._tree.lineage(node_id)[0]
        return (first.extraction_view_id, first.start, first.end, node_id)

    def _rank(self, ids, sims: dict[str, float]) -> list[str]:
        return sorted(ids, key=lambda i: (-sims[i], self._order(i)))

    async def _chunks(self) -> dict:
        version = await self._index.get_index_version(self._scope, self._version.id)
        if getattr(version.state, "value", version.state) != "ready":
            raise IndexingError("not_ready")
        chunks, offset = {}, 0
        while True:
            page = await self._index.list_chunks(self._scope, self._version.id, offset=offset, limit=_PAGE)
            chunks.update({c.id: c for c in page})
            if len(page) < _PAGE:
                return chunks
            offset += _PAGE

    def _collapsed(self, sims: dict[str, float]) -> tuple[list[str], list[tuple[str, str]]]:
        selected = self._rank(sims, sims)[: self.config.top_nodes]
        picks: list[tuple[str, str]] = []  # (leaf node id, via node id)
        for node_id in selected:
            if self._nodes[node_id].kind == "leaf":
                picks.append((node_id, node_id))
            else:
                best = self._rank(self._tree.leaf_ids(node_id), sims)[: self.config.leaves_per_node]
                picks.extend((leaf, node_id) for leaf in best)
        return selected, picks

    def _traversal(self, sims: dict[str, float]) -> tuple[list[str], list[tuple[str, str]]]:
        selected: list[str] = []
        picks: list[tuple[str, str]] = []
        pool = [(root, root) for root in self._tree.roots()]  # (node, via)
        while pool:
            via_of = dict(pool)
            kept = self._rank(via_of, sims)[: self.config.beam]
            selected.extend(kept)
            pool = []
            for node_id in kept:
                if self._nodes[node_id].kind == "leaf":
                    picks.append((node_id, via_of[node_id]))
                else:
                    pool.extend((child, node_id) for child in self._tree.children_of(node_id))
        return selected, picks

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        chunks = await self._chunks()
        embedded = await self._embedder.embed([query], input_type="query")
        if len(embedded.vectors) != 1:
            raise IndexingError("embedding_count_mismatch")
        vector = tuple(embedded.vectors[0])
        if len(vector) != len(self._tree.nodes[0].embedding):
            raise IndexingError("incompatible_embedding")
        sims = {n.node_id: _cosine(vector, n.embedding) for n in self._tree.nodes}
        selected, picks = self._collapsed(sims) if self.config.mode == "collapsed" else self._traversal(sims)

        hits: list[RetrievedChunk] = []
        lineage: list[tuple[str, str]] = []
        seen: set[str] = set()
        for leaf_id, via in picks:
            span = self._nodes[leaf_id].span
            if span is None:
                raise IndexingError("tree_invalid")
            chunk = chunks.get(span.chunk_id)
            if chunk is None or (chunk.extraction_view_id, chunk.start, chunk.end) != (
                span.extraction_view_id,
                span.start,
                span.end,
            ):
                raise IndexingError("tree_source_mismatch")
            if chunk.id in seen or len(hits) >= self.config.max_hits:
                continue
            seen.add(chunk.id)
            via_node = self._nodes[via]
            leaf_sim = sims[leaf_id]
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    extraction_view_id=chunk.extraction_view_id,
                    start=chunk.start,
                    end=chunk.end,
                    block_ids=chunk.block_ids,
                    text=chunk.text,
                    score=leaf_sim,
                    rank=len(hits) + 1,
                    score_components=(
                        ScoreComponent(name="node_level", value=float(via_node.level)),
                        ScoreComponent(name="node_similarity", value=sims[via]),
                        ScoreComponent(name="leaf_similarity", value=leaf_sim),
                    ),
                    parent_start=chunk.parent_start_cp,
                    parent_end=chunk.parent_end_cp,
                )
            )
            lineage.append((chunk.id, via))
        self.last_trace = RaptorTrace(
            mode=self.config.mode,
            selected_nodes=tuple((i, self._nodes[i].level, sims[i], self._nodes[i].synthetic) for i in selected),
            hit_lineage=tuple(lineage),
        )
        return RetrievalResult(
            technique_id="R37",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=tuple(hits),
            scoring="raptor-tree",
        )
