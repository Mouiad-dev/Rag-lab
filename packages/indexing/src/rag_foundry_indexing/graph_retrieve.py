"""R38 entity-graph retrieval: source chunks along the shortest relation path between two named entities.

This is graph traversal over a grounded entity/relation graph. It is not neighbor expansion (R21 and
friends widen adjacent text). Hits are the stored chunks that hold the relation evidence spans, with their
original text. A question naming exactly one known entity uses neighborhood mode; a question naming none
returns no hits with an explicit status, and nothing else is substituted.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.graphs import EntityGraph, GraphSpan, normalize_name
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

_PAGE = 100


class EntityGraphConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_hops: int = Field(default=3, ge=1, le=4)
    neighborhood_hops: int = Field(default=1, ge=1, le=2)
    max_matched_entities: int = Field(default=5, ge=1, le=10)
    max_hits: int = Field(default=10, ge=1, le=50)


class GraphRetrievalTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["ok", "no_entity_match", "no_path_within_hops"]
    mode: Literal["path", "neighborhood", "none"]
    matched_entity_ids: tuple[str, ...]
    path: tuple[tuple[str, Literal["forward", "reverse"]], ...]
    evidence: tuple[GraphSpan, ...]


class EntityGraphRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R38",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=EntityGraphConfig,
    )

    def __init__(
        self,
        config: EntityGraphConfig,
        *,
        scope: Scope,
        graph: EntityGraph,
        index: IndexRepository,
        index_version: IndexVersion,
    ) -> None:
        graph.check_scope(scope, index_version.id)
        self.config, self._scope, self._graph = config, scope, graph
        self._index, self._version = index, index_version
        self.last_trace: GraphRetrievalTrace | None = None

    def _match(self, query: str) -> list[str]:
        text = normalize_name(query)
        found: list[tuple[int, int, str]] = []  # (start, end, entity_id)
        for entity in sorted(self._graph.entities, key=lambda e: (-len(normalize_name(e.name)), e.entity_id)):
            name = normalize_name(entity.name)
            at = text.find(name)
            if at < 0 or any(at < b and a < at + len(name) for a, b, _ in found):
                continue
            if len(found) >= self.config.max_matched_entities:
                break
            found.append((at, at + len(name), entity.entity_id))
        return [eid for _, _, eid in sorted(found)]

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

    def _neighborhood(self, entity_id: str) -> list[tuple[str, int]]:
        """Relation ids reachable within the configured hops, with the hop at which each is first seen."""
        seen_entities, seen_relations = {entity_id}, {}
        frontier = [entity_id]
        for hop in range(1, self.config.neighborhood_hops + 1):
            following = []
            for node in frontier:
                for rel_id, other, _ in self._graph.neighbors(node):
                    seen_relations.setdefault(rel_id, hop)
                    if other not in seen_entities:
                        seen_entities.add(other)
                        following.append(other)
            frontier = following
        return sorted(seen_relations.items(), key=lambda item: (item[1], item[0]))

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        chunks = await self._chunks()
        matched = self._match(query)
        path: tuple = ()
        ordered: list[tuple[str, int]] = []  # (relation_id, hop)
        status: str = "ok"
        mode = "path" if len(matched) >= 2 else "neighborhood" if matched else "none"
        if not matched:
            status = "no_entity_match"
        elif len(matched) >= 2:
            found = self._graph.shortest_path(matched[0], matched[1], self.config.max_hops)
            if found is None:
                status = "no_path_within_hops"
            else:
                path = found
                ordered = [(rel_id, i) for i, (rel_id, _) in enumerate(found, 1)]
        else:
            ordered = self._neighborhood(matched[0])

        relations = {r.relation_id: r for r in self._graph.relations}
        evidence: list[GraphSpan] = []
        hits: list[RetrievedChunk] = []
        seen_chunks: set[str] = set()
        for rel_id, hop in ordered:
            for span in relations[rel_id].evidence:
                chunk = chunks.get(span.chunk_id)
                if chunk is None:
                    raise IndexingError("graph_evidence_missing")
                evidence.append(span)
                if chunk.id in seen_chunks or len(hits) >= self.config.max_hits:
                    continue
                seen_chunks.add(chunk.id)
                score = 1.0 if path else 1.0 / (1 + hop)
                hits.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        extraction_view_id=chunk.extraction_view_id,
                        start=chunk.start,
                        end=chunk.end,
                        block_ids=chunk.block_ids,
                        text=chunk.text,
                        score=score,
                        rank=len(hits) + 1,
                        score_components=(
                            ScoreComponent(name="graph_hop", value=float(hop)),
                            ScoreComponent(name="path_edges", value=float(len(path))),
                        ),
                        parent_start=chunk.parent_start_cp,
                        parent_end=chunk.parent_end_cp,
                    )
                )
        self.last_trace = GraphRetrievalTrace(
            status=status,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            matched_entity_ids=tuple(matched),
            path=tuple(path),
            evidence=tuple(evidence),
        )
        return RetrievalResult(
            technique_id="R38",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=tuple(hits),
            scoring="entity-graph",
        )
