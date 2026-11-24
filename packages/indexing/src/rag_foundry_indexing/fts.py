"""R04: PostgreSQL full-text retrieval (`to_tsvector`/`websearch_to_tsquery`, `ts_rank_cd`), not BM25.

The 'simple' configuration only lowercases: no stemming, no stop words, and no Arabic
normalization (diacritics, tatweel and alef forms are not folded). Scores are not comparable
across queries.
"""

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState, IndexVersion
from rag_foundry_domain.ports import FullTextIndex
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent


class FtsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=5, ge=1, le=50)
    operator: Literal["and", "or"] = "and"
    text_search_config: Literal["simple"] = "simple"


class FtsRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R04",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=FtsConfig,
    )

    def __init__(
        self,
        config: FtsConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: FullTextIndex,
    ) -> None:
        self.config, self._scope, self._version, self._index = config, scope, index_version, index

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        if self._version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        # Keep local tokenization aligned with the SQL OR normalization.
        if self.config.operator == "or" and not [
            term for term in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if term.lower() != "or"
        ]:
            raise IndexingError("no_query_terms")
        scored = await self._index.fts_search(
            self._scope, self._version.id, query, self.config.k, operator=self.config.operator
        )
        hits = tuple(
            RetrievedChunk(
                chunk_id=item.chunk.id,
                extraction_view_id=item.chunk.extraction_view_id,
                start=item.chunk.start,
                end=item.chunk.end,
                block_ids=item.chunk.block_ids,
                text=item.chunk.text,
                distance=None,
                score=item.ts_rank,
                rank=item.rank,
                score_components=(ScoreComponent(name="ts_rank_cd", value=item.ts_rank),),
                parent_start=getattr(item.chunk, "parent_start_cp", None),
                parent_end=getattr(item.chunk, "parent_end_cp", None),
            )
            for item in scored
        )
        return RetrievalResult(
            technique_id="R04",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=hits,
            scoring="pg-fts-simple-ts_rank_cd",
        )
