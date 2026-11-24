"""R05 learned sparse indexing and retrieval; test doubles are never registered here."""

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState, IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.ports import SparseEncoder
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent
from rag_foundry_techniques.errors import TechniqueUnavailableError


def sparse_encoder_key(encoder: SparseEncoder) -> str:
    return hashlib.sha256(
        json.dumps([encoder.profile.provider, encoder.profile.model], separators=(",", ":")).encode()
    ).hexdigest()


class SparseIndexService:
    def __init__(self, index: IndexRepository) -> None:
        self._index = index

    async def build(self, scope: Scope, version: IndexVersion, encoder: SparseEncoder) -> dict[str, int]:
        key = sparse_encoder_key(encoder)
        offset = 0
        stats = {"chunks": 0, "nonzero": 0, "terms": 0}
        while True:
            page = await self._index.list_chunks(scope, version.id, offset=offset, limit=500)
            if not page:
                break
            result = await encoder.encode([chunk.text for chunk in page], input_type="document")
            if len(result.vectors) != len(page):
                raise IndexingError("sparse_encoding_count_mismatch")
            rows = tuple((chunk.id, vector) for chunk, vector in zip(page, result.vectors, strict=True))
            await self._index.add_sparse(
                scope,
                version.id,
                key,
                tuple((cid, (vector.terms, vector.weights)) for cid, vector in rows),
            )
            stats["chunks"] += len(page)
            stats["nonzero"] += sum(bool(vector.terms) for vector in result.vectors)
            stats["terms"] += sum(len(vector.terms) for vector in result.vectors)
            offset += len(page)
            if len(page) < 500:
                break
        return stats


class SparseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)


class LearnedSparseRetriever:
    """R05; query encoding uses static vocabulary weights and does not run transformer inference."""

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R05",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=SparseConfig,
    )

    def __init__(
        self,
        config: SparseConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        encoder: SparseEncoder | None = None,
    ) -> None:
        self.config, self._scope, self._version, self._index, self._encoder = (
            config,
            scope,
            index_version,
            index,
            encoder,
        )

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        if self._version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        if self._encoder is None:
            raise TechniqueUnavailableError("R05 needs the [sparse] extra and approved local model files")
        key = sparse_encoder_key(self._encoder)
        try:
            result = await self._encoder.encode([query], input_type="query")
        except AdapterDependencyError as exc:
            raise TechniqueUnavailableError("R05 needs the [sparse] extra and approved local model files") from exc
        if len(result.vectors) != 1:
            raise IndexingError("sparse_encoding_count_mismatch")
        vector = result.vectors[0]
        scored = await self._index.sparse_search(
            self._scope, self._version.id, key, (vector.terms, vector.weights), self.config.k
        )
        hits = tuple(
            RetrievedChunk(
                chunk_id=x.chunk.id,
                extraction_view_id=x.chunk.extraction_view_id,
                start=x.chunk.start,
                end=x.chunk.end,
                block_ids=x.chunk.block_ids,
                text=x.chunk.text,
                distance=None,
                score=x.score,
                rank=x.rank,
                score_components=tuple(ScoreComponent(name="sparse_term", term=t, value=w) for t, w in x.terms),
                parent_start=getattr(x.chunk, "parent_start_cp", None),
                parent_end=getattr(x.chunk, "parent_end_cp", None),
            )
            for x in scored
        )
        return RetrievalResult(
            technique_id="R05",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=hits,
            scoring="learned-sparse-dot",
        )
