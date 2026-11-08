"""Built-in dense retrieval and explicit no-grounding control."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexVersion, embedding_key_for
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_providers.ports import EmbeddingModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent


class DenseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)


class DenseRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R02",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=DenseConfig,
    )

    def __init__(
        self,
        config: DenseConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        embedder: EmbeddingModel,
    ) -> None:
        self.config, self._scope, self._version = config, scope, index_version
        self._index, self._embedder = index, embedder

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        profile = self._embedder.profile
        dim = profile.capabilities.embedding_dim
        if dim is None or embedding_key_for(profile.provider, profile.model, dim) != self._version.embedding_key:
            raise IndexingError("incompatible_embedding")
        embedded = await self._embedder.embed([query], input_type="query")
        if len(embedded.vectors) != 1:
            raise IndexingError("embedding_count_mismatch")
        results = await self._index.search(self._scope, self._version.id, embedded.vectors[0], self.config.k)
        hits = tuple(
            RetrievedChunk(
                chunk_id=item.chunk.id,
                extraction_view_id=item.chunk.extraction_view_id,
                start=item.chunk.start,
                end=item.chunk.end,
                block_ids=item.chunk.block_ids,
                text=item.chunk.text,
                distance=item.distance,
                score=1.0 - item.distance,
                rank=item.rank,
                score_components=(ScoreComponent(name="cosine_similarity", value=1.0 - item.distance),),
            )
            for item in results
        )
        return RetrievalResult(
            technique_id="R02",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=hits,
            scoring="cosine",
        )


class NoRetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NoRetrieval:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R01",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=NoRetrievalConfig,
    )

    def __init__(self, config: NoRetrievalConfig) -> None:
        self.config = config

    async def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(technique_id="R01", grounding="none", index_version_id=None, hits=())


def register_builtin_plugins(registry) -> None:
    registry.register(DenseRetriever)
    registry.register(NoRetrieval)
    from rag_foundry_indexing.bm25 import BM25Retriever

    registry.register(BM25Retriever)
