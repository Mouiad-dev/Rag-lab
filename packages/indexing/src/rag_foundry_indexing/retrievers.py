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
                parent_start=getattr(item.chunk, "parent_start_cp", None),
                parent_end=getattr(item.chunk, "parent_end_cp", None),
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
    from rag_foundry_indexing.fts import FtsRetriever

    registry.register(BM25Retriever)
    registry.register(FtsRetriever)
    from rag_foundry_indexing.sparse import LearnedSparseRetriever

    registry.register(LearnedSparseRetriever)
    from rag_foundry_indexing.hybrid import HybridRetriever

    registry.register(HybridRetriever)
    from rag_foundry_indexing.fusion import RrfFusion, WeightedFusion

    registry.register(RrfFusion)
    registry.register(WeightedFusion)
    from rag_foundry_indexing.selection import EvidenceDedup

    registry.register(EvidenceDedup)
    from rag_foundry_indexing.metadata_filter import MetadataFilteredRetriever

    registry.register(MetadataFilteredRetriever)
    from rag_foundry_indexing.mmr import MmrSelector

    registry.register(MmrSelector)
    from rag_foundry_indexing.self_query import SelfQuery

    registry.register(SelfQuery)
    from rag_foundry_indexing.rewrite import QuestionRewriter

    registry.register(QuestionRewriter)
    from rag_foundry_indexing.conversation import ConversationalRewriter

    registry.register(ConversationalRewriter)
    from rag_foundry_indexing.multi_query import MultiQueryExpansion
    from rag_foundry_indexing.rag_fusion import RagFusion

    registry.register(MultiQueryExpansion)
    registry.register(RagFusion)
    from rag_foundry_indexing.decompose import QuestionDecomposition
    from rag_foundry_indexing.step_back import StepBackRetrieval

    registry.register(QuestionDecomposition)
    registry.register(StepBackRetrieval)
    from rag_foundry_indexing.sentence_window import SentenceWindowExpander

    registry.register(SentenceWindowExpander)
    from rag_foundry_indexing.auto_merge import AutoMergingExpander
    from rag_foundry_indexing.parent_document import ParentDocumentExpander

    registry.register(ParentDocumentExpander)
    registry.register(AutoMergingExpander)
    from rag_foundry_indexing.hyde import HydeRetriever

    registry.register(HydeRetriever)
    from rag_foundry_indexing.representations import ContextualRetriever, MultiRepresentationRetriever

    registry.register(MultiRepresentationRetriever)
    registry.register(ContextualRetriever)
    from rag_foundry_indexing.cross_encoder_rerank import CrossEncoderReranker
    from rag_foundry_indexing.llm_rerank import LlmListwiseReranker

    registry.register(LlmListwiseReranker)
    registry.register(CrossEncoderReranker)
    from rag_foundry_indexing.compression import ExtractiveCompressor

    registry.register(ExtractiveCompressor)
    from rag_foundry_indexing.packing import ContextPacker

    registry.register(ContextPacker)
    from rag_foundry_indexing.graph_retrieve import EntityGraphRetriever

    registry.register(EntityGraphRetriever)
    from rag_foundry_indexing.graph_global import CommunityGlobalSearch

    registry.register(CommunityGlobalSearch)
