"""Hybrid lexical and dense retrieval with explicit rank or weighted fusion."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState, IndexVersion
from rag_foundry_domain.ports import FullTextIndex, IndexRepository
from rag_foundry_providers.ports import EmbeddingModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest

from rag_foundry_indexing.bm25 import BM25Config, BM25Retriever
from rag_foundry_indexing.fts import FtsConfig, FtsRetriever
from rag_foundry_indexing.fusion import RrfConfig, RrfFusion, WeightedConfig, WeightedFusion
from rag_foundry_indexing.retrievers import DenseConfig, DenseRetriever


class HybridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=5, ge=1, le=50)
    branch_k: int = Field(default=20, ge=1, le=50)
    lexical: Literal["R03", "R04"] = "R03"
    fts_operator: Literal["and", "or"] = "or"
    fusion: Literal["R07", "R08"] = "R07"
    rrf_k: int = Field(default=60, ge=1, le=1000)
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class HybridRetriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R06",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=HybridConfig,
    )

    def __init__(
        self,
        config: HybridConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        embedder: EmbeddingModel,
        full_text: FullTextIndex | None = None,
    ) -> None:
        if config.lexical == "R04" and full_text is None:
            raise IndexingError("fts_unavailable")
        self.config, self._scope, self._version = config, scope, index_version
        self._index = index
        if config.lexical == "R03":
            self._lexical = BM25Retriever(
                BM25Config(k=config.branch_k), scope=scope, index_version=index_version, index=index
            )
        else:
            self._lexical = FtsRetriever(
                FtsConfig(k=config.branch_k, operator=config.fts_operator),
                scope=scope,
                index_version=index_version,
                index=full_text,
            )
        self._dense = DenseRetriever(
            DenseConfig(k=config.branch_k),
            scope=scope,
            index_version=index_version,
            index=index,
            embedder=embedder,
        )

    async def retrieve(self, query: str):
        version = await self._index.get_index_version(self._scope, self._version.id)
        if version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        lexical_result = await self._lexical.retrieve(query)
        dense_result = await self._dense.retrieve(query)
        if self.config.fusion == "R07":
            result = RrfFusion(RrfConfig(k=self.config.k, rrf_k=self.config.rrf_k)).fuse([lexical_result, dense_result])
            scoring = f"hybrid[{self.config.lexical}+R02]/R07:rrf-k{self.config.rrf_k}"
        else:
            result = WeightedFusion(
                WeightedConfig(
                    k=self.config.k,
                    weights={self.config.lexical: self.config.lexical_weight, "R02": 1 - self.config.lexical_weight},
                )
            ).fuse([lexical_result, dense_result])
            scoring = f"hybrid[{self.config.lexical}+R02]/R08:weighted-minmax:wl={self.config.lexical_weight:.2f}"
        return result.model_copy(update={"technique_id": "R06", "scoring": scoring})
