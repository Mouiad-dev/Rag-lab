"""R19 local cross-encoder reranking after retrieval."""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.errors import IndexingError
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.ports import Reranker
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, ScoreComponent
from rag_foundry_techniques.errors import TechniqueUnavailableError

from rag_foundry_indexing.rerank import RerankedRetrieval, RerankTrace, apply_order, check_rerank_input


class CrossEncoderRerankConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)
    max_candidates: int = Field(default=50, ge=1, le=100)


class CrossEncoderReranker:
    """Joint query-passage scoring follows the mechanism in arXiv 1901.04085; this model differs."""

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R19",
        kind=PluginKind.RERANKER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=CrossEncoderRerankConfig,
    )

    def __init__(self, config: CrossEncoderRerankConfig, *, reranker: Reranker | None = None) -> None:
        self.config = config
        self._reranker = reranker

    async def rerank(self, question: str, result: RetrievalResult) -> RerankedRetrieval:
        if not question.strip():
            raise IndexingError("empty_query")
        if len(question) > 2000:
            raise IndexingError("query_too_long")
        check_rerank_input(result, self.config.max_candidates)
        if self._reranker is None:
            raise TechniqueUnavailableError("R19 needs the [sparse] extra and an approved local cross-encoder model")
        if self._reranker.profile.kind.value != "reranker":
            raise IndexingError("rerank_wrong_model_kind")
        if not result.hits:
            return RerankedRetrieval(
                trace=RerankTrace(
                    technique_id="R19",
                    input_technique_id=result.technique_id,
                    status="applied",
                    reason=None,
                    profile_id=self._reranker.profile.id,
                    model=self._reranker.profile.model,
                    input_count=0,
                    output_count=0,
                    dropped=(),
                    usage=(),
                ),
                input=result,
                result=result,
                changes=(),
            )
        try:
            ranked = await self._reranker.rerank(question, [hit.text for hit in result.hits], top_n=self.config.k)
        except AdapterDependencyError as exc:
            raise TechniqueUnavailableError(
                "R19 needs the [sparse] extra and an approved local cross-encoder model"
            ) from exc
        items = ranked.items
        indexes = [item.index for item in items]
        scores = [item.score for item in items]
        if (
            len(set(indexes)) != len(indexes)
            or any(i < 0 or i >= len(result.hits) for i in indexes)
            or not all(math.isfinite(score) for score in scores)
            or any(
                first.score < second.score or (first.score == second.score and first.index > second.index)
                for first, second in zip(items, items[1:], strict=False)
            )
        ):
            raise IndexingError("rerank_invalid_order")
        order = [(item.index, item.score, (ScoreComponent(name="rerank_score", value=item.score),)) for item in items]
        output, changes, dropped = apply_order(result, order, technique_id="R19", scoring="cross-encoder")
        changes_by_id = {change.chunk_id: change for change in changes}
        changes = tuple(changes_by_id[hit.chunk_id] for hit in output.hits) + tuple(
            change for change in changes if change.chunk_id in dropped
        )
        profile = self._reranker.profile
        trace = RerankTrace(
            technique_id="R19",
            input_technique_id=result.technique_id,
            status="applied",
            reason=None,
            profile_id=profile.id,
            model=profile.model,
            input_count=len(result.hits),
            output_count=len(output.hits),
            dropped=dropped,
            usage=(ranked.usage,),
        )
        return RerankedRetrieval(trace=trace, input=result, result=output, changes=changes)
