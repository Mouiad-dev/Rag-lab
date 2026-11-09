"""Maximal Marginal Relevance result selection (R09)."""

from collections import Counter
from collections.abc import Sequence
from math import sqrt
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.ports import ChunkVectorReader, IndexRepository
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.lexical_text import lexical_tokens
from rag_foundry_indexing.selection import check_single_input


class MmrConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=5, ge=1, le=50)
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity: Literal["embedding", "lexical"] = "embedding"


class MmrSelector:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R09",
        kind=PluginKind.FUSION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=MmrConfig,
    )

    def __init__(
        self,
        config: MmrConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
        vectors: ChunkVectorReader | None = None,
    ) -> None:
        if config.similarity == "embedding" and vectors is None:
            raise IndexingError("selection_vectors_unavailable")
        self.config, self._scope, self._version = config, scope, index_version
        self._index, self._vectors = index, vectors

    async def select(self, result: RetrievalResult) -> RetrievalResult:
        check_single_input(result)
        if result.index_version_id != self._version.id:
            raise IndexingError("selection_version_mismatch")
        hits = result.hits
        vectors: dict[str, tuple[float, ...]] = {}
        if self.config.similarity == "embedding":
            await self._index.get_index_version(self._scope, self._version.id)
            vectors = await self._vectors.chunk_embeddings(
                self._scope, self._version.id, tuple(hit.chunk_id for hit in hits)
            )
            if any(hit.chunk_id not in vectors for hit in hits):
                raise IndexingError("selection_missing_vector")
        scores = [hit.score for hit in hits]
        low, high = min(scores, default=0.0), max(scores, default=0.0)
        relevance = [1.0 if high == low else (score - low) / (high - low) for score in scores]
        lexical = [Counter(lexical_tokens(hit.text)) for hit in hits] if self.config.similarity == "lexical" else []
        selected: list[int] = []
        mmr_values: dict[int, tuple[float, float, float]] = {}
        while len(selected) < min(self.config.k, len(hits)):
            options = []
            for i in range(len(hits)):
                if i in selected:
                    continue
                maximum = max(
                    (
                        _cosine(vectors[hits[i].chunk_id], vectors[hits[j].chunk_id])
                        if not lexical
                        else _counter_cosine(lexical[i], lexical[j])
                        for j in selected
                    ),
                    default=0.0,
                )
                score = self.config.lambda_mult * relevance[i] - (1.0 - self.config.lambda_mult) * maximum
                options.append((round(score, 12), -i, i, maximum, score))
            _, _, picked, maximum, score = max(options)
            selected.append(picked)
            mmr_values[picked] = (relevance[picked], maximum, score)
        output = []
        for rank, i in enumerate(selected, 1):
            hit = hits[i]
            rel, maximum, score = mmr_values[i]
            components = (
                *hit.score_components,
                ScoreComponent(
                    name="input_rank", value=float(hit.rank), source=result.technique_id, source_rank=hit.rank
                ),
                ScoreComponent(name="mmr_relevance", value=rel),
                ScoreComponent(name="mmr_max_similarity", value=maximum),
                ScoreComponent(name="mmr_score", value=score),
            )
            output.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    extraction_view_id=hit.extraction_view_id,
                    start=hit.start,
                    end=hit.end,
                    block_ids=hit.block_ids,
                    text=hit.text,
                    distance=hit.distance,
                    score=score,
                    rank=rank,
                    score_components=components,
                    parent_start=hit.parent_start,
                    parent_end=hit.parent_end,
                )
            )
        return RetrievalResult(
            technique_id="R09",
            grounding="corpus",
            index_version_id=result.index_version_id,
            hits=tuple(output),
            scoring=f"mmr[{result.technique_id}]:{self.config.similarity}:lambda={self.config.lambda_mult:.2f}",
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(left, right, strict=True)) / denominator


def _counter_cosine(left: Counter, right: Counter) -> float:
    denominator = sqrt(sum(x * x for x in left.values()) * sum(x * x for x in right.values()))
    return 0.0 if denominator == 0 else sum(value * right.get(key, 0) for key, value in left.items()) / denominator
