"""Rank fusion plugins for results from one index version."""

# ruff: noqa: E501

from collections import OrderedDict
from collections.abc import Sequence
from math import isfinite
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rag_foundry_domain.errors import IndexingError
from rag_foundry_techniques import (
    Fidelity,
    PluginKind,
    PluginManifest,
    RetrievalResult,
    RetrievedChunk,
    ScoreComponent,
)


def check_inputs(results: Sequence[RetrievalResult]) -> None:
    if len(results) < 2:
        raise IndexingError("fusion_too_few_branches")
    ids = [result.technique_id for result in results]
    if len(set(ids)) != len(ids):
        raise IndexingError("fusion_duplicate_branch")
    versions = [result.index_version_id for result in results]
    if (
        any(result.grounding != "corpus" for result in results)
        or any(v is None for v in versions)
        or len(set(versions)) != 1
    ):
        raise IndexingError("fusion_version_mismatch")
    candidates = {}
    for result in results:
        ranks = [hit.rank for hit in result.hits]
        if ranks != list(range(1, len(ranks) + 1)):
            raise IndexingError("fusion_invalid_ranks")
        seen = set()
        for hit in result.hits:
            if hit.chunk_id in seen:
                raise IndexingError("fusion_duplicate_candidate")
            seen.add(hit.chunk_id)
            fields = (hit.extraction_view_id, hit.start, hit.end, hit.text, hit.parent_start, hit.parent_end)
            if hit.chunk_id in candidates and candidates[hit.chunk_id] != fields:
                raise IndexingError("fusion_candidate_conflict")
            candidates[hit.chunk_id] = fields


class RrfConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)


class WeightedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    k: int = Field(default=5, ge=1, le=50)
    weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def valid_weights(cls, value):
        if any(not isfinite(weight) or weight < 0 for weight in value.values()):
            raise ValueError("weights must be finite and nonnegative")
        if value and sum(value.values()) <= 0:
            raise ValueError("weights must have a positive sum")
        return value


class RrfFusion:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R07", kind=PluginKind.FUSION, version="1.0.0", fidelity=Fidelity.FAITHFUL, config_model=RrfConfig
    )

    def __init__(self, config: RrfConfig):
        self.config = config

    def fuse(self, results: Sequence[RetrievalResult]) -> RetrievalResult:
        check_inputs(results)
        records = _records(results)
        for result in results:
            for hit in result.hits:
                records[hit.chunk_id]["score"] += 1 / (self.config.rrf_k + hit.rank)
                records[hit.chunk_id]["rank"] = min(records[hit.chunk_id]["rank"], hit.rank)
                records[hit.chunk_id]["components"].extend(_tag(hit, result.technique_id))
                records[hit.chunk_id]["components"].append(
                    ScoreComponent(
                        name="rrf_contribution",
                        value=1 / (self.config.rrf_k + hit.rank),
                        source=result.technique_id,
                        source_rank=hit.rank,
                    )
                )
        return _result(records, results, "R07", f"rrf-k{self.config.rrf_k}", self.config.k)


class WeightedFusion:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R08",
        kind=PluginKind.FUSION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=WeightedConfig,
    )

    def __init__(self, config: WeightedConfig):
        self.config = config

    def fuse(self, results: Sequence[RetrievalResult]) -> RetrievalResult:
        check_inputs(results)
        ids = {result.technique_id for result in results}
        if self.config.weights and set(self.config.weights) != ids:
            raise IndexingError("fusion_weights_mismatch")
        # Configured weights are used as given (score = sum of weight * norm); no config means 1/n each.
        weights = self.config.weights or {result.technique_id: 1 / len(results) for result in results}
        records = _records(results)
        for result in results:
            scores = [hit.score for hit in result.hits]
            low, high = (min(scores), max(scores)) if scores else (0, 0)
            degenerate = len(scores) <= 1 or low == high
            for hit in result.hits:
                norm = 1.0 if degenerate else (hit.score - low) / (high - low)
                rec = records[hit.chunk_id]
                rec["rank"] = min(rec["rank"], hit.rank)
                rec["components"].extend(_tag(hit, result.technique_id))
                if degenerate:
                    rec["components"].append(
                        ScoreComponent(name="normalization_degenerate", value=1.0, source=result.technique_id)
                    )
                rec["score"] += weights[result.technique_id] * norm
                rec["components"].append(
                    ScoreComponent(name="min_max_score", value=norm, source=result.technique_id, source_rank=hit.rank)
                )
                rec["components"].append(
                    ScoreComponent(
                        name="weighted_contribution",
                        value=weights[result.technique_id] * norm,
                        source=result.technique_id,
                        source_rank=hit.rank,
                    )
                )
        return _result(records, results, "R08", "weighted-minmax", self.config.k)


def _records(results):
    records = OrderedDict()
    for result in results:
        for hit in result.hits:
            records.setdefault(hit.chunk_id, {"hit": hit, "score": 0.0, "rank": hit.rank, "components": []})
    return records


def _tag(hit, source):
    return [
        component.model_copy(update={"source": source, "source_rank": hit.rank}) for component in hit.score_components
    ]


def _result(records, results, technique_id, scoring, k):
    ordered = sorted(records.items(), key=lambda item: (-round(item[1]["score"], 12), item[1]["rank"], item[0]))
    hits = []
    for rank, (chunk_id, record) in enumerate(ordered[:k], start=1):
        hit = record["hit"]
        hits.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                extraction_view_id=hit.extraction_view_id,
                start=hit.start,
                end=hit.end,
                block_ids=hit.block_ids,
                text=hit.text,
                distance=None,
                score=record["score"],
                rank=rank,
                score_components=tuple(record["components"]),
                parent_start=hit.parent_start,
                parent_end=hit.parent_end,
            )
        )
    return RetrievalResult(
        technique_id=technique_id,
        grounding="corpus",
        index_version_id=results[0].index_version_id,
        hits=tuple(hits),
        scoring=scoring,
    )
