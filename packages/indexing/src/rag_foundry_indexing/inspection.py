"""Read-only, scope-checked retrieval stage inspection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rag_foundry_domain.errors import IndexingError
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

from rag_foundry_indexing.rerank import RankChange, rank_changes


class InspectionStage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["retrieve", "fuse", "select", "transform", "rerank"]
    technique_id: str
    scoring: str | None
    candidates: tuple[RetrievedChunk, ...]
    changes: tuple[RankChange, ...]


class RetrievalInspection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inspection_version: Literal["1"] = "1"
    index_version_id: str
    question: str
    stages: tuple[InspectionStage, ...]


async def inspect_retrieval(
    scope, index, question: str, stages: Sequence[tuple[str, RetrievalResult]]
) -> RetrievalInspection:
    if not stages:
        raise IndexingError("inspection_empty")
    version_id = stages[0][1].index_version_id
    if any(result.grounding != "corpus" or result.index_version_id != version_id for _, result in stages):
        raise IndexingError("inspection_version_mismatch")
    await index.get_index_version(scope, version_id)
    output = []
    previous = None
    for name, result in stages:
        changes = () if previous is None else rank_changes(previous, result, allow_added=name != "rerank")
        output.append(
            InspectionStage(
                name=name,
                technique_id=result.technique_id,
                scoring=result.scoring,
                candidates=result.hits,
                changes=changes,
            )
        )
        previous = result
    return RetrievalInspection(index_version_id=version_id, question=question, stages=tuple(output))
