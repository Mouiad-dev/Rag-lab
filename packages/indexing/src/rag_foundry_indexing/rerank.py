"""Shared reranking primitives for ordered retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rag_foundry_domain.errors import IndexingError
from rag_foundry_providers.usage import Usage
from rag_foundry_techniques import RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.selection import check_single_input


class RankChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    extraction_view_id: str
    start: int
    end: int
    before_rank: int | None
    after_rank: int | None
    before_score: float | None
    after_score: float | None
    moved: int | None


class RerankTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    technique_id: str
    input_technique_id: str
    status: Literal["applied", "rejected"]
    reason: str | None
    profile_id: str
    model: str
    input_count: int
    output_count: int
    dropped: tuple[str, ...]
    unranked: tuple[str, ...] = ()
    usage: tuple[Usage, ...]
    prompt_sha256: str | None = None


class RerankedRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: RerankTrace
    input: RetrievalResult
    result: RetrievalResult
    changes: tuple[RankChange, ...]


def rank_changes(before: RetrievalResult, after: RetrievalResult, *, allow_added: bool) -> tuple[RankChange, ...]:
    old = {hit.chunk_id: hit for hit in before.hits}
    new = {hit.chunk_id: hit for hit in after.hits}
    rows: list[RankChange] = []
    for hit in before.hits:
        other = new.get(hit.chunk_id)
        if other is not None and (hit.extraction_view_id, hit.start, hit.end) != (
            other.extraction_view_id,
            other.start,
            other.end,
        ):
            raise IndexingError("rerank_added_candidate")
        rows.append(_change(hit, other))
    for hit in after.hits:
        if hit.chunk_id not in old:
            if not allow_added:
                raise IndexingError("rerank_added_candidate")
            rows.append(_change(None, hit))
    return tuple(rows)


def _change(before: RetrievedChunk | None, after: RetrievedChunk | None) -> RankChange:
    hit = before or after
    assert hit is not None
    old_rank = before.rank if before else None
    new_rank = after.rank if after else None
    return RankChange(
        chunk_id=hit.chunk_id,
        extraction_view_id=hit.extraction_view_id,
        start=hit.start,
        end=hit.end,
        before_rank=old_rank,
        after_rank=new_rank,
        before_score=before.score if before else None,
        after_score=after.score if after else None,
        moved=old_rank - new_rank if old_rank is not None and new_rank is not None else None,
    )


def apply_order(
    input: RetrievalResult,
    order: Sequence[tuple[int, float, tuple[ScoreComponent, ...]]],
    *,
    technique_id: str,
    scoring: str | None,
) -> tuple[RetrievalResult, tuple[RankChange, ...], tuple[str, ...]]:
    indexes = [item[0] for item in order]
    if len(set(indexes)) != len(indexes) or any(i < 0 or i >= len(input.hits) for i in indexes):
        raise IndexingError("rerank_invalid_order")
    output = []
    for rank, (i, score, extras) in enumerate(order, 1):
        hit = input.hits[i]
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
                score_components=(
                    *hit.score_components,
                    ScoreComponent(
                        name="input_rank", value=float(hit.rank), source=input.technique_id, source_rank=hit.rank
                    ),
                    *extras,
                ),
                parent_start=hit.parent_start,
                parent_end=hit.parent_end,
            )
        )
    result = RetrievalResult(
        technique_id=technique_id,
        grounding=input.grounding,
        index_version_id=input.index_version_id,
        hits=tuple(output),
        scoring=scoring,
    )
    changes = rank_changes(input, result, allow_added=False)
    output_ids = {hit.chunk_id for hit in result.hits}
    return (
        result,
        changes,
        tuple(hit.chunk_id for hit in input.hits if hit.chunk_id not in output_ids),
    )


def check_rerank_input(result: RetrievalResult, max_candidates: int) -> None:
    try:
        check_single_input(result)
    except IndexingError as exc:
        if exc.code == "selection_invalid_input":
            raise IndexingError("rerank_invalid_input") from exc
        raise
    if len(result.hits) > max_candidates:
        raise IndexingError("rerank_too_many_candidates")
