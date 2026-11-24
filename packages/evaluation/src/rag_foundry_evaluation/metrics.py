"""Span overlap retrieval metrics.

A single chunk that covers several labelled spans counts each span for recall,
but contributes only its highest grade at that rank; therefore nDCG cannot
reach 1 when one chunk covers several labelled spans.
"""

from collections.abc import Sequence
from math import log2

from pydantic import BaseModel, ConfigDict, Field

from .dataset import EvalCase
from .errors import EvaluationError


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalHit(_FrozenModel):
    doc_key: str | None
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    rank: int = Field(ge=1)
    score: float
    text_sha256: str


class CaseMetrics(_FrozenModel):
    answerable: bool
    found_ranks: tuple[int | None, ...]
    first_relevant_rank: int | None
    hit_rate: dict[int, float] | None
    recall: dict[int, float] | None
    ndcg: dict[int, float] | None
    rr: float | None
    returned_hits: int
    top1_score: float | None


class GroupSummary(_FrozenModel):
    split: str
    group: str
    n_cases: int
    n_answerable: int
    n_unanswerable: int
    n_errors: int
    hit_rate: dict[int, float] | None
    recall: dict[int, float] | None
    ndcg: dict[int, float] | None
    mrr: float | None


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    values = tuple(ks)
    if not values or values != tuple(sorted(set(values))) or any(k < 1 or k > 50 for k in values):
        raise ValueError("ks must be sorted, unique, and in 1..50")
    return values


def _covered(span, hits) -> float:
    intervals = sorted(
        (max(span.start, hit.start), min(span.end, hit.end))
        for hit in hits
        if hit.doc_key == span.doc_key and hit.start < span.end and span.start < hit.end
    )
    cursor = span.start
    total = 0
    for start, end in intervals:
        total += max(0, end - max(cursor, start))
        cursor = max(cursor, end)
    return total / (span.end - span.start)


def case_metrics(case: EvalCase, hits: Sequence[EvalHit], *, ks=(1, 3, 5), threshold: float = 0.8) -> CaseMetrics:
    ks = _validate_ks(ks)
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if any(hit.rank != index for index, hit in enumerate(hits, 1)):
        raise EvaluationError("invalid_ranking")
    returned = len(hits)
    top_score = hits[0].score if hits else None
    if not case.answerable:
        return CaseMetrics(
            answerable=False,
            found_ranks=(),
            first_relevant_rank=None,
            hit_rate=None,
            recall=None,
            ndcg=None,
            rr=None,
            returned_hits=returned,
            top1_score=top_score,
        )

    max_k = max(ks)
    found = []
    for span in case.relevant:
        rank = None
        for k in range(1, max_k + 1):
            if _covered(span, hits[:k]) >= threshold:
                rank = k
                break
        found.append(rank)
    found_ranks = tuple(found)
    relevant_ranks = [rank for rank in found_ranks if rank is not None]
    first = min(relevant_ranks) if relevant_ranks else None
    hit_rate = {k: float(any(rank is not None and rank <= k for rank in found_ranks)) for k in ks}
    recall = {k: sum(rank is not None and rank <= k for rank in found_ranks) / len(found_ranks) for k in ks}
    rr = 1.0 / first if first is not None else 0.0
    by_rank = {}
    for span, rank in zip(case.relevant, found_ranks, strict=True):
        if rank is not None:
            by_rank[rank] = max(by_rank.get(rank, 0), span.grade)
    ideal = sorted((span.grade for span in case.relevant), reverse=True)
    ndcg = {}
    for k in ks:
        dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in by_rank.items() if rank <= k)
        idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal[:k], 1))
        ndcg[k] = dcg / idcg
    return CaseMetrics(
        answerable=True,
        found_ranks=found_ranks,
        first_relevant_rank=first,
        hit_rate=hit_rate,
        recall=recall,
        ndcg=ndcg,
        rr=rr,
        returned_hits=returned,
        top1_score=top_score,
    )


def summarize_group(split, group, members, ks) -> GroupSummary:
    answerable = [(case, metrics, error) for case, metrics, error in members if case.answerable]
    count = len(answerable)

    def means(field):
        if not count:
            return None
        return {k: round(sum(getattr(metric, field)[k] for _, metric, _ in answerable) / count, 6) for k in ks}

    mrr = round(sum(metric.rr for _, metric, _ in answerable) / count, 6) if count else None
    return GroupSummary(
        split=split,
        group=group,
        n_cases=len(members),
        n_answerable=count,
        n_unanswerable=sum(not case.answerable for case, _, _ in members),
        n_errors=sum(error for _, _, error in members),
        hit_rate=means("hit_rate"),
        recall=means("recall"),
        ndcg=means("ndcg"),
        mrr=mrr,
    )


def aggregate(rows, ks) -> tuple[GroupSummary, ...]:
    ks = _validate_ks(ks)
    group_names = (
        "all",
        "language:en",
        "language:ar",
        "language:mixed",
        "tag:identifier",
        "tag:paraphrase",
        "tag:diacritics",
        "tag:table",
        "tag:ocr",
        "unanswerable",
    )
    splits = ("dev", "validation", "held_out")
    present = [split for split in splits if any(case.split == split for case, _, _ in rows)]
    output = []
    for split in present:
        for group in group_names:
            members = []
            for case, metrics, error in rows:
                matches = (
                    group == "all"
                    or group == f"language:{case.language}"
                    or group == "unanswerable"
                    and not case.answerable
                    or group.startswith("tag:")
                    and group[4:] in case.tags
                )
                if case.split == split and matches:
                    members.append((case, metrics, error))
            output.append(summarize_group(split, group, members, ks))
    return tuple(output)
