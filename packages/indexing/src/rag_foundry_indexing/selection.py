"""Deterministic evidence deduplication (R10).

Every candidate is compared with every kept candidate: at most n(n-1)/2 pairs for n input hits, so the
input is expected to stay small (about 50 hits; `k` caps only the output). Tokens and shingles are computed
once per hit. Number protection compares the ordered number tokens, so swapped figures stay apart.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_foundry_domain.errors import IndexingError
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.lexical_text import lexical_tokens


def check_single_input(result: RetrievalResult) -> None:
    ranks = [hit.rank for hit in result.hits]
    if (
        result.grounding != "corpus"
        or result.index_version_id is None
        or ranks != list(range(1, len(ranks) + 1))
        or len({hit.chunk_id for hit in result.hits}) != len(result.hits)
    ):
        raise IndexingError("selection_invalid_input")


class DedupConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[Literal["exact_text", "span_overlap", "near_text"], ...] = (
        "exact_text",
        "span_overlap",
        "near_text",
    )
    near_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    shingle_size: int = Field(default=3, ge=1, le=5)
    span_overlap_min: float = Field(default=0.5, ge=0.1, le=1.0)
    protect_numbers: bool = True
    k: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _valid_rules(self):
        if not 1 <= len(self.rules) <= 3 or len(set(self.rules)) != len(self.rules):
            raise ValueError("rules must contain one to three unique rules")
        return self


class DroppedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    extraction_view_id: str
    start: int
    end: int
    input_rank: int
    kept_chunk_id: str
    rule: Literal["exact_text", "span_overlap", "near_text"]
    similarity: float = Field(ge=0, le=1)


class DedupResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: RetrievalResult
    dropped: tuple[DroppedCandidate, ...]
    # Survivors cut by `k`, in input order. A dropped candidate's `kept_chunk_id` may name one of these.
    truncated: tuple[str, ...] = ()


class EvidenceDedup:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R10",
        kind=PluginKind.FUSION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=DedupConfig,
    )

    def __init__(self, config: DedupConfig):
        self.config = config

    def select(self, result: RetrievalResult) -> DedupResult:
        check_single_input(result)
        kept: list[dict] = []
        dropped = []
        input_tid = result.technique_id
        features: dict[str, tuple] = {}
        for hit in result.hits:
            match = None
            for previous in kept:
                for rule in ("exact_text", "span_overlap", "near_text"):
                    if rule not in self.config.rules:
                        continue
                    similarity = _similarity(hit, previous["hit"], rule, self.config, features)
                    if similarity is not None:
                        match = (previous, rule, similarity)
                        break
                if match is not None:
                    break
            if match is None:
                kept.append({"hit": hit, "absorbed": []})
            else:
                previous, rule, similarity = match
                dropped.append(
                    DroppedCandidate(
                        chunk_id=hit.chunk_id,
                        extraction_view_id=hit.extraction_view_id,
                        start=hit.start,
                        end=hit.end,
                        input_rank=hit.rank,
                        kept_chunk_id=previous["hit"].chunk_id,
                        rule=rule,
                        similarity=similarity,
                    )
                )
                previous["absorbed"].append((hit, rule, similarity))

        output_hits = []
        for rank, record in enumerate(kept[: self.config.k], start=1):
            hit = record["hit"]
            components = [
                *hit.score_components,
                ScoreComponent(name="input_rank", value=float(hit.rank), source=input_tid, source_rank=hit.rank),
            ]
            components.extend(
                ScoreComponent(name=f"dedup_{rule}", value=similarity, source=input_tid, source_rank=absorbed.rank)
                for absorbed, rule, similarity in record["absorbed"]
            )
            output_hits.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    extraction_view_id=hit.extraction_view_id,
                    start=hit.start,
                    end=hit.end,
                    block_ids=hit.block_ids,
                    text=hit.text,
                    distance=hit.distance,
                    score=hit.score,
                    rank=rank,
                    score_components=tuple(components),
                    parent_start=hit.parent_start,
                    parent_end=hit.parent_end,
                )
            )
        selected = RetrievalResult(
            technique_id="R10",
            grounding="corpus",
            index_version_id=result.index_version_id,
            hits=tuple(output_hits),
            scoring=f"dedup[{input_tid}]:{'+'.join(self.config.rules)}",
        )
        truncated = tuple(record["hit"].chunk_id for record in kept[len(output_hits) :])
        return DedupResult(result=selected, dropped=tuple(dropped), truncated=truncated)


def _similarity(
    candidate: RetrievedChunk, kept: RetrievedChunk, rule: str, config: DedupConfig, features: dict[str, tuple]
) -> float | None:
    if rule == "exact_text":
        left = hashlib.sha256(unicodedata.normalize("NFC", candidate.text).encode("utf-8")).digest()
        right = hashlib.sha256(unicodedata.normalize("NFC", kept.text).encode("utf-8")).digest()
        return 1.0 if left == right else None
    if rule == "span_overlap":
        if candidate.extraction_view_id != kept.extraction_view_id:
            return None
        overlap = max(0, min(candidate.end, kept.end) - max(candidate.start, kept.start))
        ratio = overlap / min(candidate.end - candidate.start, kept.end - kept.start)
        return ratio if ratio >= config.span_overlap_min else None
    left_tokens, left_numbers, left = _features(candidate, config, features)
    right_tokens, right_numbers, right = _features(kept, config, features)
    if not left_tokens or not right_tokens:
        return None
    if config.protect_numbers and left_numbers != right_numbers:
        return None
    score = len(left & right) / len(left | right)
    return score if score >= config.near_threshold else None


def _features(hit: RetrievedChunk, config: DedupConfig, cache: dict[str, tuple]) -> tuple:
    if hit.chunk_id not in cache:
        tokens = lexical_tokens(hit.text)
        cache[hit.chunk_id] = (tokens, _numbers(tokens), _shingles(tokens, config.shingle_size))
    return cache[hit.chunk_id]


def _numbers(tokens: Iterable[str]) -> tuple[str, ...]:
    """Number-bearing tokens in order (whole identifiers like ab-1234 included)."""
    return tuple(token for token in tokens if any(char.isdigit() for char in token))


def _shingles(tokens: tuple[str, ...], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tokens}
    return {tokens[index : index + size] for index in range(len(tokens) - size + 1)}
