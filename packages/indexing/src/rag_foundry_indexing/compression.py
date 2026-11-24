"""Extractive context compression (R26). Sentence selection only; no model call, no rewriting."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_chunking.boundaries import sentence_ranges
from rag_foundry_domain.errors import IndexingError
from rag_foundry_techniques import (
    Fidelity,
    PluginKind,
    PluginManifest,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    ScoreComponent,
)

from rag_foundry_indexing.evidence_constraints import content_terms, evidence_constraints, is_qualifier
from rag_foundry_indexing.selection import check_single_input

RULE_ID = "lexical-overlap-v1"
_MAX_QUESTION_CHARS = 2000


class ExtractiveCompressionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_overlap: int = Field(default=1, ge=1, le=10)
    keep_qualifiers: bool = True
    max_total_chars: int = Field(default=8000, ge=20, le=100000)
    max_hits: int = Field(default=50, ge=1, le=50)


class SentenceDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    start: int
    end: int
    overlap: int
    reason: Literal["overlap", "qualifier", "no_overlap", "budget"]
    constraints: tuple[str, ...]


class CompressionTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    technique_id: str
    input_technique_id: str
    rule: str = RULE_ID
    question_terms: tuple[str, ...]
    input_chars: int
    output_chars: int
    kept: tuple[SentenceDecision, ...]
    dropped: tuple[SentenceDecision, ...]
    removed_hits: tuple[str, ...]
    lost_constraints: tuple[str, ...]


class CompressedRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: CompressionTrace
    input: RetrievalResult
    result: RetrievalResult


def _check(question: str, result: RetrievalResult, config: ExtractiveCompressionConfig) -> None:
    if not question.strip():
        raise IndexingError("empty_query")
    if len(question) > _MAX_QUESTION_CHARS:
        raise IndexingError("query_too_long")
    try:
        check_single_input(result)
    except IndexingError as exc:
        if exc.code == "selection_invalid_input":
            raise IndexingError("compression_invalid_input") from exc
        raise
    if len(result.hits) > config.max_hits or any(hit.end - hit.start != len(hit.text) for hit in result.hits):
        raise IndexingError("compression_invalid_input")


def _piece_id(hit: RetrievedChunk, start: int, end: int) -> str:
    if start == hit.start and end == hit.end:
        return hit.chunk_id
    payload = json.dumps(["R26", hit.chunk_id, start, end], separators=(",", ":"))
    return "cmp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class ExtractiveCompressor:
    """Extractive compression: keep whole sentences of each hit, never rewrite them.

    Rule `lexical-overlap-v1`:
    - Sentences come from `sentence_ranges(hit.text, 0, len(hit.text))`, shifted by `hit.start`.
    - Keep a sentence when `|content_terms(sentence) & content_terms(question)| >= min_overlap`
      (reason `overlap`).
    - With `keep_qualifiers`, also keep the next sentence of the same hit when `is_qualifier` holds
      (reason `qualifier`). This is not chained: only an `overlap` sentence pulls in its successor.
    - Budget: walk kept sentences in (hit rank, start) order. Consecutive kept sentences form one piece
      `[first.start, last.end)`. Stop at the first sentence that would push the sum of piece lengths above
      `max_total_chars`; it and all later kept sentences get reason `budget`.
    - A hit with no remaining kept sentence is listed in `removed_hits`.
    - `lost_constraints`: constraints of all input hit texts absent from the output hits, in input order.
    """

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R26",
        kind=PluginKind.CONTEXT_BUILDER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ExtractiveCompressionConfig,
    )

    def __init__(self, config: ExtractiveCompressionConfig):
        self.config = config

    def compress(self, question: str, result: RetrievalResult) -> CompressedRetrieval:
        _check(question, result, self.config)
        config = self.config
        terms = content_terms(question)
        # per hit: rows [start, end, overlap, reason, text] with absolute offsets
        table: list[list[list]] = []
        for hit in result.hits:
            rows = []
            for s, e in sentence_ranges(hit.text, 0, len(hit.text)):
                text = hit.text[s:e]
                overlap = len(content_terms(text) & terms)
                rows.append(
                    [
                        hit.start + s,
                        hit.start + e,
                        overlap,
                        "overlap" if overlap >= config.min_overlap else "no_overlap",
                        text,
                    ]
                )
            if config.keep_qualifiers:
                for i, row in enumerate(rows[:-1]):
                    nxt = rows[i + 1]
                    if row[3] == "overlap" and nxt[3] == "no_overlap" and is_qualifier(nxt[4]):
                        nxt[3] = "qualifier"
            table.append(rows)

        total = 0
        stopped = False
        pieces: list[list[tuple[int, int, int]]] = [[] for _ in result.hits]  # (start, end, sentence_count)
        last_kept: tuple[int, int] | None = None  # (hit index, sentence index)
        for h, rows in enumerate(table):
            for i, row in enumerate(rows):
                if row[3] not in ("overlap", "qualifier"):
                    continue
                if stopped:
                    row[3] = "budget"
                    continue
                extend = last_kept == (h, i - 1) and bool(pieces[h])
                if extend:
                    ps, pe, count = pieces[h][-1]
                    new_total = total - (pe - ps) + (row[1] - ps)
                    update = (ps, row[1], count + 1)
                else:
                    new_total = total + (row[1] - row[0])
                    update = (row[0], row[1], 1)
                if new_total > config.max_total_chars:
                    stopped = True
                    row[3] = "budget"
                    continue
                total = new_total
                if extend:
                    pieces[h][-1] = update
                else:
                    pieces[h].append(update)
                last_kept = (h, i)

        output: list[RetrievedChunk] = []
        removed: list[str] = []
        for hit, hit_pieces in zip(result.hits, pieces, strict=True):
            if not hit_pieces:
                removed.append(hit.chunk_id)
                continue
            for start, end, count in hit_pieces:
                text = hit.text[start - hit.start : end - hit.start]
                output.append(
                    RetrievedChunk(
                        chunk_id=_piece_id(hit, start, end),
                        extraction_view_id=hit.extraction_view_id,
                        start=start,
                        end=end,
                        block_ids=hit.block_ids,
                        text=text,
                        distance=hit.distance,
                        score=hit.score,
                        rank=len(output) + 1,
                        score_components=(
                            *hit.score_components,
                            ScoreComponent(
                                name="input_rank",
                                value=float(hit.rank),
                                source=result.technique_id,
                                source_rank=hit.rank,
                            ),
                            ScoreComponent(name="kept_sentences", value=float(count)),
                        ),
                        parent_start=hit.parent_start,
                        parent_end=hit.parent_end,
                    )
                )
        compressed = RetrievalResult(
            technique_id="R26",
            grounding=result.grounding,
            index_version_id=result.index_version_id,
            hits=tuple(output),
            scoring=result.scoring,
        )
        kept: list[SentenceDecision] = []
        dropped: list[SentenceDecision] = []
        for hit, rows in zip(result.hits, table, strict=True):
            for start, end, overlap, reason, text in rows:
                decision = SentenceDecision(
                    chunk_id=hit.chunk_id,
                    start=start,
                    end=end,
                    overlap=overlap,
                    reason=reason,
                    constraints=evidence_constraints(text),
                )
                (kept if reason in ("overlap", "qualifier") else dropped).append(decision)
        present = {item for hit in output for item in evidence_constraints(hit.text)}
        original = dict.fromkeys(item for hit in result.hits for item in evidence_constraints(hit.text))
        trace = CompressionTrace(
            technique_id="R26",
            input_technique_id=result.technique_id,
            question_terms=tuple(sorted(terms)),
            input_chars=sum(len(hit.text) for hit in result.hits),
            output_chars=sum(len(hit.text) for hit in output),
            kept=tuple(kept),
            dropped=tuple(dropped),
            removed_hits=tuple(removed),
            lost_constraints=tuple(item for item in original if item not in present),
        )
        return CompressedRetrieval(trace=trace, input=result, result=compressed)


class CompressingRetriever:
    def __init__(self, base: Retriever, compressor: ExtractiveCompressor):
        self.base = base
        self.compressor = compressor
        self.last: CompressedRetrieval | None = None

    async def retrieve(self, query: str) -> RetrievalResult:
        self.last = self.compressor.compress(query, await self.base.retrieve(query))
        return self.last.result
