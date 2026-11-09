"""Shared source-backed context expansion primitives."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexVersion
from rag_foundry_domain.ports import IndexRepository, ObjectStore
from rag_foundry_ingestion.extraction import read_payload
from rag_foundry_techniques import RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.selection import check_single_input


class SourceBlockRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    block_id: str
    start: int
    end: int


class ViewSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    extraction_view_id: str
    text: str
    blocks: tuple[SourceBlockRef, ...]


class SourceTextReader(Protocol):
    async def view_source(self, scope: Scope, view_id: str) -> ViewSource: ...


class PayloadSourceReader:
    def __init__(self, store: ObjectStore, uow, *, max_cached_views: int = 16):
        self.store, self.uow = store, uow
        self.max_cached_views = max_cached_views
        self._cache: OrderedDict[tuple[str, str, str], ViewSource] = OrderedDict()

    async def view_source(self, scope: Scope, view_id: str) -> ViewSource:
        key = (scope.tenant_id, scope.project_id, view_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        async with self.uow(read_only=True) as uow:
            view = await uow.extractions.get_view(scope, view_id)
        payload = await read_payload(self.store, scope, view)
        source = ViewSource(
            extraction_view_id=view.id,
            text=payload.canonical_text,
            blocks=tuple(SourceBlockRef(block_id=b.block_id, start=b.start, end=b.end) for b in payload.blocks),
        )
        if self.max_cached_views > 0:
            self._cache[key] = source
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cached_views:
                self._cache.popitem(last=False)
        return source


class ExpansionSeed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    start: int
    end: int
    rank: int


class ExpandedContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    extraction_view_id: str
    start: int
    end: int
    seeds: tuple[ExpansionSeed, ...]
    flags: tuple[str, ...] = ()


class ExpansionTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    technique_id: str
    input_technique_id: str
    index_version_id: str
    rule: str
    input_chars: int
    output_chars: int
    contexts: tuple[ExpandedContext, ...]
    dropped_for_budget: tuple[str, ...]


class ExpandedRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trace: ExpansionTrace
    input: RetrievalResult
    result: RetrievalResult


async def load_sources(
    result: RetrievalResult,
    *,
    scope: Scope,
    index: IndexRepository,
    index_version: IndexVersion,
    source: SourceTextReader,
    max_seeds: int,
):
    try:
        check_single_input(result)
    except IndexingError as exc:
        if exc.code == "selection_invalid_input":
            raise IndexingError("expansion_invalid_input") from exc
        raise
    if len(result.hits) > max_seeds:
        raise IndexingError("expansion_too_many_candidates")
    if result.index_version_id != index_version.id:
        raise IndexingError("expansion_version_mismatch")
    version = await index.get_index_version(scope, index_version.id)
    if version.state != "ready":
        raise IndexingError("not_ready")
    sources = {}
    for hit in result.hits:
        if hit.extraction_view_id not in version.extraction_view_ids:
            raise IndexingError("expansion_view_not_in_index")
        if hit.extraction_view_id not in sources:
            sources[hit.extraction_view_id] = await source.view_source(scope, hit.extraction_view_id)
        text = sources[hit.extraction_view_id].text
        if hit.end > len(text) or text[hit.start : hit.end] != hit.text:
            raise IndexingError("expansion_source_mismatch")
    return version, sources


def context_id_for(technique_id: str, view_id: str, start: int, end: int) -> str:
    raw = json.dumps([technique_id, view_id, start, end], separators=(",", ":"), ensure_ascii=False)
    return "ctx_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def merge_overlapping(candidates: Sequence[tuple[str, int, int, tuple[ExpansionSeed, ...], tuple[str, ...]]]):
    rows = [list(item) for item in candidates]
    rows.sort(key=lambda row: min(seed.rank for seed in row[3]))
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(rows):
            for j in range(i + 1, len(rows)):
                right = rows[j]
                if left[0] == right[0] and max(left[1], right[1]) < min(left[2], right[2]):
                    left[1], left[2] = min(left[1], right[1]), max(left[2], right[2])
                    left[3] = tuple(sorted((*left[3], *right[3]), key=lambda seed: seed.rank))
                    left[4] = tuple(dict.fromkeys((*left[4], *right[4])))
                    rows.pop(j)
                    changed = True
                    break
            if changed:
                break
    return tuple((r[0], r[1], r[2], tuple(r[3]), tuple(r[4])) for r in rows)


def assemble(
    input: RetrievalResult, candidates, sources, *, technique_id: str, rule: str, max_total_chars: int, scoring: str
):
    rows = sorted(candidates, key=lambda row: min(s.rank for s in row[3]))
    emitted, contexts, dropped = [], [], []
    total = 0
    for pos, (view_id, start, end, seeds, flags) in enumerate(rows):
        seed_hits = [h for h in input.hits if any(s.chunk_id == h.chunk_id for s in seeds)]
        best = min(seed_hits, key=lambda h: h.rank)
        length = end - start
        if total + length > max_total_chars:
            dropped.extend(h.chunk_id for h in seed_hits)
            for later in rows[pos + 1 :]:
                dropped.extend(h.chunk_id for h in input.hits if any(s.chunk_id == h.chunk_id for s in later[3]))
            break
        total += length
        source = sources[view_id]
        block_ids = tuple(b.block_id for b in source.blocks if max(start, b.start) < min(end, b.end))
        exact = len(seeds) == 1 and (start, end) == (seeds[0].start, seeds[0].end)
        chunk_id = best.chunk_id if exact else context_id_for(technique_id, view_id, start, end)
        extras = (
            ScoreComponent(name="seed_rank", value=float(best.rank), source=input.technique_id, source_rank=best.rank),
            ScoreComponent(name="expanded_chars", value=float(length)),
        )
        emitted.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                extraction_view_id=view_id,
                start=start,
                end=end,
                block_ids=block_ids,
                text=source.text[start:end],
                distance=None,
                score=best.score,
                rank=len(emitted) + 1,
                score_components=(*best.score_components, *extras),
                parent_start=best.parent_start if exact else None,
                parent_end=best.parent_end if exact else None,
            )
        )
        contexts.append(
            ExpandedContext(
                chunk_id=chunk_id, extraction_view_id=view_id, start=start, end=end, seeds=seeds, flags=flags
            )
        )
    result = RetrievalResult(
        technique_id=technique_id,
        grounding="corpus",
        index_version_id=input.index_version_id,
        hits=tuple(emitted),
        scoring=scoring,
    )
    trace = ExpansionTrace(
        technique_id=technique_id,
        input_technique_id=input.technique_id,
        index_version_id=input.index_version_id or "",
        rule=rule,
        input_chars=sum(len(h.text) for h in input.hits),
        output_chars=sum(len(h.text) for h in emitted),
        contexts=tuple(contexts),
        dropped_for_budget=tuple(dict.fromkeys(dropped)),
    )
    return ExpandedRetrieval(trace=trace, input=input, result=result)


class ExpandingRetriever:
    def __init__(self, base, expander):
        self.base, self.expander, self.last = base, expander, None

    async def retrieve(self, q: str) -> RetrievalResult:
        self.last = await self.expander.expand(await self.base.retrieve(q))
        return self.last.result
