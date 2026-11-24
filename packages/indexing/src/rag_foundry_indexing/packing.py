"""Ranked evidence packing with estimated lab-regex token costs (R27)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_chunking._units import LabRegexTokenizer
from rag_foundry_domain.errors import IndexingError
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk


class ContextPackConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order: Literal["rank", "lost_in_middle", "source_order"] = "rank"
    context_window_tokens: int | None = Field(default=None, ge=256, le=2_000_000)
    prompt_overhead_tokens: int = Field(default=256, ge=0, le=100000)
    per_item_overhead_tokens: int = Field(default=16, ge=0, le=1000)
    drop_contained: bool = True


class PlacedEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    rank: int
    position: int
    cost_tokens: int


class DroppedEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    rank: int
    reason: Literal["budget", "contained"]
    cost_tokens: int
    kept_chunk_id: str | None


class PackTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    technique_id: str
    input_technique_id: str
    order: Literal["rank", "lost_in_middle", "source_order"]
    tokenizer: str = "lab-regex-v1"
    context_window: int
    reserve_output_tokens: int
    fixed_tokens: int
    budget_tokens: int
    used_tokens: int
    placed: tuple[PlacedEvidence, ...]
    dropped: tuple[DroppedEvidence, ...]


class PackedContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: PackTrace
    items: tuple[RetrievedChunk, ...]


def _tokens(text: str) -> int:
    return len(LabRegexTokenizer().token_spans(text))


class ContextPacker:
    """Pack whole hits within budget, selecting by rank then arranging prompt order.

    `rank` sorts ascending. `lost_in_middle` sends rank-order indexes at even
    positions to the front and odd positions to the back, then reverses the back.
    `source_order` sorts by (extraction_view_id, start, end, rank).
    """

    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R27",
        kind=PluginKind.CONTEXT_BUILDER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ContextPackConfig,
    )

    def __init__(self, config: ContextPackConfig):
        self.config = config

    def pack(
        self,
        result: RetrievalResult,
        *,
        model_context_window: int | None,
        reserve_output_tokens: int,
        fixed_text: str = "",
    ) -> PackedContext:
        hits = result.hits
        if (
            result.grounding != "corpus"
            or result.index_version_id is None
            or len(hits) > 50
            or len({hit.rank for hit in hits}) != len(hits)
            or len({hit.chunk_id for hit in hits}) != len(hits)
        ):
            raise IndexingError("packing_invalid_input")
        configured = self.config.context_window_tokens
        if configured is None and model_context_window is None:
            raise IndexingError("packing_no_context_window")
        if configured is not None and model_context_window is not None and configured > model_context_window:
            raise IndexingError("packing_window_exceeds_model")
        window = configured if configured is not None else model_context_window
        assert window is not None
        fixed_tokens = _tokens(fixed_text)
        budget = window - reserve_output_tokens - self.config.prompt_overhead_tokens - fixed_tokens
        if budget <= 0:
            raise IndexingError("packing_no_budget")

        ordered = sorted(hits, key=lambda hit: hit.rank)
        selected: list[tuple[RetrievedChunk, int]] = []
        dropped: list[DroppedEvidence] = []
        used = 0
        stopped = False
        for hit in ordered:
            cost = _tokens(hit.text) + self.config.per_item_overhead_tokens
            container = None
            if self.config.drop_contained:
                for kept, _ in selected:
                    if (
                        hit.extraction_view_id == kept.extraction_view_id
                        and kept.start <= hit.start
                        and hit.end <= kept.end
                    ):
                        container = kept
                        break
            if container is not None:
                dropped.append(
                    DroppedEvidence(
                        chunk_id=hit.chunk_id,
                        rank=hit.rank,
                        reason="contained",
                        cost_tokens=cost,
                        kept_chunk_id=container.chunk_id,
                    )
                )
                continue
            if stopped or used + cost > budget:
                stopped = True
                dropped.append(
                    DroppedEvidence(
                        chunk_id=hit.chunk_id, rank=hit.rank, reason="budget", cost_tokens=cost, kept_chunk_id=None
                    )
                )
                continue
            selected.append((hit, cost))
            used += cost

        if self.config.order == "rank":
            arranged = selected
        elif self.config.order == "source_order":
            arranged = sorted(
                selected, key=lambda row: (row[0].extraction_view_id, row[0].start, row[0].end, row[0].rank)
            )
        else:
            front = selected[::2]
            back = selected[1::2]
            arranged = front + list(reversed(back))
        placed = tuple(
            PlacedEvidence(chunk_id=hit.chunk_id, rank=hit.rank, position=position, cost_tokens=cost)
            for position, (hit, cost) in enumerate(arranged, start=1)
        )
        trace = PackTrace(
            technique_id="R27",
            input_technique_id=result.technique_id,
            order=self.config.order,
            context_window=window,
            reserve_output_tokens=reserve_output_tokens,
            fixed_tokens=fixed_tokens,
            budget_tokens=budget,
            used_tokens=used,
            placed=placed,
            dropped=tuple(dropped),
        )
        return PackedContext(trace=trace, items=tuple(hit for hit, _ in arranged))
