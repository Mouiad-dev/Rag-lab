"""Rule-based adaptive segmentation (C19).

Policies map to existing chunkers: C02 uses words_per_chunk derived as
chunk_size // 6; C05 uses RecursiveConfig with zero overlap; C06 uses
SentenceConfig; C07 uses ParagraphConfig; and C21 uses MultilingualConfig.
The latter three receive chunk_size and otherwise keep their defaults.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rag_foundry_techniques import ChunkInput, ChunkSpan, Fidelity, PluginKind, PluginManifest, TextBlock

from rag_foundry_chunking.fixed import FixedWordChunker, FixedWordConfig
from rag_foundry_chunking.multilingual import MultilingualChunker, MultilingualConfig, script_of
from rag_foundry_chunking.paragraph import ParagraphChunker, ParagraphConfig, paragraph_ranges
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.sentence import SentenceChunker, SentenceConfig

RegionKind = Literal["prose", "mixed_script", "list", "table", "code"]
PolicyId = Literal["C02", "C05", "C06", "C07", "C21"]
_KINDS = frozenset(("prose", "mixed_script", "list", "table", "code"))
_POLICIES = frozenset(("C02", "C05", "C06", "C07", "C21"))
_DEFAULT_POLICIES: dict[RegionKind, PolicyId] = {
    "prose": "C06",
    "mixed_script": "C21",
    "list": "C07",
    "table": "C07",
    "code": "C05",
}


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(default=800, ge=32, le=8000)
    use_blocks: bool = True
    policies: dict[RegionKind, PolicyId] = Field(default_factory=lambda: dict(_DEFAULT_POLICIES))
    fallback_policy: PolicyId = "C05"
    mixed_min_share: float = Field(default=0.2, ge=0, le=0.5)
    router: Literal["rules"] = "rules"

    @field_validator("policies")
    @classmethod
    def valid_policies(cls, value: dict[RegionKind, PolicyId]) -> dict[RegionKind, PolicyId]:
        if set(value) != _KINDS:
            raise ValueError("policies must define exactly the five region kinds")
        if any(policy not in _POLICIES for policy in value.values()):
            raise ValueError("unknown policy id")
        return value


class RegionDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: RegionKind
    policy: PolicyId
    reason: str
    features: dict[str, float | int]
    failed: bool = False
    error: str | None = None
    chunk_count: int = Field(ge=0)


class PolicyComparison(BaseModel):
    adaptive_chunks: int
    fixed_chunks: int
    fixed_policy: PolicyId
    fixed_cross_region: int
    regions: tuple[RegionDecision, ...]
    failures: int
    oversized_adaptive: int
    oversized_fixed: int


def classify(text: str, start: int, end: int, mixed_min_share: float) -> tuple[RegionKind, dict[str, float | int], str]:
    sample = text[start:end]
    lines = sample.splitlines() or ([sample] if sample else [])
    count = len(lines)
    list_lines = sum(bool(re.match(r"\s*(?:[-*\u2022]|\d+[.)]|[\u0660-\u0669]+[.)])", line)) for line in lines)
    table_lines = sum(line.count("|") >= 2 or ("\t" in line and line.count("\t") >= 2) for line in lines)
    code_lines = sum(
        bool(
            line.startswith("\t")
            or len(line) - len(line.lstrip(" ")) >= 4
            or line.rstrip().endswith(("{", "}", ";"))
            or line.startswith("```")
        )
        for line in lines
    )
    letters = [script_of(char) for char in sample if script_of(char) != "other"]
    total = len(letters)
    arabic = sum(script == "arabic" for script in letters) / total if total else 0.0
    latin = sum(script == "latin" for script in letters) / total if total else 0.0
    features: dict[str, float | int] = {
        "lines": count,
        "chars": len(sample),
        "list_line_share": list_lines / count if count else 0.0,
        "table_line_share": table_lines / count if count else 0.0,
        "code_line_share": code_lines / count if count else 0.0,
        "arabic_share": arabic,
        "latin_share": latin,
    }
    if count >= 2 and features["table_line_share"] >= 0.5:
        return "table", features, "at least half of lines have table separators"
    if count and features["code_line_share"] >= 0.5:
        return "code", features, "at least half of lines look like code"
    if count and features["list_line_share"] >= 0.5:
        return "list", features, "at least half of lines are list items"
    if arabic >= mixed_min_share and latin >= mixed_min_share:
        return "mixed_script", features, "Arabic and Latin letters both meet the minimum share"
    return "prose", features, "no structural or mixed-script rule matched"


_POLICY_CLASSES = {
    "C02": (FixedWordChunker, lambda size: FixedWordConfig(words_per_chunk=max(1, size // 6))),
    "C05": (RecursiveChunker, lambda size: RecursiveConfig(chunk_size=size, chunk_overlap=0)),
    "C06": (SentenceChunker, lambda size: SentenceConfig(chunk_size=size)),
    "C07": (ParagraphChunker, lambda size: ParagraphConfig(chunk_size=size)),
    "C21": (MultilingualChunker, lambda size: MultilingualConfig(chunk_size=size)),
}


def _policy(policy: PolicyId, size: int):
    cls, config = _POLICY_CLASSES[policy]
    return cls(config(size))


class AdaptiveChunker:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="C19",
        kind=PluginKind.CHUNKER,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=AdaptiveConfig,
    )

    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config

    def _segment(self, doc: ChunkInput) -> tuple[tuple[ChunkSpan, ...], tuple[RegionDecision, ...]]:
        raw_regions = paragraph_ranges(doc, self.config.use_blocks)
        regions: list[tuple[int, int, RegionKind, dict[str, float | int], str]] = []
        for start, end in raw_regions:
            kind, features, reason = classify(doc.text, start, end, self.config.mixed_min_share)
            if regions and regions[-1][2] == kind and regions[-1][1] <= start:
                prior_start, _, prior_kind, _, prior_reason = regions.pop()
                start = prior_start
                _, features, _ = classify(doc.text, start, end, self.config.mixed_min_share)
                kind, reason = prior_kind, prior_reason
            regions.append((start, end, kind, features, reason))
        spans: list[ChunkSpan] = []
        decisions: list[RegionDecision] = []
        for start, end, kind, features, reason in regions:
            selected = self.config.policies[kind]
            local_blocks = tuple(
                TextBlock(
                    block_id=block.block_id, start=max(block.start, start) - start, end=min(block.end, end) - start
                )
                for block in doc.blocks
                if block.start < end and block.end > start
            )
            local_doc = ChunkInput(text=doc.text[start:end], blocks=local_blocks)
            failed = False
            error = None
            try:
                local_spans = _policy(selected, self.config.chunk_size).chunk(local_doc)
                if any(span.start < 0 or span.end > len(local_doc.text) for span in local_spans):
                    raise ValueError("policy returned a span outside the region")
            except Exception as exc:
                failed = True
                error = f"{type(exc).__name__}: {exc}"[:200]
                selected = self.config.fallback_policy
                local_spans = _policy(selected, self.config.chunk_size).chunk(local_doc)
                if any(span.start < 0 or span.end > len(local_doc.text) for span in local_spans):
                    raise ValueError("fallback policy returned a span outside the region") from exc
            spans.extend(
                ChunkSpan(start=span.start + start, end=span.end + start, split_level=span.split_level)
                for span in local_spans
            )
            decisions.append(
                RegionDecision(
                    start=start,
                    end=end,
                    kind=kind,
                    policy=selected,
                    reason=reason,
                    features=features,
                    failed=failed,
                    error=error,
                    chunk_count=len(local_spans),
                )
            )
        return tuple(sorted(spans, key=lambda span: (span.start, span.end))), tuple(decisions)

    def segment(self, doc: ChunkInput) -> tuple[tuple[ChunkSpan, ...], tuple[RegionDecision, ...]]:
        return self._segment(doc)

    def chunk(self, doc: ChunkInput) -> tuple[ChunkSpan, ...]:
        return self._segment(doc)[0]


def compare_with_fixed(doc: ChunkInput, config: AdaptiveConfig, fixed_policy: PolicyId = "C05") -> PolicyComparison:
    adaptive, regions = AdaptiveChunker(config).segment(doc)
    fixed = _policy(fixed_policy, config.chunk_size).chunk(doc)
    boundaries = {point for region in regions for point in (region.start, region.end)}
    cross = sum(any(span.start < point < span.end for point in boundaries) for span in fixed)
    return PolicyComparison(
        adaptive_chunks=len(adaptive),
        fixed_chunks=len(fixed),
        fixed_policy=fixed_policy,
        fixed_cross_region=cross,
        regions=regions,
        failures=sum(region.failed for region in regions),
        oversized_adaptive=sum(span.end - span.start > config.chunk_size for span in adaptive),
        oversized_fixed=sum(span.end - span.start > config.chunk_size for span in fixed),
    )
