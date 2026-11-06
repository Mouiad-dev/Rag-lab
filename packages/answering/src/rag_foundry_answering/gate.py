"""Score threshold gate for retrieved evidence."""

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rag_foundry_techniques.kinds import RetrievedChunk

# Unvalidated coarse floor; see ADR-0013 D5.
DEFAULT_MIN_SCORES: Mapping[tuple[str, str, int], float] = {("ollama", "bge-m3", 1024): 0.45}


def default_min_score(provider: str, model: str, dim: int) -> float | None:
    """Return the configured default score floor, if known."""
    return DEFAULT_MIN_SCORES.get((provider, model, dim))


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["pass", "no_evidence", "low_score"]
    used: tuple[RetrievedChunk, ...]
    truncated: bool


def apply_gate(hits: Sequence[RetrievedChunk], *, min_score: float, max_evidence_chars: int) -> GateResult:
    """Filter low scoring hits and cap evidence characters while retaining the first hit."""
    if not hits:
        return GateResult(outcome="no_evidence", used=(), truncated=False)
    if hits[0].score < min_score:
        return GateResult(outcome="low_score", used=(), truncated=False)
    qualified = [hit for hit in hits if hit.score >= min_score]
    used: list[RetrievedChunk] = []
    total = 0
    for hit in qualified:
        if used and total + len(hit.text) > max_evidence_chars:
            break
        used.append(hit)
        total += len(hit.text)
    return GateResult(outcome="pass", used=tuple(used), truncated=len(used) < len(qualified))
