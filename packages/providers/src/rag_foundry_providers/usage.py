"""Usage record, cost estimation and latency measurement."""

from collections.abc import Callable
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_foundry_providers.capabilities import ModelProfile

Clock = Callable[[], float]
"""Monotonic seconds."""


class Operation(StrEnum):
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    EMBED = "embed"
    RERANK = "rerank"


class TokenSource(StrEnum):
    """Where token counts came from, so cost reports can separate estimates."""

    PROVIDER = "provider"  # reported by the provider API
    ESTIMATED = "estimated"  # counted locally (for example by the fake provider)


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    provider: str
    model: str
    operation: Operation
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    token_source: TokenSource
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


_MTOK = Decimal(1_000_000)


def estimate_cost(profile: ModelProfile, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Exact tokens * rate / 1e6; None when a needed rate is unknown (never invented).

    Callers must not pass 0 for counts the provider did not report: estimate them and
    mark the Usage token_source as estimated, so a zero cost is never made up.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be >= 0")
    rate_in = profile.cost_per_mtok_in
    rate_out = profile.cost_per_mtok_out
    total = Decimal(0)
    if input_tokens > 0:
        if rate_in is None:
            return None
        total += Decimal(input_tokens) * rate_in / _MTOK
    if output_tokens > 0:
        if rate_out is None:
            return None
        total += Decimal(output_tokens) * rate_out / _MTOK
    return total


def measure_latency_ms(clock: Clock, start: float) -> float:
    return (clock() - start) * 1000.0
