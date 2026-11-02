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
    # Uncached input only. Adapters whose provider counts cached tokens inside its prompt
    # count must subtract them here, or estimate_cost charges them twice.
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(0, ge=0)
    cache_write_input_tokens: int = Field(0, ge=0)
    token_source: TokenSource
    latency_ms: float = Field(ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


_MTOK = Decimal(1_000_000)


def estimate_cost(
    profile: ModelProfile,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    """Exact tokens * rate / 1e6; None when a needed rate is unknown (never invented).

    `input_tokens` is uncached input only; cache read and write tokens are priced at their
    own rates. Callers must not pass 0 for counts the provider did not report: estimate
    them and mark the Usage token_source as estimated, so a zero cost is never made up.
    """
    if min(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens) < 0:
        raise ValueError("token counts must be >= 0")
    total = Decimal(0)
    for tokens, rate in (
        (input_tokens, profile.cost_per_mtok_in),
        (output_tokens, profile.cost_per_mtok_out),
        (cache_read_tokens, profile.cost_per_mtok_cache_read),
        (cache_write_tokens, profile.cost_per_mtok_cache_write),
    ):
        if tokens > 0:
            if rate is None:
                return None
            total += Decimal(tokens) * rate / _MTOK
    return total


def measure_latency_ms(clock: Clock, start: float) -> float:
    return (clock() - start) * 1000.0
