"""Test-only deterministic token encoder. Simulation, never registered as C20 or R45."""

import re
from collections.abc import Sequence
from typing import Literal

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.messages import TokenEncoding, TokenEncodingResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class FakeTokenEncoder:
    """One token per whitespace-separated word; vector is derived from the word's code points."""

    profile = ModelProfile(
        id="test-fake-tokens",
        provider="test",
        model="test-fake-tokens",
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(local=True, embedding_dim=4),
    )

    def __init__(self, max_tokens: int = 8, dim: int = 4) -> None:
        self.max_tokens = max_tokens
        self.dim = dim
        self.calls = 0

    async def encode_tokens(
        self,
        texts: Sequence[str],
        *,
        head: Literal["hidden", "colbert"],
        input_type: Literal["document", "query"],
    ) -> TokenEncodingResult:
        self.calls += 1
        items = []
        total = 0
        for text in texts:
            spans = [m.span() for m in re.finditer(r"\S+", text)]
            truncated = len(spans) > self.max_tokens
            spans = spans[: self.max_tokens]
            total += len(spans)
            vectors = tuple(
                tuple(float((sum(ord(c) for c in text[s:e]) + k) % 7) for k in range(self.dim)) for s, e in spans
            )
            items.append(TokenEncoding(vectors=vectors, offsets=tuple(spans), truncated=truncated))
        return TokenEncodingResult(
            items=tuple(items),
            dim=self.dim,
            usage=Usage(
                profile_id=self.profile.id,
                provider="test",
                model=self.profile.model,
                operation=Operation.EMBED,
                input_tokens=total,
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=0,
                estimated_cost_usd=0,
            ),
        )
