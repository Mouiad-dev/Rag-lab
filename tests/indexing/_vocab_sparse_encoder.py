"""Test-only vocabulary counter; it is not R05 and must never be registered as R05."""

from collections import Counter
from typing import Literal

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.messages import SparseEncodingResult, SparseVector
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class VocabSparseEncoder:
    """D13 fixture only. Never register this lexical test double as the learned R05 plugin."""

    profile = ModelProfile(
        id="test-vocab-sparse",
        provider="test",
        model="test-vocab-sparse",
        kind=ModelKind.SPARSE,
        capabilities=Capabilities(local=True),
    )

    async def encode(self, texts, *, input_type: Literal["document", "query"]):
        vectors = []
        for text in texts:
            counts = Counter(text.lower().split())
            vectors.append(SparseVector(terms=tuple(counts), weights=tuple(float(x) for x in counts.values())))
        return SparseEncodingResult(
            vectors=tuple(vectors),
            usage=Usage(
                profile_id=self.profile.id,
                provider="test",
                model=self.profile.model,
                operation=Operation.EMBED,
                input_tokens=0,
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=0,
                estimated_cost_usd=0,
            ),
        )
