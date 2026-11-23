"""Test-only deterministic page-image encoder. Simulation, never registered as R41."""

import hashlib
from collections.abc import Sequence

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.colpali_pages import png_pixel_count
from rag_foundry_providers.messages import Multivector, MultivectorResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage


def _vector(seed: bytes, dim: int) -> tuple[float, ...]:
    digest = hashlib.sha256(seed).digest()
    return tuple(float(digest[i % len(digest)] % 7) for i in range(dim))


class FakePageEncoder:
    """Pages: `patches` vectors derived from the PNG byte hash. Queries: one vector per whitespace word."""

    profile = ModelProfile(
        id="test-fake-pages",
        provider="test",
        model="test-fake-pages",
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(local=True, embedding_dim=4),
    )

    def __init__(self, dim: int = 4, patches: int = 3) -> None:
        self.dim = dim
        self.patches = patches
        self.calls = 0

    def _usage(self, total: int) -> Usage:
        return Usage(
            profile_id=self.profile.id,
            provider="test",
            model=self.profile.model,
            operation=Operation.EMBED,
            input_tokens=total,
            output_tokens=0,
            token_source=TokenSource.ESTIMATED,
            latency_ms=0,
            estimated_cost_usd=0,
        )

    async def encode_pages(self, pngs: Sequence[bytes]) -> MultivectorResult:
        self.calls += 1
        for png in pngs:
            png_pixel_count(png)
        items = tuple(
            Multivector(vectors=tuple(_vector(png + bytes([k]), self.dim) for k in range(self.patches))) for png in pngs
        )
        return MultivectorResult(items=items, dim=self.dim, usage=self._usage(len(items) * self.patches))

    async def encode_queries(self, texts: Sequence[str]) -> MultivectorResult:
        self.calls += 1
        items = tuple(
            Multivector(vectors=tuple(_vector(w.encode("utf-8"), self.dim) for w in text.split())) for text in texts
        )
        return MultivectorResult(items=items, dim=self.dim, usage=self._usage(sum(len(i.vectors) for i in items)))
