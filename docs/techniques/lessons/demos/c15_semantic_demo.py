"""C15 lesson demo: toy embedder, two texts, absolute vs percentile thresholds. No network."""

import asyncio

from rag_foundry_chunking import InMemorySegmentationCache, ModelChunkContext, SemanticChunker, SemanticConfig
from rag_foundry_providers.fake import FakeEmbeddingModel, fake_embedding_profile
from rag_foundry_providers.messages import EmbeddingResult
from rag_foundry_techniques import ChunkInput


class TopicEmbedder:  # toy model: axis 0 = cats, axis 1 = markets, axis 2 = length
    profile = fake_embedding_profile(dim=3)

    async def embed(self, texts, *, input_type="document"):
        usage = (await FakeEmbeddingModel(self.profile).embed(texts)).usage
        vec = lambda t: (t.count("cat") + 0.1, t.count("market") + 0.1, len(t) / 10)  # noqa: E731
        return EmbeddingResult(vectors=tuple(vec(t) for t in texts), usage=usage)


async def main():
    topic = "The cat sleeps. The cat eats. The market rises. The market falls."
    cohesive = "The cat sleeps. The cat eats. The cat plays. The cat purrs."
    for name, text in (("topic change", topic), ("cohesive", cohesive)):
        for kind, amount in (("absolute", 0.5), ("percentile", 90.0)):
            cfg = SemanticConfig(buffer_size=0, threshold_type=kind, threshold_amount=amount)
            ctx = ModelChunkContext(embedder=TopicEmbedder(), cache=InMemorySegmentationCache())
            r = await SemanticChunker(cfg).chunk_with_models(ChunkInput(text=text), ctx)
            d = [round(x, 3) for x in r.report.details["distances"]]
            print(f"{name:12} {kind:10} distances={d} threshold={r.report.details['threshold']:.3f}")
            print("   ", [text[s.start : s.end] for s in r.spans])


asyncio.run(main())
