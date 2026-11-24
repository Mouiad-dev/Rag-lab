"""Test-only deterministic lexical vectorizer; this is not a retriever."""

import hashlib
import unicodedata

from rag_foundry_providers.fake import fake_embedding_profile
from rag_foundry_providers.messages import EmbeddingResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class LexicalTestEmbedder:
    def __init__(self):
        self.profile = fake_embedding_profile("test-lexical", dim=256).model_copy(update={"model": "lexical-test-v1"})
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((tuple(texts), input_type))
        vectors = []
        for text in texts:
            vector = [0.01] + [0.0] * 255
            cleaned = "".join(
                c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c) and c != "\u0640"
            ).casefold()
            for token in _tokens(cleaned):
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest(), "big") % 256
                vector[bucket] += 1.0
            vectors.append(tuple(vector))
        usage = Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=Operation.EMBED,
            input_tokens=0,
            output_tokens=0,
            token_source=TokenSource.ESTIMATED,
            latency_ms=0,
            estimated_cost_usd=0,
        )
        return EmbeddingResult(vectors=tuple(vectors), usage=usage)


def _tokens(text):
    token = ""
    for char in text:
        if char.isalnum():
            token += char
        elif token:
            yield token
            token = ""
    if token:
        yield token
