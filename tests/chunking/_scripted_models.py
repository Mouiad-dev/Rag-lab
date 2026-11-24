"""Deterministic scripted model ports for chunking tests."""

from rag_foundry_providers.fake import FakeChatModel, FakeEmbeddingModel, fake_chat_profile, fake_embedding_profile
from rag_foundry_providers.messages import ChatResult, EmbeddingResult


class ScriptedEmbedder:
    def __init__(self, vector_for, profile=None):
        self.profile = profile or fake_embedding_profile(dim=4)
        self.vector_for = vector_for
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(tuple(texts))
        usage = (await FakeEmbeddingModel(self.profile).embed(texts, input_type=input_type)).usage
        return EmbeddingResult(vectors=tuple(self.vector_for(text) for text in texts), usage=usage)


class ScriptedChat:
    def __init__(self, responder, profile=None):
        self.profile = profile or fake_chat_profile()
        self.responder = responder
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        result = self.responder(request)
        if isinstance(result, ChatResult):
            return result
        text, finish_reason = result
        usage = (await FakeChatModel(self.profile).complete(request)).usage
        return ChatResult(text=text, finish_reason=finish_reason, usage=usage)


def topic_vector(text):
    import hashlib

    lowered = text.casefold()
    words = set(lowered.split())
    cats = sum(word in words for word in ("cat", "cats", "kitten")) + lowered.count("\u0642\u0637\u0629")
    markets = sum(word in words for word in ("market", "stock", "shares")) + lowered.count("\u0633\u0648\u0642")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    noise = tuple(byte / 2550 for byte in digest[:4])
    return (float(cats) + noise[0], float(markets) + noise[1], noise[2], noise[3])
