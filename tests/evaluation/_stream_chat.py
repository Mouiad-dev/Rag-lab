"""Scripted chat model with complete and stream, for timing tests."""

from decimal import Decimal

from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.messages import ChatChunk, ChatResult
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class StreamScriptedChat:
    def __init__(self, pieces, profile=None):
        self.profile = profile or fake_chat_profile()
        self.pieces = list(pieces)
        self.requests = []

    def _usage(self, operation):
        return Usage(
            profile_id=self.profile.id,
            provider=self.profile.provider,
            model=self.profile.model,
            operation=operation,
            input_tokens=1,
            output_tokens=1,
            token_source=TokenSource.ESTIMATED,
            latency_ms=0,
            estimated_cost_usd=Decimal(0),
        )

    async def complete(self, request):
        self.requests.append(request)
        return ChatResult(text="".join(self.pieces), finish_reason="stop", usage=self._usage(Operation.CHAT))

    async def stream(self, request):
        self.requests.append(request)
        for piece in self.pieces:
            yield ChatChunk(delta=piece)
        yield ChatChunk(delta="", usage=self._usage(Operation.CHAT_STREAM))
