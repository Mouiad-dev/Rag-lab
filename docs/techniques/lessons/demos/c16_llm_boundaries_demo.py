"""C16 lesson demo: a scripted chat model (no network) proposes two breaks; one quote is invented."""

import asyncio
import importlib.util
import json
from pathlib import Path

from rag_foundry_chunking import InMemorySegmentationCache, LlmBoundaryChunker, LlmBoundaryConfig, ModelChunkContext
from rag_foundry_domain.prompts import PromptTemplate, prompt_sha256
from rag_foundry_providers.fake import FakeChatModel, fake_chat_profile
from rag_foundry_providers.messages import ChatResult
from rag_foundry_techniques import ChunkInput

MIGRATION = Path("packages/storage/src/rag_foundry_storage/sql/migrations/versions/0006_chunking_prompts.py")
spec = importlib.util.spec_from_file_location("seed_0006", MIGRATION)
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)
PROMPT = PromptTemplate(
    id="chunk-boundaries-v1",
    family="chunk-boundaries",
    version=1,
    purpose="chunk-boundaries",
    system_text=seed._CHUNK_BOUNDARIES_V1,
    content_sha256=prompt_sha256(seed._CHUNK_BOUNDARIES_V1),
)
TEXT = (
    "The cat sleeps on the warm mat. The cat wakes up at noon. "
    "The market opened higher today. Shares of banks rose. "
    "The weather turns cold tonight. Rain is expected tomorrow."
)
REPLY = {
    "breaks": [
        {"unit": 2, "starts_with": "The market opened higher"},
        {"unit": 4, "starts_with": "Snow falls in the city"},
    ]
}


class ScriptedChat:
    profile = fake_chat_profile()

    async def complete(self, request):
        usage = (await FakeChatModel(self.profile).complete(request)).usage
        return ChatResult(text=json.dumps(REPLY), finish_reason="stop", usage=usage)


async def main():
    ctx = ModelChunkContext(chat=ScriptedChat(), prompt=PROMPT, cache=InMemorySegmentationCache())
    chunker = LlmBoundaryChunker(LlmBoundaryConfig(chunk_size=200))
    first = await chunker.chunk_with_models(ChunkInput(text=TEXT), ctx)
    print("spans:", [TEXT[s.start : s.end] for s in first.spans])
    print("accepted break offsets:", first.report.details["break_offsets"])
    for r in first.report.rejections:
        print("rejected:", r.reason, r.detail)
    again = await chunker.chunk_with_models(ChunkInput(text=TEXT), ctx)
    print("second run: cache_hit =", again.report.cache_hit, "model_calls =", again.report.model_calls)


asyncio.run(main())
