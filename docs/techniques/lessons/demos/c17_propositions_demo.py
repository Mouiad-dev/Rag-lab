"""C17 lesson demo: a scripted chat model (no network) returns propositions; the lexical checks judge them."""

import asyncio
import importlib.util
import json
from pathlib import Path

from rag_foundry_chunking import InMemorySegmentationCache, ModelChunkContext, PropositionChunker, PropositionConfig
from rag_foundry_domain.prompts import PromptTemplate, prompt_sha256
from rag_foundry_providers.fake import FakeChatModel, fake_chat_profile
from rag_foundry_providers.messages import ChatResult
from rag_foundry_techniques import ChunkInput

MIGRATION = Path("packages/storage/src/rag_foundry_storage/sql/migrations/versions/0006_chunking_prompts.py")
spec = importlib.util.spec_from_file_location("seed_0006", MIGRATION)
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)
PROMPT = PromptTemplate(
    id="propositions-v1",
    family="propositions",
    version=1,
    purpose="propositions",
    system_text=seed._PROPOSITIONS_V1,
    content_sha256=prompt_sha256(seed._PROPOSITIONS_V1),
)
TEXT = (
    "The city library opens at 9 am on weekdays. It does not open on Sundays. "
    "If it rains, the garden closes early. The reading room has twelve large tables."
)
REPLY = {
    "propositions": [
        {"text": "The city library does not open on Sundays.", "quotes": ["It does not open on Sundays"]},
        {"text": "The city library opens on Sundays.", "quotes": ["It does not open on Sundays"]},
        {"text": "The garden closes early.", "quotes": ["the garden closes early"]},
        {"text": "The library opens at 10 am on weekdays.", "quotes": ["opens at 9 am on weekdays"]},
        {"text": "The library has a cafe.", "quotes": ["Library has a cafe"]},
        {"text": "The city library reading room has twelve large tables.", "quotes": ["twelve large tables"]},
        {"text": "The reading room has twelve large wooden tables.", "quotes": ["twelve large tables"]},
        {"text": "The reading room has 12 big desks.", "quotes": ["twelve large tables"]},
    ]
}


class ScriptedChat:
    profile = fake_chat_profile()

    async def complete(self, request):
        usage = (await FakeChatModel(self.profile).complete(request)).usage
        return ChatResult(text=json.dumps(REPLY), finish_reason="stop", usage=usage)


async def main():
    ctx = ModelChunkContext(chat=ScriptedChat(), prompt=PROMPT, cache=InMemorySegmentationCache())
    result = await PropositionChunker(PropositionConfig()).chunk_with_models(ChunkInput(text=TEXT), ctx)
    for p in result.propositions:
        print(f"accepted: {p.text!r} -> source span {TEXT[p.start : p.end]!r}")
    for r in result.report.rejections:
        print(f"rejected: {r.reason}: {r.detail}")
    print("chunk spans:", [TEXT[s.start : s.end] for s in result.spans])


asyncio.run(main())
