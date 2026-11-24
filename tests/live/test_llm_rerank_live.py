import asyncio
import os

import pytest
from _fixture import DISTRACTORS, FACT_EN, QUERY_EN
from rag_foundry_core.settings import load_settings
from rag_foundry_indexing import LlmListwiseReranker, LlmRerankConfig, order_probe
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile
from rag_foundry_techniques import RetrievalResult, RetrievedChunk

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]


def test_llm_rerank_live():
    async def run():
        settings = load_settings()
        chat = OllamaChatModel(
            ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300
        )
        texts = (*DISTRACTORS[:4], FACT_EN)
        hits = tuple(
            RetrievedChunk(
                chunk_id=f"live-{i}",
                extraction_view_id="view",
                start=i,
                end=i + len(text),
                block_ids=(),
                text=text,
                distance=None,
                score=1 / (i + 1),
                rank=i + 1,
                score_components=(),
            )
            for i, text in enumerate(texts)
        )
        result = RetrievalResult(technique_id="R02", grounding="corpus", index_version_id="live", hits=hits)
        reranker = LlmListwiseReranker(LlmRerankConfig(k=5))
        output = await reranker.rerank(QUERY_EN, result, chat=chat)
        probe = await order_probe(QUERY_EN, result, chat=chat, reranker=reranker)
        assert output.trace.status in {"applied", "rejected"}
        assert {h.chunk_id for h in output.result.hits} <= {h.chunk_id for h in result.hits}
        assert len(output.trace.usage) == 1 and len(probe.usage) == 2
        print("order", [h.chunk_id for h in output.result.hits], "probe", probe.forward_status, probe.reversed_status)
        for item in (*output.trace.usage, *probe.usage):
            print("tokens", item.input_tokens, item.output_tokens, "latency_ms", item.latency_ms)

    asyncio.run(run())
