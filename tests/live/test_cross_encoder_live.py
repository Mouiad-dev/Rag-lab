import asyncio
import os

import pytest
from _fixture import DISTRACTORS, FACT_AR, FACT_EN, QUERY_AR, QUERY_EN
from rag_foundry_providers.contract_checks import RERANK_CHECKS
from rag_foundry_providers.sentence_transformers_rerank import SentenceTransformersCrossEncoder

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("RAG_LIVE_RERANK") != "1" or not os.environ.get("RAG_RERANK_MODEL_DIR"),
    reason="set RAG_LIVE_RERANK=1 and RAG_RERANK_MODEL_DIR to run the local cross-encoder",
)
def test_local_cross_encoder_contract_and_facts():
    adapter = SentenceTransformersCrossEncoder()
    asyncio.run(_check(adapter))


async def _check(adapter):
    for check in RERANK_CHECKS:
        await check(adapter)
    assert adapter.activation == "Sigmoid"
    for query, fact in (
        (QUERY_EN, FACT_EN),
        (QUERY_AR, FACT_AR),
    ):
        documents = [*DISTRACTORS, fact]
        result = await adapter.rerank(query, documents)
        assert result.items[0].index == len(DISTRACTORS)
        assert {item.index for item in result.items} <= set(range(len(documents)))
        assert result.usage.input_tokens > 0
        print(
            "query",
            query,
            "order",
            [item.index for item in result.items],
            "scores",
            [round(item.score, 6) for item in result.items],
            "input_tokens",
            result.usage.input_tokens,
            "latency_ms",
            result.usage.latency_ms,
        )
