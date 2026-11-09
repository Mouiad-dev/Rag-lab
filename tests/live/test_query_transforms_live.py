"""Opt-in smoke coverage for query transformation retrievers on local Ollama."""

import asyncio
import os

import pytest
from _fixture import DISTRACTORS, FACT_AR, FACT_EN
from _inmemory_index import ready_index
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.errors import IndexingError, QueryTransformError
from rag_foundry_indexing import (
    DecomposeConfig,
    HydeConfig,
    HydeRetriever,
    MultiQueryConfig,
    MultiQueryExpansion,
    QuestionDecomposition,
    QuestionRewriter,
    RagFusion,
    RagFusionConfig,
    RewriteConfig,
    StepBackConfig,
    StepBackRetrieval,
)
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]


class RecordingRetriever:
    async def retrieve(self, query):
        from rag_foundry_techniques import RetrievalResult

        return RetrievalResult(technique_id="R02", grounding="corpus", index_version_id=None, hits=())


def test_query_transforms_live_smoke():
    async def run():
        settings = load_settings()
        chat = OllamaChatModel(
            ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300
        )
        question = "How often does the Port Sallow observatory record sea levels?"
        recorder = RecordingRetriever()
        cases = (
            ("R13", QuestionRewriter(RewriteConfig(), recorder), (question,)),
            ("R14", MultiQueryExpansion(MultiQueryConfig(n_queries=2), recorder), (question,)),
            ("R15", QuestionDecomposition(DecomposeConfig(), recorder), (question,)),
            ("R16", StepBackRetrieval(StepBackConfig(), recorder), (question,)),
            ("R18", RagFusion(RagFusionConfig(), recorder), (question,)),
        )
        for technique, transform, args in cases:
            try:
                result = await transform.retrieve(*args, chat=chat)
            except QueryTransformError as exc:
                print(f"{technique} error={exc.code}")
                continue
            assert result.trace.status in {"applied", "rejected", "constraint_loss"}
            assert result.trace.technique_id == technique
            assert result.trace.original_question == question
            print(f"{technique} status={result.trace.status}")
            print(f"generated={[q.text for q in result.trace.generated]}")
            for item in result.trace.usage:
                print(
                    "usage",
                    f"operation={item.operation.value}",
                    f"model={item.model}",
                    f"input_tokens={item.input_tokens}",
                    f"output_tokens={item.output_tokens}",
                    f"latency_ms={item.latency_ms}",
                )
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(model="bge-m3"), api_base=settings.ollama_base_url)
        texts = (FACT_EN, FACT_AR, *DISTRACTORS[:4])
        owner, _, index, version = await ready_index([(0, len(text), text) for text in texts], embedder)
        try:
            result = await HydeRetriever(
                HydeConfig(), scope=owner, index_version=version, index=index, embedder=embedder
            ).retrieve(question, chat=chat)
        except (QueryTransformError, IndexingError) as exc:
            print(f"R17 error={exc.code}")
        else:
            assert result.trace.status in {"applied", "rejected", "constraint_loss"}
            assert result.trace.technique_id == "R17"
            assert result.trace.original_question == question
            print(f"R17 status={result.trace.status}")
            print(f"generated={[q.text for q in result.trace.generated]}")
            for item in result.trace.usage:
                print(
                    "usage",
                    f"operation={item.operation.value}",
                    f"model={item.model}",
                    f"input_tokens={item.input_tokens}",
                    f"output_tokens={item.output_tokens}",
                    f"latency_ms={item.latency_ms}",
                )

    asyncio.run(run())
