import asyncio
import os

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_indexing import ContextualPrefixBuildConfig, ContextualPrefixBuilder
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

DOC = (
    "Harbor Lights Annual Report. Section 2: Operations. "
    "The Port of Marlow handled 4.2 million containers in 2023. "
    "Section 3: Staff. The company employs 310 people at the Marlow site."
)


def test_representation_builders_live():
    async def run():
        settings = load_settings()
        chat = OllamaChatModel(
            ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300
        )
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url)
        a = DOC.index("The Port")
        b = DOC.index(" Section 3")
        c = DOC.index("The company")
        spans = [(a, b, DOC[a:b]), (c, len(DOC), DOC[c:])]
        scope, _, index, version = await ready_index(spans, embedder)
        view = version.extraction_view_ids[0]
        source = StaticSourceReader(
            {
                (scope.tenant_id, scope.project_id, view): ViewSource(
                    extraction_view_id=view,
                    text=DOC,
                    blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(DOC)),),
                )
            }
        )
        builder = ContextualPrefixBuilder(
            ContextualPrefixBuildConfig(max_chunks=3, on_invalid="skip"),
            scope=scope,
            index_version=version,
            index=index,
            source=source,
            chat=chat,
            embedder=embedder,
        )
        report = await builder.build()
        print("statuses", "written", report.rows_written, "rejected", report.rejected)
        for item in report.usage:
            print(
                "usage", item.operation, "tokens", item.input_tokens, item.output_tokens, "latency_ms", item.latency_ms
            )
        assert report.rows_written + len(report.rejected) == 2
        assert report.rows_written >= 1
        hits = await index.representation_search(
            scope, version.id, report.representation_key, (await embedder.embed(["port containers"])).vectors[0], 5
        )
        assert hits and all(h.representation.synthetic is True for h in hits)
        assert all(h.chunk.text == DOC[h.chunk.start : h.chunk.end] for h in hits)
        for h in hits:
            print("context", h.representation.text[:200], "| chunk", h.chunk.text[:60])

    asyncio.run(run())
