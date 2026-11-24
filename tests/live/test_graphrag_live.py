"""Opt-in live smoke for R38 extraction, R39 summaries and map-reduce on local Ollama (gemma4:e4b).

One local run, not an evaluation; output not bit-identical across runs. Only grounding and tracing invariants are
asserted. Embeddings are the offline scripted vectors of the shared fixture (no embedding model is called).
"""

import asyncio
import json
import os

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_indexing import (
    CommunityGlobalSearch,
    CommunitySummaryBuilder,
    CommunitySummaryConfig,
    EntityGraphConfig,
    EntityGraphRetriever,
    EntityRelationExtractConfig,
    EntityRelationGraphBuilder,
    GlobalSearchConfig,
    detect_communities,
    trace_claims,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger

from tests.chunking._scripted_models import ScriptedEmbedder
from tests.indexing._graph_fixture import GOLD_SPANS, QUESTION, SPANS, TEXT, evidence_span_recall
from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

GLOBAL_QUESTION = "Which organisations work with the Harbor Authority?"


def _stage(name, usage):
    print(
        f"stage={name}",
        f"calls={len(usage)}",
        f"input_tokens={sum(u.input_tokens for u in usage)}",
        f"output_tokens={sum(u.output_tokens for u in usage)}",
        f"latency_ms={round(sum(u.latency_ms for u in usage))}",
    )


def test_graphrag_live_smoke():
    async def run():
        settings = load_settings()
        ledger = UsageLedger()
        chat = RecordingChatModel(
            OllamaChatModel(ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300),
            ledger,
        )
        embedder = ScriptedEmbedder(lambda text: (1.0, 0.0, 0.0, 0.0))
        scope, _stranger, index, version = await ready_index(SPANS, embedder)
        chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
        view_id = chunks[0].extraction_view_id
        view = ViewSource(
            extraction_view_id=view_id,
            text=TEXT,
            blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(TEXT)),),
        )
        source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})

        report = await EntityRelationGraphBuilder(
            EntityRelationExtractConfig(on_invalid="skip"),
            scope=scope,
            index_version=version,
            index=index,
            source=source,
            chat=chat,
        ).build()
        graph = await index.get_entity_graph(scope, version.id, report.graph_key)
        spans = [s for e in graph.entities for s in e.mentions] + [s for r in graph.relations for s in r.evidence]
        assert all(TEXT[s.start : s.end] == s.quote for s in spans)
        print(f"graph entities={len(graph.entities)} relations={len(graph.relations)} chunks={len(chunks)}")
        print(f"extraction rejected={len(report.rejected)} reasons={sorted({r.reason for r in report.rejected})}")
        _stage("extraction", report.usage)

        retriever = EntityGraphRetriever(
            EntityGraphConfig(), scope=scope, graph=graph, index=index, index_version=version
        )
        result = await retriever.retrieve(QUESTION)
        recall = evidence_span_recall(result, GOLD_SPANS, view_id)
        print(f"r38 status={retriever.last_trace.status} mode={retriever.last_trace.mode} recall={recall:.2f}")

        communities = detect_communities(graph)
        summaries = await CommunitySummaryBuilder(
            CommunitySummaryConfig(on_invalid="skip"), graph=graph, chat=chat
        ).build()
        for summary in summaries.summaries:
            assert summary.synthetic is True
            assert all(TEXT[s.start : s.end] == s.quote for s in trace_claims(summary, graph))
        print(
            f"communities={len(communities)} summaries={len(summaries.summaries)} "
            f"skipped_small={len(summaries.skipped_small)} rejected_claims={len(summaries.rejected_claims)} "
            f"rejected_summaries={summaries.rejected_summaries}"
        )
        _stage("summaries", summaries.usage)

        map_calls = reduce_calls = 0
        if summaries.summaries:
            answer = await CommunityGlobalSearch(
                GlobalSearchConfig(on_invalid="skip"),
                scope=scope,
                graph=graph,
                summaries=summaries.summaries,
                index=index,
                index_version=version,
                chat=chat,
            ).answer(GLOBAL_QUESTION)
            map_calls, reduce_calls = answer.map_calls, answer.reduce_calls
            assert answer.status == "abstained" or all(s.sources for s in answer.sentences)
            for sentence in answer.sentences:
                assert all(TEXT[s.start : s.end] == s.quote for s in sentence.sources)
            print(f"r39 status={answer.status} reason={answer.reason} sentences={len(answer.sentences)}")
            print(
                f"r39 rejected_points={len(answer.rejected_points)} invalid_map_replies={answer.invalid_map_replies} "
                f"dropped_sentences={answer.dropped_sentences}"
            )
            print("r39 answer=" + json.dumps([s.text for s in answer.sentences], ensure_ascii=False))
            _stage("map", answer.map_usage)
            _stage("reduce", answer.reduce_usage)
        else:
            print("r39 not run: no community summary was kept")

        totals = ledger.totals()
        print(f"ledger calls={totals.calls} input_tokens={totals.input_tokens} output_tokens={totals.output_tokens}")
        assert totals.calls == report.chunks_processed + summaries.chat_calls + map_calls + reduce_calls

    asyncio.run(run())
