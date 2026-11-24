"""Opt-in live run of R48 (KAG-style graph and planner) on local Ollama: gemma4:e4b (chat) and bge-m3 (embeddings).

One local run, not an evaluation; output not bit-identical across runs. Only grounding, tracing and call-count
invariants are asserted; no extraction, planning or answer quality is asserted. A rejected plan, an incomplete or
conflict status and a skipped invalid extraction item are valid outcomes. The embeddings serve only the R02 dense
baseline (KAG does not use them). Local Ollama only: no keys, no paid calls.
"""

import asyncio
import os
from collections import Counter

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.knowledge import KnowledgeGraph
from rag_foundry_indexing import (
    DenseConfig,
    DenseRetriever,
    KagConfig,
    KagExtractConfig,
    KagGraphBuilder,
    KagRetriever,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger

from tests.indexing._graph_fixture import evidence_span_recall
from tests.indexing._inmemory_index import ready_index
from tests.indexing._kag_fixture import QUESTIONS, SCHEMA, SPANS, TEXT
from tests.indexing._source_reader import StaticSourceReader

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

STATUSES = {"answered", "incomplete", "conflict", "rejected"}


def _stage(name, usage):
    print(
        f"stage={name}",
        f"calls={len(usage)}",
        f"input_tokens={sum(u.input_tokens for u in usage)}",
        f"output_tokens={sum(u.output_tokens for u in usage)}",
        f"latency_ms={round(sum(u.latency_ms for u in usage))}",
    )


def test_kag_live_smoke():
    async def run():
        settings = load_settings()
        chat_ledger, embed_ledger = UsageLedger(), UsageLedger()
        chat = RecordingChatModel(
            OllamaChatModel(ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=300),
            chat_ledger,
        )
        embedder = RecordingEmbeddingModel(
            OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url), embed_ledger
        )
        scope, _stranger, index, version = await ready_index(SPANS, embedder)
        chunks = await index.list_chunks(scope, version.id, offset=0, limit=500)
        by_id = {c.id: c for c in chunks}
        view_id = chunks[0].extraction_view_id
        view = ViewSource(
            extraction_view_id=view_id,
            text=TEXT,
            blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(TEXT)),),
        )
        source = StaticSourceReader({(scope.tenant_id, scope.project_id, view_id): view})
        _stage("index_embed", embed_ledger.records())

        report = await KagGraphBuilder(
            KagExtractConfig(on_invalid="skip"),
            scope=scope,
            index_version=version,
            index=index,
            source=source,
            chat=chat,
            knowledge_schema=SCHEMA,
        ).build()
        graph = await index.get_knowledge_graph(scope, version.id, report.kg_key)
        assert KnowledgeGraph.from_json(graph.to_json()) == graph
        spans = [s for e in graph.entities for s in e.mentions] + [s for f in graph.facts for s in f.evidence]
        assert all(TEXT[s.start : s.end] == s.quote for s in spans)
        assert chat_ledger.totals().calls == report.chat_calls == report.chunks_processed == len(chunks)
        reasons = Counter(r.reason for r in report.rejected)
        print(f"graph entities={len(graph.entities)} facts={len(graph.facts)} chunks={len(chunks)}")
        print(f"extraction rejected={len(report.rejected)} by_reason={dict(sorted(reasons.items()))}")
        print(f"conflicts={len(report.conflicts)}")
        _stage("extraction", report.usage)

        dense = DenseRetriever(DenseConfig(k=3), scope=scope, index_version=version, index=index, embedder=embedder)
        plan_calls = first_plans = replans = 0
        plan_usage = []
        plan_reasons: Counter = Counter()
        for q in QUESTIONS:
            retriever = KagRetriever(
                KagConfig(max_replans=1),
                scope=scope,
                graph=graph,
                index=index,
                index_version=version,
                chat=chat,
            )
            print(f"question {q['id']}")
            result = await retriever.retrieve(q["text"])
            trace = retriever.last_trace
            assert trace is not None
            assert trace.status in STATUSES
            assert 1 <= len(trace.attempts) <= 2
            first_plans += 1
            replans += len(trace.attempts) - 1
            plan_calls += len(trace.attempts)
            plan_usage.extend(trace.usage)
            if trace.status == "answered":
                backed = any(b.fact_ids for b in trace.answer)
                assert bool(result.hits) == backed
                if not backed:
                    print("  answered_without_sources")
                for hit in result.hits:
                    chunk = by_id[hit.chunk_id]
                    assert (hit.start, hit.end, hit.text) == (chunk.start, chunk.end, chunk.text)
                    assert TEXT[hit.start : hit.end] == hit.text
            else:
                assert not result.hits
            print(f"{q['id']} {q['kind']}: {q['text']}")
            for n, attempt in enumerate(trace.attempts, 1):
                if attempt.rejection:
                    plan_reasons[attempt.rejection[0]] += 1
                print(
                    f"  attempt={n} prompt_sha256={attempt.prompt_sha256[:12]} rejection={attempt.rejection} "
                    f"execution_status={attempt.execution_status}"
                )
                print(f"  plan_json={attempt.plan_json}")
            for step in trace.steps:
                print(f"  step={step.step_id} op={step.op} status={step.status} bindings={len(step.bindings)}")
            print(
                f"  status={trace.status} hits={len(result.hits)} "
                f"answer={[str(b.label) for b in trace.answer]} conflicts={len(trace.conflicts)}"
            )
            baseline = await dense.retrieve(q["text"])
            print(
                f"  recall R48={evidence_span_recall(result, q['gold'], view_id):.2f} (hits={len(result.hits)}) "
                f"R02 k=3={evidence_span_recall(baseline, q['gold'], view_id):.2f} (hits={len(baseline.hits)})"
            )
        print(f"plan rejection reasons by count={dict(sorted(plan_reasons.items()))}")
        print(f"plan calls={plan_calls} first_plans={first_plans} replans={replans}")
        _stage("planning", plan_usage)

        totals = chat_ledger.totals()
        print(
            f"chat ledger calls={totals.calls} input_tokens={totals.input_tokens} output_tokens={totals.output_tokens}"
        )
        assert totals.calls == len(chunks) + plan_calls

    asyncio.run(run())
