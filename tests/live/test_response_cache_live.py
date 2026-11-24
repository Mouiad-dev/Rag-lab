"""Opt-in live run of R50 (exact response cache) and R51 (semantic calibration) on local Ollama.

Models: gemma4:e4b (chat) and bge-m3 (embeddings). One local run, not an evaluation; numbers vary between runs;
no quality or speed assertion. R50 asserts only that a hit returns the cold answer with zero usage. R51 asserts
only that calibration runs over every pair; the table, threshold (or None) and held-out rates are printed.
The store is in memory; the SQL store is covered by `-m db`. Local Ollama only: no keys, no paid calls.
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _fixture import QUERY_AR, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _qa import corpus_markdown
from _templates import prompt_templates
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_cache import (
    AllowAllSourceGate,
    ExactCacheConfig,
    ExactResponseCache,
    GenerationParams,
    ResponseKeyInputs,
    calibrate,
)
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.index_testing import InMemoryIndexRepository
from rag_foundry_domain.response_cache import InMemoryResponseCacheStore
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger
from test_extraction_service import ingest, setup

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

PAIRS = Path(__file__).resolve().parents[1] / "cache" / "data" / "semantic_pairs_v1.json"
GRID = tuple(round(0.5 + i * 0.01, 2) for i in range(51))
PROMPTS = RunPrompts(*prompt_templates())
QUESTIONS = (
    ("EN", QUERY_EN),
    ("AR", QUERY_AR),
    ("EN2", "How many manuscripts does the oasis library keep?"),
)


def test_r50_cold_then_warm_live():
    async def run():
        settings = load_settings()
        profile = ollama_chat_profile(model="gemma4:e4b")
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url)
        with tempfile.TemporaryDirectory() as tmp:
            extraction, intake, _spy, store, _docs, views, scope = setup(Path(tmp))
            doc_id, version_id = await ingest(
                intake, scope, corpus_markdown().encode("utf-8"), name="qa-notes.md", media="text/markdown"
            )
            view = (await extraction.extract(scope, doc_id, version_id)).view
            index = InMemoryIndexRepository(views)
            anchors = AnchorService(
                store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace=settings.provider_namespace
            )
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            version = await IndexService(
                store, InMemoryUnitOfWorkFactory(extractions=views, indexes=index), anchors
            ).build(scope, (view.id,), chunker, embedder)

            defaults = AnswerConfig()
            inputs = ResponseKeyInputs(
                authz_revision="live-1",
                history=(),
                technique_id="R02",
                technique_version="1.0.0",
                retriever_config_sha256="a" * 64,
                index_version_id=version.id,
                embedding_key=version.embedding_key,
                prompt_template=PROMPTS.grounded.id,
                prompt_sha256=PROMPTS.grounded.content_sha256,
                chat_profile_id=profile.id,
                chat_provider=profile.provider,
                chat_model=profile.model,
                generation=GenerationParams(
                    max_output_tokens=defaults.max_output_tokens,
                    temperature=defaults.temperature,
                    k=3,
                    min_score=0.0,
                    max_evidence_chars=defaults.max_evidence_chars,
                    packer="none",
                    citation_check="off",
                ),
                settings_sha256=None,
                mode="source_only",
                freshness="static_snapshot",
                tools=(),
                is_action_run=False,
                output_language="auto",
                contract_version="1.1",
            )
            cache = ExactResponseCache(ExactCacheConfig(ttl_s=86400), store=InMemoryResponseCacheStore())
            answer_calls = []

            def answer_fn(question):
                async def fn():
                    ledger = UsageLedger()
                    answer_calls.append(ledger)
                    retriever = DenseRetriever(
                        DenseConfig(k=3),
                        scope=scope,
                        index_version=version,
                        index=index,
                        embedder=RecordingEmbeddingModel(embedder, ledger),
                    )
                    chat = OllamaChatModel(profile, api_base=settings.ollama_base_url, timeout_s=600)
                    return await AnswerService(anchors).answer(
                        scope,
                        question,
                        prompts=PROMPTS,
                        retriever=retriever,
                        technique_version="1.0.0",
                        index_version=version,
                        chat=RecordingChatModel(chat, ledger),
                        ledger=ledger,
                        config=AnswerConfig(min_score=0.0, k=3),
                    )

                return fn

            async def go(question):
                started = time.perf_counter()
                result = await cache.run(
                    scope,
                    question,
                    inputs,
                    answer_fn(question),
                    expected_index_version_id=version.id,
                    anchors=anchors,
                    gate=AllowAllSourceGate(),
                    clock=lambda: datetime.now(UTC),
                    run_id_factory=lambda: str(uuid.uuid4()),
                )
                return result, (time.perf_counter() - started) * 1000

            pairs = 0
            for name, question in QUESTIONS:
                cold, cold_ms = await go(question)
                calls_after_cold = len(answer_calls)
                warm, warm_ms = await go(question)
                print(
                    f"R50 {name} cold outcome={cold.trace.outcome} status={cold.response.status}",
                    f"total_ms={cold_ms:.0f} lookup_ms={cold.trace.lookup_ms:.1f}",
                    f"model_calls={cold.response.usage.model_calls}",
                    f"input_tokens={cold.response.usage.input_tokens}",
                    f"store_reason={cold.trace.store_reason}",
                )
                print(
                    f"R50 {name} warm outcome={warm.trace.outcome} reason={warm.trace.reason}",
                    f"total_ms={warm_ms:.0f} lookup_ms={warm.trace.lookup_ms:.1f}",
                    f"calls_avoided={warm.trace.calls_avoided}",
                    f"input_tokens_avoided={warm.trace.input_tokens_avoided}",
                    f"model_calls={warm.response.usage.model_calls}",
                )
                assert cold.trace.outcome == "miss" and calls_after_cold >= 1
                if warm.trace.outcome == "hit":
                    pairs += 1
                    assert len(answer_calls) == calls_after_cold  # no answer call on the hit
                    assert warm.response.answer == cold.response.answer
                    assert warm.response.usage.model_calls == 0
                    assert warm.response.usage.input_tokens == 0 and warm.response.usage.output_tokens == 0
                else:
                    # Only an unstored (not cacheable, e.g. abstained) cold answer may miss again.
                    assert cold.trace.store_reason is not None
            print(f"R50 hits={pairs} of {len(QUESTIONS)}")
            assert pairs >= 1

    asyncio.run(run())


def test_r51_bge_m3_calibration_live():
    async def run():
        settings = load_settings()
        cache_ledger = UsageLedger()
        embedder = RecordingEmbeddingModel(
            OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url), cache_ledger
        )
        pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
        vectors = {}
        for pair in pairs:
            result = await embedder.embed([pair["a"], pair["b"]])
            vectors[pair["id"]] = (result.vectors[0], result.vectors[1])
        print(f"R51 pairs={len(pairs)} embed_calls={cache_ledger.totals().calls} (separate cache ledger)")
        report = calibrate(pairs, vectors, GRID)
        print("threshold  dev paraphrase hits (ug=unguarded g=guarded)  dev false hits (ug g)")
        for row in report.rows:
            d = row.dev
            para = f"ug={d.paraphrase_hits_unguarded} g={d.paraphrase_hits_guarded} of {d.paraphrase_total}"
            false = f"ug={d.false_hits_unguarded} g={d.false_hits_guarded} of {d.adversarial_total}"
            print(f"{row.threshold:<9}  {para}  {false}")
        print(f"R51 chosen_threshold={report.chosen_threshold} reason={report.reason}")
        held = report.held_out
        if held is None:
            print("R51 held_out=None (nothing chosen, R51 must not be used)")
        else:
            print(
                f"R51 held_out paraphrase_hits_guarded={held.paraphrase_hits_guarded}/{held.paraphrase_total}",
                f"paraphrase_hits_unguarded={held.paraphrase_hits_unguarded}/{held.paraphrase_total}",
                f"false_hits_guarded={held.false_hits_guarded}/{held.adversarial_total}",
                f"false_hits_unguarded={held.false_hits_unguarded}/{held.adversarial_total}",
            )
        assert len(report.rows) == len(GRID) and len(vectors) == len(pairs)
        assert (report.chosen_threshold is None) == (report.held_out is None)

    asyncio.run(run())
