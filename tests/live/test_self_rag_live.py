"""Opt-in local R31 run. This records one model run and asserts response structure only."""

import asyncio
import os
import time

import pytest
from rag_foundry_answering.self_rag import SelfRagConfig, SelfRagWorkflow

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RAG_LIVE_OLLAMA") != "1" or not os.environ.get("RAG_SELFRAG_MODEL"),
        reason="set RAG_LIVE_OLLAMA=1 and RAG_SELFRAG_MODEL",
    ),
]

_RUNTIME = os.environ.get("RAG_SELFRAG_RUNTIME", "ollama")
if _RUNTIME not in {"ollama", "llamacpp"}:
    raise ValueError(f"unknown RAG_SELFRAG_RUNTIME: {_RUNTIME}")


DOCUMENT = """# Northbridge reference

The Northbridge observatory is on the western coast. It records the tide at 06:00 and 18:00 daily.
Its public archive opened in 2018 and contains monthly summaries of coastal water levels.
The observatory uses a pressure sensor calibrated each January.
"""

QUESTIONS = (
    ("retrieval", "At what times does the Northbridge observatory record the tide?"),
    ("no_retrieval", "Write a friendly one-sentence greeting."),
    ("unanswerable", "What is the observatory director's favorite food?"),
    (
        "arabic_losing_case",
        "\u0645\u062a\u0649 \u064a\u0633\u062c\u0644 "
        "\u0627\u0644\u0645\u0631\u0635\u062f \u0627\u0644\u0645\u062f\u064a\u061f",
    ),
)


def test_self_rag_short_form_live(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    async def run():
        from rag_foundry_answering.self_rag import REFLECTION_TOKENS
        from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
        from rag_foundry_core.settings import load_settings
        from rag_foundry_domain.index_testing import InMemoryIndexRepository
        from rag_foundry_domain.testing import make_scope
        from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
        from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService
        from rag_foundry_ingestion.anchor_service import AnchorService
        from rag_foundry_providers.adapters.ollama import (
            OllamaEmbeddingModel,
            OllamaRawCompletionModel,
            ollama_embedding_profile,
            ollama_raw_completion_profile,
        )
        from rag_foundry_providers.messages import RawCompletionRequest
        from test_extraction_service import ingest, setup

        model_name = os.environ["RAG_SELFRAG_MODEL"]
        runtime = _RUNTIME
        scope = make_scope()
        extraction, intake, _spy, store, _docs, views, _scope = setup(tmp_path, scope)
        doc_id, version_id = await ingest(
            intake, scope, DOCUMENT.encode("utf-8"), name="northbridge.md", media="text/markdown"
        )
        view = (await extraction.extract(scope, doc_id, version_id)).view
        index = InMemoryIndexRepository(views)
        anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="live")
        settings = load_settings()
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url)
        index_service = IndexService(store, InMemoryUnitOfWorkFactory(extractions=views, indexes=index), anchors)
        chunker = RecursiveChunker(RecursiveConfig(chunk_size=500, chunk_overlap=0))
        version = await index_service.build(scope, (view.id,), chunker, embedder)
        retriever = DenseRetriever(DenseConfig(k=5), scope=scope, index_version=version, index=index, embedder=embedder)
        if runtime == "ollama":
            raw_model = OllamaRawCompletionModel(
                ollama_raw_completion_profile(model=model_name), api_base=settings.ollama_base_url
            )
            config = SelfRagConfig()
            check_k = 20
        else:
            from rag_foundry_providers.adapters.llamacpp import (
                LlamaCppRawCompletionModel,
                llamacpp_raw_completion_profile,
            )

            raw_model = LlamaCppRawCompletionModel(
                llamacpp_raw_completion_profile(model=model_name),
                vocab_size=32016,
                base_url=os.environ.get("RAG_LLAMACPP_BASE_URL", "http://127.0.0.1:8091"),
            )
            config = SelfRagConfig(decision_top_logprobs=32016, evidence_top_logprobs=5000)
            check_k = 32016
        workflow = SelfRagWorkflow(config)
        print(f"model={model_name} embedding_model={embedder.profile.model} reflection_tokens={len(REFLECTION_TOKENS)}")
        total_scored = total_missing = 0
        for name, question in QUESTIONS:
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            # Warm the prompt cache first: cold and warm calls differ by up to ~0.07 logprob (measured 2026-11-24),
            # so both compared calls must run in the same cache state; only top_k may differ.
            await raw_model.complete_raw(
                RawCompletionRequest(
                    prompt=prompt, max_output_tokens=1, temperature=0.0, top_p=1.0, top_k=0, top_logprobs=check_k
                )
            )
            top_one = await raw_model.complete_raw(
                RawCompletionRequest(
                    prompt=prompt, max_output_tokens=1, temperature=0.0, top_p=1.0, top_k=1, top_logprobs=check_k
                )
            )
            top_zero = await raw_model.complete_raw(
                RawCompletionRequest(
                    prompt=prompt, max_output_tokens=1, temperature=0.0, top_p=1.0, top_k=0, top_logprobs=check_k
                )
            )
            a = top_one.tokens[0].top if top_one.tokens else ()
            b = top_zero.tokens[0].top if top_zero.tokens else ()
            ids_a = top_one.tokens[0].top_ids if top_one.tokens else None
            ids_b = top_zero.tokens[0].top_ids if top_zero.tokens else None
            if ids_a is not None and ids_b is not None:
                same_order = bool(a) and list(ids_a) == list(ids_b)
            else:
                same_order = bool(a) and [x[0] for x in a] == [x[0] for x in b]
            max_diff = max((abs(x[1] - y[1]) for x, y in zip(a, b, strict=False)), default=float("inf"))
            # Post-sampler top_k=1 would leave one token; tiny float noise from prompt-cache reuse is tolerated.
            equal = same_order and max_diff <= 1e-3
            print(
                f"{name} pre_sampler_same_order={same_order} pre_sampler_max_abs_diff={max_diff:.6g} "
                f"top_logprobs={check_k}"
            )
            started = time.perf_counter()
            result = await workflow.answer(question, retriever=retriever, raw_model=raw_model)
            elapsed = time.perf_counter() - started
            trace = workflow.last_trace
            decision = trace["retrieval_decision"]
            print(
                f"{name} pre_sampler_check={'equal' if equal else 'different'} "
                f"rule={decision['rule']} p_ret={decision['p_retrieval']} "
                f"alternatives={decision['alternatives_returned']} match_mode={decision['match_mode']} "
                f"decision={decision} scored_positions={trace['scored_positions']} "
                f"positions_missing_group_tokens={trace['positions_missing_group_tokens']}"
            )
            for candidate in trace["candidates"]:
                print(f"{name} candidate={candidate}")
            tokens = sum(call["usage"].get("output_tokens", 0) for call in trace["calls"])
            citation = result.citations[0].chunk_id if result.citations else None
            print(
                f"{name} answer={result.answer!r} citation={citation} latency_s={elapsed:.3f} "
                f"tokens={tokens} status={result.status}"
            )
            total_scored += trace["scored_positions"]
            total_missing += trace["positions_missing_group_tokens"]
            assert result.status in {"answered", "abstained"}
            assert decision["rule"] in {"probability_threshold", "undetermined_topk_generated_token"}
            assert trace["scored_positions"] >= 1
            if result.citations:
                assert len(result.citations) == 1
        share = total_missing / max(total_scored, 1)
        print(f"run scored_positions={total_scored} positions_missing_group_tokens={total_missing} share={share:.2f}")

    asyncio.run(run())
