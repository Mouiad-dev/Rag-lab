"""Opt-in live run of R49 (prefix-cache probe) and R47 (CAG) on local Ollama: gemma4:e4b and bge-m3.

One local run, not an evaluation; numbers vary between runs; no quality or speed assertion. Only structural
invariants are asserted: record counts, distinct cold prefixes, exact citations when answered, and that the
prompt fits the context window. Local Ollama only: no keys, no paid calls. Response caches stay off here (ADR-0021).

R47 comparison: the arms run as three blocks in this order: (1) all `cag` calls, (2) all `full_context_uncached`
calls, (3) R02 k=3. The uncached arm still shares the system prompt and the text before the first nonce with the
CAG prefix, so the runtime may reuse that part; it is uncached only after that point.
"""

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from _fixture import FACT_AR, FACT_EN, TEST_CHUNK_SIZE, TEST_OVERLAP
from _qa import corpus_markdown
from _templates import prompt_templates
from rag_foundry_answering.cag import CagConfig, CagWorkflow, build_cag_snapshot
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts
from rag_foundry_cache import PrefixCacheProbe, PrefixProbeConfig
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.index_testing import InMemoryIndexRepository
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

EN_Q = "How often does the Port Sallow observatory record sea levels?"
AR_Q = "كم مخطوطة تحفظ مكتبة الواحة؟"
OOC_Q = "What colour is the moon base?"
QUESTIONS = (("EN", EN_Q, FACT_EN), ("AR", AR_Q, FACT_AR), ("OUT_OF_CORPUS", OOC_Q, None))
PROMPTS = RunPrompts(*prompt_templates())
RESERVE = 512


def _chat(settings, ledger=None):
    model = OllamaChatModel(ollama_chat_profile(model="gemma4:e4b"), api_base=settings.ollama_base_url, timeout_s=600)
    return model if ledger is None else RecordingChatModel(model, ledger)


def _num(value):
    return "n/a" if value is None else f"{value:.0f}"


def _timings(timings):
    if timings is None:
        return "prefill_ms=n/a decode_ms=n/a load_ms=n/a"
    return f"prefill_ms={_num(timings.prefill_ms)} decode_ms={_num(timings.decode_ms)} load_ms={_num(timings.load_ms)}"


def test_r49_prefix_probe_live():
    async def run():
        settings = load_settings()
        chat = _chat(settings)
        questions = (EN_Q, AR_Q, OOC_Q)
        # One salt sequence for both modes, and a per-run, per-mode tag at the start of the prefix: no prompt of
        # this run repeats a prompt of an earlier mode or an earlier run, so cold calls are really cold.
        salts = iter(f"{i:016x}" for i in range(1, 1000))
        run_tag = uuid.uuid4().hex
        for mode in ("complete", "stream"):
            prefix = f"[probe run {run_tag} mode {mode}]\n{corpus_markdown()}"
            probe = PrefixCacheProbe(PrefixProbeConfig(prefix_text=prefix, questions=questions, mode=mode))
            report = await probe.run(chat, clock=time.perf_counter, salt_factory=salts.__next__)
            print(f"R49 mode={mode} cache_metric={report.cache_metric} note={report.note}")
            for r in report.records:
                print(
                    f"R49 {mode} arm={r.arm} label={r.label} ttft_ms={_num(r.ttft_ms)} total_ms={_num(r.total_ms)}",
                    _timings(r.timings),
                    f"input_tokens={r.input_tokens} cache_read_input_tokens={r.cache_read_input_tokens}",
                    f"cache_read_reported={r.cache_read_reported} output_tokens={r.output_tokens}",
                )
            for s in report.arms:
                print(
                    f"R49 {mode} summary arm={s.arm} calls={s.calls} ttft_median={s.ttft_ms_median}",
                    f"prefill_median={s.prefill_ms_median} decode_median={s.decode_ms_median}",
                    f"cache_read_median={s.cache_read_tokens_median}",
                )
            # 3 cold + (prime + 3) warm + 1 changed + 2 eviction attempt.
            assert len(report.records) == 3 + 4 + 1 + 2
            assert [s.calls for s in report.arms] == [3, 4, 1, 2]
            cold = [r for r in report.records if r.arm == "cold"]
            assert len({r.prefix_sha256 for r in cold}) == len(cold) == 3  # a fresh salt per cold call
            assert report.cache_metric in ("reported", "unavailable")
            if mode == "stream":
                assert all(r.ttft_ms is not None for r in report.records)

    asyncio.run(run())


def test_r47_cag_vs_uncached_vs_r02_live():
    async def run():
        settings = load_settings()
        profile = ollama_chat_profile(model="gemma4:e4b")
        num_ctx = profile.capabilities.context_window
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
            index_ledger = UsageLedger()
            version = await IndexService(
                store, InMemoryUnitOfWorkFactory(extractions=views, indexes=index), anchors
            ).build(scope, (view.id,), chunker, RecordingEmbeddingModel(embedder, index_ledger))
            print(f"index chunks_embed_calls={index_ledger.totals().calls} num_ctx={num_ctx} reserve={RESERVE}")

            snapshot = await build_cag_snapshot(
                scope,
                index,
                version.id,
                template=PROMPTS.grounded,
                chat_profile=profile,
                reserve_output_tokens=RESERVE,
                max_question_tokens=64,
                max_evidence_chars=100_000,
            )
            print(
                f"snapshot items={len(snapshot.items)} estimated_prompt_tokens={snapshot.estimated_prompt_tokens}",
                f"margin={snapshot.estimate_margin}",
            )
            print("arm order: (1) cag (prime, EN, AR, OUT_OF_CORPUS) (2) full_context_uncached (3) R02 k=3")

            async def resolve_exact(result):
                if result.status != "answered":
                    return
                resolved = await anchors.resolve(scope, tuple(a for c in result.citations for a in c.anchors))
                assert resolved and all(item.status == "exact" for item in resolved)

            def show(arm, label, result, ledger, wall_ms):
                usage = ledger.records()
                assert usage, "no ledger record for this call"
                last = usage[-1]  # the answer chat call is the last ledger record
                print(
                    f"R47cmp arm={arm} label={label} status={result.status}",
                    f"markers={[c.marker for c in result.citations]} wall_ms={wall_ms:.0f}",
                    _timings(last.timings),
                    f"input_tokens={last.input_tokens} cache_read_input_tokens={last.cache_read_input_tokens}",
                    f"cache_read_reported={last.cache_read_reported} output_tokens={last.output_tokens}",
                )
                return last

            def fits(arm, label, last):
                total = last.input_tokens + last.cache_read_input_tokens
                double = last.cache_read_input_tokens > 0 and last.input_tokens >= last.cache_read_input_tokens
                assert total < num_ctx - RESERVE, (
                    f"{arm}/{label}: input+cached={total} >= num_ctx-reserve={num_ctx - RESERVE}; "
                    f"input_tokens={last.input_tokens} cached={last.cache_read_input_tokens} "
                    f"(input>=cached, possible double counting of cached tokens: {double}); "
                    f"snapshot estimate={snapshot.estimated_prompt_tokens}"
                )

            for arm in ("cag", "full_context_uncached"):
                workflow = CagWorkflow(CagConfig(arm=arm), snapshot=snapshot, anchors=anchors, index=index)
                calls = [("prime", EN_Q)] if arm == "cag" else []
                calls += [(name, question) for name, question, _fact in QUESTIONS]
                first = True
                for label, question in calls:
                    ledger = UsageLedger()
                    started = time.perf_counter()
                    result = await workflow.answer(
                        scope, question, prompts=PROMPTS, chat=_chat(settings, ledger), ledger=ledger
                    )
                    wall = (time.perf_counter() - started) * 1000
                    last = show(arm, label, result, ledger, wall)
                    await resolve_exact(result)
                    assert ledger.totals().calls == 1
                    if arm == "full_context_uncached" or first:
                        fits(arm, label, last)
                    first = False

            retriever = DenseRetriever(
                DenseConfig(k=3), scope=scope, index_version=version, index=index, embedder=embedder
            )
            for name, question, _fact in QUESTIONS:
                ledger = UsageLedger()
                started = time.perf_counter()
                result = await AnswerService(anchors).answer(
                    scope,
                    question,
                    prompts=PROMPTS,
                    retriever=retriever,
                    technique_version="1.0.0",
                    index_version=version,
                    chat=_chat(settings, ledger),
                    ledger=ledger,
                    config=AnswerConfig(min_score=0.0, k=3),
                )
                wall = (time.perf_counter() - started) * 1000
                show("R02_k3", name, result, ledger, wall)
                await resolve_exact(result)

    asyncio.run(run())
