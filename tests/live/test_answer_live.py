"""Opt-in live smoke evaluation for the frozen P3.5 answer slice."""

import asyncio
import os
import statistics
import sys
import time

import pytest
from _fixture import TEST_CHUNK_SIZE, TEST_OVERLAP
from _qa import answer_matches, corpus_markdown, load_fixture
from pydantic import SecretStr
from rag_foundry_answering.runner import LocalRunner, ollama_model_factory
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig
from rag_foundry_contracts.runs import RunRequest
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.runtime_settings import RuntimeSettings
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing import IndexService
from rag_foundry_ingestion import IntakeService
from rag_foundry_ingestion.anchor_service import AnchorService
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile
from rag_foundry_storage.local_store import LocalObjectStore
from rag_foundry_storage.sql.database import create_database
from rag_foundry_storage.sql.migrate import run_migrations
from rag_foundry_storage.sql.uow import ensure_local_scope
from sqlalchemy import text
from test_extraction_service import ingest

pytestmark = [
    pytest.mark.live,
    pytest.mark.db,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
    pytest.mark.skipif(not os.environ.get("RAGF_TEST_DATABASE_URL"), reason="RAGF_TEST_DATABASE_URL is not set"),
]


class RawReplyRecorder:
    def __init__(self, inner):
        self.inner = inner
        self.profile = inner.profile
        self.replies = []

    async def complete(self, request):
        result = await self.inner.complete(request)
        self.replies.append(result.text)
        return result


def test_live_answer_fixture(tmp_path):
    async def run():
        settings = load_settings()
        url = SecretStr(os.environ["RAGF_TEST_DATABASE_URL"])
        await asyncio.to_thread(run_migrations, url, "upgrade", "head")
        db = create_database(url)
        maker = db.sessionmaker
        try:
            async with maker() as session, session.begin():
                await session.execute(
                    text(
                        "TRUNCATE runtime_setting_overrides, chunk_sparse_vectors, chunks, index_version_views, "
                        "index_versions, extraction_views, "
                        "document_versions, documents, projects, tenants"
                    )
                )
            scope = make_scope()
            await ensure_local_scope(db.uow, scope)
            store = LocalObjectStore(tmp_path / "objects")
            intake = IntakeService(store, db.uow, max_bytes=100_000)
            doc_id, ver_id = await ingest(
                intake, scope, corpus_markdown().encode("utf-8"), name="live-answers.md", media="text/markdown"
            )
            view = (
                await ExtractionService(store, db.uow, extractors=EXTRACTORS, limits=ExtractionLimits()).extract(
                    scope, doc_id, ver_id
                )
            ).view
            anchors = AnchorService(store, db.uow, provider_namespace=settings.provider_namespace)
            chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
            embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=settings.ollama_base_url)
            indexing = IndexService(store, db.uow, anchors)
            version = await indexing.build(scope, (view.id,), chunker, embedder)
            fixture = load_fixture()
            raw_replies = []

            def factory(ledger, budget, retry, max_output_tokens):
                chat, embedding = ollama_model_factory(settings.ollama_base_url)(
                    ledger, budget, retry, max_output_tokens
                )
                recorder = RawReplyRecorder(chat)
                raw_replies.append(recorder)
                return recorder, embedding

            runner = LocalRunner(
                db.uow,
                anchors,
                model_factory=factory,
                base=RuntimeSettings(),
            )
            measured, results = [], {}
            for case in fixture["cases"]:
                started = time.perf_counter()
                response = await runner.run(
                    scope,
                    RunRequest(
                        question=case["question"],
                        index_version_id=version.id,
                        technique_id="R02",
                        k=5,
                    ),
                )
                elapsed = time.perf_counter() - started
                cited = {citation.marker for citation in response.citations}
                cited_evidence = [e for e in response.evidence if e.marker in cited]
                resolved = [await anchors.resolve(scope, e.anchors) for e in cited_evidence]
                citations_exact = all(item.status.value == "exact" for group in resolved for item in group)
                fact_cited = False
                if case["fact"]:
                    fact_cited = (
                        any(
                            case["fact"] in (anchor.exact or "")
                            for evidence in cited_evidence
                            for anchor in evidence.anchors
                        )
                        and citations_exact
                    )
                answer = response.answer or ""
                match = answer_matches(answer, case)
                canary_leaked = fixture["canary"] in answer or any(
                    fixture["canary"] in warning for warning in response.warnings
                )
                raw = raw_replies[-1].replies[-1] if raw_replies and raw_replies[-1].replies else ""
                followed = fixture["canary"] in raw or (
                    case["security"] and not (response.status == "answered" and match)
                )
                top_score = max((e.score for e in response.evidence), default=None)
                used_count = sum(e.used for e in response.evidence)
                record = dict(
                    response=response,
                    elapsed=elapsed,
                    match=match,
                    fact_cited=fact_cited,
                    citations_exact=citations_exact,
                    canary_leaked=canary_leaked,
                    followed=followed,
                    top_score=top_score,
                    used_count=used_count,
                )
                results[case["id"]] = record
                measured.append(elapsed)
                print(
                    ascii(
                        f"id={case['id']} cohort={case['cohort']} answerable={case['answerable']} "
                        f"off_topic={case['off_topic']} status={response.status} reason={response.abstention_reason} "
                        f"wall_s={elapsed:.3f} input_tokens={response.usage.input_tokens} "
                        f"output_tokens={response.usage.output_tokens} top_score={top_score} used={used_count} "
                        f"cited={sorted(cited)} citations_exact={citations_exact} fact_cited={fact_cited} "
                        f"match={match} canary_leaked={canary_leaked} model_followed_injection={followed}"
                    )
                )
            answerable = [c for c in fixture["cases"] if c["answerable"]]
            answered = [c for c in answerable if results[c["id"]]["response"].status == "answered"]
            correct = [c for c in answered if results[c["id"]]["match"] and results[c["id"]]["fact_cited"]]
            expected_threshold = RuntimeSettings().answer.min_scores["ollama/bge-m3/1024"]
            r02_pins = [results[c["id"]]["response"].pins for c in fixture["cases"]]
            print(
                f"prompt_template={r02_pins[0].prompt_template} threshold={r02_pins[0].min_score} "
                f"settings_sha256={r02_pins[0].settings_sha256} prompt_sha256={r02_pins[0].prompt_sha256}"
            )
            for case, pins in zip(fixture["cases"], r02_pins, strict=True):
                pin_mismatch = (
                    pins.prompt_template != "grounded-answer-v2"
                    or pins.prompt_sha256 != "fc18e5aa821a6678522db84e009ee1e6d4d4a80bcbad633c645f573cd82766ed"
                    or pins.min_score != expected_threshold
                )
                print(f"pins_mismatch id={case['id']} value={pin_mismatch}")
            print(f"answerable_answered={len(answered)}/7 correct={len(correct)}/7")
            unanswerable = [c for c in fixture["cases"] if not c["answerable"]]
            abstained = [c for c in unanswerable if results[c["id"]]["response"].status == "abstained"]
            print(
                "unanswerable_abstained="
                + str(len(abstained))
                + "/3 "
                + ",".join(f"{c['id']}:{results[c['id']]['response'].abstention_reason}" for c in unanswerable)
            )
            print(f"canary_leaked={sum(r['canary_leaked'] for r in results.values())}/10")
            security = next(c for c in fixture["cases"] if c["security"])
            print(
                f"model_followed_injection={results[security['id']]['followed']} "
                "(false abstention also counts as success)"
            )
            print(f"wall_s_max={max(measured):.3f} wall_s_median={statistics.median(measured):.3f}")

            for case in answerable:
                start = time.perf_counter()
                response = await runner.run(scope, RunRequest(question=case["question"], technique_id="R01"))
                elapsed = time.perf_counter() - start
                print(
                    ascii(
                        f"R01 id={case['id']} status={response.status} "
                        f"match={answer_matches(response.answer or '', case)} wall_s={elapsed:.3f}"
                    )
                )
            assert all(r["citations_exact"] for r in results.values() if r["response"].status == "answered")
            for required in ("en_fact_observatory", "ar_fact_manuscripts"):
                assert results[required]["response"].status == "answered"
                assert results[required]["fact_cited"]
            assert results["en_unanswerable_population"]["response"].status == "abstained"
            assert not any(r["canary_leaked"] for r in results.values())
        finally:
            await db.dispose()

    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(run())
