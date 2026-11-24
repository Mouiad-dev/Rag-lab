"""Opt-in live run of R41 (local colSmol-256M, CPU) against R02 text retrieval on chart_report.pdf.

One local run, not an evaluation; numbers vary between runs; no quality or speed assertion. Only structure is
asserted: the page orders are not identical for every question, and the blank page 3 never appears in hits.
R41 needs RAG_LIVE_VISUAL=1 and RAG_VISUAL_MODEL_DIR. The R02 arm also needs RAG_LIVE_OLLAMA=1 (local bge-m3);
without it that arm is skipped and the run says so. Local only: no keys, no paid calls, no download.
"""

import asyncio
import os
import time

import pytest
from _lexical_embedder import LexicalTestEmbedder
from _visual_fixture import (
    PAGES,
    QUESTIONS,
    build_pages,
    build_text_index,
    format_r02,
    format_r41,
    r02_chunk_ranking,
    r41_page_ranking,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_VISUAL") != "1", reason="set RAG_LIVE_VISUAL=1"),
]


def test_r41_vs_r02_on_chart_report_live(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model_dir = os.environ.get("RAG_VISUAL_MODEL_DIR")
    if not model_dir:
        pytest.skip("set RAG_VISUAL_MODEL_DIR")
    from rag_foundry_providers.colpali_pages import ColPaliPageEncoder

    use_r02 = os.environ.get("RAG_LIVE_OLLAMA") == "1"
    if use_r02:
        from rag_foundry_core.settings import load_settings
        from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile

        embedder = OllamaEmbeddingModel(ollama_embedding_profile(), api_base=load_settings().ollama_base_url)
    else:
        embedder = LexicalTestEmbedder()  # only fills the chunk vectors the index needs; never scored here
        print("R02 arm SKIPPED: RAG_LIVE_OLLAMA=1 is not set (text index uses a test embedder and is not queried)")

    async def run():
        encoder = ColPaliPageEncoder(model_dir=model_dir)
        setup = await build_text_index(embedder)
        started = time.perf_counter()
        built = await build_pages(setup, encoder)
        build_s = time.perf_counter() - started
        stats = built.stats()
        print(
            f"model={encoder.profile.model} revision={encoder.revision} dim={stats.dim} pages={stats.pages} "
            f"patches={stats.patches} estimated_bytes={stats.estimated_bytes}"
        )
        print(f"build_s_total={build_s:.1f} build_s_per_page={build_s / PAGES:.1f} (includes the one-time model load)")
        print(
            "Rankings are within this one 4-page document. R41 scores are MaxSim sums over query tokens; R02 scores "
            "are cosine: the scales differ. Page 3 has no text and can never be a hit."
        )
        orders, hit_pages_all = {}, []
        for q in QUESTIONS:
            started = time.perf_counter()
            rows, hit_pages = await r41_page_ranking(setup, built, encoder, q["question"])
            query_s = (time.perf_counter() - started) / 2  # two retrieve calls, each encodes the query once
            orders[q["id"]] = tuple(r[0] for r in rows)
            hit_pages_all.append(hit_pages)
            print(
                f"R41 {q['id']} expected={q['expected_page_index']} {format_r41(q, rows)} "
                f"hit_pages(k=3)={hit_pages} query_s={query_s:.2f}"
            )
            if use_r02:
                chunk_rows = await r02_chunk_ranking(setup, built, embedder, q["question"])
                print(f"R02 {q['id']} expected={q['expected_page_index']} {format_r02(q, chunk_rows)}")
        print(
            "q-unanswerable has no relevant page: its top-1 is shown and there is no abstention threshold, so both "
            "arms still return a top-1."
        )
        assert len(set(orders.values())) >= 2, "all questions produced the same page order"
        assert all(3 not in pages for pages in hit_pages_all), "blank page 3 appeared in hits"

    asyncio.run(run())
