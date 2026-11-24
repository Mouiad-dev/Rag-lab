"""Opt-in local colSmol-256M page-image check (CPU, safetensors only, offline)."""

import asyncio
import os
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_VISUAL") != "1", reason="set RAG_LIVE_VISUAL=1"),
]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "multimodal" / "chart_report.pdf"
QUERIES = ["quarterly revenue chart", "which region grew the most"]


def test_local_page_encoder_fixture_pages(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # scoped; also proves nothing needs the network
    from rag_foundry_ingestion.page_images import render_pdf_pages
    from rag_foundry_providers.colpali_pages import ColPaliPageEncoder

    model_dir = os.environ.get("RAG_VISUAL_MODEL_DIR")
    if not model_dir:
        pytest.skip("set RAG_VISUAL_MODEL_DIR")
    pages = render_pdf_pages(FIXTURE.read_bytes())
    assert len(pages) == 4
    encoder = ColPaliPageEncoder(model_dir=model_dir)
    start = time.monotonic()
    result = asyncio.run(encoder.encode_pages([p.png for p in pages]))
    per_page = (time.monotonic() - start) / len(pages)  # includes the one-time model load
    queries = asyncio.run(encoder.encode_queries(QUERIES))
    assert result.dim == queries.dim == 128
    assert len(result.items) == 4 and len(queries.items) == 2
    assert all(len(i.vectors) > 0 and all(len(v) == 128 for v in i.vectors) for i in (*result.items, *queries.items))
    try:
        import psutil

        rss = f"{psutil.Process().memory_info().rss / 1e6:.0f} MB (current RSS at end)"
    except ImportError:
        rss = "not measured"
    print(
        f"revision={encoder.revision} dim={result.dim} patches_per_page={[len(i.vectors) for i in result.items]} "
        f"query_tokens={[len(i.vectors) for i in queries.items]} seconds_per_page={per_page:.1f} peak_rss={rss}"
    )
