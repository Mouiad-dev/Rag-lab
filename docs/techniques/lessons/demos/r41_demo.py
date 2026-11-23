"""P7.5 demo: R41 visual page retrieval and R02 text retrieval on chart_report.pdf, with test-only fakes.

Simulation only: the page encoder is a test-only fake (patch vectors from a hash of the PNG bytes, one vector per
query word from a hash of the word) and the R02 arm uses a test-only lexical hash embedder. Neither sees the chart
or understands language, so the ranks show the mechanics (render, encode, MaxSim, text anchoring), never a model
result. Output has no timings, IDs or paths and is byte-identical between runs.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for rel in ("", "tests/indexing"):
    sys.path.insert(0, str(ROOT / rel))

from _lexical_embedder import LexicalTestEmbedder  # noqa: E402
from _visual_fixture import (  # noqa: E402
    PAGES,
    QUESTIONS,
    build_pages,
    build_text_index,
    format_r02,
    format_r41,
    r02_chunk_ranking,
    r41_page_ranking,
)

from tests.providers._fake_page_encoder import FakePageEncoder  # noqa: E402


async def main() -> None:
    print("SIMULATION: fake page encoder, not a model result")
    print(
        "Fixture: chart_report.pdf, 4 pages (0 text, 1 bar chart, 2 table, 3 no text layer), one text chunk per PDF "
        "block. R41 ranks all 4 pages by MaxSim; R02 (lexical test embedder) ranks the text chunks, shown per page "
        "(best chunk per page) and per chunk (c<ordinal>@p<page>). Both rankings are within this one document. "
        "MaxSim scores are sums over query tokens, so they are not on the cosine scale of R02."
    )
    embedder = LexicalTestEmbedder()
    encoder = FakePageEncoder(dim=4, patches=3)
    setup = await build_text_index(embedder)
    built = await build_pages(setup, encoder)
    stats = built.stats()
    print()
    print("## Index size")
    print(f"pages={stats.pages} patches={stats.patches} dim={stats.dim} estimated_bytes={stats.estimated_bytes}")
    print()
    print("## Relevant-page rank and score per question")
    print("| question | arm | result |")
    print("|---|---|---|")
    top1_pages = {}
    for q in QUESTIONS:
        rows, hit_pages = await r41_page_ranking(setup, built, encoder, q["question"])
        chunk_rows = await r02_chunk_ranking(setup, built, embedder, q["question"])
        top1_pages[q["id"]] = (rows[0][0], hit_pages)
        print(f"| {q['id']} | r41_visual | {format_r41(q, rows)} hit_pages(k=3)={hit_pages} |")
        print(f"| {q['id']} | r02_text | {format_r02(q, chunk_rows)} |")
    print()
    print(f"Pages scored: {PAGES}. Page 3 has no text chunk: it is scored and reported as (no text), never a hit.")
    print(
        "q-unanswerable has no relevant page: only the top-1 is shown, and it still gets a top-1 in both arms; "
        "there is no abstention threshold here."
    )
    losing = [
        q["id"]
        for q in QUESTIONS
        if q["expected_page_index"] is not None and top1_pages[q["id"]][0] != q["expected_page_index"]
    ]
    print(f"Losing case: R41 top1_page differs from the expected page for {', '.join(losing) if losing else 'none'}.")


if __name__ == "__main__":
    asyncio.run(main())
