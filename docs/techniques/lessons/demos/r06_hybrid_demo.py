"""R06 lesson demo: hybrid R03 + R02 with RRF on a tiny in-memory index. Offline; test-only embedder."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root, for the tests.indexing helpers

from rag_foundry_domain.errors import IndexingError  # noqa: E402
from rag_foundry_indexing import HybridConfig, HybridRetriever  # noqa: E402

from tests.indexing._inmemory_index import ready_index  # noqa: E402
from tests.indexing._lexical_embedder import LexicalTestEmbedder  # noqa: E402

TEXTS = (
    "the tidal observatory records sea levels",
    "رقم القطعة AB-1234",
    "رقم القطعة AB-1243",
    "chess club meets on tuesday",
)


def spans():
    out, offset = [], 0
    for text in TEXTS:
        out.append((offset, offset + len(text), text))
        offset += len(text) + 2
    return out


async def main():
    embedder = LexicalTestEmbedder()
    owner, _, index, version = await ready_index(spans(), embedder)
    hybrid = HybridRetriever(HybridConfig(), scope=owner, index_version=version, index=index, embedder=embedder)
    result = await hybrid.retrieve("AB-1234")
    print("scoring:", result.scoring)
    for hit in result.hits:
        parts = [
            (c.source, c.source_rank, round(c.value, 6)) for c in hit.score_components if c.name == "rrf_contribution"
        ]
        print(hit.rank, round(hit.score, 6), ascii(hit.text), parts)
    try:
        await hybrid.retrieve("?!")
    except IndexingError as error:
        print("losing case: query '?!' ->", error.code, "(no lexical terms; the dense branch is not used alone)")


asyncio.run(main())
