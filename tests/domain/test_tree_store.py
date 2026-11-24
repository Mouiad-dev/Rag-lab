import asyncio

from rag_foundry_domain.index_testing import TREE_STORE_CHECKS, InMemoryIndexRepository
from rag_foundry_domain.ports import IndexRepository, SummaryTreeStore


def test_in_memory_summary_tree_store_checks() -> None:
    from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository

    async def factory(scopes):
        docs = InMemoryDocumentRepository(scopes)
        views = InMemoryExtractionRepository(docs)
        repo = InMemoryIndexRepository(views)
        assert isinstance(repo, IndexRepository) and isinstance(repo, SummaryTreeStore)
        return docs, views, repo

    for check in TREE_STORE_CHECKS:
        asyncio.run(check(factory))
