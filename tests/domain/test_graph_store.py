import asyncio

from rag_foundry_domain.index_testing import GRAPH_STORE_CHECKS, InMemoryIndexRepository
from rag_foundry_domain.ports import EntityGraphStore, IndexRepository


def test_in_memory_entity_graph_store_checks() -> None:
    from rag_foundry_domain.testing import InMemoryDocumentRepository, InMemoryExtractionRepository

    async def factory(scopes):
        docs = InMemoryDocumentRepository(scopes)
        views = InMemoryExtractionRepository(docs)
        repo = InMemoryIndexRepository(views)
        assert isinstance(repo, IndexRepository) and isinstance(repo, EntityGraphStore)
        return docs, views, repo

    for check in GRAPH_STORE_CHECKS:
        asyncio.run(check(factory))
