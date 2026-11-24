"""Small deterministic retriever for query transformation tests."""

from rag_foundry_techniques import RetrievalResult, RetrievedChunk


class RecordingRetriever:
    def __init__(self, hits_by_query: dict[str, tuple[str, ...]], technique_id="R02", index_version_id="v1"):
        self.hits_by_query = hits_by_query
        self.technique_id = technique_id
        self.index_version_id = index_version_id
        self.queries = []

    async def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        hits = tuple(
            RetrievedChunk(
                chunk_id=chunk_id,
                extraction_view_id="view",
                start=0,
                end=max(1, len(chunk_id)),
                block_ids=(),
                text=chunk_id,
                score=1.0,
                rank=rank,
            )
            for rank, chunk_id in enumerate(self.hits_by_query.get(query, ()), start=1)
        )
        return RetrievalResult(
            technique_id=self.technique_id, grounding="corpus", index_version_id=self.index_version_id, hits=hits
        )
