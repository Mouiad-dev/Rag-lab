"""In-process Okapi BM25 retrieval for one immutable index version."""

from collections import Counter
from math import log
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState, IndexVersion
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_techniques import (
    Fidelity,
    PluginKind,
    PluginManifest,
    RetrievalResult,
    RetrievedChunk,
    ScoreComponent,
)

from rag_foundry_indexing.lexical_text import lexical_tokens


class BM25Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(default=5, ge=1, le=50)
    k1: float = Field(default=1.2, ge=0.0, le=3.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    max_chunks: int = Field(default=20000, ge=1, le=50000)


class BM25Retriever:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R03",
        kind=PluginKind.RETRIEVER,
        version="1.0.0",
        fidelity=Fidelity.FAITHFUL,
        config_model=BM25Config,
    )

    def __init__(
        self,
        config: BM25Config,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index: IndexRepository,
    ) -> None:
        self.config, self._scope, self._version, self._index = config, scope, index_version, index
        self._stats: tuple[list, list[Counter[str]], list[int], Counter[str], float] | None = None

    async def _build_stats(self):
        docs = []
        term_freqs = []
        lengths = []
        document_freq = Counter()
        offset = 0
        while True:
            page = await self._index.list_chunks(self._scope, self._version.id, offset=offset, limit=500)
            if not page:
                break
            for chunk in page:
                tokens = lexical_tokens(chunk.text)
                frequencies = Counter(tokens)
                docs.append(chunk)
                term_freqs.append(frequencies)
                lengths.append(len(tokens))
                document_freq.update(frequencies.keys())
                if len(docs) > self.config.max_chunks:
                    raise IndexingError("lexical_corpus_too_large")
            offset += len(page)
            if len(page) < 500:
                break
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        self._stats = docs, term_freqs, lengths, document_freq, average_length
        return self._stats

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise IndexingError("empty_query")
        if len(query) > 2000:
            raise IndexingError("query_too_long")
        query_terms = tuple(dict.fromkeys(lexical_tokens(query)))
        if not query_terms:
            raise IndexingError("no_query_terms")

        version = await self._index.get_index_version(self._scope, self._version.id)
        if version.state is not IndexState.READY:
            raise IndexingError("not_ready")
        stats = self._stats if self._stats is not None else await self._build_stats()
        docs, term_freqs, lengths, document_freq, average_length = stats
        count = len(docs)
        idfs = {
            term: log(1.0 + (count - document_freq[term] + 0.5) / (document_freq[term] + 0.5)) for term in query_terms
        }
        scored = []
        for chunk, frequencies, length in zip(docs, term_freqs, lengths, strict=True):
            components = []
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                idf = idfs[term]
                denominator = (
                    tf + self.config.k1 * (1.0 - self.config.b + self.config.b * length / average_length)
                    if average_length
                    else float(tf)
                )
                value = idf * tf * (self.config.k1 + 1.0) / denominator
                components.append(ScoreComponent(name="bm25_term", term=term, value=value))
            score = sum(component.value for component in components)
            if score > 0:
                scored.append((score, chunk.ordinal, chunk, tuple(components)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = tuple(
            RetrievedChunk(
                chunk_id=chunk.id,
                extraction_view_id=chunk.extraction_view_id,
                start=chunk.start,
                end=chunk.end,
                block_ids=chunk.block_ids,
                text=chunk.text,
                distance=None,
                score=score,
                rank=rank,
                score_components=components,
                parent_start=getattr(chunk, "parent_start_cp", None),
                parent_end=getattr(chunk, "parent_end_cp", None),
            )
            for rank, (score, _, chunk, components) in enumerate(scored[: self.config.k], start=1)
        )
        return RetrievalResult(
            technique_id="R03",
            grounding="corpus",
            index_version_id=self._version.id,
            hits=hits,
            scoring="bm25-okapi-lucene-idf",
        )
