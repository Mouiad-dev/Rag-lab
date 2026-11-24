"""R18 query expansion with reciprocal rank fusion."""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict, Field
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.fusion import fusion_sort_key, rrf_contribution
from rag_foundry_indexing.multi_query import MultiQueryConfig, generate_variants, retrieve_all
from rag_foundry_indexing.query_transform import TransformedRetrieval


class RagFusionConfig(MultiQueryConfig):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rrf_k: int = Field(default=60, ge=1, le=1000)


class RagFusion:
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R18",
        kind=PluginKind.QUERY_TRANSFORM,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=RagFusionConfig,
    )

    def __init__(self, config=None):
        self.config = config or RagFusionConfig()

    async def run(self, question: str, *, chat, retriever) -> TransformedRetrieval:
        trace = await generate_variants(question, chat=chat, config=self.config, technique_id="R18")
        per_query = await retrieve_all(trace.retrieval_queries, retriever)
        records = {}
        for qi, result in enumerate(per_query):
            for hit in result.hits:
                rec = records.setdefault(hit.chunk_id, {"hit": hit, "rank": hit.rank, "score": 0.0, "components": []})
                rec["rank"] = min(rec["rank"], hit.rank)
                contribution = rrf_contribution(hit.rank, self.config.rrf_k)
                rec["score"] += contribution
                rec["components"].append(
                    ScoreComponent(
                        name=f"q{qi}_rrf", value=contribution, source=result.technique_id, source_rank=hit.rank
                    )
                )
        ordered = sorted(
            records.items(),
            key=lambda pair: fusion_sort_key(pair[1]["score"], pair[1]["rank"], pair[1]["hit"], pair[0]),
        )[: self.config.k]
        hits = tuple(
            RetrievedChunk(
                chunk_id=cid,
                extraction_view_id=rec["hit"].extraction_view_id,
                start=rec["hit"].start,
                end=rec["hit"].end,
                block_ids=rec["hit"].block_ids,
                text=rec["hit"].text,
                distance=None,
                score=rec["score"],
                rank=i,
                score_components=tuple(rec["components"]),
                parent_start=rec["hit"].parent_start,
                parent_end=rec["hit"].parent_end,
            )
            for i, (cid, rec) in enumerate(ordered, 1)
        )
        result = RetrievalResult(
            technique_id="R18",
            grounding="corpus",
            index_version_id=per_query[0].index_version_id,
            hits=hits,
            scoring=f"rag-fusion-rrf-k{self.config.rrf_k}",
        )
        return TransformedRetrieval(trace=trace, result=result, per_query=per_query)
