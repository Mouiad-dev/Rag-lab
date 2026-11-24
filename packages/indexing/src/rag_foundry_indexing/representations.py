"""R23 and R24 search over stored synthetic representations; hits are the original chunks."""

from __future__ import annotations

import hashlib
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rag_foundry_domain import Scope
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState, IndexVersion, embedding_key_for
from rag_foundry_providers.ports import EmbeddingModel
from rag_foundry_techniques import Fidelity, PluginKind, PluginManifest, RetrievalResult, RetrievedChunk, ScoreComponent

from rag_foundry_indexing.query_transform import check_question

MatchKind = Literal["original", "summary", "question", "context_prefix"]


class RepresentationSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    representation_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=100)
    include_original: bool = True

    @model_validator(mode="after")
    def _candidate_k_covers_k(self) -> RepresentationSearchConfig:
        if self.candidate_k < self.k:
            raise ValueError("candidate_k must be >= k")
        return self


class ContextualSearchConfig(RepresentationSearchConfig):
    include_original: bool = False


class RepresentationMatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    rank: int = Field(ge=1)
    kind: MatchKind
    ordinal: int = Field(ge=0)
    synthetic: bool
    text: str | None
    distance: float


class RepresentationRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    result: RetrievalResult
    matches: tuple[RepresentationMatch, ...]


class _RepresentationRetriever:
    allowed_kinds: ClassVar[frozenset[str]]
    manifest: ClassVar[PluginManifest]

    def __init__(
        self,
        config: RepresentationSearchConfig,
        *,
        scope: Scope,
        index_version: IndexVersion,
        index,
        embedder: EmbeddingModel,
    ) -> None:
        self.config, self._scope, self._version = config, scope, index_version
        self._index, self._embedder = index, embedder

    async def retrieve(self, question: str) -> RetrievalResult:
        return (await self.retrieve_traced(question)).result

    async def retrieve_traced(self, question: str) -> RepresentationRetrieval:
        check_question(question)
        if self._version.state != IndexState.READY:
            raise IndexingError("not_ready")
        profile = self._embedder.profile
        dim = profile.capabilities.embedding_dim
        if dim is None or embedding_key_for(profile.provider, profile.model, dim) != self._version.embedding_key:
            raise IndexingError("incompatible_embedding")
        technique_id = self.manifest.technique_id
        key = self.config.representation_key
        rep_set = await self._index.get_representation_set(self._scope, self._version.id, key)
        if rep_set.technique_id != technique_id:
            raise IndexingError("representation_technique_mismatch")

        embedded = await self._embedder.embed([question], input_type="query")
        if len(embedded.vectors) != 1:
            raise IndexingError("embedding_count_mismatch")
        vector = embedded.vectors[0]
        found = await self._index.representation_search(
            self._scope, self._version.id, key, vector, self.config.candidate_k
        )
        originals = []
        if self.config.include_original:
            originals = await self._index.search(self._scope, self._version.id, vector, self.config.candidate_k)

        # chunk id -> (sort key, chunk, kind, ordinal, synthetic text, distance)
        best: dict[str, tuple] = {}

        def offer(chunk, kind: str, ordinal: int, text: str | None, distance: float) -> None:
            order = (distance, 0 if kind == "original" else 1, kind, ordinal)
            current = best.get(chunk.id)
            if current is None or order < current[0]:
                best[chunk.id] = (order, chunk, kind, ordinal, text, distance)

        for item in found:
            kind = item.representation.kind
            if kind not in self.allowed_kinds:
                raise IndexingError("representation_kind_mismatch")
            offer(item.chunk, kind, item.representation.ordinal, item.representation.text, item.distance)
        for item in originals:
            offer(item.chunk, "original", 0, None, item.distance)

        chosen = sorted(best.values(), key=lambda row: (row[5], row[1].ordinal))[: self.config.k]
        hits = []
        matches = []
        for rank, (_, chunk, kind, ordinal, text, distance) in enumerate(chosen, start=1):
            if hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() != chunk.text_sha256:
                raise IndexingError("representation_source_mismatch")
            score = 1.0 - distance
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    extraction_view_id=chunk.extraction_view_id,
                    start=chunk.start,
                    end=chunk.end,
                    block_ids=chunk.block_ids,
                    text=chunk.text,
                    distance=distance,
                    score=score,
                    rank=rank,
                    score_components=(
                        ScoreComponent(name="representation_cosine", value=score, term=f"{kind}:{ordinal}"),
                    ),
                    parent_start=chunk.parent_start_cp,
                    parent_end=chunk.parent_end_cp,
                )
            )
            matches.append(
                RepresentationMatch(
                    chunk_id=chunk.id,
                    rank=rank,
                    kind=kind,
                    ordinal=ordinal,
                    synthetic=kind != "original",
                    text=text,
                    distance=distance,
                )
            )
        result = RetrievalResult(
            technique_id=technique_id,
            grounding="corpus",
            index_version_id=self._version.id,
            hits=tuple(hits),
            scoring="representation-max-cosine",
        )
        return RepresentationRetrieval(result=result, matches=tuple(matches))


class MultiRepresentationRetriever(_RepresentationRetriever):
    allowed_kinds: ClassVar[frozenset[str]] = frozenset({"summary", "question"})
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R23",
        kind=PluginKind.REPRESENTATION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=RepresentationSearchConfig,
    )


class ContextualRetriever(_RepresentationRetriever):
    allowed_kinds: ClassVar[frozenset[str]] = frozenset({"context_prefix"})
    manifest: ClassVar[PluginManifest] = PluginManifest(
        technique_id="R24",
        kind=PluginKind.REPRESENTATION,
        version="1.0.0",
        fidelity=Fidelity.ADAPTATION,
        config_model=ContextualSearchConfig,
    )
