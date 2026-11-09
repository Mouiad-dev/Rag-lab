# ruff: noqa: E501
"""Scoped SQL implementation of IndexRepository."""

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from rag_foundry_domain import NotFoundError, Scope, new_id
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import (
    ChunkRecord,
    FtsScoredChunk,
    IndexVersion,
    NewChunk,
    NewIndexVersion,
    ScoredChunk,
    check_vector,
)
from rag_foundry_domain.representations import (
    NewRepresentation,
    RepresentationRecord,
    RepresentationSet,
    ScoredRepresentation,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_foundry_storage.sql._tx import read, write


def _uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise NotFoundError() from None
    if parsed.version != 4 or str(parsed) != value:
        raise NotFoundError()
    return value


def _version(row: Any, view_ids: tuple[str, ...]) -> IndexVersion:
    return IndexVersion(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        chunker_id=row["chunker_id"],
        chunker_version=row["chunker_version"],
        chunker_config_json=row["chunker_config_json"],
        chunker_config_sha256=row["chunker_config_sha256"],
        embedding_profile_id=row["embedding_profile_id"],
        embedding_provider=row["embedding_provider"],
        embedding_model=row["embedding_model"],
        embedding_dim=row["embedding_dim"],
        distance=row["distance"],
        embedding_key=row["embedding_key"],
        extraction_view_ids=view_ids,
        corpus_sha256=row["corpus_sha256"],
        unanchored_chunk_count=row["unanchored_chunk_count"],
        state=row["state"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"].astimezone(UTC),
        ready_at=None if row["ready_at"] is None else row["ready_at"].astimezone(UTC),
    )


def _chunk(row: Any) -> ChunkRecord:
    return ChunkRecord(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        index_version_id=str(row["index_version_id"]),
        extraction_view_id=str(row["extraction_view_id"]),
        ordinal=row["ordinal"],
        start=row["start_cp"],
        end=row["end_cp"],
        split_level=row["split_level"],
        block_ids=tuple(row["block_ids"]),
        text=row["text"],
        text_sha256=row["text_sha256"],
        parent_start_cp=row["parent_start_cp"],
        parent_end_cp=row["parent_end_cp"],
    )


class SqlIndexRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_sparse(self, scope: Scope, index_version_id: str, encoder_key: str, rows: tuple) -> None:
        iid = _uuid(index_version_id)
        async with write(self._session) as session:
            state = await session.scalar(
                text("SELECT state FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p FOR UPDATE"),
                {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
            )
            if state is None:
                raise NotFoundError()
            if state != "ready":
                raise IndexingError("not_ready")
            for cid, vector in rows:
                try:
                    inserted = await session.execute(
                        text("""INSERT INTO chunk_sparse_vectors (tenant_id,project_id,index_version_id,chunk_id,encoder_key,terms,weights)
                    SELECT :t,:p,:i,id,:e,:terms,:weights FROM chunks WHERE id=:c AND tenant_id=:t AND project_id=:p AND index_version_id=:i"""),
                        {
                            "t": scope.tenant_id,
                            "p": scope.project_id,
                            "i": iid,
                            "c": _uuid(cid),
                            "e": encoder_key,
                            "terms": list(vector[0]),
                            "weights": list(vector[1]),
                        },
                    )
                except IntegrityError as exc:
                    if "pk_chunk_sparse_vectors" in str(exc):
                        raise IndexingError("duplicate_sparse_vector") from None
                    raise
                if inserted.rowcount != 1:
                    raise NotFoundError()

    async def sparse_search(
        self,
        scope: Scope,
        index_version_id: str,
        encoder_key: str,
        query: tuple[tuple[str, ...], tuple[float, ...]],
        k: int,
    ) -> list:
        iid = _uuid(index_version_id)
        if not 1 <= k <= 100:
            raise IndexingError("invalid_k")
        async with read(self._session) as session:
            state = await session.scalar(
                text("SELECT state FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
            )
            if state is None:
                raise NotFoundError()
            if state != "ready":
                raise IndexingError("not_ready")
            exists = await session.scalar(
                text(
                    "SELECT 1 FROM chunk_sparse_vectors WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND encoder_key=:e LIMIT 1"
                ),
                {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "e": encoder_key},
            )
            if exists is None:
                other_key = await session.scalar(
                    text(
                        "SELECT 1 FROM chunk_sparse_vectors WHERE tenant_id=:t AND project_id=:p "
                        "AND index_version_id=:i AND encoder_key<>:e LIMIT 1"
                    ),
                    {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "e": encoder_key},
                )
                raise IndexingError("incompatible_sparse_encoder" if other_key is not None else "sparse_unavailable")
            rows = (
                (
                    await session.execute(
                        text("""WITH q AS (SELECT * FROM unnest(CAST(:qt AS text[]),CAST(:qw AS real[])) q(term,weight)),
                d AS (SELECT c.*,sum(v.weight*q.weight) score FROM chunk_sparse_vectors s JOIN chunks c ON c.id=s.chunk_id AND c.tenant_id=s.tenant_id AND c.project_id=s.project_id AND c.index_version_id=s.index_version_id
                CROSS JOIN LATERAL unnest(s.terms,s.weights) v(term,weight) JOIN q ON q.term=v.term WHERE s.tenant_id=:t AND s.project_id=:p AND s.index_version_id=:i AND s.encoder_key=:e
                GROUP BY c.id), ranked_terms AS (SELECT s.chunk_id,q.term,sum(q.weight*v.weight) term_weight,row_number() OVER (PARTITION BY s.chunk_id ORDER BY sum(q.weight*v.weight) DESC) rn FROM chunk_sparse_vectors s CROSS JOIN LATERAL unnest(s.terms,s.weights) v(term,weight) JOIN q ON q.term=v.term WHERE s.tenant_id=:t AND s.project_id=:p AND s.index_version_id=:i AND s.encoder_key=:e GROUP BY s.chunk_id,q.term), m AS (SELECT chunk_id,array_agg(term ORDER BY term_weight DESC) terms,array_agg(term_weight ORDER BY term_weight DESC) weights FROM ranked_terms WHERE rn<=8 GROUP BY chunk_id) SELECT d.*,m.terms AS matched_terms,m.weights AS matched_weights FROM d JOIN m ON m.chunk_id=d.id ORDER BY score DESC,ordinal LIMIT :k"""),
                        {
                            "qt": list(query[0]),
                            "qw": list(query[1]),
                            "t": scope.tenant_id,
                            "p": scope.project_id,
                            "i": iid,
                            "e": encoder_key,
                            "k": k,
                        },
                    )
                )
                .mappings()
                .all()
            )
            from types import SimpleNamespace

            result = []
            for rank, row in enumerate(rows, 1):
                result.append(
                    SimpleNamespace(
                        chunk=_chunk(row),
                        score=float(row["score"]),
                        rank=rank,
                        terms=tuple(
                            zip(row["matched_terms"][:8], (float(w) for w in row["matched_weights"][:8]), strict=True)
                        ),
                    )
                )
            return result

    async def add_representations(
        self,
        scope: Scope,
        index_version_id: str,
        rep_set: RepresentationSet,
        rows: tuple[NewRepresentation, ...],
    ) -> None:
        iid = _uuid(index_version_id)
        async with write(self._session) as session:
            version = (
                (
                    await session.execute(
                        text(
                            "SELECT state,embedding_dim FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p FOR UPDATE"
                        ),
                        {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise NotFoundError()
            if version["state"] != "ready":
                raise IndexingError("not_ready")
            if len(rows) > 500:
                raise IndexingError("too_many_rows")
            if not rows:
                return
            wanted = {r.chunk_id for r in rows}
            found = await session.scalar(
                text(
                    "SELECT count(*) FROM chunks WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "ids": sorted(wanted)},
            )
            if found != len(wanted):
                raise NotFoundError()
            if any(len(r.embedding) != version["embedding_dim"] for r in rows):
                raise IndexingError("dimension_mismatch")
            stored = (
                (
                    await session.execute(
                        text(
                            "SELECT technique_id,generator_json FROM representation_sets WHERE index_version_id=:i AND representation_key=:k"
                        ),
                        {"i": iid, "k": rep_set.representation_key},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if stored is not None and (
                stored["technique_id"] != rep_set.technique_id or stored["generator_json"] != rep_set.generator_json
            ):
                raise IndexingError("representation_set_conflict")
            slots = [(r.chunk_id, r.kind, r.ordinal) for r in rows]
            if len(set(slots)) != len(slots):
                raise IndexingError("duplicate_representation")
            if stored is None:
                await session.execute(
                    text("""INSERT INTO representation_sets (tenant_id,project_id,index_version_id,representation_key,technique_id,generator_json)
                    VALUES (:t,:p,:i,:k,:tech,:g)"""),
                    {
                        "t": scope.tenant_id,
                        "p": scope.project_id,
                        "i": iid,
                        "k": rep_set.representation_key,
                        "tech": rep_set.technique_id,
                        "g": rep_set.generator_json,
                    },
                )
            for r in rows:
                try:
                    async with session.begin_nested():
                        await session.execute(
                            text("""INSERT INTO chunk_representations
                            (tenant_id,project_id,index_version_id,chunk_id,representation_key,kind,ordinal,text,text_sha256,embedded_input_sha256,embedding)
                            VALUES (:t,:p,:i,:c,:k,:kind,:o,:text,:th,:eh,CAST(:e AS vector))"""),
                            {
                                "t": scope.tenant_id,
                                "p": scope.project_id,
                                "i": iid,
                                "c": r.chunk_id,
                                "k": rep_set.representation_key,
                                "kind": r.kind,
                                "o": r.ordinal,
                                "text": r.text,
                                "th": r.text_sha256,
                                "eh": r.embedded_input_sha256,
                                "e": "[" + ",".join(repr(x) for x in r.embedding) + "]",
                            },
                        )
                except IntegrityError as exc:
                    if "pk_chunk_representations" in str(exc):
                        raise IndexingError("duplicate_representation") from None
                    raise

    async def get_representation_set(self, scope: Scope, index_version_id: str, key: str) -> RepresentationSet:
        iid = _uuid(index_version_id)
        async with read(self._session) as session:
            if (
                await session.scalar(
                    text("SELECT 1 FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                    {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                )
                is None
            ):
                raise NotFoundError()
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT representation_key,technique_id,generator_json FROM representation_sets "
                            "WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND representation_key=:k"
                        ),
                        {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "k": key},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise NotFoundError()
            return RepresentationSet(
                representation_key=row["representation_key"],
                technique_id=row["technique_id"],
                generator_json=row["generator_json"],
            )

    async def representation_search(
        self, scope: Scope, index_version_id: str, key: str, query: tuple[float, ...], k: int
    ) -> list[ScoredRepresentation]:
        iid = _uuid(index_version_id)
        async with read(self._session) as session:
            version = (
                (
                    await session.execute(
                        text(
                            "SELECT state,embedding_dim FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"
                        ),
                        {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise NotFoundError()
            if version["state"] != "ready":
                raise IndexingError("not_ready")
            if not 1 <= k <= 100:
                raise IndexingError("invalid_k")
            vector = check_vector(query)
            if len(vector) != version["embedding_dim"]:
                raise IndexingError("dimension_mismatch")
            if (
                await session.scalar(
                    text(
                        "SELECT 1 FROM representation_sets WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND representation_key=:k"
                    ),
                    {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "k": key},
                )
                is None
            ):
                raise NotFoundError()
            literal = "[" + ",".join(repr(x) for x in vector) + "]"
            rows = (
                await session.execute(
                    text("""SELECT c.*, r.kind AS rep_kind, r.ordinal AS rep_ordinal, r.text AS rep_text,
                       r.text_sha256 AS rep_text_sha256, r.embedded_input_sha256 AS rep_embedded_input_sha256,
                       r.representation_key AS rep_key, r.embedding <=> CAST(:q AS vector) AS distance
                FROM chunk_representations r
                JOIN chunks c ON c.id=r.chunk_id AND c.tenant_id=r.tenant_id AND c.project_id=r.project_id AND c.index_version_id=r.index_version_id
                JOIN index_versions v ON v.id=r.index_version_id AND v.tenant_id=r.tenant_id AND v.project_id=r.project_id
                WHERE r.tenant_id=:t AND r.project_id=:p AND r.index_version_id=:i AND r.representation_key=:k AND v.state='ready'
                ORDER BY distance, c.ordinal, r.kind COLLATE "C", r.ordinal LIMIT :n"""),
                    {"q": literal, "t": scope.tenant_id, "p": scope.project_id, "i": iid, "k": key, "n": k},
                )
            ).mappings()
            return [
                ScoredRepresentation(
                    chunk=_chunk(row),
                    representation=RepresentationRecord(
                        chunk_id=str(row["id"]),
                        kind=row["rep_kind"],
                        ordinal=row["rep_ordinal"],
                        text=row["rep_text"],
                        text_sha256=row["rep_text_sha256"],
                        embedded_input_sha256=row["rep_embedded_input_sha256"],
                        index_version_id=str(row["index_version_id"]),
                        tenant_id=str(row["tenant_id"]),
                        project_id=str(row["project_id"]),
                        representation_key=row["rep_key"],
                    ),
                    distance=float(row["distance"]),
                    rank=n,
                )
                for n, row in enumerate(rows, 1)
            ]

    async def _locked(self, session: AsyncSession, scope: Scope, iid: str):
        row = (
            (
                await session.execute(
                    text("SELECT * FROM index_versions WHERE id=:id AND tenant_id=:t AND project_id=:p FOR UPDATE"),
                    {"id": iid, "t": scope.tenant_id, "p": scope.project_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise NotFoundError()
        return row

    async def _views(self, session: AsyncSession, scope: Scope, version_id: str) -> tuple[str, ...]:
        rows = (
            await session.execute(
                text(
                    "SELECT extraction_view_id FROM index_version_views WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i ORDER BY extraction_view_id"
                ),
                {"t": scope.tenant_id, "p": scope.project_id, "i": version_id},
            )
        ).scalars()
        return tuple(str(x) for x in rows)

    async def create_index_version(self, scope: Scope, new: NewIndexVersion) -> IndexVersion:
        async with write(self._session) as session:
            for view_id in new.extraction_view_ids:
                found = await session.scalar(
                    text("SELECT id FROM extraction_views WHERE id=:v AND tenant_id=:t AND project_id=:p"),
                    {"v": view_id, "t": scope.tenant_id, "p": scope.project_id},
                )
                if found is None:
                    raise NotFoundError()
            iid = new_id()
            values = new.model_dump(exclude={"extraction_view_ids"})
            await session.execute(
                text("""INSERT INTO index_versions
                (id,tenant_id,project_id,chunker_id,chunker_version,chunker_config_json,chunker_config_sha256,
                 embedding_profile_id,embedding_provider,embedding_model,embedding_dim,distance,embedding_key,
                 corpus_sha256,unanchored_chunk_count,state,chunk_count)
                VALUES (:id,:t,:p,:chunker_id,:chunker_version,:chunker_config_json,:chunker_config_sha256,
                 :embedding_profile_id,:embedding_provider,:embedding_model,:embedding_dim,:distance,:embedding_key,
                 :corpus_sha256,:unanchored_chunk_count,'building',0)"""),
                {**values, "id": iid, "t": scope.tenant_id, "p": scope.project_id},
            )
            for view_id in new.extraction_view_ids:
                await session.execute(
                    text(
                        "INSERT INTO index_version_views (tenant_id,project_id,index_version_id,extraction_view_id) VALUES (:t,:p,:i,:v)"
                    ),
                    {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "v": view_id},
                )
            row = (await session.execute(text("SELECT * FROM index_versions WHERE id=:i"), {"i": iid})).mappings().one()
            return _version(row, tuple(new.extraction_view_ids))

    async def add_chunks(self, scope: Scope, index_version_id: str, chunks: tuple[NewChunk, ...]) -> None:
        iid = _uuid(index_version_id)
        async with write(self._session) as session:
            version = await self._locked(session, scope, iid)
            if version["state"] != "building":
                raise IndexingError("not_building")
            for chunk in chunks:
                vector = check_vector(chunk.embedding)
                if len(vector) != version["embedding_dim"]:
                    raise IndexingError("dimension_mismatch")
                if (
                    await session.scalar(
                        text(
                            "SELECT 1 FROM index_version_views WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND extraction_view_id=:v"
                        ),
                        {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "v": chunk.extraction_view_id},
                    )
                    is None
                ):
                    raise NotFoundError()
                try:
                    await session.execute(
                        text("""INSERT INTO chunks
                        (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding,parent_start_cp,parent_end_cp)
                        VALUES (:id,:t,:p,:i,:v,:ordinal,:start,:end,:split_level,:block_ids,:text,:text_sha256,CAST(:embedding AS vector),:parent_start_cp,:parent_end_cp)"""),
                        {
                            "id": new_id(),
                            "t": scope.tenant_id,
                            "p": scope.project_id,
                            "i": iid,
                            "v": chunk.extraction_view_id,
                            "ordinal": chunk.ordinal,
                            "start": chunk.start,
                            "end": chunk.end,
                            "split_level": chunk.split_level,
                            "block_ids": list(chunk.block_ids),
                            "text": chunk.text,
                            "text_sha256": chunk.text_sha256,
                            "embedding": "[" + ",".join(repr(x) for x in vector) + "]",
                            "parent_start_cp": chunk.parent_start_cp,
                            "parent_end_cp": chunk.parent_end_cp,
                        },
                    )
                except IntegrityError as exc:
                    if "uq_chunks_version_ordinal" in str(exc):
                        raise IndexingError("duplicate_ordinal") from None
                    raise

    async def _transition(self, scope: Scope, iid: str, target: str) -> IndexVersion:
        async with write(self._session) as session:
            row = await self._locked(session, scope, iid)
            if row["state"] != "building":
                raise IndexingError("not_building")
            count = await session.scalar(
                text("SELECT count(*) FROM chunks WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i"),
                {"t": scope.tenant_id, "p": scope.project_id, "i": iid},
            )
            if target == "ready" and count == 0:
                raise IndexingError("empty_index")
            await session.execute(
                text(
                    "UPDATE index_versions SET state=:s,chunk_count=:c,ready_at=:r WHERE id=:i AND tenant_id=:t AND project_id=:p"
                ),
                {
                    "s": target,
                    "c": count if target == "ready" else row["chunk_count"],
                    "r": datetime.now(UTC) if target == "ready" else None,
                    "i": iid,
                    "t": scope.tenant_id,
                    "p": scope.project_id,
                },
            )
            updated = (
                (await session.execute(text("SELECT * FROM index_versions WHERE id=:i"), {"i": iid})).mappings().one()
            )
            return _version(updated, await self._views(session, scope, iid))

    async def mark_ready(self, scope: Scope, index_version_id: str) -> IndexVersion:
        return await self._transition(scope, _uuid(index_version_id), "ready")

    async def mark_failed(self, scope: Scope, index_version_id: str) -> IndexVersion:
        return await self._transition(scope, _uuid(index_version_id), "failed")

    async def get_index_version(self, scope: Scope, index_version_id: str) -> IndexVersion:
        iid = _uuid(index_version_id)
        async with read(self._session) as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                        {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise NotFoundError()
            return _version(row, await self._views(session, scope, iid))

    async def chunk_embeddings(
        self, scope: Scope, index_version_id: str, chunk_ids: tuple[str, ...]
    ) -> dict[str, tuple[float, ...]]:
        iid = _uuid(index_version_id)
        async with read(self._session) as session:
            if (
                await session.scalar(
                    text("SELECT 1 FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                    {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                )
                is None
            ):
                raise NotFoundError()
            if len(chunk_ids) > 100:
                raise IndexingError("too_many_ids")
            ids = []
            for value in chunk_ids:
                try:
                    parsed = uuid.UUID(value)
                except (ValueError, AttributeError, TypeError):
                    continue
                if parsed.version == 4 and str(parsed) == value:
                    ids.append(parsed)
            if not ids:
                return {}
            rows = (
                await session.execute(
                    text(
                        "SELECT id, embedding::text AS e FROM chunks WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i AND id = ANY(CAST(:ids AS uuid[]))"
                    ),
                    {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "ids": ids},
                )
            ).mappings()
            return {str(row["id"]): tuple(float(value) for value in json.loads(row["e"])) for row in rows}

    async def list_chunks(
        self, scope: Scope, index_version_id: str, *, offset: int = 0, limit: int = 100
    ) -> list[ChunkRecord]:
        iid = _uuid(index_version_id)
        if offset < 0 or not 1 <= limit <= 500:
            raise IndexingError("invalid_page")
        async with read(self._session) as session:
            if (
                await session.scalar(
                    text("SELECT 1 FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                    {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                )
                is None
            ):
                raise NotFoundError()
            rows = (
                await session.execute(
                    text(
                        "SELECT * FROM chunks WHERE tenant_id=:t AND project_id=:p AND index_version_id=:i ORDER BY ordinal OFFSET :o LIMIT :l"
                    ),
                    {"t": scope.tenant_id, "p": scope.project_id, "i": iid, "o": offset, "l": limit},
                )
            ).mappings()
            return [_chunk(row) for row in rows]

    async def search(self, scope: Scope, index_version_id: str, query: tuple[float, ...], k: int) -> list[ScoredChunk]:
        iid = _uuid(index_version_id)
        if not 1 <= k <= 100:
            raise IndexingError("invalid_k")
        vector = check_vector(query)
        literal = "[" + ",".join(repr(x) for x in vector) + "]"
        async with read(self._session) as session:
            version = (
                (
                    await session.execute(
                        text(
                            "SELECT state,embedding_dim FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"
                        ),
                        {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise NotFoundError()
            if version["state"] != "ready":
                raise IndexingError("not_ready")
            if version["embedding_dim"] != len(vector):
                raise IndexingError("dimension_mismatch")
            rows = (
                await session.execute(
                    text("""SELECT c.*, c.embedding <=> CAST(:q AS vector) AS distance
                FROM chunks c JOIN index_versions v ON v.id=c.index_version_id AND v.tenant_id=c.tenant_id AND v.project_id=c.project_id
                WHERE c.tenant_id=:t AND c.project_id=:p AND c.index_version_id=:i AND v.state='ready'
                ORDER BY c.embedding <=> CAST(:q AS vector), c.ordinal LIMIT :k"""),
                    {"q": literal, "t": scope.tenant_id, "p": scope.project_id, "i": iid, "k": k},
                )
            ).mappings()
            return [
                ScoredChunk(chunk=_chunk(row), distance=float(row["distance"]), rank=n) for n, row in enumerate(rows, 1)
            ]

    async def fts_search(
        self,
        scope: Scope,
        index_version_id: str,
        query: str,
        k: int,
        *,
        operator: str,
    ) -> list[FtsScoredChunk]:
        iid = _uuid(index_version_id)
        if not 1 <= k <= 100:
            raise IndexingError("invalid_k")
        if operator == "and":
            sql_query = query
        elif operator == "or":
            terms = [term for term in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if term.lower() != "or"]
            if not terms:
                raise IndexingError("no_query_terms")
            sql_query = " or ".join(terms)
        else:
            raise ValueError("operator must be 'and' or 'or'")
        async with read(self._session) as session:
            version = (
                (
                    await session.execute(
                        text("SELECT state FROM index_versions WHERE id=:i AND tenant_id=:t AND project_id=:p"),
                        {"i": iid, "t": scope.tenant_id, "p": scope.project_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise NotFoundError()
            if version["state"] != "ready":
                raise IndexingError("not_ready")
            # ts_rank_cd normalization 0: no length normalization; scores are not comparable across queries.
            rows = (
                await session.execute(
                    text("""SELECT c.*, ts_rank_cd(c.fts_simple, q, 0) AS ts_rank
                FROM chunks c JOIN index_versions v ON v.id=c.index_version_id AND v.tenant_id=c.tenant_id AND v.project_id=c.project_id,
                     websearch_to_tsquery('simple', :q) q
                WHERE c.fts_simple @@ q AND c.tenant_id=:t AND c.project_id=:p
                  AND c.index_version_id=:i AND v.state='ready'
                ORDER BY ts_rank DESC, c.ordinal LIMIT :k"""),
                    {"q": sql_query, "t": scope.tenant_id, "p": scope.project_id, "i": iid, "k": k},
                )
            ).mappings()
            return [
                FtsScoredChunk(chunk=_chunk(row), ts_rank=float(row["ts_rank"]), rank=rank)
                for rank, row in enumerate(rows, 1)
            ]
