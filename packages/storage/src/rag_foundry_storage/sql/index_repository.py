# ruff: noqa: E501
"""Scoped SQL implementation of IndexRepository."""

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
