"""SQL implementation of the scoped response-cache store port (R50/R51, ADR-0021).

Each call uses its own short transaction. Rows are inserted and deleted, never updated. Errors are mapped to
content-free `CacheError` codes; row content and driver messages are never logged or chained.

pgvector stores float4: a vector read back equals the stored one only for values exact in float4, and
similarity can differ from the in-memory store by about 1e-7. Do not compare stored vectors for equality.
A zero vector has a NaN cosine distance in PostgreSQL; it is ranked as distance 1.0 (similarity 0.0), as in
the in-memory store. Source version ids must be canonical UUIDs (they reference document_versions).
"""

import math
import uuid
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import CacheEntry
from sqlalchemy import text
from sqlalchemy.exc import DataError, IntegrityError, InterfaceError, OperationalError

from rag_foundry_storage.sql.database import Database

_COLUMNS = """e.tenant_id, e.project_id, e.key, e.partition, e.response_json, e.source_run_id, e.created_at,
    e.expires_at, e.features_json, e.question_vector::text AS question_vector, e.vector_key,
    (SELECT array_agg(s.document_version_id::text ORDER BY s.ordinal) FROM response_cache_sources s
     WHERE s.tenant_id = e.tenant_id AND s.project_id = e.project_id AND s.key = e.key) AS versions"""


def _sqlstate(exc: IntegrityError) -> str | None:
    original = exc.orig
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)


def _canonical_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(repr(x) for x in vector) + "]"


def _row_entry(row: Any) -> CacheEntry:
    try:
        vector = row["question_vector"]
        return CacheEntry(
            key=row["key"],
            partition=row["partition"],
            tenant_id=str(row["tenant_id"]),
            project_id=str(row["project_id"]),
            response_json=row["response_json"],
            source_run_id=row["source_run_id"],
            source_version_ids=tuple(row["versions"] or ()),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            features_json=row["features_json"],
            question_vector=None if vector is None else tuple(float(x) for x in vector.strip("[]").split(",")),
            vector_key=row["vector_key"],
        )
    except (ValidationError, ValueError, TypeError):
        raise CacheError("cache_entry_invalid") from None


class SqlResponseCacheStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def _run(self, work):
        try:
            async with self._database.sessionmaker() as session, session.begin():
                return await work(session)
        except (OperationalError, InterfaceError):
            raise CacheError("cache_unavailable") from None
        except DataError:
            # Includes a pgvector dimension mismatch inside one partition: surfaced, never skipped.
            raise CacheError("cache_entry_invalid") from None

    async def get(self, scope: Scope, key: str) -> CacheEntry | None:
        async def work(session):
            result = await session.execute(
                text(
                    f"SELECT {_COLUMNS} FROM response_cache_entries e "
                    "WHERE e.tenant_id = :t AND e.project_id = :p AND e.key = :k"
                ),
                {"t": scope.tenant_id, "p": scope.project_id, "k": key},
            )
            return result.mappings().first()

        row = await self._run(work)
        return None if row is None else _row_entry(row)

    async def put(self, scope: Scope, entry: CacheEntry) -> None:
        if entry.tenant_id != scope.tenant_id or entry.project_id != scope.project_id:
            raise CacheError("cache_scope_mismatch")
        if not all(_canonical_uuid(v) for v in entry.source_version_ids):
            raise CacheError("cache_entry_invalid")

        async def work(session):
            await session.execute(
                text(
                    """INSERT INTO response_cache_entries
                    (tenant_id, project_id, key, partition, response_json, source_run_id, created_at, expires_at,
                     features_json, question_vector, vector_key)
                    VALUES (:t, :p, :k, :part, :resp, :run, :created, :expires, :features,
                            CAST(:vec AS vector), :vk)"""
                ),
                {
                    "t": entry.tenant_id,
                    "p": entry.project_id,
                    "k": entry.key,
                    "part": entry.partition,
                    "resp": entry.response_json,
                    "run": entry.source_run_id,
                    "created": entry.created_at,
                    "expires": entry.expires_at,
                    "features": entry.features_json,
                    "vec": None if entry.question_vector is None else _vector_literal(entry.question_vector),
                    "vk": entry.vector_key,
                },
            )
            for ordinal, version_id in enumerate(entry.source_version_ids):
                await session.execute(
                    text(
                        """INSERT INTO response_cache_sources (tenant_id, project_id, key, document_version_id, ordinal)
                        VALUES (:t, :p, :k, CAST(:v AS uuid), :o)"""
                    ),
                    {"t": entry.tenant_id, "p": entry.project_id, "k": entry.key, "v": version_id, "o": ordinal},
                )

        try:
            await self._run(work)
        except IntegrityError as exc:
            code = _sqlstate(exc)
            if code == "23505":
                raise CacheError("cache_exists") from None
            if code in ("23503", "23000"):
                # Unknown project, or a source version outside this scope.
                raise CacheError("cache_scope_mismatch") from None
            raise CacheError("cache_entry_invalid") from None

    async def _delete(self, sql: str, params: dict[str, Any]) -> int:
        async def work(session):
            return (await session.execute(text(sql), params)).rowcount

        return int(await self._run(work))

    async def delete(self, scope: Scope, key: str) -> bool:
        removed = await self._delete(
            "DELETE FROM response_cache_entries WHERE tenant_id = :t AND project_id = :p AND key = :k",
            {"t": scope.tenant_id, "p": scope.project_id, "k": key},
        )
        return removed > 0

    async def similar(
        self, scope: Scope, partition: str, vector_key: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[CacheEntry, float]]:
        if k < 1 or not vector or not all(math.isfinite(x) for x in vector):
            raise CacheError("cache_policy_invalid")

        async def work(session):
            result = await session.execute(
                text(
                    f"""SELECT {_COLUMNS}, e.question_vector <=> CAST(:q AS vector) AS distance
                    FROM response_cache_entries e
                    WHERE e.tenant_id = :t AND e.project_id = :p AND e.partition = :part AND e.vector_key = :vk
                    ORDER BY COALESCE(NULLIF(e.question_vector <=> CAST(:q AS vector), 'NaN'), 1.0), e.key LIMIT :k"""
                ),
                {
                    "t": scope.tenant_id,
                    "p": scope.project_id,
                    "part": partition,
                    "vk": vector_key,
                    "q": _vector_literal(vector),
                    "k": k,
                },
            )
            return result.mappings().all()

        rows = await self._run(work)
        scored = []
        for row in rows:
            distance = row["distance"]
            # A zero vector has no direction: cosine 0.0, as in the in-memory store.
            similarity = 0.0 if distance is None or math.isnan(distance) else 1.0 - float(distance)
            scored.append((_row_entry(row), similarity))
        return scored

    async def invalidate_sources(self, scope: Scope, version_ids: tuple[str, ...]) -> int:
        # An ID that is not a canonical UUID can never have been stored, so it matches nothing.
        usable = sorted({v for v in version_ids if _canonical_uuid(v)})
        if not usable:
            return 0
        return await self._delete(
            """DELETE FROM response_cache_entries e WHERE e.tenant_id = :t AND e.project_id = :p AND e.key IN (
                SELECT s.key FROM response_cache_sources s
                WHERE s.tenant_id = :t AND s.project_id = :p AND s.document_version_id = ANY(CAST(:ids AS uuid[])))""",
            {"t": scope.tenant_id, "p": scope.project_id, "ids": "{" + ",".join(usable) + "}"},
        )

    async def invalidate_partition(self, scope: Scope, partition: str) -> int:
        return await self._delete(
            "DELETE FROM response_cache_entries WHERE tenant_id = :t AND project_id = :p AND partition = :part",
            {"t": scope.tenant_id, "p": scope.project_id, "part": partition},
        )

    async def purge_expired(self, scope: Scope, now: datetime) -> int:
        if now.tzinfo is None:
            raise CacheError("cache_policy_invalid")
        return await self._delete(
            "DELETE FROM response_cache_entries WHERE tenant_id = :t AND project_id = :p AND expires_at <= :now",
            {"t": scope.tenant_id, "p": scope.project_id, "now": now},
        )
