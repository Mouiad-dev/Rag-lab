# ruff: noqa: E501
"""SQL IndexRepository checks and database-enforced immutability tests."""

import asyncio
import functools
import os
import sys

import pytest
from pydantic import SecretStr

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]

if _URL:
    from rag_foundry_domain.errors import NotFoundError
    from rag_foundry_domain.index_testing import (
        GRAPH_STORE_CHECKS,
        INDEX_REPOSITORY_CHECKS,
        KG_STORE_CHECKS,
        REPRESENTATION_STORE_CHECKS,
        TREE_STORE_CHECKS,
    )
    from rag_foundry_domain.testing import CheckSkippedError, make_new_extraction, make_new_version, make_scope
    from rag_foundry_domain.uow_adapters import per_call_documents, per_call_extractions, per_call_indexes
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.uow import ensure_local_scope
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError


def run(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro_fn(*args, **kwargs))

    return wrapper


def _url():
    assert _URL
    return SecretStr(_URL)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    run_migrations(_url(), "upgrade", "head")


class _Factory:
    def __init__(self):
        self.databases = []

    def database(self):
        db = create_database(_url())
        self.databases.append(db)
        return db

    async def __call__(self, scopes):
        db = self.database()
        maker = db.sessionmaker
        async with maker() as session, session.begin():
            await session.execute(
                text(
                    "TRUNCATE runtime_setting_overrides, response_cache_sources, response_cache_entries, kg_spans, kg_facts, kg_entities, knowledge_graphs, "
                    "tree_edges, tree_nodes, summary_trees, graph_spans, graph_relations, graph_entities, entity_graphs, chunk_representations, representation_sets, chunk_sparse_vectors, chunks, index_version_views, index_versions, extraction_views, document_versions, documents, projects, tenants"
                )
            )
        docs = per_call_documents(db.uow)
        for scope in scopes:
            await ensure_local_scope(db.uow, scope)
        return docs, per_call_extractions(db.uow), per_call_indexes(db.uow)

    async def close(self):
        for db in self.databases:
            await db.dispose()


@pytest.mark.parametrize(
    "check",
    (INDEX_REPOSITORY_CHECKS + REPRESENTATION_STORE_CHECKS + GRAPH_STORE_CHECKS + TREE_STORE_CHECKS + KG_STORE_CHECKS)
    if _URL
    else [None],
    ids=lambda c: getattr(c, "__name__", "none"),
)
@run
async def test_index_repository_check(check):
    factory = _Factory()
    try:
        await check(factory)
    except CheckSkippedError as exc:
        pytest.skip(str(exc))
    finally:
        await factory.close()


@run
async def test_database_guards_and_migration_roundtrip():
    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("index-guard-a"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="guard-a"))).view
        foreign_out = await docs.create_document_or_get(other, make_new_version("index-guard-b"), display_name="d")
        foreign_view = (
            await views.create_view_or_get(other, make_new_extraction(foreign_out.version, tag="guard-b"))
        ).view
        from rag_foundry_domain.index_testing import make_chunk, make_new_index_version

        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        async with factory.database().sessionmaker() as session:
            version_count = await session.scalar(text("SELECT count(*) FROM index_versions"))
        with pytest.raises(NotFoundError):
            await repo.create_index_version(scope, make_new_index_version((foreign_view.id,)))
        async with factory.database().sessionmaker() as session:
            assert await session.scalar(text("SELECT count(*) FROM index_versions")) == version_count

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        base = {
            "id": "00000000-0000-4000-8000-000000000001",
            "t": scope.tenant_id,
            "p": scope.project_id,
            "i": version.id,
            "v": view.id,
            "h": "a" * 64,
        }
        common = "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding) VALUES (:id,:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST(:e AS vector))"
        await rejected(
            "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding,parent_start_cp,parent_end_cp) VALUES (gen_random_uuid(),:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST('[1,0,0,0]' AS vector),1,2)",
            {key: base[key] for key in ("t", "p", "i", "v", "h")},
            "ck_chunks_parent_span",
        )
        await rejected(common, {**base, "e": "[1,0,0]"}, "embedding dimension mismatch")
        await rejected(
            "UPDATE index_versions SET state='failed', corpus_sha256=:h WHERE id=:i",
            {"h": "b" * 64, "i": version.id},
            "invalid index version update",
        )
        second = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        async with factory.database().sessionmaker() as session, session.begin():
            await session.execute(text("UPDATE index_versions SET state='failed' WHERE id=:i"), {"i": second.id})
        await rejected(
            "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding) VALUES (gen_random_uuid(),:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST('[1,0,0,0]' AS vector))",
            {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "v": foreign_view.id, "h": "a" * 64},
            "fk_chunks_version_view",
        )

        await repo.add_chunks(scope, version.id, (make_chunk(view.id, 0, (1, 0, 0, 0)),))
        await repo.mark_ready(scope, version.id)
        await rejected(
            common,
            {**base, "id": "00000000-0000-4000-8000-000000000002", "e": "[1,0,0,0]"},
            "index version is not building",
        )
        await rejected(
            "UPDATE chunks SET text='z' WHERE index_version_id=:i", {"i": version.id}, "index rows are immutable"
        )
        await rejected(
            "UPDATE index_versions SET corpus_sha256=:h WHERE id=:i",
            {"h": "b" * 64, "i": version.id},
            "invalid index version update",
        )
        # Alembic runs its own event loop, so it must not run on this test's loop.
        await asyncio.to_thread(run_migrations, _url(), "downgrade", "0002_extraction_views")
        async with factory.databases[0].engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('chunks') IS NULL"))
            assert await connection.scalar(text("SELECT to_regclass('index_versions') IS NULL"))
        await asyncio.to_thread(run_migrations, _url(), "upgrade", "head")
    finally:
        await factory.close()


@run
async def test_representation_database_guards():
    from rag_foundry_domain.index_testing import make_chunk, make_new_index_version, make_representation
    from rag_foundry_domain.representations import RepresentationSet

    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("repr-guard"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="repr-guard"))).view
        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        await repo.add_chunks(scope, version.id, (make_chunk(view.id, 0, (1, 0, 0, 0), "alpha"),))
        chunk = (await repo.list_chunks(scope, version.id))[0]
        key, h = "c" * 64, "d" * 64
        row = make_representation(chunk.id, "summary", 0, "summary", (1, 0, 0, 0))

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        set_sql = "INSERT INTO representation_sets (tenant_id,project_id,index_version_id,representation_key,technique_id,generator_json) VALUES (:t,:p,:i,:k,'R23','{}')"
        rep_sql = "INSERT INTO chunk_representations (tenant_id,project_id,index_version_id,chunk_id,representation_key,kind,ordinal,text,text_sha256,embedded_input_sha256,embedding) VALUES (:t,:p,:i,:c,:k,'summary',0,'s',:h,:h,CAST(:e AS vector))"
        params = {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "k": key, "c": chunk.id, "h": h}
        await rejected(set_sql, params, "index version is not ready")
        await repo.mark_ready(scope, version.id)
        await rejected(set_sql, {**params, "t": other.tenant_id, "p": other.project_id}, "index version is not ready")
        await rejected(set_sql, {**params, "k": "xyz"}, "ck_representation_sets_key")
        await repo.add_representations(
            scope,
            version.id,
            RepresentationSet(representation_key=key, technique_id="R23", generator_json="{}"),
            (row,),
        )
        await rejected(rep_sql, {**params, "e": "[1,0,0]"}, "embedding dimension mismatch")
        await rejected(
            rep_sql,
            {**params, "t": other.tenant_id, "p": other.project_id, "e": "[1,0,0,0]"},
            "index version is not ready",
        )
        await rejected(
            "UPDATE chunk_representations SET text='z' WHERE index_version_id=:i",
            {"i": version.id},
            "index rows are immutable",
        )
        await rejected(
            "UPDATE representation_sets SET technique_id='R24' WHERE index_version_id=:i",
            {"i": version.id},
            "index rows are immutable",
        )
    finally:
        await factory.close()


@run
async def test_graph_database_guards():
    from rag_foundry_domain.index_testing import make_chunk, make_entity_graph, make_new_index_version

    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("graph-guard"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="graph-guard"))).view
        foreign_out = await docs.create_document_or_get(other, make_new_version("graph-guard-b"), display_name="d")
        foreign_view = (
            await views.create_view_or_get(other, make_new_extraction(foreign_out.version, tag="graph-guard-b"))
        ).view
        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        await repo.add_chunks(scope, version.id, (make_chunk(view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),))
        await repo.mark_ready(scope, version.id)
        chunk = (await repo.list_chunks(scope, version.id))[0]
        building = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        foreign_version = await repo.create_index_version(other, make_new_index_version((foreign_view.id,)))
        await repo.add_chunks(
            other, foreign_version.id, (make_chunk(foreign_view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),)
        )
        foreign_chunk = (await repo.list_chunks(other, foreign_version.id))[0]

        graph = make_entity_graph(scope, version, (chunk,))
        await repo.add_entity_graph(scope, version.id, graph)
        entity, relation = graph.entities[0], graph.relations[0]

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        async def accepted(sql, params):
            async with factory.database().sessionmaker() as session, session.begin():
                await session.execute(text(sql), params)

        base = {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "g": graph.graph_key}
        span_sql = (
            "INSERT INTO graph_spans (tenant_id,project_id,index_version_id,graph_key,owner_kind,owner_id,ordinal,"
            "chunk_id,extraction_view_id,start_cp,end_cp,quote) VALUES (:t,:p,:i,:g,:kind,:oid,:o,:c,:v,:s,:e,:q)"
        )
        span = {
            **base,
            "kind": "entity",
            "oid": entity.entity_id,
            "o": 10,
            "c": chunk.id,
            "v": view.id,
            "s": 0,
            "e": 3,
            "q": "Ada",
        }
        # The trigger accepts a valid span, so each rejection below is caused by its own defect.
        await accepted(span_sql, span)

        # Cross-tenant chunk: the trigger leaves a missing chunk to the foreign key.
        await rejected(span_sql, {**span, "o": 11, "c": foreign_chunk.id}, "fk_graph_spans_chunk")
        # Wrong quote length with a chunk the trigger cannot find: only the check constraint can reject it.
        await rejected(
            span_sql,
            {**span, "o": 12, "c": "00000000-0000-4000-8000-0000000000aa", "q": "Ad"},
            "ck_graph_spans_quote",
        )
        await rejected(span_sql, {**span, "o": 13, "v": foreign_view.id}, "graph span view does not match chunk")
        await rejected(span_sql, {**span, "o": 14, "e": 20, "q": "x" * 20}, "graph span outside chunk")
        await rejected(span_sql, {**span, "o": 15, "q": "Adb"}, "graph span quote does not match chunk text")
        await rejected(span_sql, {**span, "o": 16, "oid": "ent_" + "f" * 16}, "graph span owner not found")
        await rejected(
            span_sql,
            {**span, "o": 17, "kind": "relation", "oid": "rel_" + "f" * 16},
            "graph span owner not found",
        )

        # A relation endpoint that exists only in another graph of the same version.
        other_key = "b" * 64
        await accepted(
            "INSERT INTO entity_graphs (tenant_id,project_id,index_version_id,graph_key,technique_id,generator_json) VALUES (:t,:p,:i,:k,'R38','{}')",
            {**base, "k": other_key},
        )
        await accepted(
            "INSERT INTO graph_entities (tenant_id,project_id,index_version_id,graph_key,entity_id,name,type) VALUES (:t,:p,:i,:k,:e,'x','thing')",
            {**base, "k": other_key, "e": "ent_" + "1" * 16},
        )
        relation_sql = (
            "INSERT INTO graph_relations (tenant_id,project_id,index_version_id,graph_key,relation_id,source_id,target_id,label) "
            "VALUES (:t,:p,:i,:g,:r,:s,:d,:l)"
        )
        rel = {**base, "r": "rel_" + "2" * 16, "s": entity.entity_id, "d": relation.target_id, "l": "x"}
        await rejected(relation_sql, {**rel, "s": "ent_" + "1" * 16}, "fk_graph_relations_source")
        await rejected(relation_sql, {**rel, "d": "ent_" + "1" * 16}, "fk_graph_relations_target")
        await rejected(relation_sql, {**rel, "r": "bad"}, "ck_graph_relations_id")
        await rejected(relation_sql, {**rel, "l": ""}, "ck_graph_relations_label")

        graph_sql = "INSERT INTO entity_graphs (tenant_id,project_id,index_version_id,graph_key,technique_id,generator_json) VALUES (:t,:p,:i,:k,'R38','{}')"
        await rejected(graph_sql, {**base, "i": building.id, "k": "c" * 64}, "index version is not ready")
        await rejected(graph_sql, {**base, "k": graph.graph_key}, "pk_entity_graphs")

        for sql in (
            "UPDATE entity_graphs SET generator_json='z' WHERE index_version_id=:i",
            "UPDATE graph_entities SET name='z' WHERE index_version_id=:i",
            "UPDATE graph_relations SET label='z' WHERE index_version_id=:i",
            "UPDATE graph_spans SET quote='z' WHERE index_version_id=:i",
        ):
            await rejected(sql, {"i": version.id}, "index rows are immutable")

        # Alembic runs its own event loop, so it must not run on this test's loop.
        await asyncio.to_thread(run_migrations, _url(), "downgrade", "0010_chunk_representations")
        async with factory.databases[0].engine.connect() as connection:
            for table in ("entity_graphs", "graph_entities", "graph_relations", "graph_spans"):
                assert await connection.scalar(text("SELECT to_regclass(:n) IS NULL"), {"n": table})
            assert await connection.scalar(text("SELECT to_regclass('chunk_representations') IS NOT NULL"))
            assert (
                await connection.scalar(text("SELECT count(*) FROM pg_proc WHERE proname='reject_index_update'")) == 1
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_proc WHERE proname IN ('graph_spans_validate_insert','entity_graphs_validate_insert')"
                    )
                )
                == 0
            )
        await asyncio.to_thread(run_migrations, _url(), "upgrade", "head")
        async with factory.databases[0].engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('graph_spans') IS NOT NULL"))
    finally:
        await factory.close()


@run
async def test_tree_database_guards():
    from rag_foundry_domain.errors import IndexingError
    from rag_foundry_domain.index_testing import make_chunk, make_new_index_version, make_summary_tree

    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("tree-guard"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="tree-guard"))).view
        foreign_out = await docs.create_document_or_get(other, make_new_version("tree-guard-b"), display_name="d")
        foreign_view = (
            await views.create_view_or_get(other, make_new_extraction(foreign_out.version, tag="tree-guard-b"))
        ).view
        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        await repo.add_chunks(
            scope,
            version.id,
            (
                make_chunk(view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),
                make_chunk(view.id, 1, (0, 1, 0, 0), "Bob read them"),
                make_chunk(view.id, 2, (0, 0, 1, 0), "Eve audited both"),
            ),
        )
        await repo.mark_ready(scope, version.id)
        chunk, chunk2, chunk3 = await repo.list_chunks(scope, version.id)
        building = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        foreign_version = await repo.create_index_version(other, make_new_index_version((foreign_view.id,)))
        await repo.add_chunks(
            other, foreign_version.id, (make_chunk(foreign_view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),)
        )
        foreign_chunk = (await repo.list_chunks(other, foreign_version.id))[0]

        tree = make_summary_tree(scope, version, (chunk,))
        await repo.add_summary_tree(scope, version.id, tree)
        leaf = next(n for n in tree.nodes if n.kind == "leaf")
        summary = next(n for n in tree.nodes if n.kind == "summary")

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        async def accepted(sql, params):
            async with factory.database().sessionmaker() as session, session.begin():
                await session.execute(text(sql), params)

        base = {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "g": tree.tree_key}
        node_sql = (
            "INSERT INTO tree_nodes (tenant_id,project_id,index_version_id,tree_key,node_id,level,kind,text,synthetic,"
            "embedding,chunk_id,extraction_view_id,start_cp,end_cp) "
            "VALUES (:t,:p,:i,:g,:n,:lv,:kind,:txt,:syn,CAST(:emb AS double precision[]),:c,:v,:s,:e)"
        )
        leaf2 = {
            **base,
            "n": "rnd_" + "a" * 16,
            "lv": 0,
            "kind": "leaf",
            "txt": "Bob",
            "syn": False,
            "emb": [0.5, 0.5, 0.5, 0.5],
            "c": chunk2.id,
            "v": view.id,
            "s": 0,
            "e": 3,
        }
        # The trigger accepts a valid leaf, so each rejection below is caused by its own defect.
        await accepted(node_sql, leaf2)
        count = 0

        def variant(**changes):
            nonlocal count
            count += 1
            node_id = "rnd_" + format(count, "x").rjust(16, "b")
            return {**leaf2, "n": node_id, "c": chunk3.id, "txt": "Eve", **changes}

        # Cross-tenant chunk: the trigger leaves a missing chunk to the foreign key (the view is valid here).
        await rejected(node_sql, variant(c=foreign_chunk.id), "fk_tree_nodes_chunk")
        await rejected(node_sql, variant(txt="Evx"), "tree leaf text does not match chunk text")
        await rejected(node_sql, variant(e=40, txt="x" * 40), "tree leaf outside chunk")
        await rejected(node_sql, variant(v=foreign_view.id), "tree leaf view does not match chunk")
        await rejected(node_sql, variant(emb=[0.5, 0.5]), "tree node embedding length does not match tree")
        await rejected(node_sql, variant(c=None), "ck_tree_nodes_leaf")
        # A chunk is at most one leaf per tree.
        await rejected(node_sql, variant(c=chunk2.id, txt="Bob"), "uq_tree_nodes_chunk")
        summary_row = {**variant(), "n": "rnd_" + "c" * 16, "lv": 1, "kind": "summary", "txt": "s", "syn": True}
        no_source = {"c": None, "v": None, "s": None, "e": None}
        await rejected(node_sql, summary_row, "ck_tree_nodes_summary")
        await rejected(node_sql, {**summary_row, **no_source, "lv": 0}, "ck_tree_nodes_summary")
        await accepted(node_sql, {**summary_row, **no_source})

        edge_sql = (
            "INSERT INTO tree_edges (tenant_id,project_id,index_version_id,tree_key,parent_id,child_id,position) "
            "VALUES (:t,:p,:i,:g,:par,:chd,:pos)"
        )
        extra = "rnd_" + "c" * 16
        # A second parent for a child that already has one.
        await rejected(edge_sql, {**base, "par": extra, "chd": leaf.node_id, "pos": 1}, "pk_tree_edges")
        await rejected(
            edge_sql, {**base, "par": leaf.node_id, "chd": extra, "pos": 0}, "tree edge parent must be a summary"
        )
        # Both are level-1 summaries, so the child is not below the parent.
        await rejected(edge_sql, {**base, "par": summary.node_id, "chd": extra, "pos": 5}, "tree edge child level")
        await accepted(edge_sql, {**base, "par": extra, "chd": "rnd_" + "a" * 16, "pos": 0})
        # A fresh leaf (no parent yet) isolates the position rules from the one-parent key.
        fresh = variant()
        await accepted(node_sql, fresh)
        await rejected(edge_sql, {**base, "par": extra, "chd": fresh["n"], "pos": 0}, "uq_tree_edges_position")
        await rejected(edge_sql, {**base, "par": extra, "chd": fresh["n"], "pos": 50}, "ck_tree_edges_position")

        tree_sql = (
            "INSERT INTO summary_trees (tenant_id,project_id,index_version_id,tree_key,technique_id,embedding_key,"
            "embedding_dim,generator_json) VALUES (:t,:p,:i,:k,'R37','e',4,'{}')"
        )
        await rejected(tree_sql, {**base, "i": building.id, "k": "c" * 64}, "index version is not ready")
        await rejected(tree_sql, {**base, "k": tree.tree_key}, "pk_summary_trees")

        for sql in (
            "UPDATE summary_trees SET generator_json='z' WHERE index_version_id=:i",
            "UPDATE tree_nodes SET text='z' WHERE index_version_id=:i",
            "UPDATE tree_edges SET parent_id=child_id WHERE index_version_id=:i",
        ):
            await rejected(sql, {"i": version.id}, "index rows are immutable")

        # A row set that satisfies every database rule but breaks a domain rule (summary without children).
        orphan_key = "d" * 64
        await accepted(tree_sql, {**base, "k": orphan_key})
        await accepted(node_sql, {**summary_row, **no_source, "g": orphan_key, "n": "rnd_" + "e" * 16})
        with pytest.raises(IndexingError) as info:
            await repo.get_summary_tree(scope, version.id, orphan_key)
        assert info.value.code == "tree_invalid"

        # Alembic runs its own event loop, so it must not run on this test's loop.
        await asyncio.to_thread(run_migrations, _url(), "downgrade", "0011_entity_graphs")
        async with factory.databases[0].engine.connect() as connection:
            for table in ("summary_trees", "tree_nodes", "tree_edges"):
                assert await connection.scalar(text("SELECT to_regclass(:n) IS NULL"), {"n": table})
            assert await connection.scalar(text("SELECT to_regclass('graph_spans') IS NOT NULL"))
            assert (
                await connection.scalar(text("SELECT count(*) FROM pg_proc WHERE proname='reject_index_update'")) == 1
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_proc WHERE proname IN ('tree_nodes_validate_insert','tree_edges_validate_insert','summary_trees_validate_insert')"
                    )
                )
                == 0
            )
        await asyncio.to_thread(run_migrations, _url(), "upgrade", "head")
        async with factory.databases[0].engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('tree_nodes') IS NOT NULL"))
            assert (
                await connection.scalar(text("SELECT count(*) FROM pg_proc WHERE proname='reject_index_update'")) == 1
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_trigger WHERE tgname IN ('summary_trees_no_update','tree_nodes_no_update','tree_edges_no_update')"
                    )
                )
                == 3
            )
    finally:
        await factory.close()


@run
async def test_kg_database_guards():
    from rag_foundry_domain.errors import IndexingError
    from rag_foundry_domain.index_testing import make_chunk, make_knowledge_graph, make_new_index_version

    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("kg-guard"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="kg-guard"))).view
        foreign_out = await docs.create_document_or_get(other, make_new_version("kg-guard-b"), display_name="d")
        foreign_view = (
            await views.create_view_or_get(other, make_new_extraction(foreign_out.version, tag="kg-guard-b"))
        ).view
        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        await repo.add_chunks(
            scope,
            version.id,
            (
                make_chunk(view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),
                make_chunk(view.id, 1, (0, 1, 0, 0), "Bob read them"),
            ),
        )
        await repo.mark_ready(scope, version.id)
        chunk, chunk2 = await repo.list_chunks(scope, version.id)
        building = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        foreign_version = await repo.create_index_version(other, make_new_index_version((foreign_view.id,)))
        await repo.add_chunks(
            other, foreign_version.id, (make_chunk(foreign_view.id, 0, (1, 0, 0, 0), "Ada wrote notes"),)
        )
        foreign_chunk = (await repo.list_chunks(other, foreign_version.id))[0]

        graph = make_knowledge_graph(scope, version, (chunk,))
        graph_b = make_knowledge_graph(scope, version, (chunk2,))
        await repo.add_knowledge_graph(scope, version.id, graph)
        await repo.add_knowledge_graph(scope, version.id, graph_b)
        person = next(e for e in graph.entities if e.type == "Person")
        foreign_entity = graph_b.entities[0]

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        async def accepted(sql, params):
            async with factory.database().sessionmaker() as session, session.begin():
                await session.execute(text(sql), params)

        base = {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "g": graph.kg_key}
        span_sql = (
            "INSERT INTO kg_spans (tenant_id,project_id,index_version_id,kg_key,owner_kind,owner_id,ordinal,"
            "chunk_id,extraction_view_id,start_cp,end_cp,quote) VALUES (:t,:p,:i,:g,:k,:o,:n,:c,:v,:s,:e,:q)"
        )
        good_span = {
            **base,
            "k": "entity",
            "o": person.entity_id,
            "n": 1,
            "c": chunk.id,
            "v": view.id,
            "s": 0,
            "e": 3,
            "q": "Ada",
        }
        # The trigger and keys accept a valid span, so each rejection below is caused by its own defect.
        await accepted(span_sql, good_span)

        def span(**changes):
            return {**good_span, "n": 2, **changes}

        await rejected(span_sql, span(c=foreign_chunk.id), "fk_kg_spans_chunk")
        await rejected(span_sql, span(q="Adx"), "kg span quote does not match chunk text")
        await rejected(span_sql, span(s=chunk.end, e=chunk.end + 2, q="xx"), "kg span outside chunk")
        await rejected(span_sql, span(v=foreign_view.id), "kg span view does not match chunk")
        await rejected(span_sql, span(o="ken_" + "0" * 16), "kg span owner not found")
        await rejected(span_sql, span(k="fact", o="fct_" + "0" * 16), "kg span owner not found")
        await rejected(span_sql, span(k="relation"), "ck_kg_spans_kind")

        fact_sql = (
            "INSERT INTO kg_facts (tenant_id,project_id,index_version_id,kg_key,fact_id,subject_id,predicate,"
            "object_id,literal_text,literal_int) VALUES (:t,:p,:i,:g,:f,:s,'founded',:o,:lt,:li)"
        )
        fact = {**base, "f": "fct_" + "a" * 16, "s": person.entity_id, "o": None, "lt": None, "li": 5}
        await rejected(fact_sql, {**fact, "o": foreign_entity.entity_id, "li": None}, "fk_kg_facts_object")
        await rejected(fact_sql, {**fact, "o": person.entity_id}, "ck_kg_facts_value")
        await rejected(fact_sql, {**fact, "li": None}, "ck_kg_facts_value")
        await rejected(fact_sql, {**fact, "f": "fct_xyz"}, "ck_kg_facts_id")
        await accepted(fact_sql, fact)

        kg_sql = (
            "INSERT INTO knowledge_graphs (tenant_id,project_id,index_version_id,kg_key,technique_id,schema_json,"
            "schema_sha256,generator_json) VALUES (:t,:p,:i,:g,'R48',:sj,:sha,'{}')"
        )
        schema = graph.knowledge_schema
        kg_row = {**base, "sj": schema._canonical_json(), "sha": schema.schema_sha256()}
        await rejected(kg_sql, {**kg_row, "i": building.id, "g": "c" * 64}, "index version is not ready")
        await rejected(kg_sql, {**kg_row, "g": graph.kg_key}, "pk_knowledge_graphs")

        for sql in (
            "UPDATE knowledge_graphs SET generator_json='z' WHERE index_version_id=:i",
            "UPDATE kg_entities SET name='z' WHERE index_version_id=:i",
            "UPDATE kg_facts SET predicate='z' WHERE index_version_id=:i",
            "UPDATE kg_spans SET quote='z' WHERE index_version_id=:i",
        ):
            await rejected(sql, {"i": version.id}, "index rows are immutable")

        # A row set that satisfies every database rule but breaks a domain rule (entity ID is not derived).
        bad_key = "d" * 64
        await accepted(kg_sql, {**kg_row, "g": bad_key})
        bad_id = "ken_" + "1" * 16
        await accepted(
            "INSERT INTO kg_entities (tenant_id,project_id,index_version_id,kg_key,entity_id,type,name) "
            "VALUES (:t,:p,:i,:g,:e,'Person','Zed')",
            {**base, "g": bad_key, "e": bad_id},
        )
        await accepted(span_sql, {**good_span, "g": bad_key, "o": bad_id, "n": 0})
        with pytest.raises(IndexingError) as info:
            await repo.get_knowledge_graph(scope, version.id, bad_key)
        assert info.value.code == "kag_invalid"

        # Alembic runs its own event loop, so it must not run on this test's loop.
        await asyncio.to_thread(run_migrations, _url(), "downgrade", "0012_summary_trees")
        async with factory.databases[0].engine.connect() as connection:
            for table in ("knowledge_graphs", "kg_entities", "kg_facts", "kg_spans"):
                assert await connection.scalar(text("SELECT to_regclass(:n) IS NULL"), {"n": table})
            assert await connection.scalar(text("SELECT to_regclass('tree_nodes') IS NOT NULL"))
            assert (
                await connection.scalar(text("SELECT count(*) FROM pg_proc WHERE proname='reject_index_update'")) == 1
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_proc WHERE proname IN ('kg_spans_validate_insert','knowledge_graphs_validate_insert')"
                    )
                )
                == 0
            )
        await asyncio.to_thread(run_migrations, _url(), "upgrade", "head")
        async with factory.databases[0].engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('kg_spans') IS NOT NULL"))
            assert (
                await connection.scalar(text("SELECT count(*) FROM pg_proc WHERE proname='reject_index_update'")) == 1
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_trigger WHERE tgname IN ('knowledge_graphs_no_update','kg_entities_no_update','kg_facts_no_update','kg_spans_no_update')"
                    )
                )
                == 4
            )
    finally:
        await factory.close()
