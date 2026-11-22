# ruff: noqa: E501
"""RAPTOR summary trees (R37) belong to ready, immutable index versions (ADR-0020).

Raw SQL schema; not mirrored in models.py. reject_index_update() is owned by 0003 and kept on downgrade.
"""

from alembic import op

revision = "0012_summary_trees"
down_revision = "0011_entity_graphs"
branch_labels = None
depends_on = None

_SCOPE = (
    "tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL, tree_key varchar(64) NOT NULL"
)
_TREE_FK = "(tenant_id, project_id, index_version_id, tree_key) REFERENCES summary_trees (tenant_id, project_id, index_version_id, tree_key)"
_NODE_REF = "REFERENCES tree_nodes (tenant_id, project_id, index_version_id, tree_key, node_id)"


def upgrade() -> None:
    op.execute(f"""CREATE TABLE summary_trees (
        {_SCOPE}, technique_id varchar(8) NOT NULL, embedding_key varchar(500) NOT NULL, embedding_dim smallint NOT NULL,
        generator_json text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_summary_trees PRIMARY KEY (index_version_id, tree_key),
        CONSTRAINT uq_summary_trees_tree_key UNIQUE (tree_key),
        CONSTRAINT uq_summary_trees_scope UNIQUE (tenant_id, project_id, index_version_id, tree_key),
        CONSTRAINT fk_summary_trees_scope FOREIGN KEY (tenant_id, project_id, index_version_id)
          REFERENCES index_versions (tenant_id, project_id, id),
        CONSTRAINT ck_summary_trees_key CHECK (tree_key ~ '^[0-9a-f]{{64}}$'),
        CONSTRAINT ck_summary_trees_technique CHECK (technique_id ~ '^R[0-9]{{2}}$'),
        CONSTRAINT ck_summary_trees_dim CHECK (embedding_dim BETWEEN 1 AND 8192),
        CONSTRAINT ck_summary_trees_generator CHECK (char_length(generator_json) <= 8000)
    )""")
    op.execute(f"""CREATE TABLE tree_nodes (
        {_SCOPE}, node_id varchar(20) NOT NULL, level smallint NOT NULL, kind varchar(8) NOT NULL, text text NOT NULL,
        synthetic boolean NOT NULL, embedding double precision[] NOT NULL,
        chunk_id uuid NULL, extraction_view_id uuid NULL, start_cp integer NULL, end_cp integer NULL,
        CONSTRAINT pk_tree_nodes PRIMARY KEY (index_version_id, tree_key, node_id),
        CONSTRAINT uq_tree_nodes_scope UNIQUE (tenant_id, project_id, index_version_id, tree_key, node_id),
        CONSTRAINT uq_tree_nodes_chunk UNIQUE (tenant_id, project_id, index_version_id, tree_key, chunk_id),
        CONSTRAINT fk_tree_nodes_tree FOREIGN KEY {_TREE_FK},
        CONSTRAINT fk_tree_nodes_chunk FOREIGN KEY (tenant_id, project_id, index_version_id, chunk_id)
          REFERENCES chunks (tenant_id, project_id, index_version_id, id),
        CONSTRAINT fk_tree_nodes_view FOREIGN KEY (tenant_id, project_id, index_version_id, extraction_view_id)
          REFERENCES index_version_views (tenant_id, project_id, index_version_id, extraction_view_id),
        CONSTRAINT ck_tree_nodes_id CHECK (node_id ~ '^rnd_[0-9a-f]{{16}}$'),
        CONSTRAINT ck_tree_nodes_level CHECK (level BETWEEN 0 AND 8),
        CONSTRAINT ck_tree_nodes_kind CHECK (kind IN ('leaf', 'summary')),
        CONSTRAINT ck_tree_nodes_leaf CHECK (kind <> 'leaf' OR (
          chunk_id IS NOT NULL AND extraction_view_id IS NOT NULL AND start_cp IS NOT NULL AND end_cp IS NOT NULL
          AND level = 0 AND NOT synthetic AND start_cp >= 0 AND start_cp < end_cp AND char_length(text) = end_cp - start_cp)),
        CONSTRAINT ck_tree_nodes_summary CHECK (kind <> 'summary' OR (
          chunk_id IS NULL AND extraction_view_id IS NULL AND start_cp IS NULL AND end_cp IS NULL
          AND level >= 1 AND synthetic AND char_length(text) BETWEEN 1 AND 4000))
    )""")
    op.execute(f"""CREATE TABLE tree_edges (
        {_SCOPE}, parent_id varchar(20) NOT NULL, child_id varchar(20) NOT NULL,
        position smallint NOT NULL,
        CONSTRAINT pk_tree_edges PRIMARY KEY (index_version_id, tree_key, child_id),
        CONSTRAINT uq_tree_edges_position UNIQUE (index_version_id, tree_key, parent_id, position),
        CONSTRAINT ck_tree_edges_position CHECK (position BETWEEN 0 AND 49),
        CONSTRAINT fk_tree_edges_tree FOREIGN KEY {_TREE_FK},
        CONSTRAINT fk_tree_edges_parent FOREIGN KEY (tenant_id, project_id, index_version_id, tree_key, parent_id)
          {_NODE_REF},
        CONSTRAINT fk_tree_edges_child FOREIGN KEY (tenant_id, project_id, index_version_id, tree_key, child_id)
          {_NODE_REF}
    )""")
    for table in ("tree_nodes", "tree_edges"):
        op.execute(f"CREATE INDEX ix_{table}_scope ON {table} (tenant_id, project_id, index_version_id, tree_key)")
    op.execute("""CREATE FUNCTION summary_trees_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE current_state text;
    BEGIN
        SELECT state INTO current_state FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'ready' THEN RAISE EXCEPTION 'index version is not ready' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER summary_trees_validate BEFORE INSERT ON summary_trees "
        "FOR EACH ROW EXECUTE FUNCTION summary_trees_validate_insert()"
    )
    # A missing tree or chunk is left to the foreign keys so they report scope violations;
    # incomplete leaf columns are left to ck_tree_nodes_leaf.
    op.execute("""CREATE FUNCTION tree_nodes_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE t record; c record;
    BEGIN
        SELECT embedding_dim INTO t FROM summary_trees
        WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND tree_key = NEW.tree_key;
        IF FOUND AND array_length(NEW.embedding, 1) IS DISTINCT FROM t.embedding_dim THEN RAISE EXCEPTION 'tree node embedding length does not match tree' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.kind IS DISTINCT FROM 'leaf' OR NEW.chunk_id IS NULL OR NEW.extraction_view_id IS NULL OR NEW.start_cp IS NULL OR NEW.end_cp IS NULL THEN RETURN NEW; END IF;
        SELECT extraction_view_id, start_cp, end_cp, text INTO c FROM chunks
        WHERE id = NEW.chunk_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id;
        IF NOT FOUND THEN RETURN NEW; END IF;
        IF c.extraction_view_id <> NEW.extraction_view_id THEN RAISE EXCEPTION 'tree leaf view does not match chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.start_cp < c.start_cp OR NEW.end_cp > c.end_cp THEN RAISE EXCEPTION 'tree leaf outside chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.text IS DISTINCT FROM substr(c.text, NEW.start_cp - c.start_cp + 1, NEW.end_cp - NEW.start_cp) THEN RAISE EXCEPTION 'tree leaf text does not match chunk text' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER tree_nodes_validate BEFORE INSERT ON tree_nodes "
        "FOR EACH ROW EXECUTE FUNCTION tree_nodes_validate_insert()"
    )
    # A missing parent or child node is left to the foreign keys.
    op.execute("""CREATE FUNCTION tree_edges_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE parent record; child record;
    BEGIN
        SELECT kind, level INTO parent FROM tree_nodes
        WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND tree_key = NEW.tree_key AND node_id = NEW.parent_id;
        IF NOT FOUND THEN RETURN NEW; END IF;
        SELECT level INTO child FROM tree_nodes
        WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND tree_key = NEW.tree_key AND node_id = NEW.child_id;
        IF NOT FOUND THEN RETURN NEW; END IF;
        IF parent.kind <> 'summary' THEN RAISE EXCEPTION 'tree edge parent must be a summary' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF parent.level <= child.level THEN RAISE EXCEPTION 'tree edge child level must be lower than parent level' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER tree_edges_validate BEFORE INSERT ON tree_edges "
        "FOR EACH ROW EXECUTE FUNCTION tree_edges_validate_insert()"
    )
    for table in ("summary_trees", "tree_nodes", "tree_edges"):
        op.execute(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
        )


def downgrade() -> None:
    for table in ("tree_edges", "tree_nodes", "summary_trees"):
        op.execute(f"DROP TRIGGER {table}_no_update ON {table}")
    op.execute("DROP TRIGGER tree_edges_validate ON tree_edges")
    op.execute("DROP TRIGGER tree_nodes_validate ON tree_nodes")
    op.execute("DROP TRIGGER summary_trees_validate ON summary_trees")
    op.execute("DROP FUNCTION tree_edges_validate_insert()")
    op.execute("DROP FUNCTION tree_nodes_validate_insert()")
    op.execute("DROP FUNCTION summary_trees_validate_insert()")
    for table in ("tree_edges", "tree_nodes"):
        op.execute(f"DROP INDEX ix_{table}_scope")
    op.execute("DROP TABLE tree_edges")
    op.execute("DROP TABLE tree_nodes")
    op.execute("DROP TABLE summary_trees")
