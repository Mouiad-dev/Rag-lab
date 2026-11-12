# ruff: noqa: E501
"""Entity graphs (R38/R39) belong to ready, immutable index versions (ADR-0019).

Raw SQL schema; not mirrored in models.py. reject_index_update() is owned by 0003 and kept on downgrade.
"""

from alembic import op

revision = "0011_entity_graphs"
down_revision = "0010_chunk_representations"
branch_labels = None
depends_on = None

_SCOPE = (
    "tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL, graph_key varchar(64) NOT NULL"
)
_GRAPH_FK = "(tenant_id, project_id, index_version_id, graph_key) REFERENCES entity_graphs (tenant_id, project_id, index_version_id, graph_key)"


def upgrade() -> None:
    op.execute(f"""CREATE TABLE entity_graphs (
        {_SCOPE}, technique_id varchar(8) NOT NULL,
        generator_json text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_entity_graphs PRIMARY KEY (index_version_id, graph_key),
        CONSTRAINT uq_entity_graphs_graph_key UNIQUE (graph_key),
        CONSTRAINT uq_entity_graphs_scope UNIQUE (tenant_id, project_id, index_version_id, graph_key),
        CONSTRAINT fk_entity_graphs_scope FOREIGN KEY (tenant_id, project_id, index_version_id)
          REFERENCES index_versions (tenant_id, project_id, id),
        CONSTRAINT ck_entity_graphs_key CHECK (graph_key ~ '^[0-9a-f]{{64}}$'),
        CONSTRAINT ck_entity_graphs_technique CHECK (technique_id ~ '^R[0-9]{{2}}$'),
        CONSTRAINT ck_entity_graphs_generator CHECK (char_length(generator_json) <= 8000)
    )""")
    op.execute(f"""CREATE TABLE graph_entities (
        {_SCOPE}, entity_id varchar(20) NOT NULL, name text NOT NULL, type varchar(64) NOT NULL,
        CONSTRAINT pk_graph_entities PRIMARY KEY (index_version_id, graph_key, entity_id),
        CONSTRAINT uq_graph_entities_scope UNIQUE (tenant_id, project_id, index_version_id, graph_key, entity_id),
        CONSTRAINT fk_graph_entities_graph FOREIGN KEY {_GRAPH_FK},
        CONSTRAINT ck_graph_entities_id CHECK (entity_id ~ '^ent_[0-9a-f]{{16}}$'),
        CONSTRAINT ck_graph_entities_name CHECK (char_length(name) BETWEEN 1 AND 500),
        CONSTRAINT ck_graph_entities_type CHECK (char_length(type) BETWEEN 1 AND 64)
    )""")
    op.execute(f"""CREATE TABLE graph_relations (
        {_SCOPE}, relation_id varchar(20) NOT NULL, source_id varchar(20) NOT NULL, target_id varchar(20) NOT NULL,
        label varchar(200) NOT NULL,
        CONSTRAINT pk_graph_relations PRIMARY KEY (index_version_id, graph_key, relation_id),
        CONSTRAINT uq_graph_relations_scope UNIQUE (tenant_id, project_id, index_version_id, graph_key, relation_id),
        CONSTRAINT fk_graph_relations_graph FOREIGN KEY {_GRAPH_FK},
        CONSTRAINT fk_graph_relations_source FOREIGN KEY (tenant_id, project_id, index_version_id, graph_key, source_id)
          REFERENCES graph_entities (tenant_id, project_id, index_version_id, graph_key, entity_id),
        CONSTRAINT fk_graph_relations_target FOREIGN KEY (tenant_id, project_id, index_version_id, graph_key, target_id)
          REFERENCES graph_entities (tenant_id, project_id, index_version_id, graph_key, entity_id),
        CONSTRAINT ck_graph_relations_id CHECK (relation_id ~ '^rel_[0-9a-f]{{16}}$'),
        CONSTRAINT ck_graph_relations_label CHECK (char_length(label) BETWEEN 1 AND 200)
    )""")
    op.execute(f"""CREATE TABLE graph_spans (
        {_SCOPE}, owner_kind varchar(8) NOT NULL, owner_id varchar(20) NOT NULL, ordinal smallint NOT NULL,
        chunk_id uuid NOT NULL, extraction_view_id uuid NOT NULL, start_cp integer NOT NULL, end_cp integer NOT NULL,
        quote text NOT NULL,
        CONSTRAINT pk_graph_spans PRIMARY KEY (index_version_id, graph_key, owner_kind, owner_id, ordinal),
        CONSTRAINT fk_graph_spans_graph FOREIGN KEY {_GRAPH_FK},
        CONSTRAINT fk_graph_spans_chunk FOREIGN KEY (tenant_id, project_id, index_version_id, chunk_id)
          REFERENCES chunks (tenant_id, project_id, index_version_id, id),
        CONSTRAINT fk_graph_spans_view FOREIGN KEY (tenant_id, project_id, index_version_id, extraction_view_id)
          REFERENCES index_version_views (tenant_id, project_id, index_version_id, extraction_view_id),
        CONSTRAINT ck_graph_spans_kind CHECK (owner_kind IN ('entity', 'relation')),
        CONSTRAINT ck_graph_spans_range CHECK (start_cp >= 0 AND start_cp < end_cp),
        CONSTRAINT ck_graph_spans_quote CHECK (char_length(quote) = end_cp - start_cp),
        CONSTRAINT ck_graph_spans_ordinal CHECK (ordinal BETWEEN 0 AND 49)
    )""")
    for table in ("graph_entities", "graph_relations", "graph_spans"):
        op.execute(f"CREATE INDEX ix_{table}_scope ON {table} (tenant_id, project_id, index_version_id, graph_key)")
    op.execute("""CREATE FUNCTION entity_graphs_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE current_state text;
    BEGIN
        SELECT state INTO current_state FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'ready' THEN RAISE EXCEPTION 'index version is not ready' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER entity_graphs_validate BEFORE INSERT ON entity_graphs "
        "FOR EACH ROW EXECUTE FUNCTION entity_graphs_validate_insert()"
    )
    # A missing chunk is left to fk_graph_spans_chunk so the foreign key reports scope violations.
    op.execute("""CREATE FUNCTION graph_spans_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE c record;
    BEGIN
        SELECT extraction_view_id, start_cp, end_cp, text INTO c FROM chunks
        WHERE id = NEW.chunk_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id;
        IF NOT FOUND THEN RETURN NEW; END IF;
        IF c.extraction_view_id <> NEW.extraction_view_id THEN RAISE EXCEPTION 'graph span view does not match chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.start_cp < c.start_cp OR NEW.end_cp > c.end_cp THEN RAISE EXCEPTION 'graph span outside chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.quote IS DISTINCT FROM substr(c.text, NEW.start_cp - c.start_cp + 1, NEW.end_cp - NEW.start_cp) THEN RAISE EXCEPTION 'graph span quote does not match chunk text' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.owner_kind = 'entity' THEN
            PERFORM 1 FROM graph_entities WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND graph_key = NEW.graph_key AND entity_id = NEW.owner_id;
        ELSE
            PERFORM 1 FROM graph_relations WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND graph_key = NEW.graph_key AND relation_id = NEW.owner_id;
        END IF;
        IF NOT FOUND THEN RAISE EXCEPTION 'graph span owner not found' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER graph_spans_validate BEFORE INSERT ON graph_spans "
        "FOR EACH ROW EXECUTE FUNCTION graph_spans_validate_insert()"
    )
    for table in ("entity_graphs", "graph_entities", "graph_relations", "graph_spans"):
        op.execute(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
        )


def downgrade() -> None:
    for table in ("graph_spans", "graph_relations", "graph_entities", "entity_graphs"):
        op.execute(f"DROP TRIGGER {table}_no_update ON {table}")
    op.execute("DROP TRIGGER graph_spans_validate ON graph_spans")
    op.execute("DROP TRIGGER entity_graphs_validate ON entity_graphs")
    op.execute("DROP FUNCTION graph_spans_validate_insert()")
    op.execute("DROP FUNCTION entity_graphs_validate_insert()")
    for table in ("graph_spans", "graph_relations", "graph_entities"):
        op.execute(f"DROP INDEX ix_{table}_scope")
    op.execute("DROP TABLE graph_spans")
    op.execute("DROP TABLE graph_relations")
    op.execute("DROP TABLE graph_entities")
    op.execute("DROP TABLE entity_graphs")
