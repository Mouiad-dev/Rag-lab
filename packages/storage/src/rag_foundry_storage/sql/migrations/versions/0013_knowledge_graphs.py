# ruff: noqa: E501
"""KAG knowledge graphs (R48) belong to ready, immutable index versions (ADR-0020).

Raw SQL schema; not mirrored in models.py. reject_index_update() is owned by 0003 and kept on downgrade.
Schema semantics (domain/range, literal kinds) are checked in the domain model, not here.
"""

from alembic import op

revision = "0013_knowledge_graphs"
down_revision = "0012_summary_trees"
branch_labels = None
depends_on = None

_SCOPE = (
    "tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL, kg_key varchar(64) NOT NULL"
)
_KG_FK = "(tenant_id, project_id, index_version_id, kg_key) REFERENCES knowledge_graphs (tenant_id, project_id, index_version_id, kg_key)"
_ENTITY_REF = "REFERENCES kg_entities (tenant_id, project_id, index_version_id, kg_key, entity_id)"


def upgrade() -> None:
    op.execute(f"""CREATE TABLE knowledge_graphs (
        {_SCOPE}, technique_id varchar(8) NOT NULL, schema_json text NOT NULL, schema_sha256 char(64) NOT NULL,
        generator_json text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_knowledge_graphs PRIMARY KEY (index_version_id, kg_key),
        CONSTRAINT uq_knowledge_graphs_kg_key UNIQUE (kg_key),
        CONSTRAINT uq_knowledge_graphs_scope UNIQUE (tenant_id, project_id, index_version_id, kg_key),
        CONSTRAINT fk_knowledge_graphs_scope FOREIGN KEY (tenant_id, project_id, index_version_id)
          REFERENCES index_versions (tenant_id, project_id, id),
        CONSTRAINT ck_knowledge_graphs_key CHECK (kg_key ~ '^[0-9a-f]{{64}}$'),
        CONSTRAINT ck_knowledge_graphs_sha CHECK (schema_sha256 ~ '^[0-9a-f]{{64}}$'),
        CONSTRAINT ck_knowledge_graphs_technique CHECK (technique_id ~ '^R[0-9]{{2}}$'),
        CONSTRAINT ck_knowledge_graphs_schema CHECK (char_length(schema_json) <= 64000),
        CONSTRAINT ck_knowledge_graphs_generator CHECK (char_length(generator_json) <= 8000)
    )""")
    op.execute(f"""CREATE TABLE kg_entities (
        {_SCOPE}, entity_id varchar(20) NOT NULL, type varchar(40) NOT NULL, name text NOT NULL,
        CONSTRAINT pk_kg_entities PRIMARY KEY (index_version_id, kg_key, entity_id),
        CONSTRAINT uq_kg_entities_scope UNIQUE (tenant_id, project_id, index_version_id, kg_key, entity_id),
        CONSTRAINT fk_kg_entities_graph FOREIGN KEY {_KG_FK},
        CONSTRAINT ck_kg_entities_id CHECK (entity_id ~ '^ken_[0-9a-f]{{16}}$'),
        CONSTRAINT ck_kg_entities_name CHECK (char_length(name) BETWEEN 1 AND 500)
    )""")
    op.execute(f"""CREATE TABLE kg_facts (
        {_SCOPE}, fact_id varchar(20) NOT NULL, subject_id varchar(20) NOT NULL, predicate varchar(40) NOT NULL,
        object_id varchar(20) NULL, literal_text text NULL, literal_int bigint NULL,
        CONSTRAINT pk_kg_facts PRIMARY KEY (index_version_id, kg_key, fact_id),
        CONSTRAINT uq_kg_facts_scope UNIQUE (tenant_id, project_id, index_version_id, kg_key, fact_id),
        CONSTRAINT fk_kg_facts_graph FOREIGN KEY {_KG_FK},
        CONSTRAINT fk_kg_facts_subject FOREIGN KEY (tenant_id, project_id, index_version_id, kg_key, subject_id)
          {_ENTITY_REF},
        CONSTRAINT fk_kg_facts_object FOREIGN KEY (tenant_id, project_id, index_version_id, kg_key, object_id)
          {_ENTITY_REF},
        CONSTRAINT ck_kg_facts_id CHECK (fact_id ~ '^fct_[0-9a-f]{{16}}$'),
        CONSTRAINT ck_kg_facts_value CHECK (
          (object_id IS NOT NULL)::int + (literal_text IS NOT NULL)::int + (literal_int IS NOT NULL)::int = 1),
        CONSTRAINT ck_kg_facts_text CHECK (literal_text IS NULL OR char_length(literal_text) BETWEEN 1 AND 500)
    )""")
    op.execute(f"""CREATE TABLE kg_spans (
        {_SCOPE}, owner_kind varchar(8) NOT NULL, owner_id varchar(20) NOT NULL, ordinal smallint NOT NULL,
        chunk_id uuid NOT NULL, extraction_view_id uuid NOT NULL, start_cp integer NOT NULL, end_cp integer NOT NULL,
        quote text NOT NULL,
        CONSTRAINT pk_kg_spans PRIMARY KEY (index_version_id, kg_key, owner_kind, owner_id, ordinal),
        CONSTRAINT fk_kg_spans_graph FOREIGN KEY {_KG_FK},
        CONSTRAINT fk_kg_spans_chunk FOREIGN KEY (tenant_id, project_id, index_version_id, chunk_id)
          REFERENCES chunks (tenant_id, project_id, index_version_id, id),
        CONSTRAINT fk_kg_spans_view FOREIGN KEY (tenant_id, project_id, index_version_id, extraction_view_id)
          REFERENCES index_version_views (tenant_id, project_id, index_version_id, extraction_view_id),
        CONSTRAINT ck_kg_spans_kind CHECK (owner_kind IN ('entity', 'fact')),
        CONSTRAINT ck_kg_spans_range CHECK (start_cp >= 0 AND start_cp < end_cp),
        CONSTRAINT ck_kg_spans_quote CHECK (char_length(quote) = end_cp - start_cp),
        CONSTRAINT ck_kg_spans_ordinal CHECK (ordinal BETWEEN 0 AND 49)
    )""")
    for table in ("kg_entities", "kg_facts", "kg_spans"):
        op.execute(f"CREATE INDEX ix_{table}_scope ON {table} (tenant_id, project_id, index_version_id, kg_key)")
    op.execute("""CREATE FUNCTION knowledge_graphs_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE current_state text;
    BEGIN
        SELECT state INTO current_state FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'ready' THEN RAISE EXCEPTION 'index version is not ready' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER knowledge_graphs_validate BEFORE INSERT ON knowledge_graphs "
        "FOR EACH ROW EXECUTE FUNCTION knowledge_graphs_validate_insert()"
    )
    # A missing chunk is left to fk_kg_spans_chunk so the foreign key reports scope violations.
    op.execute("""CREATE FUNCTION kg_spans_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE c record;
    BEGIN
        SELECT extraction_view_id, start_cp, end_cp, text INTO c FROM chunks
        WHERE id = NEW.chunk_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id;
        IF NOT FOUND THEN RETURN NEW; END IF;
        IF c.extraction_view_id <> NEW.extraction_view_id THEN RAISE EXCEPTION 'kg span view does not match chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.start_cp < c.start_cp OR NEW.end_cp > c.end_cp THEN RAISE EXCEPTION 'kg span outside chunk' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.quote IS DISTINCT FROM substr(c.text, NEW.start_cp - c.start_cp + 1, NEW.end_cp - NEW.start_cp) THEN RAISE EXCEPTION 'kg span quote does not match chunk text' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF NEW.owner_kind = 'entity' THEN
            PERFORM 1 FROM kg_entities WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND kg_key = NEW.kg_key AND entity_id = NEW.owner_id;
        ELSIF NEW.owner_kind = 'fact' THEN
            PERFORM 1 FROM kg_facts WHERE tenant_id = NEW.tenant_id AND project_id = NEW.project_id AND index_version_id = NEW.index_version_id AND kg_key = NEW.kg_key AND fact_id = NEW.owner_id;
        ELSE
            RETURN NEW;  -- unknown kind: ck_kg_spans_kind rejects the row
        END IF;
        IF NOT FOUND THEN RAISE EXCEPTION 'kg span owner not found' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER kg_spans_validate BEFORE INSERT ON kg_spans "
        "FOR EACH ROW EXECUTE FUNCTION kg_spans_validate_insert()"
    )
    for table in ("knowledge_graphs", "kg_entities", "kg_facts", "kg_spans"):
        op.execute(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
        )


def downgrade() -> None:
    for table in ("kg_spans", "kg_facts", "kg_entities", "knowledge_graphs"):
        op.execute(f"DROP TRIGGER {table}_no_update ON {table}")
    op.execute("DROP TRIGGER kg_spans_validate ON kg_spans")
    op.execute("DROP TRIGGER knowledge_graphs_validate ON knowledge_graphs")
    op.execute("DROP FUNCTION kg_spans_validate_insert()")
    op.execute("DROP FUNCTION knowledge_graphs_validate_insert()")
    for table in ("kg_spans", "kg_facts", "kg_entities"):
        op.execute(f"DROP INDEX ix_{table}_scope")
    op.execute("DROP TABLE kg_spans")
    op.execute("DROP TABLE kg_facts")
    op.execute("DROP TABLE kg_entities")
    op.execute("DROP TABLE knowledge_graphs")
