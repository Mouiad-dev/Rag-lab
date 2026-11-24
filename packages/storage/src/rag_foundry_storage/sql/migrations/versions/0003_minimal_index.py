# ruff: noqa: E501
"""Minimal immutable vector index (ADR-0012 D6-D8).

Revision ID: 0003_minimal_index
Revises: 0002_extraction_views

Raw SQL schema; not mirrored in models.py; no autogenerate.
"""

from alembic import op

revision = "0003_minimal_index"
down_revision = "0002_extraction_views"
branch_labels = None
depends_on = None

_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "ALTER TABLE extraction_views ADD CONSTRAINT uq_extraction_views_tenant_id_project_id_id UNIQUE (tenant_id, project_id, id)",
    """CREATE TABLE index_versions (
        id uuid PRIMARY KEY, tenant_id uuid NOT NULL, project_id uuid NOT NULL,
        chunker_id varchar(3) NOT NULL, chunker_version varchar(128) NOT NULL,
        chunker_config_json text NOT NULL, chunker_config_sha256 varchar(64) NOT NULL,
        embedding_profile_id varchar(128) NOT NULL, embedding_provider varchar(64) NOT NULL,
        embedding_model varchar(256) NOT NULL, embedding_dim integer NOT NULL,
        distance varchar(16) NOT NULL, embedding_key varchar(64) NOT NULL,
        corpus_sha256 varchar(64) NOT NULL, unanchored_chunk_count integer NOT NULL,
        state varchar(16) NOT NULL, chunk_count integer NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(), ready_at timestamptz,
        CONSTRAINT fk_index_versions_project FOREIGN KEY (tenant_id, project_id) REFERENCES projects (tenant_id, id),
        CONSTRAINT uq_index_versions_tenant_project_id UNIQUE (tenant_id, project_id, id),
        CONSTRAINT ck_index_versions_state CHECK (state IN ('building', 'ready', 'failed')),
        CONSTRAINT ck_index_versions_distance CHECK (distance = 'cosine'),
        CONSTRAINT ck_index_versions_embedding_dim CHECK (embedding_dim BETWEEN 1 AND 16000),
        CONSTRAINT ck_index_versions_hashes CHECK (chunker_config_sha256 ~ '^[0-9a-f]{64}$' AND embedding_key ~ '^[0-9a-f]{64}$' AND corpus_sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_index_versions_counts CHECK (unanchored_chunk_count >= 0 AND chunk_count >= 0)
    )""",
    """CREATE TABLE index_version_views (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL, extraction_view_id uuid NOT NULL,
        CONSTRAINT pk_index_version_views PRIMARY KEY (index_version_id, extraction_view_id),
        CONSTRAINT uq_index_version_views_scope UNIQUE (tenant_id, project_id, index_version_id, extraction_view_id),
        CONSTRAINT fk_index_version_views_version FOREIGN KEY (tenant_id, project_id, index_version_id) REFERENCES index_versions (tenant_id, project_id, id),
        CONSTRAINT fk_index_version_views_view FOREIGN KEY (tenant_id, project_id, extraction_view_id) REFERENCES extraction_views (tenant_id, project_id, id)
    )""",
    """CREATE TABLE chunks (
        id uuid PRIMARY KEY, tenant_id uuid NOT NULL, project_id uuid NOT NULL,
        index_version_id uuid NOT NULL, extraction_view_id uuid NOT NULL,
        ordinal integer NOT NULL, start_cp integer NOT NULL, end_cp integer NOT NULL,
        split_level integer NOT NULL, block_ids text[] NOT NULL, text text NOT NULL,
        text_sha256 varchar(64) NOT NULL, embedding vector NOT NULL,
        CONSTRAINT uq_chunks_version_ordinal UNIQUE (index_version_id, ordinal),
        CONSTRAINT fk_chunks_version_view FOREIGN KEY (tenant_id, project_id, index_version_id, extraction_view_id)
            REFERENCES index_version_views (tenant_id, project_id, index_version_id, extraction_view_id),
        CONSTRAINT ck_chunks_range CHECK (end_cp > start_cp AND char_length(text) = end_cp - start_cp),
        CONSTRAINT ck_chunks_ordinal CHECK (ordinal >= 0),
        CONSTRAINT ck_chunks_text_sha256 CHECK (text_sha256 ~ '^[0-9a-f]{64}$')
    )""",
    "CREATE INDEX ix_chunks_scope_version_ordinal ON chunks (tenant_id, project_id, index_version_id, ordinal)",
    """CREATE FUNCTION chunks_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE expected_dim integer; current_state text;
    BEGIN
        SELECT state, embedding_dim INTO current_state, expected_dim FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'building' THEN RAISE EXCEPTION 'index version is not building' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF vector_dims(NEW.embedding) <> expected_dim THEN RAISE EXCEPTION 'embedding dimension mismatch' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""",
    "CREATE TRIGGER chunks_validate BEFORE INSERT ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_validate_insert()",
    """CREATE FUNCTION reject_index_update() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'index rows are immutable' USING ERRCODE = 'integrity_constraint_violation'; END $$""",
    "CREATE TRIGGER chunks_no_update BEFORE UPDATE ON chunks FOR EACH ROW EXECUTE FUNCTION reject_index_update()",
    "CREATE TRIGGER index_version_views_no_update BEFORE UPDATE ON index_version_views FOR EACH ROW EXECUTE FUNCTION reject_index_update()",
    """CREATE FUNCTION index_versions_transition() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        IF OLD.state <> 'building' OR NEW.state NOT IN ('ready', 'failed') OR
           (to_jsonb(NEW) - ARRAY['state','chunk_count','ready_at']) <> (to_jsonb(OLD) - ARRAY['state','chunk_count','ready_at']) THEN
            RAISE EXCEPTION 'invalid index version update' USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
    END $$""",
    "CREATE TRIGGER index_versions_transition BEFORE UPDATE ON index_versions FOR EACH ROW EXECUTE FUNCTION index_versions_transition()",
)


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TRIGGER index_versions_transition ON index_versions")
    op.execute("DROP TRIGGER index_version_views_no_update ON index_version_views")
    op.execute("DROP TRIGGER chunks_no_update ON chunks")
    op.execute("DROP TRIGGER chunks_validate ON chunks")
    op.execute("DROP FUNCTION index_versions_transition()")
    op.execute("DROP FUNCTION reject_index_update()")
    op.execute("DROP FUNCTION chunks_validate_insert()")
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE index_version_views")
    op.execute("DROP TABLE index_versions")
    op.execute("ALTER TABLE extraction_views DROP CONSTRAINT uq_extraction_views_tenant_id_project_id_id")
