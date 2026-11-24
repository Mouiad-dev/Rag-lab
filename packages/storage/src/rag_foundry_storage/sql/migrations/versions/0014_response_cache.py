# ruff: noqa: E501
"""R50/R51 response-cache tables (ADR-0021).

Raw SQL schema; not mirrored in models.py. Rows are scoped, never updated and removed only by explicit
invalidation, delete or expiry purge. No question text is stored. question_vector is a pgvector `vector`
without a fixed dimension and without an ANN index; similarity is an exact scan inside one partition.
reject_index_update() is owned by 0003 and kept on downgrade.
"""

from alembic import op

revision = "0014_response_cache"
down_revision = "0013_knowledge_graphs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE response_cache_entries (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, key char(64) NOT NULL, partition char(64) NOT NULL,
        response_json text NOT NULL, source_run_id varchar(64) NOT NULL,
        created_at timestamptz NOT NULL, expires_at timestamptz NOT NULL,
        features_json text NULL, question_vector vector NULL, vector_key varchar(200) NULL,
        CONSTRAINT pk_response_cache_entries PRIMARY KEY (tenant_id, project_id, key),
        CONSTRAINT fk_response_cache_entries_scope FOREIGN KEY (tenant_id, project_id)
          REFERENCES projects (tenant_id, id),
        CONSTRAINT ck_response_cache_entries_key CHECK (key ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_response_cache_entries_partition CHECK (partition ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_response_cache_entries_expiry CHECK (expires_at > created_at),
        CONSTRAINT ck_response_cache_entries_response CHECK (char_length(response_json) BETWEEN 1 AND 200000),
        CONSTRAINT ck_response_cache_entries_run CHECK (char_length(source_run_id) BETWEEN 1 AND 64),
        CONSTRAINT ck_response_cache_entries_features CHECK (features_json IS NULL OR char_length(features_json) BETWEEN 1 AND 4000),
        CONSTRAINT ck_response_cache_entries_vector_pair CHECK ((question_vector IS NULL) = (vector_key IS NULL)),
        CONSTRAINT ck_response_cache_entries_vector_key CHECK (vector_key IS NULL OR char_length(vector_key) BETWEEN 1 AND 200),
        CONSTRAINT ck_response_cache_entries_vector_dims CHECK (question_vector IS NULL OR vector_dims(question_vector) BETWEEN 1 AND 8192)
    )""")
    op.execute(
        "CREATE INDEX ix_response_cache_entries_partition ON response_cache_entries (tenant_id, project_id, partition, vector_key)"
    )
    op.execute(
        "CREATE INDEX ix_response_cache_entries_expiry ON response_cache_entries (tenant_id, project_id, expires_at)"
    )
    op.execute("""CREATE TABLE response_cache_sources (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, key char(64) NOT NULL, document_version_id uuid NOT NULL,
        ordinal smallint NOT NULL,
        CONSTRAINT pk_response_cache_sources PRIMARY KEY (tenant_id, project_id, key, document_version_id),
        CONSTRAINT fk_response_cache_sources_entry FOREIGN KEY (tenant_id, project_id, key)
          REFERENCES response_cache_entries (tenant_id, project_id, key) ON DELETE CASCADE,
        CONSTRAINT ck_response_cache_sources_ordinal CHECK (ordinal BETWEEN 0 AND 199)
    )""")
    op.execute(
        "CREATE INDEX ix_response_cache_sources_version ON response_cache_sources (tenant_id, project_id, document_version_id)"
    )
    op.execute("""CREATE FUNCTION response_cache_sources_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        PERFORM 1 FROM document_versions
        WHERE id = NEW.document_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF NOT FOUND THEN RAISE EXCEPTION 'cache source version not in scope' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER response_cache_sources_validate BEFORE INSERT ON response_cache_sources "
        "FOR EACH ROW EXECUTE FUNCTION response_cache_sources_validate_insert()"
    )
    for table in ("response_cache_entries", "response_cache_sources"):
        op.execute(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
        )


def downgrade() -> None:
    for table in ("response_cache_sources", "response_cache_entries"):
        op.execute(f"DROP TRIGGER {table}_no_update ON {table}")
    op.execute("DROP TRIGGER response_cache_sources_validate ON response_cache_sources")
    op.execute("DROP FUNCTION response_cache_sources_validate_insert()")
    op.execute("DROP INDEX ix_response_cache_sources_version")
    op.execute("DROP TABLE response_cache_sources")
    op.execute("DROP INDEX ix_response_cache_entries_expiry")
    op.execute("DROP INDEX ix_response_cache_entries_partition")
    op.execute("DROP TABLE response_cache_entries")
