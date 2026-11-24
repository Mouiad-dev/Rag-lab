# ruff: noqa: E501
"""Synthetic chunk representations (R23/R24) belong to ready, immutable index versions."""

from alembic import op

revision = "0010_chunk_representations"
down_revision = "0009_chunk_sparse_vectors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE representation_sets (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL,
        representation_key varchar(64) NOT NULL, technique_id varchar(8) NOT NULL,
        generator_json text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_representation_sets PRIMARY KEY (index_version_id, representation_key),
        CONSTRAINT fk_representation_sets_scope FOREIGN KEY (tenant_id, project_id, index_version_id)
          REFERENCES index_versions (tenant_id, project_id, id),
        CONSTRAINT ck_representation_sets_key CHECK (representation_key ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_representation_sets_technique CHECK (technique_id ~ '^R[0-9]{2}$'),
        CONSTRAINT ck_representation_sets_generator CHECK (char_length(generator_json) <= 8000)
    )""")
    op.execute("""CREATE TABLE chunk_representations (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL,
        chunk_id uuid NOT NULL, representation_key varchar(64) NOT NULL,
        kind varchar(16) NOT NULL, ordinal smallint NOT NULL, text text NOT NULL,
        text_sha256 varchar(64) NOT NULL, embedded_input_sha256 varchar(64) NOT NULL,
        embedding vector NOT NULL,
        CONSTRAINT pk_chunk_representations PRIMARY KEY (chunk_id, representation_key, kind, ordinal),
        CONSTRAINT fk_chunk_representations_chunk FOREIGN KEY (tenant_id, project_id, index_version_id, chunk_id)
          REFERENCES chunks (tenant_id, project_id, index_version_id, id),
        CONSTRAINT fk_chunk_representations_set FOREIGN KEY (index_version_id, representation_key)
          REFERENCES representation_sets (index_version_id, representation_key),
        CONSTRAINT ck_chunk_representations_kind CHECK (kind IN ('summary', 'question', 'context_prefix')),
        CONSTRAINT ck_chunk_representations_ordinal CHECK (ordinal BETWEEN 0 AND 15),
        CONSTRAINT ck_chunk_representations_text CHECK (char_length(text) BETWEEN 1 AND 4000),
        CONSTRAINT ck_chunk_representations_hashes CHECK (text_sha256 ~ '^[0-9a-f]{64}$' AND embedded_input_sha256 ~ '^[0-9a-f]{64}$')
    )""")
    op.execute(
        "CREATE INDEX ix_chunk_repr_scope_key ON chunk_representations "
        "(tenant_id, project_id, index_version_id, representation_key)"
    )
    op.execute("""CREATE FUNCTION representation_sets_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE current_state text;
    BEGIN
        SELECT state INTO current_state FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'ready' THEN RAISE EXCEPTION 'index version is not ready' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER representation_sets_validate BEFORE INSERT ON representation_sets "
        "FOR EACH ROW EXECUTE FUNCTION representation_sets_validate_insert()"
    )
    op.execute("""CREATE FUNCTION chunk_representations_validate_insert() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE expected_dim integer; current_state text;
    BEGIN
        SELECT state, embedding_dim INTO current_state, expected_dim FROM index_versions
        WHERE id = NEW.index_version_id AND tenant_id = NEW.tenant_id AND project_id = NEW.project_id;
        IF current_state IS DISTINCT FROM 'ready' THEN RAISE EXCEPTION 'index version is not ready' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        IF vector_dims(NEW.embedding) <> expected_dim THEN RAISE EXCEPTION 'embedding dimension mismatch' USING ERRCODE = 'integrity_constraint_violation'; END IF;
        RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER chunk_representations_validate BEFORE INSERT ON chunk_representations "
        "FOR EACH ROW EXECUTE FUNCTION chunk_representations_validate_insert()"
    )
    op.execute(
        "CREATE TRIGGER representation_sets_no_update BEFORE UPDATE ON representation_sets "
        "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
    )
    op.execute(
        "CREATE TRIGGER chunk_representations_no_update BEFORE UPDATE ON chunk_representations "
        "FOR EACH ROW EXECUTE FUNCTION reject_index_update()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER chunk_representations_no_update ON chunk_representations")
    op.execute("DROP TRIGGER representation_sets_no_update ON representation_sets")
    op.execute("DROP TRIGGER chunk_representations_validate ON chunk_representations")
    op.execute("DROP TRIGGER representation_sets_validate ON representation_sets")
    op.execute("DROP FUNCTION chunk_representations_validate_insert()")
    op.execute("DROP FUNCTION representation_sets_validate_insert()")
    op.execute("DROP INDEX ix_chunk_repr_scope_key")
    op.execute("DROP TABLE chunk_representations")
    op.execute("DROP TABLE representation_sets")
