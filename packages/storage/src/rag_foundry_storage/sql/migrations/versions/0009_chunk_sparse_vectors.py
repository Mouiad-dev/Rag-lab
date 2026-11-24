"""Learned sparse vectors belong to immutable chunks."""

from alembic import op

revision = "0009_chunk_sparse_vectors"
down_revision = "0008_chunks_fts_simple"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT uq_chunks_scope_id UNIQUE (tenant_id, project_id, index_version_id, id)"
    )
    op.execute("""CREATE TABLE chunk_sparse_vectors (
        tenant_id uuid NOT NULL, project_id uuid NOT NULL, index_version_id uuid NOT NULL,
        chunk_id uuid NOT NULL, encoder_key varchar(64) NOT NULL,
        terms text[] NOT NULL, weights real[] NOT NULL,
        CONSTRAINT pk_chunk_sparse_vectors PRIMARY KEY (chunk_id, encoder_key),
        CONSTRAINT fk_sparse_chunk_scope FOREIGN KEY (tenant_id, project_id, index_version_id, chunk_id)
          REFERENCES chunks (tenant_id, project_id, index_version_id, id),
        CONSTRAINT ck_sparse_lengths CHECK (cardinality(terms) = cardinality(weights))
    )""")
    op.execute(
        "CREATE INDEX ix_sparse_scope_encoder ON chunk_sparse_vectors "
        "(tenant_id, project_id, index_version_id, encoder_key)"
    )
    op.execute("""CREATE FUNCTION reject_sparse_index_update() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'sparse vectors are immutable' USING ERRCODE='integrity_constraint_violation'; END $$""")
    op.execute(
        "CREATE TRIGGER chunk_sparse_vectors_no_update BEFORE UPDATE ON chunk_sparse_vectors "
        "FOR EACH ROW EXECUTE FUNCTION reject_sparse_index_update()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER chunk_sparse_vectors_no_update ON chunk_sparse_vectors")
    op.execute("DROP FUNCTION reject_sparse_index_update()")
    op.execute("DROP TABLE chunk_sparse_vectors")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT uq_chunks_scope_id")
