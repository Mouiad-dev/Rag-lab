"""PostgreSQL FTS, not BM25; 'simple' = lowercase only, no stemming/stop words/Arabic normalization"""

from alembic import op

revision = "0008_chunks_fts_simple"
down_revision = "0007_chunk_parent_spans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD COLUMN fts_simple tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_fts_simple ON chunks USING gin (fts_simple)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunks_fts_simple")
    op.execute("ALTER TABLE chunks DROP COLUMN fts_simple")
