"""Store parent spans on chunks."""

from alembic import op

revision = "0007_chunk_parent_spans"
down_revision = "0006_chunking_prompts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN parent_start_cp integer")
    op.execute("ALTER TABLE chunks ADD COLUMN parent_end_cp integer")
    op.execute("""ALTER TABLE chunks ADD CONSTRAINT ck_chunks_parent_span CHECK (
        (parent_start_cp IS NULL AND parent_end_cp IS NULL) OR
        (0 <= parent_start_cp AND parent_start_cp <= start_cp AND start_cp < end_cp AND end_cp <= parent_end_cp)
    )""")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP CONSTRAINT ck_chunks_parent_span")
    op.execute("ALTER TABLE chunks DROP COLUMN parent_end_cp")
    op.execute("ALTER TABLE chunks DROP COLUMN parent_start_cp")
