"""Extraction views owned by document versions (ADR-0010).

Revision ID: 0002_extraction_views
Revises: 0001_document_intake
"""

from alembic import op

revision = "0002_extraction_views"
down_revision = "0001_document_intake"
branch_labels = None
depends_on = None

_STATEMENTS = (
    """ALTER TABLE document_versions ADD CONSTRAINT uq_document_versions_tenant_id_project_id_document_id_id
        UNIQUE (tenant_id, project_id, document_id, id)""",
    """CREATE TABLE extraction_views (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        project_id uuid NOT NULL,
        document_id uuid NOT NULL,
        document_version_id uuid NOT NULL,
        extractor varchar(64) NOT NULL,
        extractor_version varchar(128) NOT NULL,
        profile_id varchar(128) NOT NULL,
        unicode_version varchar(32) NOT NULL,
        canonical_sha256 varchar(64) NOT NULL,
        char_count integer NOT NULL,
        block_count integer NOT NULL,
        page_count integer,
        payload_sha256 varchar(64) NOT NULL,
        payload_key text NOT NULL,
        payload_schema_version integer NOT NULL,
        warnings text[] NOT NULL DEFAULT '{}',
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_extraction_views PRIMARY KEY (id),
        CONSTRAINT fk_extraction_views_scope_version FOREIGN KEY
            (tenant_id, project_id, document_id, document_version_id)
            REFERENCES document_versions (tenant_id, project_id, document_id, id),
        CONSTRAINT uq_extraction_views_identity_key UNIQUE
            (document_version_id, extractor, extractor_version, profile_id, unicode_version),
        CONSTRAINT ck_extraction_views_canonical_sha256_format CHECK (canonical_sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_extraction_views_payload_sha256_format CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_extraction_views_char_count_positive CHECK (char_count >= 1),
        CONSTRAINT ck_extraction_views_block_count_positive CHECK (block_count >= 1),
        CONSTRAINT ck_extraction_views_page_count_positive CHECK (page_count IS NULL OR page_count >= 1),
        CONSTRAINT ck_extraction_views_payload_schema_version_known CHECK (payload_schema_version = 1),
        CONSTRAINT ck_extraction_views_extractor_format CHECK (extractor ~ '^[a-z0-9._-]{1,64}$')
    )""",
    """CREATE INDEX ix_extraction_views_version_created
        ON extraction_views (tenant_id, project_id, document_version_id, created_at)""",
    """CREATE FUNCTION extraction_views_reject_update() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        RAISE EXCEPTION 'extraction_views rows are immutable' USING ERRCODE = 'integrity_constraint_violation';
    END $$""",
    """CREATE TRIGGER extraction_views_no_update BEFORE UPDATE ON extraction_views
        FOR EACH ROW EXECUTE FUNCTION extraction_views_reject_update()""",
)


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TRIGGER extraction_views_no_update ON extraction_views")
    op.execute("DROP TABLE extraction_views")
    op.execute("DROP FUNCTION extraction_views_reject_update()")
    op.execute("ALTER TABLE document_versions DROP CONSTRAINT uq_document_versions_tenant_id_project_id_document_id_id")
