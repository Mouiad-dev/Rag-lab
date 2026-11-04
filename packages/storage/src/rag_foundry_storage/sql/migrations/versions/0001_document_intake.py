"""Document intake schema: tenants, projects, documents, document_versions (ADR-0009).

Revision ID: 0001_document_intake
Revises:
"""

from alembic import op

revision = "0001_document_intake"
down_revision = None
branch_labels = None
depends_on = None

_STATEMENTS = (
    """CREATE TABLE tenants (
        id uuid NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_tenants PRIMARY KEY (id)
    )""",
    """CREATE TABLE projects (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_projects PRIMARY KEY (id),
        CONSTRAINT uq_projects_tenant_id_id UNIQUE (tenant_id, id),
        CONSTRAINT fk_projects_tenant_id FOREIGN KEY (tenant_id) REFERENCES tenants (id)
    )""",
    """CREATE TABLE documents (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        project_id uuid NOT NULL,
        display_name varchar(255) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_documents PRIMARY KEY (id),
        CONSTRAINT uq_documents_tenant_id_project_id_id UNIQUE (tenant_id, project_id, id),
        CONSTRAINT fk_documents_tenant_id_project_id FOREIGN KEY (tenant_id, project_id)
            REFERENCES projects (tenant_id, id),
        CONSTRAINT ck_documents_display_name_length CHECK (char_length(display_name) BETWEEN 1 AND 255)
    )""",
    """CREATE TABLE document_versions (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        project_id uuid NOT NULL,
        document_id uuid NOT NULL,
        version_number integer NOT NULL,
        sha256 varchar(64) NOT NULL,
        size_bytes bigint NOT NULL,
        media_type varchar(64) NOT NULL,
        text_encoding varchar(16),
        original_filename varchar(255) NOT NULL,
        object_key text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_document_versions PRIMARY KEY (id),
        CONSTRAINT uq_document_versions_document_id_version_number UNIQUE (document_id, version_number),
        CONSTRAINT uq_document_versions_document_id_sha256 UNIQUE (document_id, sha256),
        CONSTRAINT fk_document_versions_scope_document FOREIGN KEY (tenant_id, project_id, document_id)
            REFERENCES documents (tenant_id, project_id, id),
        CONSTRAINT ck_document_versions_sha256_format CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT ck_document_versions_size_positive CHECK (size_bytes > 0),
        CONSTRAINT ck_document_versions_version_number_positive CHECK (version_number >= 1),
        CONSTRAINT ck_document_versions_media_type_known CHECK (
            media_type IN ('text/plain', 'text/markdown', 'text/html', 'application/pdf')
        ),
        CONSTRAINT ck_document_versions_text_encoding_known CHECK (text_encoding IN ('utf-8', 'utf-8-sig')),
        CONSTRAINT ck_document_versions_text_encoding_required_for_text CHECK (
            text_encoding IS NOT NULL OR media_type = 'application/pdf'
        ),
        CONSTRAINT ck_document_versions_filename_length CHECK (char_length(original_filename) BETWEEN 1 AND 255)
    )""",
    "CREATE INDEX ix_document_versions_project_sha256 ON document_versions (tenant_id, project_id, sha256)",
    """CREATE FUNCTION document_versions_reject_update() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        RAISE EXCEPTION 'document_versions rows are immutable' USING ERRCODE = 'integrity_constraint_violation';
    END $$""",
    """CREATE TRIGGER document_versions_no_update BEFORE UPDATE ON document_versions
        FOR EACH ROW EXECUTE FUNCTION document_versions_reject_update()""",
    """CREATE FUNCTION documents_reject_scope_change() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id OR NEW.project_id IS DISTINCT FROM OLD.project_id THEN
            RAISE EXCEPTION 'documents tenant_id and project_id are immutable'
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
    END $$""",
    """CREATE TRIGGER documents_scope_immutable BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION documents_reject_scope_change()""",
)


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE document_versions")
    op.execute("DROP TABLE documents")
    op.execute("DROP TABLE projects")
    op.execute("DROP TABLE tenants")
    op.execute("DROP FUNCTION document_versions_reject_update()")
    op.execute("DROP FUNCTION documents_reject_scope_change()")
