"""Append-only runtime setting overrides.

Revision ID: 0005_runtime_setting_overrides
Revises: 0004_prompt_templates
"""

from alembic import op

revision = "0005_runtime_setting_overrides"
down_revision = "0004_prompt_templates"
branch_labels = None
depends_on = None

SETTING_KEYS = (
    "answer.max_output_tokens",
    "answer.temperature",
    "answer.max_evidence_chars",
    "answer.min_scores",
    "run.max_attempts",
    "run.attempt_timeout_s",
    "run.retry_timeouts_for_local",
    "run.deadline_s",
    "indexing.embed_batch_size",
    "indexing.c05_chunk_size",
    "indexing.c05_chunk_overlap",
)
_KEY_SQL = ", ".join("'" + key + "'" for key in SETTING_KEYS)
_MIN_SCORE_KEY = r"^[a-z0-9._-]{1,64}/[A-Za-z0-9._:-]{1,128}/[1-9][0-9]{0,4}$"


def upgrade() -> None:
    op.execute(f"""CREATE TABLE runtime_setting_overrides (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        level varchar(8) NOT NULL CHECK (level IN ('global', 'tenant', 'project')),
        tenant_id uuid NULL REFERENCES tenants(id),
        project_id uuid NULL,
        key varchar(64) NOT NULL CHECK (key IN ({_KEY_SQL})),
        value_json jsonb NULL,
        version integer NOT NULL CHECK (version >= 1),
        created_by varchar(128) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        reason varchar(500) NOT NULL,
        CHECK ((level = 'global' AND tenant_id IS NULL AND project_id IS NULL)
            OR (level = 'tenant' AND tenant_id IS NOT NULL AND project_id IS NULL)
            OR (level = 'project' AND tenant_id IS NOT NULL AND project_id IS NOT NULL)),
        CHECK (value_json IS NULL OR jsonb_typeof(value_json) IN ('number', 'boolean', 'object')),
        CHECK (value_json IS NULL OR octet_length(value_json::text) <= 4096),
        CHECK (value_json IS NULL OR jsonb_typeof(value_json) <> 'object'
            OR NOT jsonb_path_exists(value_json, '$.* ? (@.type() != "number")')),
        CHECK (value_json IS NULL OR jsonb_typeof(value_json) <> 'object'
            OR NOT jsonb_path_exists(value_json, '$.keyvalue() ? (!(@.key like_regex "{_MIN_SCORE_KEY}"))')),
        FOREIGN KEY (tenant_id, project_id) REFERENCES projects(tenant_id, id),
        UNIQUE NULLS NOT DISTINCT (level, tenant_id, project_id, key, version)
    )""")
    op.execute("""CREATE FUNCTION reject_runtime_setting_override_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        RAISE EXCEPTION 'runtime setting override rows are immutable'
            USING ERRCODE = 'integrity_constraint_violation';
    END $$""")
    op.execute("""CREATE TRIGGER runtime_setting_overrides_immutable BEFORE UPDATE OR DELETE
    ON runtime_setting_overrides FOR EACH ROW EXECUTE FUNCTION reject_runtime_setting_override_mutation()""")


def downgrade() -> None:
    op.execute("DROP TRIGGER runtime_setting_overrides_immutable ON runtime_setting_overrides")
    op.execute("DROP FUNCTION reject_runtime_setting_override_mutation()")
    op.execute("DROP TABLE runtime_setting_overrides")
