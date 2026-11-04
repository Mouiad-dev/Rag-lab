# ADR-0009: Document intake, versions and dedup

Date: 2026-11-05
Status: accepted

## Context
P3.1 adds document upload. Later steps (extraction P3.2, chunking, anchors) need stable lineage, tenant/project isolation and safe duplicate handling. Changes to schema lineage and tenant scope need an ADR (`10-architecture.md`).

## Decision
- Lineage: `Document -> DocumentVersion`. A version holds immutable bytes (sha256, size, media type, object key). There is no update path. Source anchors will belong to versions, never to chunks.
- Scope is `(tenant_id, project_id)`, taken only from authenticated context. Unknown and out-of-scope IDs raise the same `NotFoundError`.
- Dedup is atomic inside the repository, not in the caller. New-document upload dedups by sha256 within the project; new-version upload dedups within that document only.
- If several documents in a project hold the same bytes, dedup returns the earliest-created version (ties: lowest version ID).
- Re-uploading v1's bytes as a version of the same document returns v1, even when v2 exists.
- If the database fails after the object store `commit`, an unreferenced content-addressed object can remain. It is harmless; a later cleanup job removes it.
- An upload returns document and version metadata only. No ingestion task until P8.4; extraction is P3.2.

Decisions D1-D12, all decided by owner 2026-11-05:
- D1 upload is `multipart/form-data` with one file part, parsed by `python-multipart` (FastAPI's form parser), exact pin, verified in task 05. Declared media type is the part's `Content-Type`; filename is the part's filename.
- D2 SQLAlchemy 2 async + Alembic + `psycopg[binary]` 3, exact pins, verified in task 04.
- D3 dedup scopes as above (project for new documents, document for new versions).
- D4 object store root `var/objects` locally, `/data/objects` volume in compose.
- D5 formats `.txt`, `.md`, `.html`, `.pdf`; text must be UTF-8.
- D6 max upload 100 MiB, configurable, ceiling 1 GiB.
- D7 object key `t/{tenant}/p/{project}/sha256/{hh}/{hex}`.
- D8 filename is display metadata only: NFC, 1..255 code points, no separators or control characters.
- D9 IDs are UUID4 strings.
- D10 no real auth yet: dev and test map to `LOCAL_DEV_SCOPE`; production refuses with 401.
- D11 DB tests run in a CI `db-tests` job and locally against compose profile `test`, service `testdb` (tmpfs).
- D12 HTTP 201 when created, 200 on dedup.

## Alternatives
- Raw-body upload: simpler, but no filename or declared type; rejected in D1.
- Dedup in the service layer: racy under concurrency; rejected.
- Global (cross-tenant) dedup: leaks existence across tenants; rejected.

## Consequences
- New workspace member `packages/domain` (pydantic only) holds types, errors, ports and an in-memory repository with reusable checks that the SQL repository must also pass.
- Content-addressed objects are never overwritten; rollback means ignoring unreferenced objects.
- Deletion, extraction and ingestion are out of scope here.

## Evidence
Offline tests: `tests/domain` run the repository checks on the in-memory repository and on deliberately broken ones. The D1 pin is verified in task 05.

### D2 pin verification (task P3.1-04, 2026-11-05)
Sources: PyPI JSON API (release dates, `requires_python`, classifiers, `vulnerabilities`), the GitHub advisory database (`api.github.com/advisories?ecosystem=pip&affects=<pkg>`), and the installed packages (APIs imported after `uv sync --locked`). Rule: release at least two weeks old (before 2026-10-22), no advisory that affects it.

| Package | Pinned | Released | Newer releases skipped | Python | Advisories |
|---|---|---|---|---|---|
| SQLAlchemy[asyncio] | 2.1.6 | 2026-10-21 | 2.1.7, 2.1.8, 2.0.56, 2.0.57 (under 2 weeks) | >=3.11, 3.12 classifier | PyPI: none. GHSA lists only GHSA-hfg2-wf6j-x53p, GHSA-887w-45rq-vxgf, GHSA-38fc-9xqv-7f7q, fixed in 0.7.0b4, 1.3.0b3, 1.2.19; none affect 2.1.6 |
| alembic | 1.20.0 | 2026-09-11 | 1.20.1 (2026-10-22, exactly 14 days) | >=3.10, 3.12 classifier | none (PyPI, GHSA) |
| psycopg[binary] | 3.3.6 (psycopg and psycopg-binary) | 2026-09-18 | 3.3.7, 3.3.8 (under 2 weeks) | >=3.10, 3.12 classifier | none (PyPI, GHSA) |

APIs confirmed against the installed versions: `sqlalchemy.ext.asyncio.create_async_engine` and `async_sessionmaker`; `sqlalchemy.dialects.postgresql.psycopg` (driver `psycopg`, async dialect `PGDialectAsync_psycopg`, URL `postgresql+psycopg://`); `alembic.config.Config()` built in code, `alembic.command.upgrade` and `downgrade`; `alembic` 1.20.0 requires `SQLAlchemy>=2.0`; SQLAlchemy 2.1.6 `postgresql-psycopgbinary` extra accepts `psycopg[binary]>=3.0.7`. No API mismatch found. psycopg async cannot run on the Windows ProactorEventLoop; `env.py` and the tests use `SelectorEventLoop` on Windows.
Not yet evidenced: the migration and repository against a real PostgreSQL (task 06: local `testdb` and CI).

### D1 pin verification (task P3.1-05, 2026-11-05)
Sources: PyPI JSON API (release dates, `requires_python`, classifiers, `vulnerabilities`), the GitHub advisory database (`api.github.com/advisories?ecosystem=pip&affects=python-multipart`), OSV (`api.osv.dev` query for 0.0.32) and the installed packages. Rule: release at least two weeks old (before 2026-10-22), no advisory that affects it.

Pinned: `python-multipart==0.0.32`, released 2026-06-04, `requires_python >=3.11`, Python 3.12 classifier. Skipped: 0.0.33 (2026-10-31, under 2 weeks). PyPI `vulnerabilities` is empty and OSV returns none for 0.0.32.

Past advisories (all fixed before 0.0.32):

| Advisory | CVE | Severity | Affected | First fixed |
|---|---|---|---|---|
| GHSA-2jv5-9r88-3w3p | CVE-2024-24762 | high | <= 0.0.6 | 0.0.7 |
| GHSA-59g5-xgcq-4qw3 | CVE-2024-53981 | high | < 0.0.18 | 0.0.18 |
| GHSA-wp53-j4wj-2cfg | CVE-2026-24486 | high | < 0.0.22 | 0.0.22 |
| GHSA-mj87-hwqh-73pj | CVE-2026-40347 | medium | < 0.0.26 | 0.0.26 |
| GHSA-pp6c-gr5w-3c5g | CVE-2026-42561 | high | < 0.0.27 | 0.0.27 |
| GHSA-5rvq-cxj2-64vf | CVE-2026-53539 | high | < 0.0.30 | 0.0.30 |
| GHSA-6jv3-5f52-599m | CVE-2026-53538 | low | < 0.0.30 | 0.0.30 |
| GHSA-vffw-93wf-4j4q | CVE-2026-53537 | low | < 0.0.30 | 0.0.30 |
| GHSA-v9pg-7xvm-68hf | CVE-2026-53540 | low | < 0.0.31 | 0.0.31 |

Starlette 1.10.0 (locked): `Request.form(*, max_files=1000, max_fields=1000, max_part_size=1 MiB)` exists. Too many files or fields, an oversized field/header part and a malformed multipart body raise `MultiPartException`, which `Request.form` turns into `HTTPException(400)`; the upload route maps that 400 to 422. File parts are spooled to `SpooledTemporaryFile` (1 MiB in memory, then a temp file on disk); `max_part_size` does not cap file bytes, so the size limit is the ASGI middleware plus `IntakeService`. A non-multipart body yields an empty form (missing `file`, 422); urlencoded bodies with fields hit `max_fields=0` (400, mapped to 422).
