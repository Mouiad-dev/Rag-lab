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
Offline tests: `tests/domain` run the repository checks on the in-memory repository and on deliberately broken ones. Library pins for D1 and D2 are verified in tasks 04 and 05, not here.
