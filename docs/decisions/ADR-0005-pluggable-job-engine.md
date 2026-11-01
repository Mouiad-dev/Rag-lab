# ADR-0005: Pluggable job engine (PostgreSQL queue and Temporal)

Date: 2026-10-26
Status: accepted

## Context
Ingestion, batch experiments, long agent tasks and the weekly Tech Scout need durable background jobs. A PostgreSQL-backed queue needs no extra service, which suits the "clone and run" open-source goal. Temporal gives stronger durability, replay and visibility for large or long-running deployments, but adds a server to run.

## Decision
Support both behind one `JobEngine` interface in `apps/worker`:
- **`postgres` (default):** jobs, retries, cancellation and cron schedules stored in PostgreSQL. Works with `docker compose up` and no extra services.
- **`temporal` (optional):** the same job definitions run as Temporal workflows/activities. Enabled by config (`JOB_ENGINE=temporal`) plus an optional compose profile.

Job code is written once as idempotent steps; each engine adapter maps steps to its own primitives. Application run IDs are stored for both, with Temporal workflow/activity IDs recorded when that engine is used.

## Consequences
- P8.4 delivers the interface and the Postgres engine first, then the Temporal adapter, with one shared contract test suite that both must pass.
- Engine-specific features (for example Temporal signals) cannot leak into job code; they stay in the adapter.
- CI runs the contract tests on Postgres always and on Temporal in a separate (slower) job.
