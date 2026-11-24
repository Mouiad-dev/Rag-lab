# ADR-0008: RAG Foundry is an independent product; consumers integrate through contracts

Date: 2026-11-01
Status: accepted (records the 2026-10-26 decision in REUSE_AND_INTEGRATION.md §1)

## Context
RAG Foundry (the RAG Lab) must be usable as its own product. `book2video` is one possible consumer, not its domain model. Without an explicit rule, shared code, shared databases, sibling checkout paths or consumer-specific fields (course, lesson, video, TTS, timing) could couple the two repositories and make RAG Lab depend on a consumer to build, start or test. The decision and its ownership table are in [REUSE_AND_INTEGRATION.md](../architecture/REUSE_AND_INTEGRATION.md) §1 and §2. Source anchor ownership is in [SOURCE_ANCHORS_AND_MEDIA.md](../architecture/SOURCE_ANCHORS_AND_MEDIA.md).

Affected components: workspace manifests and `uv.lock` (`pyproject.toml`, `packages/*`, `apps/*`), core entities and public contracts (`packages/contracts`, P1.5-02), and the HTTP `/api/v1` surface.

## Decision
Responsible component: the RAG Lab release and its maintainers. Decision points, per REUSE_AND_INTEGRATION.md §1 to §3:

1. RAG Foundry builds, starts and tests with no consumer checkout, config, service or credential.
2. No imports of consumer code, no shared DB/migrations, no sibling-path dependencies.
3. Integration only through versioned framework-neutral JSON contracts (`packages/contracts`, P1.5-02), HTTP `/api/v1` and portable source bundles; consumers own their adapter.
4. No consumer-specific fields (course, lesson, video, TTS, timing) in core entities or public contracts.
5. Source anchors are owned by immutable document/extraction versions, not chunk/index IDs.

An enabled live integration may depend on its selected provider being reachable for that operation. That does not make RAG Lab depend on a consumer.

## Alternatives
- **Shared monorepo or shared library with book2video.** Simplest to start, but couples release schedules, dependency locks and databases, and violates points 1 and 2.
- **Sibling-path dependency (`path = "../book2video"`).** Breaks clean checkouts and CI. Rejected.
- **In-process engine package for consumers now.** REUSE §3 keeps this as an optional future extraction, only if a concrete use case warrants it. Not chosen for P1.
- **HTTP plus contracts plus bundles (chosen).** Keeps both products independent and testable on their own, at the cost of defining and versioning contracts.

## Consequences
- Static guard now: `tests/repo/test_independence.py` fails if a workspace manifest or `uv.lock` names `book2video`, if a `path` source escapes the repository root, or if Python source under `packages/` or `apps/` imports `book2video` (static `import`/`from` statements only). It guards points 1 and 2 statically only. It does not prove that the application starts or tests without a consumer.
- REUSE §7 acceptance gates are planned future application tests. They are not executed by this ADR and are not claimed as passing.
- Consumer-specific behaviour lives in the consumer repository. Adding a consumer field to a RAG Lab entity or contract needs a new ADR or contract version.
- Contract changes are additive within v1; breaking changes need v2 and migration guidance (REUSE §6).
- Rollback: this ADR can be superseded by a new ADR. Reverting independence would require a new decision, because it is a release requirement.

## Evidence
- Source decision: [REUSE_AND_INTEGRATION.md](../architecture/REUSE_AND_INTEGRATION.md) §1 (decision date 2026-10-26), §2 (ownership), §3 (supported boundaries), §7 (planned acceptance gates, not executed).
- Source anchor ownership: [SOURCE_ANCHORS_AND_MEDIA.md](../architecture/SOURCE_ANCHORS_AND_MEDIA.md).
- Static test (stdlib only): `tests/repo/test_independence.py`. Run with `uv run pytest tests/repo/test_independence.py -q`.
- Known uncertainty: the static test cannot see a consumer dependency introduced through a transitive package outside this workspace, or a runtime network dependency. Those need the §7 gates when the application exists.
