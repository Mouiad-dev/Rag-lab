# Reusable RAG Lab: independent consumers

Decision date: 2026-10-26. Status: accepted architecture; interfaces below are proposed, not implemented.

## 1. Independence is a release requirement

RAG Foundry (the RAG Lab) is an independent repository and product. `book2video` is one possible consumer, not its domain model. Neither repository imports the other's source tree, shares ORM entities/migrations, requires sibling folders, or requires the other service at startup. Independent CI, dependency locks, database ownership, credentials, deployment and release schedules are mandatory.

An enabled live integration necessarily depends on the selected provider being reachable for that operation. This is compatible with independent applications; it is not a promise of zero runtime dependencies. Without the integration, book2video must use its own basic document provider, an imported source bundle, or another provider. RAG Lab works fully without book2video. Do not silently substitute a provider if an explicitly selected provider fails.

## 2. Ownership

| Concern | Owner |
|---|---|
| Immutable document versions, extraction views, source anchors, source structure | RAG Lab when it ingests the source; exchangeable through a portable contract |
| Chunking, embeddings, retrieval, knowledge experiments and source-grounded answers | RAG Lab |
| Courses, lessons, scripts, voices, rendering, player layout and publish policy | Consumer application |
| Script statements to sources, audio alignment, final video time ranges | Consumer application |
| Public data contract and RAG Lab HTTP client | RAG Lab release; usable without its internal ORM or Lang libraries |
| Consumer-facing KnowledgeProvider port and optional RagLab adapter | Consumer repository |

The section tree is source structure. A lesson plan is a consumer decision. Retrieval chunks are search units; they are not video boundaries. One section per video is a book2video policy, not a RAG Lab rule. Long sections may span many chunks; a lesson can link many source spans.

## 3. Supported boundaries

**Default: HTTP API + JSON contracts.** Publish `/api/v1` OpenAPI, a lightweight optional HTTP SDK, and examples. The consumer can call HTTP directly; neither LangChain nor SQLAlchemy is required in its process. A provider interface owned by the consumer prevents business code from depending on RAG Lab's route names or response internals.

**Portable source bundle.** Export immutable source metadata, section hierarchy, source anchors, extraction text/layout and permitted source assets. A consumer validates and imports the bundle into its own storage, then can display sources and resolve already-built links without a live RAG Lab. This does not supply a full offline search engine, nor automatically solve synchronizing later changes.

**Optional future in-process engine.** A separately versioned installable core may be extracted if a concrete use case warrants it. It must have explicit adapters and optional dependency extras. Do not promise it exists, or make it the prerequisite for the HTTP/bundle integration. Never copy source between repos as the integration mechanism.

The lab UI is a client of the same public capabilities as external consumers. Domain/services remain framework-light; server-side adapters contain Lang integrations and persistence. Heavy KAG, visual models and local inference dependencies are opt-in profiles, not mandatory SDK dependencies.

## 4. Proposed integration API

All routes are under `/api/v1`, scoped by authenticated project grants. IDs are opaque and qualified by provider namespace in exported data. These supplement the existing upload, retrieval, run and job APIs.

| Route | Contract |
|---|---|
| `GET /capabilities` | Contract versions, supported formats/strategies, features and limits for this deployment; distinguishes configured availability from planned curriculum |
| `GET /documents/{id}/versions/{version}/structure` | Paginated immutable section tree and source block references; extraction view is explicit |
| `POST /projects/{id}/anchors:resolve` | Bounded batch of anchor references; authorized exact/ambiguous/unavailable/obsolete results with permitted text and geometry |
| `POST /projects/{id}/source-bundles` | Async export of explicit document/extraction versions; idempotency key; returns job ID |
| `GET /source-bundles/{id}/manifest` | Authorized versioned manifest, assets/hashes, inclusion policy and export snapshot |

Retrieval results include portable source references, not only ephemeral chunk IDs. A response includes `contract_version`, request/run ID and pinned pipeline/index/source versions. Optional capabilities do not silently alter a requested workflow. Unknown major versions and missing required capabilities return explicit incompatibility errors.

A service credential maps to project scopes on the server; a client-provided tenant field is not authority. Resolve, search, bundle export and asset download repeat authorization. Avoid arbitrary callback URLs and SSRF; polling existing job endpoints is sufficient for the first integration. Future webhooks must be signed, replay-protected and deduplicated.

## 5. Portable bundle v1 design

Manifest fields: `contract_version`, `bundle_id`, `provider_namespace`, creation timestamp, document-version references, extraction-view references, structure/anchor files, source-asset inclusion policy, relative asset paths, byte sizes, MIME types and SHA-256 digests. Include a complete inventory of intentionally omitted assets. References must be resolvable either within the bundle or as explicitly unavailable remote references.

Proposed members: `manifest.json`, `documents.json`, `structure.jsonl`, `anchors.jsonl`, extraction text/layout files, optional authorized originals/page renders. UTF-8; no executable content or credentials. No ORM rows, graph checkpoint internals, KV caches, permanent signed URLs or consumer video records. Hashes detect accidental content mismatch; they do not establish who authored an untrusted manifest. Authenticate transfer or verify a signature when crossing trust boundaries.

Importer requirements: supported version; bounded size/file count; validate hashes and IDs; reject traversal, absolute asset paths, symlinks, unsafe content and dangling required references; atomically activate only a valid import. Preserve foreign IDs under their provider namespace while assigning local scope. Ingestion is idempotent by bundle ID and manifest digest; same ID with different bytes is a conflict. An export is a snapshot, never a live authorization bypass.

If original bytes cannot be exported, the consumer must supply an authorized identical asset (verified hash) or display an explicit unavailable-source state. Independent copies cannot be remotely erased with certainty: recipients must implement retention/deletion propagation for managed copies; exported offline bytes cannot be reliably recalled. Explain this in product behavior without promising instantaneous cross-system revocation.

## 6. Compatibility and failure handling

- Additive optional fields within v1; breaking meaning/required-field changes need v2 and migration guidance. Publish contract fixtures and a compatibility matrix.
- SDK releases do not import server internals. Consumer adapters pin a supported contract range and tolerate documented unknown optional fields.
- Timed-out ingestion/export requests use a scoped idempotency key plus payload digest; retries return the same job. Same key with a different digest is rejected.
- HTTP timeouts, retry-after/backoff, cancellation and retry budgets are explicit. Reconcile status after uncertain completion; do not create duplicate jobs blindly.
- Missing permissions are distinct from missing data internally; externally avoid existence leaks. Version mismatches and unsupported geometry are visible, not guessed.
- An outage can leave already exported player/source assets usable; new live retrieval requires recovery or an explicitly chosen alternate provider.

## 7. Architecture acceptance gates (future application tests)

1. Start/test RAG Lab with no book2video checkout, config or service.
2. Run a consumer contract fixture against both an HTTP adapter and a local bundle provider using the same domain interface.
3. Build/install the HTTP SDK in a clean environment with no RAG server, Lang packages or database driver.
4. Export, validate and import a source bundle, disconnect the RAG service, then resolve source spans locally.
5. Rechunk/reindex a fixed extraction view and prove existing source links still resolve.
6. Exercise inaccessible source, deleted source, incompatible contract, interrupted export, duplicate retry and malformed bundle.

These are planned acceptance tests, not executed tests of an existing API. Use a generic sample consumer in this repo; building book2video itself is outside this update.

## 8. Delivery order

Roadmap phases ([ROADMAP.md](../plan/ROADMAP.md)): P1: ownership ADR, framework-light source contracts, authored fixtures. P3: source anchors and section structure in the baseline, read/resolve APIs. P4: portable bundle round trip and a minimal generic consumer fixture. P5–P6: public retrieval and optional HTTP SDK. P7: CAG/KAG profiles behind the same evidence contract. P8 and P10: compatibility/recovery/deletion and standalone release gates. Do not defer provenance until the video app is built.

See [SOURCE_ANCHORS_AND_MEDIA.md](SOURCE_ANCHORS_AND_MEDIA.md) and [CACHE_AND_KNOWLEDGE_LABS.md](../techniques/CACHE_AND_KNOWLEDGE_LABS.md).
