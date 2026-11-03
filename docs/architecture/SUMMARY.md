# Architecture summary (read this, not PROJECT_ARCH.md)

Short version of [PROJECT_ARCH.md](PROJECT_ARCH.md) for agents and new contributors. Open the full document only at a section a task spec cites (§N).

## What it is
- Open-source lab and document/data workspace for retrieval and knowledge techniques: RAG, GraphRAG/knowledge graphs, CAG/KAG, caching, agentic workflows, and future techniques.
- Users ingest documents, compare techniques on their own data, inspect every intermediate output, and ask cited questions.
- Bilingual Arabic (RTL) / English (LTR) product. Repo docs and code in English.
- Provider-agnostic: any chat/embedding/reranker provider; zero-key local install with Ollama.
- Status: planning. No application code yet. Plan: [ROADMAP.md](../plan/ROADMAP.md) (10 phases × 5 steps).

## Shape (modular monolith + worker)
```text
apps/web (React/TS/Vite) --HTTP/SSE--> apps/api (FastAPI) --> PostgreSQL + pgvector, object store
                                            |
                                            +--> packages/* (domain, ingestion, chunking, retrieval, context, evaluation)
                                            +--> packages/techniques  (plugin registry, catalogue)
                                            +--> packages/providers   (chat / embeddings / rerank ports)
                                            +--> apps/worker: ingestion, experiments, long tasks, scheduler, Tech Scout
```
Start as one deployable API plus one worker. No microservices until a measured need.

## Key decisions
| Topic | Decision | Ref |
|---|---|---|
| Models | Own ports in `packages/providers`; LiteLLM-backed default adapter; native adapters allowed; capability flags; per-stage model profiles; secrets only from env | [ADR-0001](../decisions/ADR-0001-provider-agnostic-layer.md) |
| Techniques | Plugins keyed by catalogue ID, typed config, explicit `unavailable` errors, entry-point discovery for third parties | [ADR-0002](../decisions/ADR-0002-technique-plugin-system.md) |
| Tech Scout | Weekly worker job, default Claude Opus 5.5, cost cap, dedup, cited report, `proposed` entries only; human approves | [ADR-0003](../decisions/ADR-0003-in-app-tech-scout.md) |
| License | Proposed Apache-2.0 vs current MIT; maintainer decides | [ADR-0004](../decisions/ADR-0004-open-source-license.md) |
| Backend | FastAPI, Pydantic, SQLAlchemy 2 async + Alembic, PostgreSQL + pgvector | §3 |
| Workflows | LangGraph for bounded AI workflows; Deep Agents only where file/tool planning helps | §11 |
| Jobs | Pluggable: PostgreSQL queue + scheduler (default) or Temporal (ADR-0005) | §3, §11 |
| Observability | OpenTelemetry (default) + LangSmith (optional), both at once allowed (ADR-0006) | §3 |
| Routing | 8 stage routers (3 ingest, 5 query) behind one `Router` contract; orchestrator emits one replayable plan; L0→L4 policies gated by held-out regret (ADR-0007, proposed) | [ROUTER_DESIGN.md](ROUTER_DESIGN.md) |
| Frontend | React + TypeScript + Vite + TanStack Query; SSE for live events | §3 |
| Isolation | Generated code only in a sandbox service, never in the API process | §12 |
| Independence | RAG Foundry builds, starts and tests without consumers; integration only via versioned contracts, `/api/v1` and source bundles | [ADR-0008](../decisions/ADR-0008-independent-product-and-consumers.md) |

## Core contracts
- **Provenance.** Immutable document versions → extraction views → **source anchors** (Unicode code-point offsets, page geometry with stated precision). Anchors survive rechunking; chunk IDs do not. §6, [SOURCE_ANCHORS_AND_MEDIA.md](SOURCE_ANCHORS_AND_MEDIA.md).
- **Pipelines** are immutable resolved configs: corpus snapshot, chunking, embedding profile, index version, retrieval branches, fusion, reranker, transforms, context, generator, workflow, budgets, privacy. Unknown fields or strategies are rejected. §9.
- **Rebuild rule.** Changing extraction/chunking/embedding model or preprocessing creates a new index; equal dimension is not compatibility. §9.
- **Tenant scope** comes from auth and is rechecked on retrieval, resume, download, cache hit and external action. §6, §12.
- **Answers** carry claim-to-source citations or abstain. Generated text (HyDE, summaries, prefixes) is never evidence. §8.
- **Public contracts** are versioned HTTP/JSON and portable source bundles; consumers (for example book2video) are optional and independent. [REUSE_AND_INTEGRATION.md](REUSE_AND_INTEGRATION.md).

## Technique catalogue
- 72 entries in `config/techniques.json` (schema v2) mirrored in [TECHNIQUE_CATALOG.md](../techniques/TECHNIQUE_CATALOG.md): C01–C21 chunking/representation, R01–R51 retrieval, query, reranking, context, workflows, graph, CAG/KAG, caches, multimodal, evaluation cohorts.
- Fields: `category`, `status` (implementation), `catalog_status` (accepted/proposed/rejected/deprecated), `primary_source`, `date_added`, `phase`, `roadmap_step`.
- Every entry stays in scope; losing a benchmark is a documented result. `validated` needs implementation, fidelity note, demo, tests, comparison and failure analysis. `production_eligible` is separate.
- Naming honesty: FTS is not BM25; neighbor expansion is not GraphRAG; prompt self-checks are not Self-RAG; per-chunk embedding is not late chunking; response caching is not CAG.

## Paths
- Fast Q&A: auth → scope + index version → validate config → retrieve → (rerank/compress) → pack → generate → check citations → stream + persist.
- Ingestion (worker): authorize → limits + hash → immutable version → extract → normalize with offset maps → chunk → represent → embed/index → validate → atomically activate.
- Long task (worker): idempotent job → route doc/SQL/file work → approvals → sandbox → artifacts → cleanup. Cancellation propagates.

## Evaluation (§13)
- Freeze corpus/dataset/model/config; change one factor at a time; dev/validation/held-out splits; leakage checks.
- Per-case outputs, not only averages. Cohorts: Arabic, English, mixed, tables, OCR, unanswerable.
- Retrieval: HitRate@k ≠ Recall@k with multiple relevant items; MRR, nDCG. Answers: claim support, citation accuracy, abstention.
- Count every auxiliary model call in cost and latency. Separate TTFT from total, cold from warm cache.
- LLM judges are versioned and human-calibrated. Security failures fail a release on their own.

## Security (§12)
- Secrets only in env/secret stores; never in configs, prompts, traces or the browser bundle.
- Retrieved documents, tool results, web pages and Scout fetches are untrusted data.
- SQL tool: separate readonly datasource, allowed views/functions, timeouts, result limits.
- MCP is not authorization: allowlist tools, validate schemas, enforce scope server-side.
- Approvals bind action + canonical argument hash + actor + expiry.
- Development hooks (`.claude/hooks`) are partial dev guardrails, not the app's sandbox.

## Where things go
| Path | Purpose |
|---|---|
| `apps/api`, `apps/worker`, `apps/web` | Edge API, background jobs + scheduler + Scout, UI |
| `packages/providers` | Model ports, registry, adapters, usage/cost |
| `packages/techniques` | Plugin protocol, registry, catalogue loader |
| `packages/scout` | Tech Scout fetchers, dedup, triage, reports |
| other `packages/*` | domain, ingestion, chunking, indexing, retrieval, context, agents, evaluation, storage, sandbox, contracts, provenance, export, cache (created when their phase starts) |
| `config/` | Catalogue, example experiment/provider/scout configs |
| `docs/plan/` | Roadmap, agent workflow |
| `docs/decisions/` | ADRs |
| `docs/tasks/` | Task specs written by the planner |
