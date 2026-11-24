# Roadmap: 10 phases × 5 steps

Status: planned (2026-10-26). Progress is tracked in [PROGRESS.md](../PROGRESS.md). Each step is split into small task specs by the `planner` agent ([AGENT_WORKFLOW.md](AGENT_WORKFLOW.md)). Every step is planned by `planner` (Opus) and reviewed by `reviewer` (Opus); the last column names the suggested implementer: **Sonnet** = `coder`, **Haiku** = `coder-light`.

Order rationale: tooling and the provider layer come first because every later step calls models through it. Phase 3 ends with a **thin vertical slice** (upload → chunk → embed → retrieve → cited answer on local Ollama), so the product is usable before the curriculum widens. The technique plugin contract starts in P1.5 so chunkers and retrievers are plugins from day one; P7.1 extends it to third-party packages. Catalogue entries map to steps through `roadmap_step` in `config/techniques.json`.

Router work (ADR-0007) is spread across existing steps; see [ROUTER_DESIGN.md §6](../architecture/ROUTER_DESIGN.md). Interactive pipeline view: [docs/flow/index.html](../flow/index.html).

## Cross-cutting requirement: sandbox per run

Every agent run (task or conversation turn) executes inside its own isolated, ephemeral environment. A run must not be able to affect the host, other runs, or any data or secret it was not explicitly given. When the run ends, the environment is destroyed and nothing persists except explicit outputs (PR, logs, artifacts).

This extends the code-execution sandbox in [PROJECT_ARCH.md §12](../architecture/PROJECT_ARCH.md) from "generated code" to the whole run. Delivered by: P8.3 (per-run sandbox lifecycle on the sandbox port), P8.4 (job engine creates and destroys the sandbox for each run, including on cancel, failure and worker crash), P10.1 (isolation tests gate the release). Earlier steps must not add anything that assumes runs share a writable workspace or ambient credentials.

## Phase 1: Foundation, tooling and CI (old M0)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P1.1 | Reproducible toolchain | `pyproject.toml` (uv, Python 3.12), lockfile, ruff/pytest config, task runner | `uv sync` and `uv run pytest` pass on a clean clone on Linux and Windows | Sonnet |
| P1.2 | Modular skeleton | `apps/api` FastAPI `/health`, `apps/worker` entrypoint, `packages/core` settings with strict Pydantic validation | API and worker start; unknown config keys are rejected by a test | Sonnet |
| P1.3 | Local stack | `compose.yaml`: Postgres+pgvector, api, worker, optional `ollama` profile; `.env.example` with placeholders only | `docker compose up` gives healthy services with zero API keys | Sonnet |
| P1.4 | Continuous integration | GitHub Actions: lint, tests, `verify_package.py`, hook tests, catalogue validation | CI is green on a PR; a broken catalogue entry fails CI | Haiku |
| P1.5 | Contracts and plugin contract v0 | Source/public contract stubs, independence ADR, tiny Arabic/English fixtures, technique registry loader and plugin protocol v0 | Registry loads all 72 entries; an unknown technique ID fails with an explicit error | Sonnet |

## Phase 2: Provider layer (Ollama + one paid provider)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P2.1 | Provider ports and registry | `packages/providers`: chat, embedding, reranker ports; capability flags; registry; fake provider | Contract tests pass against the fake provider with no network | Sonnet |
| P2.2 | Local default | LiteLLM-backed adapter wired to Ollama for chat and embeddings | Opt-in live test chats and embeds via local Ollama with no keys | Sonnet |
| P2.3 | First paid provider | Anthropic profile through the same adapter, env-only keys, usage and cost ledger | Opt-in live test records tokens and cost; keys never appear in logs (test) | Sonnet |
| P2.4 | Any provider, per pipeline | Provider config file + env; OpenAI-compatible endpoints (LM Studio, vLLM, Kimi/Moonshot); per-stage model selection; capability validation | Pipeline needing `tool_calling` is rejected for a model without it; two stages use different providers in one test | Sonnet |
| P2.5 | Resilience and budgets | Adapter-boundary timeouts/retries, per-run budget caps, provider contract test suite | Budget overrun stops the run with a typed error; same contract suite passes for fake and (opt-in) live providers | Sonnet |

## Phase 3: Ingestion, source anchors and the first vertical slice (old M1)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P3.1 | Document intake | Upload API, size/type limits, content hash, immutable document versions, local object store | Re-upload of same bytes dedups; changed bytes create a new version | Sonnet |
| P3.2 | Extraction views | Extractors for txt/Markdown/HTML/PDF text layer; Arabic/English normalization with offset maps | Normalized text maps back to original code-point offsets in tests | Sonnet |
| P3.3 | Source anchors | Anchor model (code-point offsets, page geometry with precision) and resolver API | Anchor resolves to the exact quote after a rechunk | Sonnet |
| P3.4 | Minimal index | C05 recursive chunker plugin, embeddings via providers, pgvector dense retrieval (R02), no-retrieval control (R01) | Known fact retrieved in top-k on the fixture; chunk spans inspectable | Sonnet |
| P3.5 | Thin slice | Grounded answer with citations and abstention via CLI and `POST /runs`; 10-question bilingual fixture | Cited answer on Ollama; unanswerable question abstains; citations resolve to anchors | Sonnet |

## Phase 4: Chunking curriculum (old M2)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P4.1 | Fixed-size family | C01–C04 chunker plugins on the P3.4 interface | Offset and overlap tests pass; lesson pages drafted | Haiku |
| P4.2 | Text-boundary family | C06, C07, C21 (sentence, paragraph, multilingual boundaries) | Arabic punctuation and mixed-script cases tested | Sonnet |
| P4.3 | Structure-aware family | C08–C14 (Markdown, HTML, JSON, layout, table, code, AST) | Tables keep headers and lineage; oversized blocks flagged, not dropped | Sonnet |
| P4.4 | Model-based family | C15, C16, C17, C19 using provider ports and caching keyed by source/model/prompt | LLM boundaries validated against source offsets; no invented text | Sonnet |
| P4.5 | Hierarchy and comparison | C18 parent-child, source-span chunk comparison report, portable source bundle round trip | Comparison report by source-span relevance; bundle export/import preserves anchors | Sonnet |

## Phase 5: Retrieval and hybrid search (old M3)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P5.1 | Lexical retrieval | R03 BM25, R04 PostgreSQL FTS, R05 learned sparse | BM25 and FTS labelled separately; Arabic exact-identifier tests | Sonnet |
| P5.2 | Hybrid and fusion | R06 hybrid, R07 RRF, R08 weighted fusion | Score components visible per candidate; paired comparison vs dense | Sonnet |
| P5.3 | Selection and filters | R09 MMR, R10 dedup, R11 metadata filters, R12 self-query | Filters never widen authorized scope (negative test) | Sonnet |
| P5.4 | Query transforms | R13–R18 (rewrite, multi-query, decomposition, step-back, HyDE, RAG fusion) | Original question kept in evidence; every extra model call metered | Sonnet |
| P5.5 | Reranking and inspection | R19 cross-encoder, R20 LLM reranker via reranker port; retrieval inspection API | Rank changes visible; reranker cannot add candidates | Sonnet |

## Phase 6: Generation, context, citations and evaluation (old M4)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P6.1 | Evaluation harness | Dataset splits, retrieval metrics, per-case reports, frozen experiment runner | Same config reruns reproduce metrics; leakage check runs | Sonnet |
| P6.2 | Context retrieval | R21–R25 (sentence windows, parent docs, multi-representation, contextual enrichment, auto-merging) | Synthetic text marked; provenance kept after expansion | Sonnet |
| P6.3 | Packing and citation checks | R26 compression, R27 ordering/packing; deterministic claim-to-source citation validator | Compression keeps dates/negation/units in tests; invalid citations flagged | Sonnet |
| P6.4 | Conversation and languages | R28 conversational RAG, R42 cross-language cohort | Per-language-pair report; history-aware tests | Sonnet |
| P6.5 | Judges and cost reports | Versioned LLM-judge rubric with human-calibrated subset; cost/latency report per run | Judge agreement reported; TTFT and total latency separated | Sonnet |

## Phase 7: Technique plugin system v1, graph and knowledge techniques (old M7, caches from M8)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P7.1 | Plugin system v1 | Entry-point discovery for third-party plugins, `new-technique` scaffold, plugin contract tests, catalogue schema v2 validation | A sample external plugin installs and runs with no core change | Sonnet |
| P7.2 | GraphRAG | R38 entity-relation GraphRAG, R39 community/global summaries | Relationship question answered that neighbor expansion fails; summary claims traced | Sonnet |
| P7.3 | Hierarchies and KAG | R37 RAPTOR, R48 KAG (schema + logical-form retrieval) | Leaf lineage preserved; KAG fidelity note vs OpenSPG | Sonnet |
| P7.4 | CAG and caches | R47 CAG, R49 prefix KV caching, R50 exact and R51 semantic response caches | Cold/warm measured; false-hit and cross-tenant cache tests pass | Sonnet |
| P7.5 | Research and visual tracks | C20 late chunking, R31 Self-RAG, R40, R41, R45, R46 as optional plugins with hardware notes | Each is implemented or recorded `blocked` with exact requirements; no simulations labelled real | Sonnet |

## Phase 8: Agents, workflows and the Tech Scout (old M5, M6, M8)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P8.1 | Bounded workflows | LangGraph runtime with budgets/checkpoints; R29, R30, R32 | Loops stop on budget and no-progress; resume from checkpoint | Sonnet |
| P8.2 | Agentic retrieval | R33 agentic, R34 multi-hop, R44 opt-in web fallback | Tool decisions traced; web evidence labelled; no private-query leakage test | Sonnet |
| P8.3 | Tools and data | R43 constrained SQL, R35 multi-source routing, MCP adapter, sandbox port with one ephemeral sandbox per run (only explicitly staged inputs and brokered credentials), R36 multi-agent | Readonly SQL denials tested; a run cannot read another run's files, host paths or unstaged secrets (tests); sandbox destroyed after the run with only declared outputs kept; multi-agent compared to single | Sonnet |
| P8.4 | Durable jobs and scheduler | `JobEngine` interface with Postgres engine (default) and Temporal adapter (ADR-0005); persisted jobs, retries, cancellation, cron scheduler; per-run sandbox create/destroy in the job lifecycle | Shared contract tests pass on both engines; cancel/retry/recovery; scheduled job survives worker restart; no sandbox left behind after success, failure, cancel or worker crash (test) | Sonnet |
| P8.5 | Tech Scout | `packages/scout`: fetchers, dedup, triage, cited report, `proposed` entries, approve/reject API, cost cap | Dry run produces a cited report under cap; duplicates skipped; nothing written outside the app DB | Sonnet |

## Phase 9: Web UI

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P9.1 | Web shell | React + TypeScript + Vite, i18n (Arabic RTL, English LTR), typed API client from OpenAPI | Language switch flips direction; client generated in CI | Sonnet |
| P9.2 | Documents and chunking | Document workspace, source inspector, chunking playground | Anchors highlight exact spans; chunk strategies compared side by side | Sonnet |
| P9.3 | Retrieval and settings | Retrieval playground, pipeline builder, provider/model settings (key status only, never values) | Incompatible pipeline shows validation error before run | Sonnet |
| P9.4 | Assistant and tasks | Streaming chat with citations (SSE), task view with approvals and cancel | SSE replay after reconnect; citation click opens source | Sonnet |
| P9.5 | Experiments, catalogue, Scout | Experiment dashboard, technique catalogue browser, Scout review/approve screen | Approving a proposal changes its status and records the reviewer | Sonnet |

## Phase 10: Open-source release hardening (old M9)

| Step | Goal | Deliverable | Done when | Implementer |
|---|---|---|---|---|
| P10.1 | Security pass | Tenant isolation tests, per-run sandbox isolation tests (host, cross-run, secrets, persistence), prompt-injection fixtures, secret scanning, dependency audit | No high findings open; security tests fail the build independently | Sonnet |
| P10.2 | One-command install | Compose quickstart with Ollama, zero paid keys; clean-machine smoke test in CI | New user reaches a cited answer following README only | Sonnet |
| P10.3 | Documentation | User guide, provider guide, plugin authoring guide, technique lesson index | Docs link check passes; a contributor adds a sample plugin from the guide | Sonnet |
| P10.4 | Release engineering | Versioning, CHANGELOG, release workflow, container images, SBOM, license applied per ADR-0004 | Tagged pre-release builds images reproducibly | Haiku |
| P10.5 | Public v0.1.0 | Representative evaluation report, consumer compatibility gates, launch checklist | Held-out results and limitations published; v0.1.0 tagged | Sonnet |

## Milestone mapping (old → new)

| Old | New | Old | New |
|---|---|---|---|
| M0 | P1, P2 | M5 | P8.1–P8.2 |
| M1 | P3 | M6 | P8.3 |
| M2 | P4 | M7 | P7 |
| M3 | P5 | M8 | P8.4, P7.4 (R49–R51) |
| M4 | P6 | M9 | P10 |

New work without an old milestone: P2 (providers), P7.1 (plugin system v1), P8.5 (Tech Scout), P9 (UI as its own phase), P10.1–P10.4 (open-source release).
