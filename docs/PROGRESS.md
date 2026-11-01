# Progress and handoff ledger

## Current state

- Current step: **P1.1** (not started). Application code: none.
- Delivered: architecture v2.0, [roadmap](plan/ROADMAP.md), [agent workflow](plan/AGENT_WORKFLOW.md), ADR-0001 to ADR-0004, 72-entry catalogue (schema v2), example configs, Claude Code rules/subagents, optional dev hooks with tests.
- Technique implementation/benchmarking: none; all 72 entries `planned`, `catalog_status: accepted`.
- Package checks: see [VALIDATION.md](VALIDATION.md).

## Next action

Run the `planner` agent on P1.1 to write the first task specs in `docs/tasks/`. Before P1.1, the maintainer should settle the license decision below.

## Open decisions (maintainer)

- License: Apache-2.0 (recommended) vs keep MIT. [ADR-0004](decisions/ADR-0004-open-source-license.md)
- Router architecture (many stage routers, one plan). [ADR-0007](decisions/ADR-0007-router-architecture.md)
- Security / Code of Conduct contact address (placeholders in SECURITY.md, CODE_OF_CONDUCT.md).

## Decided 2026-10-26

- Default paid provider: Anthropic. Local defaults: `gemma4:e4b` (chat), `bge-m3` (embeddings); re-benchmark at P2.2. See [FREE_MODELS.md](plan/FREE_MODELS.md).
- Tech Scout: enabled, Mondays 06:00 UTC, USD 2 cap per run, propose-only. [ADR-0003](decisions/ADR-0003-in-app-tech-scout.md)
- Job engine: both, PostgreSQL default + Temporal optional. [ADR-0005](decisions/ADR-0005-pluggable-job-engine.md)
- Observability: both, OpenTelemetry default + LangSmith optional. [ADR-0006](decisions/ADR-0006-observability-otel-and-langsmith.md)
- 2026-11-01: pipeline stage map `config/pipeline_stages.json` (72/72 techniques mapped to 18 stages, 8 routers), interactive view `docs/flow/index.html` built by `python scripts/build_flow.py` (CI checks it is current), router design [ROUTER_DESIGN.md](architecture/ROUTER_DESIGN.md).
- Catalogue source links audited. [SOURCE_LINK_AUDIT.md](techniques/SOURCE_LINK_AUDIT.md)

## Phases

| Phase | Title | State |
|---|---|---|
| P1 | Foundation, tooling and CI (old M0) | planned |
| P2 | Provider layer (Ollama + one paid provider) | planned |
| P3 | Ingestion, source anchors and the first vertical slice (old M1) | planned |
| P4 | Chunking curriculum (old M2) | planned |
| P5 | Retrieval and hybrid search (old M3) | planned |
| P6 | Generation, context, citations and evaluation (old M4) | planned |
| P7 | Technique plugin system v1, graph and knowledge techniques (old M7, caches from M8) | planned |
| P8 | Agents, workflows and the Tech Scout (old M5, M6, M8) | planned |
| P9 | Web UI | planned |
| P10 | Open-source release hardening (old M9) | planned |

## Steps (10 × 5)

| Step | Goal | State | Evidence |
|---|---|---|---|
| P1.1 | Reproducible toolchain | planned | - |
| P1.2 | Modular skeleton | planned | - |
| P1.3 | Local stack | planned | - |
| P1.4 | Continuous integration | planned | - |
| P1.5 | Contracts and plugin contract v0 | planned | - |
| P2.1 | Provider ports and registry | planned | - |
| P2.2 | Local default | planned | - |
| P2.3 | First paid provider | planned | - |
| P2.4 | Any provider, per pipeline | planned | - |
| P2.5 | Resilience and budgets | planned | - |
| P3.1 | Document intake | planned | - |
| P3.2 | Extraction views | planned | - |
| P3.3 | Source anchors | planned | - |
| P3.4 | Minimal index | planned | - |
| P3.5 | Thin slice | planned | - |
| P4.1 | Fixed-size family | planned | - |
| P4.2 | Text-boundary family | planned | - |
| P4.3 | Structure-aware family | planned | - |
| P4.4 | Model-based family | planned | - |
| P4.5 | Hierarchy and comparison | planned | - |
| P5.1 | Lexical retrieval | planned | - |
| P5.2 | Hybrid and fusion | planned | - |
| P5.3 | Selection and filters | planned | - |
| P5.4 | Query transforms | planned | - |
| P5.5 | Reranking and inspection | planned | - |
| P6.1 | Evaluation harness | planned | - |
| P6.2 | Context retrieval | planned | - |
| P6.3 | Packing and citation checks | planned | - |
| P6.4 | Conversation and languages | planned | - |
| P6.5 | Judges and cost reports | planned | - |
| P7.1 | Plugin system v1 | planned | - |
| P7.2 | GraphRAG | planned | - |
| P7.3 | Hierarchies and KAG | planned | - |
| P7.4 | CAG and caches | planned | - |
| P7.5 | Research and visual tracks | planned | - |
| P8.1 | Bounded workflows | planned | - |
| P8.2 | Agentic retrieval | planned | - |
| P8.3 | Tools and data | planned | - |
| P8.4 | Durable jobs and scheduler | planned | - |
| P8.5 | Tech Scout | planned | - |
| P9.1 | Web shell | planned | - |
| P9.2 | Documents and chunking | planned | - |
| P9.3 | Retrieval and settings | planned | - |
| P9.4 | Assistant and tasks | planned | - |
| P9.5 | Experiments, catalogue, Scout | planned | - |
| P10.1 | Security pass | planned | - |
| P10.2 | One-command install | planned | - |
| P10.3 | Documentation | planned | - |
| P10.4 | Release engineering | planned | - |
| P10.5 | Public v0.1.0 | planned | - |

States: `planned`, `in_progress`, `done`, `blocked`. A step is `done` only when its roadmap done-criteria were actually run; link the evidence (PR, test output, report).

## Session update template

- Date and step/task:
- Files/behavior changed:
- Exact checks run and outcomes:
- Catalogue IDs/status/evidence changes:
- Decisions and reason:
- Blockers:
- Next concrete action:

Keep entries short (3–6 lines). Archive old entries when this file grows.

## History

- 2026-10-14: v1.0 planning pack (67 catalogue entries, M0–M9 milestones).
- 2026-10-26: v1.1 added R47–R51 (CAG, KAG, prefix KV, exact and semantic response caches) and independent-consumer/source-anchor contracts.
- 2026-10-26: v2.0 restructure. Project moved to repo root; provider-agnostic layer, technique plugin system, in-app Tech Scout and open-source goal added; M0–M9 replaced by 10 phases × 5 steps (mapping in the roadmap); catalogue schema v2; planner/coder/reviewer agent workflow. No application milestone completed; no benchmark run.
