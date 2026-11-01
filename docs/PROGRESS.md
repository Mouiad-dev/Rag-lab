# Progress and handoff ledger

## Current state

- Current step: **P1.4** (in progress; 01–03 done, 04 push + PR pending). P1.2 and P1.3 done (Windows evidence); P1.1 waits on a Linux clean-clone run. P1.1 in progress (Linux clean-clone pending); P1.2 done (Windows evidence). Application code: `packages/core` (strict settings), `apps/api` (FastAPI `/health`), `apps/worker` (entrypoint, `--once`); no product features.
- Delivered: architecture v2.0, [roadmap](plan/ROADMAP.md), [agent workflow](plan/AGENT_WORKFLOW.md), ADR-0001 to ADR-0004, 72-entry catalogue (schema v2), example configs, Claude Code rules/subagents, optional dev hooks with tests.
- Technique implementation/benchmarking: none; all 72 entries `planned`, `catalog_status: accepted`.
- Package checks: see [VALIDATION.md](VALIDATION.md).

## Next action

P1.4-04: push `p1/toolchain-and-skeleton`, open a PR and record the first CI run (needs user OK). A green ubuntu `check` leg closes P1.1 (Linux clean-clone run). Then plan P1.5. The license decision below is still open.

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
- 2026-11-01: sandbox per run added to the roadmap as a cross-cutting requirement (every agent run in its own ephemeral, isolated environment; only explicit outputs persist). Delivered by P8.3, P8.4, P10.1. Roadmap only; nothing implemented.
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
| P1.1 | Reproducible toolchain | in_progress | P1.1-01 and P1.1-02 Windows runs (see History 2026-11-01); Linux clean-clone pending |
| P1.2 | Modular skeleton | done | P1.2-01 and P1.2-02 (see History 2026-11-01); `tasks.py check` 35 passed on Windows 11 |
| P1.3 | Local stack | done | P1.3-01..04 (see History 2026-11-01); `docker compose up -d --build --wait` all healthy on Windows 11 with zero API keys; Ollama profile not run |
| P1.4 | Continuous integration | in_progress | P1.4-01..03 reviewed (see History 2026-11-01); P1.4-04 push + PR needs user OK; CI not run yet |
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

- 2026-11-01: P1.1-01 uv workspace (coder). `uv sync --locked` exit 0; `uv run pytest` 22 passed (21 hooks + 1 core); `uv run ruff check .` and `ruff format --check .` pass; `verify_package.py` PASS; `unittest discover -s tests/hooks` OK (21); `build_flow.py --check` PASS. Resolved pytest 9.2.0, ruff 0.17.3.
- 2026-11-01: P1.1-02 task runner (coder-light). Added `scripts/tasks.py` (`repo`, `lint`, `format`, `test`, `check`) and CONTRIBUTING "Development setup". `uv run python scripts/tasks.py check` exit 0; `uv run python scripts/tasks.py lint` exit 0; `verify_package.py` PASS. Windows 11 only; P1.1 stays `in_progress` until a Linux clean-clone run exists.
- 2026-11-01: P1.2-01 core settings (coder; reviewer APPROVE round 1). `packages/core` `Settings`/`load_settings()` (pydantic-settings, `RAG_FOUNDRY_` prefix, frozen, unknown keys rejected). `uv sync --locked` exit 0; `uv run pytest tests/core -q` 9 passed; `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0 (30 passed: 1 import + 8 settings + 21 hooks). Review notes carried to P1.2-02: services must call `load_settings()`, not `Settings()`; future secret settings use `SecretStr`.
- 2026-11-01: P1.4-01..03 (coder-light, coder ×2; reviewer APPROVE on all three, round 1). 01: ruff now lints `scripts/` (formatting only; `verify_package.py` output and `build_flow.py --check` unchanged), hook tests run once via pytest in `tasks.py check`. 02: `tests/repo/test_catalogue_validation.py`, 10 broken-catalogue variants on a temp copy each exit 1 with the expected message, unmodified copy passes; real catalogue untouched. 03: `.github/workflows/ci.yml` with `check` (ubuntu + windows: `uv sync --locked`, `tasks.py check`, compose config on ubuntu) and `image` (ubuntu: compose build/up --wait, /health, down -v; generated password masked); actions pinned by SHA (checkout v7.0.1 3d3c42e, setup-uv v10.3.3 68559e9, uv 0.13.6), `contents: read`. Local: `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0, 75 passed; `verify_package.py` PASS. CI has not run on GitHub yet.
- 2026-11-01: P1.3-04 stack bring-up (coder; reviewer CHANGES_REQUESTED round 1 for bookkeeping only, fixed directly). Windows 11, Docker Engine 29.8.2. User approved core-stack image downloads only. `.env` created from `.env.example` with a random local password (git-ignored, not printed). `docker compose up -d --build --wait` exit 0; `docker compose ps`: api, db, worker healthy; `/health` on 127.0.0.1:8000 → `{"status":"ok","version":"0.0.0"}`; `docker compose port api 8000` → `127.0.0.1:8000`; pgvector `default_version` 0.8.7; `docker compose down` (volume kept). Reviewer re-ran up/ps/health/down with the same results. Images: rag-foundry-app:local 203MB, pgvector/pgvector:0.8.7-pg18-trixie 652MB. pg18 `VOLUME /var/lib/postgresql` confirmed in the docker-library postgres docs. `tasks.py check` exit 0, 60 passed. CONTRIBUTING "Local stack (Docker)" added. Ollama profile not run (not approved).
- 2026-11-01: P1.3-01..03 (coder ×3; reviewer: 01 APPROVE, 02 CHANGES_REQUESTED fixed directly (Dockerfile secret check now joins line continuations, negative test added), 03 APPROVE). Worker `--heartbeat-file`/`--check-heartbeat` (future mtime counts as stale); non-root multi-stage `Dockerfile` (`python:3.12.16-slim-trixie`, `ghcr.io/astral-sh/uv:0.13.6`); `compose.yaml` (`pgvector/pgvector:0.8.7-pg18-trixie`, api, worker, `ollama/ollama:0.40.13` under profile `ollama`; ports on 127.0.0.1 only, db unpublished, read-only + cap_drop ALL for api/worker); `.env.example` placeholders only, `POSTGRES_PASSWORD` required. Tags checked via registry metadata, no pulls. `docker compose --env-file .env.example config --quiet` exit 0; `uv run pytest tests/worker tests/stack/test_dockerfile.py -q` 16 passed; `tasks.py check` exit 0, 60 passed. Not built or started: P1.3-04 pending user OK for ~250–350 MB of images. Open for P1.3-04: confirm pg18 data dir against postgres image docs; consider no-new-privileges/cap_drop on ollama.
- 2026-11-01: P1.2-02 API and worker (coder; reviewer CHANGES_REQUESTED round 1, fixed directly: lower bounds `fastapi>=0.162.0`, `uvicorn>=0.54.0`; API test now captures server output on startup failure). Locked fastapi 0.162.0, starlette 1.10.0, uvicorn 0.54.0. httpx not added (needs user OK); API tests run the server in a subprocess and use urllib. `uv lock` and `uv sync --locked` exit 0; `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0, 35 passed (3 api, 9 core, 21 hooks, 2 worker); `uv run rag-foundry-worker --once` logs `worker started env=dev`, exit 0. `FastAPI(title, version)`, `app.state` and `uvicorn.run(app, host, port, log_level)` checked against fastapi.tiangolo.com and uvicorn.dev: current, not deprecated. Windows 11 only.
- 2026-10-14: v1.0 planning pack (67 catalogue entries, M0–M9 milestones).
- 2026-10-26: v1.1 added R47–R51 (CAG, KAG, prefix KV, exact and semantic response caches) and independent-consumer/source-anchor contracts.
- 2026-10-26: v2.0 restructure. Project moved to repo root; provider-agnostic layer, technique plugin system, in-app Tech Scout and open-source goal added; M0–M9 replaced by 10 phases × 5 steps (mapping in the roadmap); catalogue schema v2; planner/coder/reviewer agent workflow. No application milestone completed; no benchmark run.
