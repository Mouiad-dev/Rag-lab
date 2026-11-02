# Progress and handoff ledger

## Current state

- Current step: **P2.4** (planning). P2.1–P2.3 done locally (CI run pending push). P2.1 and P2.2 done locally (provider ports, registry, fake provider, contract checks; direct Ollama chat and embedding adapters with an opt-in live test). Phase 1 done: P1.1–P1.5 green in CI ([PR #1](https://github.com/Mouiad-dev/Rag-lab/pull/1), [run 41310101580](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41310101580)). Application code: `packages/core` (strict settings), `packages/contracts` (source anchor v1 + JSON Schema, bilingual fixtures), `packages/techniques` (catalogue loader, plugin protocol and registry v0, no plugins registered), `packages/providers` (async chat/embedding/reranker ports, model profiles with capability flags, usage records, provider registry, deterministic fake provider, reusable contract checks; direct Ollama adapters for chat and embeddings behind the optional `[ollama]` extra (httpx), Anthropic chat adapter behind the `[anthropic]` extra (same httpx, env-only key), usage ledger with cache tokens), `apps/api` (FastAPI `/health`), `apps/worker` (entrypoint, heartbeat), Docker image and compose stack; no product features.
- Delivered: architecture v2.0, [roadmap](plan/ROADMAP.md), [agent workflow](plan/AGENT_WORKFLOW.md), ADR-0001 to ADR-0004, ADR-0008 (independent product), 72-entry catalogue (schema v2), example configs, Claude Code rules/subagents, optional dev hooks with tests.
- Technique implementation/benchmarking: none; all 72 entries `planned`, `catalog_status: accepted`.
- Package checks: see [VALIDATION.md](VALIDATION.md).

## Next action

Native-speaker review of the Arabic fixtures in `packages/contracts/fixtures/sources/`. Plan and build P2.4 (any provider, per pipeline). Owner should revoke the Anthropic key used for the P2.3 live run (it was shared in chat). LiteLLM is re-decided at P2.4 planning (1.106.6 is 2 weeks old on 2026-11-11).  The license decision below is still open.

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
| P1 | Foundation, tooling and CI (old M0) | done |
| P2 | Provider layer (Ollama + one paid provider) | in_progress |
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
| P1.1 | Reproducible toolchain | done | Windows local runs plus CI clean clones on ubuntu-latest and windows-latest: [run 41307162676](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41307162676) |
| P1.2 | Modular skeleton | done | P1.2-01 and P1.2-02 (see History 2026-11-01); `tasks.py check` 35 passed on Windows 11 |
| P1.3 | Local stack | done | P1.3-01..04 (see History 2026-11-01); `docker compose up -d --build --wait` all healthy on Windows 11 with zero API keys; Ollama profile not run |
| P1.4 | Continuous integration | done | [PR #1](https://github.com/Mouiad-dev/Rag-lab/pull/1), [run 41307162676](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41307162676) green; broken catalogue fails via `tests/repo/test_catalogue_validation.py` |
| P1.5 | Contracts and plugin contract v0 | done | P1.5-01..05 (see History 2026-11-01); registry loads 72 entries, unknown ID raises `UnknownTechniqueError`; `tasks.py check` 137 passed on Windows 11; Arabic fixtures pending native-speaker review |
| P2.1 | Provider ports and registry | done | P2.1-01..05 (see History 2026-11-02); contract checks pass against the fake provider with network blocked; `tasks.py check` 297 passed on Windows 11; CI run pending push |
| P2.2 | Local default | done | P2.2-01..05 (see History 2026-11-02); opt-in live test 14 passed ×3 against native Ollama 0.40.13 with gemma4:e4b and bge-m3, no keys; `tasks.py check` 383 passed, 14 skipped; CI run pending push |
| P2.3 | First paid provider | done | P2.3-01..05 (see History 2026-11-02); live test 9 passed, 5 calls, USD 0.0000494 of a 0.05 cap, tokens from provider; offline 495 passed; CI run pending push |
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
- 2026-11-02: P2.3 live run. At the owner's explicit request the owner's key was set only for the one command and removed after (not written to any file). `RAG_LIVE_ANTHROPIC=1 uv run pytest tests/live/test_anthropic_live.py -m live -v -s`: 9 passed in 7.12 s; 5 calls on claude-haiku-5-5, input 74 / output 84 tokens, token_source=provider, total USD 0.0000494 (cap 0.05); tools/structured refused with no call. P2.3 done. The key was shared in chat, so the owner should revoke it.
- 2026-11-02: P2.3-01..05 offline (planner; coder ×5; reviewer: 01 APPROVE r1 (owner decisions added to ADR-0001); 02 APPROVE r1; 03 CHANGES_REQUESTED r1 → APPROVE r2 (malformed-key leak via httpx/h11 and leaked `__context__` fixed); 04 CHANGES_REQUESTED r1 → APPROVE r2 (billed refusals recorded via `ProviderCallError.usage`; cache writes cost None unless the 5m/1h split is present and complete); 05 APPROVE r1). 01 verified the Anthropic Messages API from official pages (ADR-0001 Evidence): claude-haiku-5-5 and claude-sonnet-5-5, prices (Haiku 0.10/0.50 per MTok up to 100k, Sonnet 2/10), anthropic-version 2026-10-21, temperature rejected, thinking on by default. Decision: direct httpx adapter reusing httpx==0.28.1 (no new package). Owner decisions: Bearer auth, temperature refused before any call, Haiku thinking disabled, Sonnet `between_tools` with default max_tokens 16384 (worst case about USD 0.16 per default call). Added: `Usage` cache read/write fields and `ModelProfile` cache rates (`estimate_cost` in Decimal, None for unknown rates), in-memory append-only `UsageLedger` + `RecordingChatModel` (records billed failures once), `MissingCredentialError` and new error reasons (auth, billing, rate_limited, overloaded, server_error, invalid_request, refused), key only from `ANTHROPIC_API_KEY` (SecretStr, printable-ASCII check), errors raised without cause/context, `AnthropicChatModel` (complete + SSE stream, Haiku cost None above the 100k tier). Leak tests with a sentinel key and DEBUG logs (MockTransport, so httpcore/h11 logging is not exercised in tests; the reviewer probed a real loopback client: no leak). Known gap for P2.5: a stream stopped early records no usage. `uv run pytest tests/providers -q` 337 passed; `tasks.py check` exit 0, 495 passed, 23 skipped; `ruff check .` and `ruff format --check .` pass; `verify_package.py` PASS. Live test (`tests/live/test_anthropic_live.py`, RAG_LIVE_ANTHROPIC=1, Haiku, ≤64 output tokens, USD 0.05 hard cap checked before each call) not yet run: needs the owner's new key. A key was pasted into chat on 2026-11-02; it was not used or stored, and the owner was asked to revoke it.
- 2026-11-02: P2.2-01..05 (planner; coder ×4, coder-light ×1; reviewer: 01, 03 APPROVE r1 (ADR wording fixed directly); 02 CHANGES_REQUESTED r1 fixed directly (URL rejects whitespace/control chars and bad ports); 04 CHANGES_REQUESTED r1 fixed directly (stream client built inside the mapped try); 05 CHANGES_REQUESTED r1 (Result record) fixed directly). 01 verified LiteLLM and Ollama against official sources (ADR-0001 Evidence): LiteLLM fits technically, but GHSA-5mg7-485q-xm76 (malware in 1.82.7/1.82.8) and no version outside GHSA-w4c7-34wj-hqm8 being 2 weeks old led to a user decision: P2.2 uses a direct Ollama adapter (ADR-0001 Update 2026-11-02); LiteLLM re-decided at P2.3/P2.4. User approved installing Ollama natively (winget, 0.40.13) and pulling gemma4:e4b (6.6 GB, dc35e8d9c606) and bge-m3 (1.2 GB, 790764642607). Added: `Settings.ollama_base_url` (strict URL validation, `hide_input_in_errors`), strict `live` pytest marker, optional extra `rag-foundry-providers[ollama]` = httpx==0.28.1 (lock adds httpx, httpcore, certifi), `OllamaEmbeddingModel` (/api/embed, truncate:false) and `OllamaChatModel` (/api/chat, NDJSON stream, num_ctx from profile = 8192 effective window, think:false; tools/structured/vision refused), token_source=provider only when Ollama returns counts, errors carry reason+status only, network guard extended to asyncio create_connection. Owner fix after a live failure (1 in 4 runs): stream-equals-complete moved to `DETERMINISTIC_CHAT_CHECKS` (fake and MockTransport only), since real models sample. `RAG_LIVE_OLLAMA=1 uv run pytest tests/live -m live -q` 14 passed in 3 consecutive runs (chat en/ar and embed en+ar, token_source=provider, embed dims 1024); without the env var 14 skipped. `uv sync --locked` ok; `uv sync --locked --extra ollama` ok; `tasks.py check` exit 0, 383 passed, 14 skipped; `ruff check .` and `ruff format --check .` pass; `verify_package.py` PASS. Windows 11 only; CI run pending push.
- 2026-11-02: P2.1-01..05 (planner; coder ×5; reviewer: 01, 03, 04 APPROVE r1; 02 CHANGES_REQUESTED r1 fixed directly (manifest scan now catches single-quoted TOML); 05 CHANGES_REQUESTED r1 → APPROVE r2 (negative tests per check, stricter rerank checks)). New workspace member `packages/providers` (pydantic only, no new third-party deps): typed errors (not KeyError/LookupError), `ModelProfile` with tri-state capability flags (undeclared never counts as supported), `Usage` with required `token_source` (provider|estimated) and Decimal-or-None cost (never invented), async ports with no LangChain types, `ProviderRegistry` with `validate_requirements` listing every gap, stdlib-only deterministic fake provider (SHA-256 embeddings, golden values stable across PYTHONHASHSEED), reusable `contract_checks` plus negative tests, test-only network guard (documented limits: not an isolation boundary; Windows proactor connect not covered). `tests/repo/test_no_vendor_sdk.py` blocks vendor SDK imports outside a future `adapters/` folder. Owner decisions: ports async-only for now (ADR-0001 sync wrappers decided at P2.2); LiteLLM re-verification moved to P2.2; `local` requirement deferred to P2.4. `uv sync --locked` ok; `uv run pytest tests/providers -q` 154 passed; `tasks.py check` exit 0, 297 passed; `ruff check .` and `ruff format --check .` pass; `verify_package.py` PASS. Windows 11 only; CI run pending push.
- 2026-11-01: P1.5 CI (user approved push). First push ([run 41309752778](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41309752778)) failed on ubuntu: `verify_package.py` "Broken local link" for a literal `[ADR-0008](...)` placeholder in the P1.5-01 spec (Windows resolves `...`, Linux does not). Fixed in c2105f1. [Run 41310101580](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41310101580): ubuntu 137 passed, windows 137 passed, image job healthy (first CI build including packages/contracts and packages/techniques). Phase 1 done.
- 2026-11-01: P1.5-01..05 (coder-light, coder ×4; reviewer: 01 APPROVE r2, 02 CHANGES_REQUESTED r1 → APPROVE r2, 03/04/05 APPROVE r1). ADR-0008 accepted (records REUSE §1) + static independence test with self-tests. `packages/contracts`: SourceAnchor v1 (frozen, extra=forbid, code-point ranges, precision must be backed by data), committed JSON Schema with drift test, LF pinned via `.gitattributes`. Bilingual fixtures (en, ar, mixed with emoji; 11 anchors; original text, pending native-speaker review; one Arabic number-agreement fix after review). `packages/techniques`: read-only strict catalogue loader (rejects proposed entries, non-int schema_version) and plugin protocol/registry v0 (12 ADR-0002 kinds; `UnknownTechniqueError`, `TechniqueUnavailableError`, `DuplicatePluginError`, none KeyError/LookupError; no plugins registered). No new third-party deps; uv.lock adds only the two workspace members; Dockerfile copies their manifests. `uv sync --locked` ok; `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0, 137 passed; `verify_package.py` PASS; `git status --porcelain config` empty. Docker build not run locally (base image not cached); the CI image job will cover it. Windows 11 only so far.
- 2026-11-01: P1.4-04 push and CI (user approved push + PR). Pushed `p1/toolchain-and-skeleton`, opened [PR #1](https://github.com/Mouiad-dev/Rag-lab/pull/1) (not merged). [Run 41307162676](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41307162676) on 3cd2e6e: `check (ubuntu-latest)` success, `tasks.py check` 75 passed, 0 skipped; `check (windows-latest)` success, 75 passed, 0 skipped; `image` success, db/api/worker Healthy, `/health` → `{"status":"ok","version":"0.0.0"}`. No fix commits needed. Closes P1.1 (clean clone on Linux and Windows) and P1.4.
- 2026-11-01: P1.4-01..03 (coder-light, coder ×2; reviewer APPROVE on all three, round 1). 01: ruff now lints `scripts/` (formatting only; `verify_package.py` output and `build_flow.py --check` unchanged), hook tests run once via pytest in `tasks.py check`. 02: `tests/repo/test_catalogue_validation.py`, 10 broken-catalogue variants on a temp copy each exit 1 with the expected message, unmodified copy passes; real catalogue untouched. 03: `.github/workflows/ci.yml` with `check` (ubuntu + windows: `uv sync --locked`, `tasks.py check`, compose config on ubuntu) and `image` (ubuntu: compose build/up --wait, /health, down -v; generated password masked); actions pinned by SHA (checkout v7.0.1 3d3c42e, setup-uv v10.3.3 68559e9, uv 0.13.6), `contents: read`. Local: `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0, 75 passed; `verify_package.py` PASS. CI has not run on GitHub yet.
- 2026-11-01: P1.3-04 stack bring-up (coder; reviewer CHANGES_REQUESTED round 1 for bookkeeping only, fixed directly). Windows 11, Docker Engine 29.8.2. User approved core-stack image downloads only. `.env` created from `.env.example` with a random local password (git-ignored, not printed). `docker compose up -d --build --wait` exit 0; `docker compose ps`: api, db, worker healthy; `/health` on 127.0.0.1:8000 → `{"status":"ok","version":"0.0.0"}`; `docker compose port api 8000` → `127.0.0.1:8000`; pgvector `default_version` 0.8.7; `docker compose down` (volume kept). Reviewer re-ran up/ps/health/down with the same results. Images: rag-foundry-app:local 203MB, pgvector/pgvector:0.8.7-pg18-trixie 652MB. pg18 `VOLUME /var/lib/postgresql` confirmed in the docker-library postgres docs. `tasks.py check` exit 0, 60 passed. CONTRIBUTING "Local stack (Docker)" added. Ollama profile not run (not approved).
- 2026-11-01: P1.3-01..03 (coder ×3; reviewer: 01 APPROVE, 02 CHANGES_REQUESTED fixed directly (Dockerfile secret check now joins line continuations, negative test added), 03 APPROVE). Worker `--heartbeat-file`/`--check-heartbeat` (future mtime counts as stale); non-root multi-stage `Dockerfile` (`python:3.12.16-slim-trixie`, `ghcr.io/astral-sh/uv:0.13.6`); `compose.yaml` (`pgvector/pgvector:0.8.7-pg18-trixie`, api, worker, `ollama/ollama:0.40.13` under profile `ollama`; ports on 127.0.0.1 only, db unpublished, read-only + cap_drop ALL for api/worker); `.env.example` placeholders only, `POSTGRES_PASSWORD` required. Tags checked via registry metadata, no pulls. `docker compose --env-file .env.example config --quiet` exit 0; `uv run pytest tests/worker tests/stack/test_dockerfile.py -q` 16 passed; `tasks.py check` exit 0, 60 passed. Not built or started: P1.3-04 pending user OK for ~250–350 MB of images. Open for P1.3-04: confirm pg18 data dir against postgres image docs; consider no-new-privileges/cap_drop on ollama.
- 2026-11-01: P1.2-02 API and worker (coder; reviewer CHANGES_REQUESTED round 1, fixed directly: lower bounds `fastapi>=0.162.0`, `uvicorn>=0.54.0`; API test now captures server output on startup failure). Locked fastapi 0.162.0, starlette 1.10.0, uvicorn 0.54.0. httpx not added (needs user OK); API tests run the server in a subprocess and use urllib. `uv lock` and `uv sync --locked` exit 0; `ruff check .` and `ruff format --check .` pass; `tasks.py check` exit 0, 35 passed (3 api, 9 core, 21 hooks, 2 worker); `uv run rag-foundry-worker --once` logs `worker started env=dev`, exit 0. `FastAPI(title, version)`, `app.state` and `uvicorn.run(app, host, port, log_level)` checked against fastapi.tiangolo.com and uvicorn.dev: current, not deprecated. Windows 11 only.
- 2026-10-14: v1.0 planning pack (67 catalogue entries, M0–M9 milestones).
- 2026-10-26: v1.1 added R47–R51 (CAG, KAG, prefix KV, exact and semantic response caches) and independent-consumer/source-anchor contracts.
- 2026-10-26: v2.0 restructure. Project moved to repo root; provider-agnostic layer, technique plugin system, in-app Tech Scout and open-source goal added; M0–M9 replaced by 10 phases × 5 steps (mapping in the roadmap); catalogue schema v2; planner/coder/reviewer agent workflow. No application milestone completed; no benchmark run.
