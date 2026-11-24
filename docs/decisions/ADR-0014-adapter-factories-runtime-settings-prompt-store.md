# ADR-0014: Adapter factories, runtime settings and prompt store

Date: 2026-11-06  
Status: accepted (decided by owner 2026-11-06)

## Context

P3.6 resolves three related design gaps before implementation: provider adapter selection, runtime tunable configuration and reviewed prompt storage. These decisions keep deployment configuration, runtime policy and safety limits in distinct homes; make settings and prompts auditable; and preserve the deterministic prompt and citation boundaries established by ADR-0013. The owner accepted D1-D16, the designs of migrations 0004 and 0005, and the plan for three live runs on 2026-11-06.

## Decision

Decisions D1-D16, accepted by owner 2026-11-06:

- **D1 Adapter factories.** Decision: define `AdapterFactory` and `AdapterRegistry` in `rag_foundry_providers.factories`. One factory per `fake`, `ollama`, `anthropic` and `openai_compatible` declares `kinds`, `requires_base_url` and `requires_api_key`, checks those requirements, and imports its adapter lazily in `build`. The registry is keyed by name; `default_adapter_registry()` explicitly registers all four, with no entry points or import-time registration. `UnknownAdapterError` and `UnsupportedModelKindError` subclass `ProviderUnavailableError` and use the same message. `build_registry(..., adapters=None)` uses the default registry; runner and `rag-foundry-index` construct through it and preserve their existing wrapper chains. Alternative: entry points, deferred to P7.1.
- **D2 Adapter names and validation.** Decision: `AdapterName` remains a closed `Literal`, and `Endpoint._check_adapter_rules` continues branching on adapter name so config validation stays independent of factories, avoids an import cycle, and returns value-free errors. Tests assert the Literal names match registry names and validator rules match factory `requires_*` flags. A new adapter adds a factory, registration, Literal member, `_IMPLEMENTED` member and validator rule; `pipeline.py` needs no edit. Alternative: validator reads factory flags.
- **D3 Configuration homes.** Decision: deployment configuration remains in environment-backed `Settings`; runtime tunables live in `rag_foundry_domain.runtime_settings` without new package edges; safety, contract and resource caps stay fixed in code with `# fixed:` comments. Alternative: place all configuration in one settings layer.
- **D4 Runtime precedence and validation.** Decision: code defaults < strict JSON file at `RAG_FOUNDRY_RUNTIME_CONFIG_PATH` < DB global < tenant < project; no per-key environment variables; each supplied value replaces the old value whole. Validate per section (`answer`, `run`, `indexing`): runs read answer and run settings, while index builds read indexing and run settings and never the answer section, so a bad section cannot break the other path. Cross-key checks such as overlap < size occur at read time only; write-time cross-key validation is deferred to the P9.3 UI. Alternative: validate every section on every read and reject writes before P9.3.
- **D5 Runtime setting scopes.** Decision: `answer.*`, `indexing.c05_chunk_size` and `indexing.c05_chunk_overlap` support global/tenant/project (G/T/P); other `run.*` settings and `indexing.embed_batch_size` are global; `run.max_concurrent` is file/default only and read at startup. Alternative: expose every tunable at every scope.
- **D6 Overrides and audit.** Decision (migration 0005): an append-only overrides table has a `key` CHECK against the allowlist and a DB CHECK limiting values to JSON numbers, booleans or objects containing numbers only; object keys follow the `provider/model/dim` pattern, so strings cannot be stored. The DB key allowlist leaves out `run.max_concurrent` (file or default only), so the database rejects it. `reason` and `created_by` are free text and must never be logged. Audit `created_by`, `created_at`, `reason` and `version`; enforce tenant/project foreign keys. P3.6 adds no write route, CLI or UI; those belong to P9.3. Alternative: permit arbitrary JSON or add writes in P3.6.
- **D7 Run pinning.** Decision: one snapshot per run. `RunPins` gains optional `settings_sha256` for answer plus run settings and `prompt_sha256`; `CONTRACT_VERSION` advances from 1.0 to 1.1 because strict `extra="forbid"` readers need a new version, while 1.0 documents remain valid. The runner logs effective answer plus run JSON once at INFO with `run_id`; after D6 these contain numbers and booleans only. This is needed because run persistence is deferred until P8.4 and a hash cannot be reversed. Alternative: pin each value in `RunPins`, requiring a contract change for each new key.
- **D8 Settings errors.** Decision: `SettingsError("store_unavailable")` maps to HTTP 503 `settings_unavailable`; any other settings code raised while resolving a bad stored row or file maps to HTTP 500 `settings_invalid` with a fixed message. Without a DB, resolve defaults plus file. Alternative: expose internal settings error details.
- **D9 Caching.** Decision: do not cache settings; perform one query per run. Cache prompt rows by ID after checking their hash, and read the active pointer once per run. Alternative: cache settings or active pointers across runs.
- **D10 Prompt store.** Decision (migration 0004): `prompt_templates` are global and immutable, with a DB CHECK that `content_sha256` equals SHA-256 of UTF-8 content. `prompt_activations` is append-only and records the latest active template per purpose, `activated_by` and `decision_ref`. Seed byte-identical `grounded-answer-v1` (activate then supersede, retaining it for the record), `grounded-answer-v2` and `ungrounded-v1`, both active. Templates are global only. Alternative: mutable templates or tenant-specific prompts.
- **D11 Prompt code and database boundary.** Decision: the database stores system text only. Code retains user-message layout, `<<<EVIDENCE S<n> nonce>>>` and question delimiters, a 16-hex nonce with eight regeneration attempts, `nonce_conflicts`, `DECLINE_TOKEN`, marker grammar, parser and nonce normalization. Parser and injection guarantees depend on these; changes require new template IDs. On load, check grounded text contains `INSUFFICIENT_EVIDENCE` and `[S1]`, contains neither `<<<` nor `>>>`, and recompute the hash. The system message never comes from request or document content. Alternative: store formatting and parser rules with prompt text.
- **D12 Prompt changes.** Decision: create versions and activations only through reviewed Alembic data migrations in P3.6; retain ADR-0013 D14. Alternative: runtime authoring or activation.
- **D13 Prompt errors.** Decision: `store_unavailable` and `no_active_template` map to 503 `prompt_unavailable`; `hash_mismatch`, `invalid_template` and other prompt errors map to 500 `prompt_invalid`. Fail before any model call (0 calls), and never fall back to code text. Alternative: silently use a built-in prompt.
- **D14 Retrieval count.** Decision: `k` remains a request field; a project default for k requires a contract change and is deferred to P9.3. Alternative: introduce an unversioned project default.
- **D15 Tunables inventory.** Decision: use the following inventory of runtime settings, fixed limits, deployment settings and provider configuration.

| Value | Where | Proposed | Reason |
|---|---|---|---|
| min_score bge-m3 0.45 (`DEFAULT_MIN_SCORES`) | `answering/gate.py` | setting `answer.min_scores` {"ollama/bge-m3/1024": 0.45}, G/T/P | calibration target (P6); a change is a recorded threshold decision (ADR-0013 D14), so `reason` is required |
| max_output_tokens 512 | `answering/service.py` AnswerConfig, `runner.py` `default_max_output_tokens=512` | setting `answer.max_output_tokens`, G/T/P, bounds 1..4096 fixed | quality/cost knob |
| temperature 0.0 | AnswerConfig, `runner.py` | setting `answer.temperature`, G/T/P, bounds 0..2 | quality knob |
| max_evidence_chars 12000 | AnswerConfig | setting `answer.max_evidence_chars`, G/T/P, bounds 500..100000 fixed | context budget knob |
| k default 5 | `contracts/runs.py` RunRequest, AnswerConfig, DenseConfig | fixed (request field, contract default) | a project default k needs RunRequest.k optional (contract change, P9.3) |
| MAX_RUN_K 20, DenseConfig k max 50, AnswerConfig k max 20 | contracts, indexing, answering | fixed | contract and resource maxima |
| `rag-foundry-ask --k 5` | `apps/worker/.../ask_cli.py` | fixed | mirrors the contract default |
| RUN_RETRY max_attempts 2 | `answering/runner.py` | setting `run.max_attempts`, G, bounds 1..5 | ops knob (ADR-0013 D7 provisional) |
| RUN_RETRY attempt_timeout_s 120 | `runner.py` | setting `run.attempt_timeout_s`, G, bounds (0, 600] | provisional, measured 1-4 s warm. Note: above 60 s it has no effect on Ollama chat today, because the adapter's own HTTP timeout (60 s) fires first; embeddings stop at 120 s |
| RUN_RETRY retry_timeouts_for_local False | `runner.py` | setting `run.retry_timeouts_for_local`, G | ADR-0013 D7 |
| run deadline 300 s (`DEFAULT_RUN_TIMEOUT_S`) | `runner.py` | setting `run.deadline_s`, G, bounds (0, 1800] | provisional |
| DEFAULT_MAX_CONCURRENT 1 | `runner.py` | setting `run.max_concurrent`, file/default only, bounds 1..8, read at startup | process-level counter; a DB override could not take effect without restart |
| run budget max_calls = 2 x attempts, USD 0 | `runner.py` | fixed (derived) | follows attempts; local-only runs (ADR-0013 D9) |
| adapter HTTP timeouts: Ollama chat 60 s (`ollama.py:68`), Ollama embed 120 s (`ollama.py:192`), Anthropic 60 s, OpenAI-compatible chat/embed 60 s | `providers/adapters/*.py` | providers config (P6) | per-endpoint transport property, not a run tunable |
| embed batch size 32 | `indexing/service.py` | setting `indexing.embed_batch_size`, G, bounds 1..256 | throughput only; outputs unchanged |
| IndexService max_views 100 | `indexing/service.py` | fixed | resource cap |
| inspect_chunks limit max 100; `rag-foundry-index inspect --limit 20` | `indexing/service.py`, `index_cli.py` | fixed | resource cap / CLI paging default |
| C05 chunk_size 800 / overlap 100 | `chunking/recursive.py` RecursiveConfig, `index_cli.py` args | setting `indexing.c05_chunk_size`, `indexing.c05_chunk_overlap`, G/T/P; RecursiveConfig keeps its defaults and bounds (32..8000) as the plugin default, a test keeps them equal | already pinned per index version (chunker_config_sha256) |
| C05 separators | `chunking/recursive.py` | fixed | technique definition (ADR-0012 D3) |
| anchor context 32 | `ingestion/anchors.py`, `anchor_service.py` | fixed | changes anchor prefix/suffix output (contract-level) |
| AnchorService max_views 10 | `ingestion/anchor_service.py` | fixed | resource cap (ADR-0011) |
| MAX_RESOLVE_ANCHORS 100, MAX_ANCHOR_ID_LENGTH 256 | `contracts/resolve.py` | fixed | contract maxima |
| non-upload body cap 1 MiB, MULTIPART_OVERHEAD 64 KiB | `api/limits.py` | fixed | security cap |
| max_upload_bytes 100 MiB | `core/settings.py` | env (unchanged) | deployment limit |
| API read chunk 1 MiB (`_READ_SIZE`) | `api/documents.py` | fixed | internal I/O size |
| ExtractionLimits max_chars 5,000,000, max_pages 2000 | `ingestion/extract/base.py` | fixed | safety caps; deployment override later if needed |
| HTML `_MAX_LT_RUN` 256, `_MAX_DEPTH` 4096 | `ingestion/extract/html.py` | fixed | security guard tied to selectolax 1.0.1 (ADR-0010 D11) |
| MAX_QUESTION_CHARS 2000 | `contracts/runs.py` | fixed | contract maximum |
| nonce 16 hex, 8 regeneration tries | `answering/prompt.py`, `service.py` | fixed | parser and injection guarantees (ADR-0013 D6) |
| provider retry defaults 3 / 20-30 s / 60 s / 300 s | `providers/resilience.py` | fixed now, setting at P6 | ADR-0001 P2.5 decisions; `build_registry` has no production caller yet |
| budget `_FRAMING_TOKENS` 64 | `providers/budget.py` | fixed | cost-safety estimate |
| providers config max 1 MiB | `providers/config.py` | fixed | parser safety |
| Ollama chat context 8192, model ids gemma4:e4b / bge-m3 | `providers/adapters/ollama.py`, `runner.py` | providers config (P6) | model profile properties, not tunables |
| Anthropic DEFAULT_MAX_TOKENS 1024 | `providers/adapters/anthropic.py` | fixed (adapter cost cap) | per-endpoint cap belongs to providers config |
| worker_poll_seconds 5.0, ollama_base_url, provider_namespace | `core/settings.py` | env (unchanged) | deployment config |

- **D16 Completion.** Decision: golden requests from P3.6-00 remain unchanged; `pytest -m db` passes on a fresh test database; three live runs on a fresh test database pass the ADR-0013 D15 gates with pins `grounded-answer-v2` plus its hash. Report per-run differences compared with the 2026-11-06 P3.5 live runs. Alternative: accept partial live-run success or alter golden requests during this task.

## Alternatives

- Use entry points for provider discovery: defer them until P7.1.
- Make config validation depend on factories: retain independent, value-free validation.
- Permit arbitrary override JSON, mutable prompts or prompt fallbacks: retain strict DB constraints, immutable reviewed prompt versions and fail-closed loading.
- Cache settings or active prompt pointers across runs: query settings and active pointers per run for current values.
- Add per-key environment variables or project-level `k`: keep deployment config separate and require a versioned contract change for project defaults.

## Consequences

- Provider construction has an explicit extension point while config validation remains independent of provider imports.
- Runtime settings have layered precedence and section-scoped validation; stored cross-key relationships are checked when read until P9.3 adds write-time validation.
- Overrides and prompt records are auditable and constrained. Override rows' foreign keys and immutability trigger block deleting a tenant or project. A privileged purge path, using the owner role with the trigger disabled inside one audited transaction, is needed before tenant deletion exists (P10.1); it is not built in P3.6.
- Run hashes pin settings and prompt versions while a single safe effective-settings log makes the non-reversible settings hash operationally useful before run persistence arrives in P8.4.
- Prompt system text can be reviewed and versioned in migrations while code retains the delimiters, parser and injection protections. Store errors fail closed before model calls.

## Evidence

- ADR-0013 D6, D14 and its amendments establish the prompt boundaries, nonce normalization and final live-run gates retained here.
- The P3.6 tunables inventory records proposed settings, scope, fixed limits and rationale copied into D15.
- Done 2026-11-07 (P3.6-06 to P3.6-09): migrations 0004 (prompts) and 0005 (settings overrides) applied on a fresh compose test database; golden ChatRequests unchanged; `uv run python scripts/tasks.py check` exit 0, 1718 passed, 55 skipped; `uv run pytest -m db -q` 54 passed, 3 skipped; three live runs (gemma4:e4b, bge-m3) all passed the ADR-0013 D15 gates with prompts loaded from the store: each run 7/7 answerable answered with exact citations, 3/3 unanswerable abstained, no canary leak, injection not followed, pins `grounded-answer-v2` and prompt_sha256 `fc18e5aa...`. Same results as the 2026-11-06 P3.5 runs.
