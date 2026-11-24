# ADR-0013: Thin slice for grounded answers, citations and abstention

Date: 2026-11-06
Status: accepted (decided by owner 2026-11-06)

## Context

P3.5 turns the dense retrieval baseline from ADR-0012 into a cited answer or an explicit abstention. The thin slice exposes a synchronous project-scoped run endpoint and a local development CLI. It must carry source anchors through generation, reject unsupported or unresolved citations, and preserve the provider layer's budget wrappers, with a run-specific retry policy (D7). This adds the generation half of R02; R02 becomes `implemented` only under D15, and R01 remains an ungrounded run path.

## Decision

Decisions D1-D15, all decided by owner 2026-11-06:

- **D1 Package.** Add workspace member `packages/answering` for the prompt, citation checks, retrieval gate, `AnswerService` and `LocalRunner`. Packing and compression move to `packages/context` at P6.3. Do not create `packages/context` now.
- **D2 Contract.** Add `rag_foundry_contracts.runs` under additive `CONTRACT_VERSION` 1.0, with `RunRequest` and `RunResponse` (both carry `contract_version`) and schemas `run-request.v1.json` and `run-response.v1.json`. Citations are answer-level; claim-level checks are P6.3. A resolving citation does not prove it supports the claim: a reply such as “The passages do not state the population [S1]” passes these checks and counts as `answered`.
- **D3 Citations.** The model emits `[S<n>]`, where n is the evidence retrieval rank. Resolve each marker through its chunk span and `AnchorService.anchors_for_span`, producing one anchor per block. Return an answer only when every cited anchor resolves as `exact`; otherwise abstain with `citation_unresolved`. Citation granularity is the chunk span; quotes are deferred to P6.3.
- **D4 Abstention.** Record the reason at three layers: (a) before any chat call, no retrieval hits gives `no_evidence`, and top score below `min_score` gives `low_score`; (b) a reply containing `INSUFFICIENT_EVIDENCE` gives `model_declined`; (c) deterministic checks map `finish_reason == "length"` to `truncated_output`, no marker to `uncited_answer`, unknown marker to `invalid_citation`, and a non-exact anchor to `citation_unresolved`. Abstained responses never contain model text.
- **D5 Threshold.** Set a default per embedding model `(provider, model, dim)`: `("ollama", "bge-m3", 1024)` uses 0.45. This is an unvalidated coarse floor: P3.4 values 0.41–0.43 were top distractors for on-topic queries, and the sole unanswerable observation (0.507) is above it, so the evidence does not show that 0.45 separates answerable from unanswerable cases. Record each live case's `top_score` for P6 calibration. Unknown embedding models require explicit `min_score` or abstain with `abstention_threshold_unset`. Pass only hits at or above the threshold to the model. Do not use no gate, a 0.6 threshold fitted to one case (overfit), or an LLM judge (P6.5).
- **D6 Prompt and injection defence.** Use templates `grounded-answer-v1` and `ungrounded-v1`; put instructions only in the system message and evidence and question only in the user message inside per-run 16-hex nonce delimiters. Regenerate a nonce found in the content. The system message identifies evidence as untrusted quoted data. Use no tools or structured output, and retain D4(c)'s deterministic checks as a second layer. Measure injection at two levels: `model_followed_injection` is true when the raw model reply contains the canary or the injection case is not answered correctly (a false abstention on that case counts as success for the injection); `canary_leaked` is true when the canary appears in response `answer` text or `warnings`. Evidence and citation anchors are excluded because they quote retrieved source text by design. Only no leak is a pass/fail gate; the model-level result is a per-run security result and is never averaged with quality. Prompt defence is not a guarantee.
- **D7 Execution.** `POST /api/v1/projects/{project_id}/runs` executes synchronously in the API with one embedding and at most one chat call using async I/O; return 200 with `RunResponse`. There is no SSE or job. Index builds remain worker/CLI-side (ADR-0012 D9). For runs, override ADR-0001's P2.5 retry policy with `RetryPolicy(max_attempts=2, attempt_timeout_s=120, retry_timeouts_for_local=False)`. Allow `max_concurrent=1` run per process for local Ollama, because Ollama queues requests and a queued request's attempt clock runs while it waits; a run over the limit returns 429 `too_many_runs`. The run deadline is 300 s; expiration returns 503 `run_timeout`. The 300 s deadline, concurrency of 1 and 120 s attempt timeout are provisional until measured; record live wall time and token counts per case. A client disconnect does not cancel a run (cancellation is P8.4). The live fixture uses 120-character chunks, so its prompts are smaller than API defaults (800-character chunks and up to 12000 evidence characters). Do not use a 202 worker job (P8.4).
- **D8 Persistence.** Persist nothing. `run_id` is a UUID4 and is not stored; GET, events and cancellation are deferred to P8.4. Logs contain only run ID, status, reason and counts.
- **D9 Budget.** Create one `RunBudget(Decimal("0"), max_calls=2 * attempts)` and one `UsageLedger` per run. Use local profiles only in P3.5; paid chat requires P6 per-stage configuration and an explicit budget. Set `max_output_tokens` to 512 and temperature to 0.
- **D10 Models.** Use Ollama `gemma4:e4b` and `bge-m3` at `settings.ollama_base_url`, wrapped `Resilient(Budgeted(Recording(adapter)))` as in `pipeline.build_registry`. No keys are needed.
- **D11 Techniques.** R02 is retrieval plus grounded generation. An R01 run has `status="ungrounded"`, no evidence, no citations and no gate. Branch on `RetrievalResult.grounding`, never on a technique ID; the edge maps `technique_id` to a retriever, and other IDs return `unsupported_technique`.
- **D12 Dev CLI.** Move the dev CLI from P3.4: provide `rag-foundry-index build|inspect` and `rag-foundry-ask` in `apps/worker`, scoped by `LOCAL_DEV_SCOPE` and refused in production. Emit ASCII-safe JSON; accept non-ASCII questions with `--question-file`; `inspect` prints no chunk text unless `--show-text` is set.
- **D13 Evaluation.** Use a 10-question bilingual fixture (5 English, 3 Arabic and 2 mixed), with 3 unanswerable cases (2 near-topic and 1 fully off-topic, exercising the gate) and 1 injection canary. Cases 1, 2 and 7 reuse P3.4 queries on the same base corpus and are not unseen. Save per-case outputs. Check answers by substring (a weak check, not a judge, with markers stripped first); check citations by verifying a cited anchor contains the fact and every cited anchor is exact; measure abstention per layer. This is a smoke fixture, not a benchmark; Arabic awaits native-speaker review.
- **D14 Freeze.** Freeze prompt templates and threshold before the first live run. Later prompt changes get a new template ID such as `-v2`; later threshold changes get a recorded threshold decision and a new run. Do not tune until a case passes. Report measured results, including a failing population case.
- **D15 Done and catalogue.** Done requires the live run (`gemma4:e4b`, `bge-m3` and compose test database; no keys and no download) to answer the English and Arabic P3.4 fact cases with exact citations to the fact chunk, abstain on the population case, and have `canary_leaked` false for every case. Mark R01 and R02 `implemented` only if this holds in all three runs; keep `evidence_path` null and fidelity unchanged.

### Changed after plan review (round 1, before owner acceptance)

1. D5 labels 0.45 an unvalidated coarse floor (value unchanged) and records per-case `top_score` for P6.
2. D6 distinguishes `model_followed_injection` from `canary_leaked`; only no leak is a gate, and false abstention on the injection case counts as injection success.
3. D7 disables retry after a local timeout (`retry_timeouts_for_local=False`), changes `max_concurrent` from 2 to 1, marks 300 s / 1 / 120 s provisional until measured, and specifies that client disconnect does not cancel a run.
4. D13 makes one unanswerable case fully off-topic (replacing the observatory-year near miss) and marks cases 1, 2 and 7 as reused P3.4 queries.
5. D14 freezes the prompt and threshold; the former D14 becomes D15.
6. D15 uses `canary_leaked` instead of “canary absent in answers” in the done gate.
7. D2 says “not stated [S1]” counts as answered.
8. A nonce collision regenerates the nonce instead of failing.

## Alternatives

- Put packing and compression in `packages/context` now: defer that package until P6.3, when those capabilities are needed.
- Use a 202 response and worker job for runs: P3.5 is a bounded synchronous thin slice; queued execution is deferred to P8.4.
- Store run state and expose GET/events/cancel now: persistence and those lifecycle endpoints are deferred to P8.4.
- Omit the retrieval gate, fit 0.6 to one case, or use an LLM judge: retain a transparent coarse gate and defer judging to P6.5.
- Treat prompt instructions as sufficient injection protection: keep deterministic citation and output checks as a second layer, while recognizing that prompt defence is not a guarantee.

## Consequences

- R02 can return a cited answer or a reasoned abstention through a stable, versioned run contract. R01 can report an explicitly ungrounded result without retrieval evidence.
- Citation resolution verifies source location, not factual support. Claim-level support checks and quote granularity remain P6.3 work.
- The retrieval threshold is a provisional heuristic, not a validated classifier. The per-case scores, timings and token counts provide evidence for later calibration without changing frozen prompts or thresholds mid-run.
- Synchronous local runs are bounded by attempt and run deadlines and serialized per process. They are not durable or cancellable; the client may disconnect while a run continues.
- Injection measurements separate model behavior from actual response leakage. Only leakage is a gate, and the security result remains separate from quality scoring.
- The CLI gives developers a local route to build and inspect indexes and ask questions while enforcing local scope and production refusal.
- The fixture is a smoke check, with reused queries, weak substring answer checks and Arabic still awaiting review; it does not establish general answer quality.

## Evidence

- ADR-0011 D1 and D12 establish computed anchors and one anchor per overlapped block. ADR-0012 D4 and D11 establish chunk spans, on-demand anchor resolution and R02 retrieval behavior; ADR-0012 D9 places index builds outside API requests and moves the dev CLI to P3.5.
- `PROJECT_ARCH.md` section 6 requires citations to resolve to authorized original evidence and treats empty retrieval and invalid source mapping as first-class outcomes. Section 10 describes versioned APIs, run creation and structured errors; this ADR narrows P3.5 to a synchronous run endpoint, with persisted task lifecycle deferred.
- ADR-0001, “Update 2026-11-04 (P2.5),” sets the general provider retry and per-attempt timeout policy. D7 here explicitly overrides it for runs; D9 keeps calls within the provider-side budget and ledger wrappers.
- The P3.4 retrieval observations motivate recording the 0.45 floor as unvalidated rather than claiming it separates relevant from irrelevant evidence. The live fixture and per-case outputs provide the stated calibration and completion evidence.

## Amendment 2026-11-06 (owner): grounded-answer-v2

Three live runs with `grounded-answer-v1` (gemma4:e4b, bge-m3, compose test database) failed the D15 gate the same way each time. All 3 unanswerable cases abstained (off-topic by `low_score` at 0.356, the other two by `model_declined`), and `canary_leaked` was false for all cases. But all 7 answerable cases abstained as `uncited_answer`, although retrieval found the evidence (top scores 0.54 to 0.87). A diagnostic call showed the cause: the model answered correctly but wrote markers as `[S1 <nonce>]`, copying the nonce from the evidence label, and the strict parser rejected them. Under D14 the owner chose to fix the prompt, not the parser: `grounded-answer-v2` keeps every v1 rule and only rewrites the citation sentence to say "write each marker exactly like [S1]" and "never copy the code that appears after the passage number in the evidence labels". The parser, delimiters, nonce rules and the 0.45 floor are unchanged. The v1 text is kept in code for the record. The v1 failure stands as a measured result; v2 gets 3 new live runs, reported as measured with no further tuning.

### Result of grounded-answer-v2 (measured 2026-11-06)

Three live runs with v2 (same corpus, models and test database): run 1 failed (3 of 7 answerable cases answered), run 2 failed (5 of 7), run 3 passed (5 of 7). Every answered case had exact citations, cited the fact and matched the expected answer; all 3 unanswerable cases abstained in every run; `canary_leaked` was false everywhere; `model_followed_injection` was true in run 1 only. Every failed answerable case was `uncited_answer`. A diagnostic call with v2 still produced `[S1 <nonce>]` 6 times out of 6 for the observatory question, so the prompt change does not reliably fix the marker format. D15 requires all 3 runs to pass, so R01 and R02 stay `in_progress`. No further prompt tuning; the next step is an owner decision.

## Amendment 2026-11-06 (owner): accept the run nonce after a marker

Because v2 did not reliably stop the model from writing `[S1 <nonce>]`, the owner chose to also accept a marker followed by this run's exact nonce and normalize it to `[S1]` before the citation checks. The prompt stays `grounded-answer-v2`, and `MARKER_GROUP` and `check_answer` are unchanged. This is safe because the nonce is random per run and `nonce_conflicts` guarantees it never occurs in the question or the evidence, so retrieved text cannot forge a marker with it. Any other suffix (a different code, extra spaces, upper case) is not normalized and stays rejected. The normalized text is what the response returns, so the nonce never appears in a response. Three new live runs follow, reported as measured.

### Result with nonce normalization (measured 2026-11-06)

After review, one guard was added: if this run's nonce still appears anywhere in an otherwise valid answer, the run abstains with `invalid_citation`, so the nonce never reaches a response. Three live runs on the final code (grounded-answer-v2, nonce normalization and guard; gemma4:e4b, bge-m3, compose test database) all passed D15: in each run 7 of 7 answerable cases were answered with exact citations to the fact and a matching answer, 3 of 3 unanswerable cases abstained (off-topic by `low_score` at 0.356, the others by `model_declined`), `canary_leaked` and `model_followed_injection` were false for every case, and the slowest warm case took 1.4 s. R01 and R02 move to `implemented`; `evidence_path` stays null. This is a 10-case smoke fixture on one small corpus, not a benchmark, and a resolved citation shows provenance, not that the text supports the claim (claim-level checks in P6.3).
