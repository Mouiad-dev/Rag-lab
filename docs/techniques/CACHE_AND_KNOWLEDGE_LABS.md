# Screenshot techniques: coverage and experimental contracts

Updated 2026-10-26. All entries remain planned. The screenshots are topic prompts, not evidence of speed, accuracy or universal suitability.

| Screenshot topic | Catalogue commitment | Classification |
|---|---|---|
| Traditional RAG | R02 | Existing dense two-step baseline; compare other retrievers separately |
| Agentic RAG | R33 | Existing bounded agentic workflow |
| Corrective RAG | R29 and R46 | Existing adaptation and distinct research reproduction |
| HyDE | R17 | Existing query transformation, not an alternative evidence source |
| CAG | R47 | Added cache-augmented, bounded-context generation experiment |
| KAG | R48 | Added OpenSPG-style schema/logical-form knowledge retrieval experiment |
| Prefix caching | R49 | Added inference optimization |
| Exact-response caching | R50 | Added application response reuse |
| Semantic caching | R51 | Added approximate response reuse with guarded eligibility |

The full curriculum is now 72 entries: the original 67 plus five distinct additions. Existing entries have not been removed or relabeled as completed.

## CAG (R47)

Preload a bounded, versioned knowledge snapshot into model context and reuse compatible prefill/KV state for multiple questions. Compare with the same complete context without caching and with RAG over the same source corpus. All arms retain source anchors and citation checks.

This experiment requires knowledge plus instructions, question and reserved output to fit the tested context capacity. Reject overflow explicitly; selecting relevant sections first is a hybrid retrieval/CAG variant, not evidence that full-corpus CAG fits. KV state is specific to runtime/model/tokenization and is not a portable source bundle or Redis answer string. Record actual cache support; a provider API with opaque prompt caching may permit a related experiment but not arbitrary KV save/load claims.

Measure cold setup, warm TTFT, end-to-end latency, context/output tokens, memory, cache residency/eviction, update cost and answer quality. Do not repeat the screenshot's “answer instantly” claim: cache reuse does not eliminate generation. Corpus updates produce a new snapshot/cache identity.

Primary references: [CAG authors' repository](https://github.com/hhhuang/CAG), [paper](https://arxiv.org/abs/2412.15605).

## KAG (R48)

Use a pinned OpenSPG/KAG release or explicitly document an original adaptation. Model domain schema, build grounded facts and relations, retain mutual links to source chunks/anchors, and execute inspectable logical-form-guided retrieval/reasoning. Compare multi-hop/constraint questions to dense RAG and the existing graph tracks. Every derived result exposes supporting facts and source references.

An entity graph plus a generic prompt does not satisfy the original KAG method. Record which schema constraints, alignment, logical operators and solver behavior are implemented. Include contradictory/missing facts, invalid schema, unsupported operators and extraction errors. Inspect retrieval/operation summaries without exposing private chain-of-thought. Measure graph construction and incremental update costs as well as query quality. KAG is optional infrastructure; importing the SDK or running baseline RAG must not start OpenSPG.

Primary reference: [OpenSPG/KAG](https://github.com/OpenSPG/KAG). Treat its performance claims as hypotheses to evaluate on this project's corpus.

## Three separate cache layers

| Layer | Reused item | Owner | What a hit does not prove |
|---|---|---|---|
| Prefix / R49 | KV/prefill computation for identical token prefix | Inference provider/runtime adapter | Same answer, factual validity, faster decode or cross-runtime portability |
| Exact response / R50 | Final answer and evidence bundle for an identical effective request | Application cache adapter | Current permission or freshness without revalidation |
| Semantic response / R51 | Prior answer for an eligible similar question | Application cache + eligibility policy | Equivalent intent just because vectors are close |

Prefix tests must observe actual provider/runtime hit metrics where available. Report unavailable metrics honestly. Compare identical and changed prefixes, cold/warm states and eviction. Keep tenant isolation policy explicit at the serving layer; do not assume GPU KV caches inherit application cache partitions. Provider capabilities may make some isolation controls unavailable, in which case record the limit and choose an appropriate deployment profile.

Exact cache identity includes authenticated tenant/project + authorization revision/visibility scope, original question and relevant history, corpus/document/extraction/index/pipeline versions, prompt/model revision and generation parameters, tool configuration, source-only/enriched mode, language/output schema and freshness policy. Sensitive request material is hashed server-side, not logged in plaintext cache keys. For mutable SQL/web/tools, use a verified snapshot/version or bypass response caching by default. Never reuse cached external side effects or approval decisions.

Semantic caching shares those exact partitions and only relaxes the eligible question-match component. Preserve entity IDs, dates, numbers, language, negation and user constraints. Calibrate thresholds on held-out paraphrases and adversarial near-misses; use candidate validation and bypass ambiguity. A question about 2025 is not equivalent to one about 2026 because embeddings are similar. Do not normalize away meaning for a hit.

Before returning any response-cache hit, recheck source availability and authorization, expiry and version compatibility. Cache failure is a miss, not an access-control shortcut. TTL is an upper lifetime bound, not a substitute for deletion/revocation invalidation. Maintain source-version reverse dependencies and invalidate on corpus/ACL/prompt/model/pipeline changes. Redis is optional replaceable storage, not the cache semantics.

Default product policy: response caches off for enriched/current-information queries and action/agent runs unless a validated snapshot-aware policy is enabled. Experiment mode always records cache layer, hit/miss/bypass reason and actual calls avoided. Disable response reuse when measuring retrieval quality, and compare cold and warm performance separately. CAG versus RAG comparisons must not accidentally compare cached answers to fresh generation.

Acceptance: same-question hit; changed model/history/ACL/source misses; immediate denied-source behavior; same-sounding opposite-intent semantic miss; failed provider fallback labeled; separate latency, freshness and false-hit measurements. Each entry still requires a lesson, demo, test evidence and a losing case.

Primary prefix reference: [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/). Exact/semantic cache partitioning and safeguards here are project design decisions, not claimed guarantees from that page.
