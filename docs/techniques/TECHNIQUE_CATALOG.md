# Technique Catalogue

72 curriculum entries: 21 chunking/embedding entries and 51 retrieval/knowledge/caching-related entries. These are different categories, not 72 interchangeable RAG architectures. All are planned at delivery.

Canonical machine-readable inventory: `../../config/techniques.json` (schema v2: adds category, catalog_status, primary_source, date_added, phase and roadmap_step). Update both representations together.

The acceptance column describes evidence that must exist before validation. Each entry also requires a lesson, demo, tests, baseline comparison and limitations record as defined by the architecture.

| ID | Family / roadmap step | Technique | Mechanism | Specific acceptance evidence |
|---|---|---|---|---|
| C01 | chunking / P4.1 | Fixed characters | Split at a configured character budget. | Verify offsets and behavior on a sentence cut in half. |
| C02 | chunking / P4.1 | Fixed words | Split by word count with explicit tokenizer assumptions. | Compare Arabic punctuation and mixed-language word boundaries. |
| C03 | chunking / P4.1 | Fixed tokens | Split using the selected model tokenizer. | Assert maximum tokens and show character/token differences. |
| C04 | chunking / P4.1 | Overlap and sliding windows | Reuse a configured span between adjacent chunks. | Quantify duplicates, storage and boundary-fact retrieval. |
| C05 | chunking / P3.4 | Recursive splitting | Try larger textual separators before smaller ones. | Show fallback on an oversized paragraph and preserved provenance. |
| C06 | chunking / P4.2 | Sentence boundaries | Segment by language-aware sentence boundaries then pack. | Test abbreviations, Arabic punctuation and quoted sentences. |
| C07 | chunking / P4.2 | Paragraph packing | Pack paragraphs within a budget with explicit overflow behavior. | Show when keeping paragraphs improves or hurts retrieval. |
| C08 | chunking / P4.3 | Markdown sections | Use heading hierarchy and keep section paths. | Test nested headings and fenced code blocks. |
| C09 | chunking / P4.3 | HTML structure | Extract meaningful DOM sections and retain locators. | Remove navigation noise without losing content or source identity. |
| C10 | chunking / P4.3 | JSON structure | Split structured objects/arrays while preserving JSON paths. | Reassociate a retrieved value with its field/object context. |
| C11 | chunking / P4.3 | Page and layout blocks | Use page regions, reading order, columns and captions. | Compare a two-column PDF with a plain-text baseline. |
| C12 | chunking / P4.3 | Table-aware segments | Preserve schema/headers and explicit row groups for large tables. | Retrieve a cell with units, row/column labels and source span. |
| C13 | chunking / P4.3 | Code separators | Split source with language-specific separators. | Inspect a class, method and docstring across boundaries. |
| C14 | chunking / P4.3 | AST-aware code | Use a real parser for declarations and nested structures. | Prove syntax-node boundaries and handle unsupported syntax explicitly. |
| C15 | chunking / P4.4 | Embedding-semantic boundaries | Detect semantic shifts and pack subject to budgets. | Compare threshold settings on topic changes and a cohesive passage. |
| C16 | chunking / P4.4 | LLM-selected boundaries | Ask for grounded source offsets using structured output. | Reject invented or invalid offsets; record cost and repeat variability. |
| C17 | chunking / P4.4 | Propositions | Generate atomic supported facts linked to original spans. | Check negation/conditions and detect unsupported generated claims. |
| C18 | chunking / P4.5 | Hierarchical parent-child | Create linked granularities with stable parent references. | Find a specific fact in a child and inspect enclosing context. |
| C19 | chunking / P4.4 | Adaptive segmentation | Route document regions to explicit segmentation policies. | Compare policy decisions and failures against one fixed policy. |
| C20 | embedding_representation / P7.5 | Late chunking | Pool token representations after wider-context encoding. | Use a compatible encoder; compare early/late with the same model. |
| C21 | chunking / P4.2 | Multilingual boundary handling | Handle Arabic/English/mixed segmentation and reversible normalization. | Preserve original locators and inspect Arabic diacritics and punctuation. |
| R01 | control / P3.4 | No-retrieval baseline | Answer without corpus retrieval as a diagnostic control. | Label absent grounding and compare on corpus-specific questions. |
| R02 | retrieval / P3.4 | Dense two-step RAG | Embed query, retrieve text vectors, generate grounded answer. | Retrieve a known fact and abstain on a missing one. |
| R03 | retrieval / P5.1 | BM25 lexical search | Implement genuine BM25 with documented tokenization. | Test exact identifiers and Arabic normalization; do not relabel ts_rank. |
| R04 | retrieval / P5.1 | PostgreSQL FTS | Use PostgreSQL lexical search as its own baseline. | Document language analyzer and contrast rank behavior with BM25. |
| R05 | retrieval / P5.1 | Learned sparse retrieval | Use an actual supported learned sparse encoder. | Persist sparse outputs and compare lexical/semantic matches and memory. |
| R06 | retrieval / P5.2 | Hybrid retrieval | Combine dense and lexical/sparse candidate branches. | Keep branch evidence, budgets and scope; measure complementary recall. |
| R07 | fusion / P5.2 | Reciprocal rank fusion | Fuse ranked lists without pretending raw scores are comparable. | Verify ranking with a hand-calculated tiny example. |
| R08 | fusion / P5.2 | Weighted score fusion | Calibrate branch scores then combine using explicit weights. | Fit/calibrate on development data only; sweep weights on validation. |
| R09 | selection / P5.3 | MMR diversity | Trade off query relevance and candidate redundancy. | Show novelty gain and possible loss of a relevant near-duplicate. |
| R10 | selection / P5.3 | Evidence deduplication | Deduplicate by source/overlap or explicit semantic rule. | Preserve distinct conflicting sources while removing repeated passages. |
| R11 | retrieval / P5.3 | Metadata filtering | Apply authorized and user-requested structured filters. | Test date/category filters and zero cross-tenant candidate exposure. |
| R12 | query / P5.3 | Self-query filters | Translate question to a validated allowlisted filter schema. | Reject invented fields and attempts to broaden access scope. |
| R13 | query / P5.4 | Question rewriting | Rewrite ambiguous/noisy questions for retrieval. | Keep original intent and detect constraint loss. |
| R14 | query / P5.4 | Multi-query expansion | Search several controlled formulations and combine evidence. | Measure incremental recall versus extra cost and duplicate queries. |
| R15 | query / P5.4 | Question decomposition | Break a composite question into answerable subquestions. | Track dependencies and ensure all original constraints are addressed. |
| R16 | query / P5.4 | Step-back retrieval | Retrieve a more general concept alongside specific evidence. | Show when abstraction helps and when it loses exact constraints. |
| R17 | query / P5.4 | HyDE | Embed a generated hypothetical document for search. | Never cite hypothetical text; compare with original-query embeddings. |
| R18 | query_fusion / P5.4 | RAG fusion | Combine multiple reformulated searches with rank fusion. | Separate expansion benefit from fusion benefit in an ablation. |
| R19 | reranking / P5.5 | Cross-encoder reranking | Score query-document pairs after candidate retrieval. | Compare before/after ranking and include reranking latency. |
| R20 | reranking / P5.5 | LLM reranking | Use a bounded structured ranking judgment. | Validate item IDs, handle ties/failures and measure order sensitivity. |
| R21 | context_retrieval / P6.2 | Sentence windows | Retrieve a sentence and expand nearby source sentences. | Measure missing context versus extra irrelevant tokens. |
| R22 | context_retrieval / P6.2 | Parent document retrieval | Search children but return authorized parent context. | Bound parent size and maintain child-to-parent source lineage. |
| R23 | representation / P6.2 | Multiple representations | Index summaries, questions or chunks linked to the original. | Show which representation retrieved the source; cite the original. |
| R24 | representation / P6.2 | Contextual enrichment | Add document-specific context before indexing a chunk. | Label generated prefixes and compare added cost and retrieval lift. |
| R25 | context_retrieval / P6.2 | Hierarchical auto-merging | Merge related retrieved children into parent context by policy. | Demonstrate threshold behavior and token growth. |
| R26 | context / P6.3 | Context compression | Extract or summarize relevant context under a budget. | Check preservation of negation, dates, conditions, units and citations. |
| R27 | context / P6.3 | Context ordering and packing | Order/deduplicate evidence and reserve model output budget. | Compare positions and record dropped evidence rather than silently truncate. |
| R28 | workflow / P6.4 | Conversational RAG | Resolve references using bounded conversation state. | Test follow-ups, topic switches and prevention of unsupported memory facts. |
| R29 | workflow / P8.1 | Corrective RAG adaptation | Grade retrieval, rewrite/retry and exit under a strict budget. | Label adaptation to CRAG, show successful recovery and no-progress termination. |
| R30 | workflow / P8.1 | Prompt-based self-checks | Validate support and revise/abstain using explicit checks. | Keep separate from trained Self-RAG and test false-positive judge decisions. |
| R31 | research / P7.5 | Faithful Self-RAG inference | Use a compatible reflection-token model and original method references. | Record checkpoint/license/hardware, actual reflection behavior and deviations. |
| R32 | workflow / P8.1 | Adaptive routing | Choose direct, retrieval or advanced workflow by explicit policy. | Evaluate router errors as well as total quality/cost. |
| R33 | workflow / P8.2 | Agentic retrieval | Let a bounded agent choose retrieval tools and subsequent steps. | Trace tool decisions, stop conditions and budget consumption. |
| R34 | workflow / P8.2 | Multi-hop retrieval | Use evidence from one hop to form the next retrieval step. | Show a genuine dependency between facts across sources. |
| R35 | workflow / P8.3 | Multi-source routing | Combine authorized document, SQL and API/MCP evidence. | Preserve source types, freshness and permission boundaries. |
| R36 | workflow / P8.3 | Multi-agent retrieval | Delegate bounded specialized retrieval tasks where justified. | Compare against one agent, including extra cost, disagreement and fan-out. |
| R37 | research / P7.3 | RAPTOR hierarchy | Recursively cluster/summarize and retrieve at multiple levels. | Preserve leaf lineage; compare against a flat summary index. |
| R38 | graph / P7.2 | Entity-relation GraphRAG | Extract grounded entities/relations and retrieve graph neighborhoods. | Answer a relationship question impossible with adjacent-chunk expansion alone. |
| R39 | graph / P7.2 | Community/global GraphRAG | Build graph communities and summaries for corpus-level questions. | Trace summary claims to sources and measure index/update cost. |
| R40 | multimodal / P7.5 | Text-table-image RAG | Route extraction and use vision-aware answers with source links. | Test chart/table questions and generated-caption errors. |
| R41 | multimodal / P7.5 | Visual multivector retrieval | Use ColPali-compatible page/query encoders and late-interaction scoring. | Question-dependent real scores, page provenance, memory/latency comparison. |
| R42 | evaluation_cohort / P6.4 | Cross-language RAG | Search across Arabic/English and mixed-language questions/documents. | Report each language pair separately, including names/numbers/negation. |
| R43 | structured_retrieval / P8.3 | Constrained SQL retrieval | Retrieve factual aggregates through a restricted readonly datasource. | Test scoped views, disallowed functions, timeout and result bounds. |
| R44 | workflow / P8.2 | Explicit web fallback | Use public web retrieval only under an enabled privacy policy. | Label web sources, avoid private query leakage, handle conflicting freshness. |
| R45 | retrieval / P7.5 | Text late-interaction retrieval | Use a compatible ColBERT-style token multivector model/scorer. | Contrast single-vector and token matching with the same question set. |
| R46 | research / P7.5 | CRAG research reproduction | Reproduce the published correction/refinement approach using primary sources. | Document evaluator/refinement/fallback fidelity separately from R29. |
| R47 | knowledge_context / P7.4 | Cache-augmented generation (CAG) | Preload bounded versioned knowledge and reuse compatible context KV state. | Compare real cold/warm CAG, uncached full context and RAG; reject overflow; retain source citations. |
| R48 | knowledge_graph_reasoning / P7.3 | Knowledge-augmented generation (KAG) | Build schema-grounded knowledge with source links and logical-form-guided retrieval/reasoning. | Inspect grounded multi-hop operations, conflicts and missing facts; document OpenSPG fidelity and compare graph/dense baselines. |
| R49 | inference_optimization / P7.4 | Prefix KV caching | Reuse shared-token-prefix prefill computation in a compatible serving runtime. | Observe actual cache metrics or declare unavailable; measure cold/warm TTFT and eviction; distinguish decode from prefill. |
| R50 | response_cache / P7.4 | Exact-response caching | Reuse an answer and evidence bundle only for identical effective scoped versioned inputs. | Test hit, changed history/model/corpus/ACL miss, deleted-source denial and freshness bypass; avoid cached side effects. |
| R51 | response_cache / P7.4 | Semantic-response caching | Reuse eligible answers after partitioned similarity search and intent/evidence validation. | Calibrate held-out paraphrase and adversarial false-hit cases including negation, dates, entity IDs and cross-tenant requests. |

See [screenshot coverage and cache/knowledge labs](CACHE_AND_KNOWLEDGE_LABS.md) for exact distinctions and safeguards. R47/R48 require compatible context/graph backends; R49 is inference caching, while R50/R51 reuse responses. All remain planned.

## Dependency guide

- P3 establishes extraction/provenance, C05 and R01–R02 before large comparisons.
- C18 supplies hierarchy for R22/R25. C11–C12 supply layout/table context for R40.
- R06 uses independently verified branches; R07/R08 fuse them. R18 combines query expansion with R07.
- R19/R20 rerank candidates; they cannot recover evidence absent from the candidate pool.
- R29/R30 are explicit workflow adaptations; R31/R46 are separate faithful-reproduction commitments.
- C20, R41 and R45 require compatible embedding/scoring capabilities, not just a dimension change.
- R38 supplies entities/relations for R39; adjacent text chunk expansion does not satisfy either.
- R36 follows a working R33/R35 baseline; measure whether delegation helps.
- R42 is a cross-language experiment track across other techniques, not a new standalone algorithm.

## Extension procedure

A new technique is a plugin plus a catalogue entry ([ADR-0002](../decisions/ADR-0002-technique-plugin-system.md)); entries suggested by the Tech Scout start as `proposed` and need human approval. Add a unique ID, category, primary source, date added, roadmap step, prerequisites and observable acceptance criterion. State whether it is a new mechanism, a variant, an integration or a benchmark cohort. Never inflate completion counts by creating aliases.
