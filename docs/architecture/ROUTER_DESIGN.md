# Router design

Status: proposed ([ADR-0007](../decisions/ADR-0007-router-architecture.md)). Interactive view: [docs/flow/index.html](../flow/index.html). The machine-readable stage and router map is `config/pipeline_stages.json`.

## 1. The question: one router or many?

**Many small routers, one shared contract, one plan.**

- **One router per decision point.** Each router makes one decision and nothing else. There are 8 routers: 3 that run at ingestion time and 5 that run at query time.
- **All routers implement the same `Router` interface.** Each one returns a `RouteDecision` with the same fields.
- **A `PipelineOrchestrator` combines the decisions.**
  - For a document, it collects the decisions into one `IngestPlan`.
  - For a question, it collects them into one `QueryPlan`.
  - Each plan uses the same schema as `config/experiment.example.json`, so any routed run can be replayed exactly as a fixed experiment.

**Why not one big router?** A single router would need to predict a combination of roughly 20 extractors × 20 chunkers × several embedders × 7 query transforms × several retrievers × several rerankers. That is thousands of combinations, and we will never have enough labelled data to learn them. Separate routers can each be measured, improved and replaced on their own. The orchestrator still adds a check across stages, for example "you cannot route to a graph index that was never built".

**Why not fully independent routers?** Their decisions depend on each other in two ways:
- **Ingestion limits query time.** The query side can only route to indexes that `RT-INDEX` decided to build.
- **Later stages can send work back.** `RT-VERIFY` can send a question back to `RT-QUERY` with a different route, at most once by default.

## 2. The routers

| Router | Time | Unit | Decides | Safe default (L0) |
|---|---|---|---|---|
| RT-EXTRACT | ingest | file / page | text layer, layout, OCR, vision LLM | text layer; OCR if text coverage < 20% |
| RT-CHUNK | ingest | document region | C01–C21 policy plus size and overlap | C05 recursive, 400 tokens, overlap 60 |
| RT-INDEX | ingest | collection | which embedder and indexes to build (dense, sparse, FTS, graph, RAPTOR, CAG preload) | bge-m3 dense + PostgreSQL FTS |
| RT-QUERY | query | question | cache / no retrieval / docs / graph / SQL / web / CAG / agentic, and the generator model | document RAG (hybrid) |
| RT-TRANSFORM | query | question | none / rewrite / multi-query / decompose / step-back / HyDE / RAG-fusion | none |
| RT-RETRIEVE | query | question | retrievers, k per branch, filters, fusion | hybrid dense + FTS, k=20, RRF |
| RT-CONTEXT | query | candidate set | rerank (which), expand, compress, final k | no rerank, final_k=5 |
| RT-VERIFY | query | draft answer | accept / retry with another route / abstain | accept if citations resolve, else 1 retry, then abstain |

The feature list for each router is in `config/pipeline_stages.json`.

```
Ingestion:  file → RT-EXTRACT → normalise/anchor → RT-CHUNK → RT-INDEX → indexes (+ graph, RAPTOR, CAG)
                                                                              │ available indexes
Query:  question → cache? → RT-QUERY ─→ RT-TRANSFORM → RT-RETRIEVE → RT-CONTEXT → generate → RT-VERIFY
                                ▲                                                              │
                                └──────────── retry with another route (bounded) ──────────────┘
Always: every decision → trace span + run record → evaluation → better router policies
```

## 3. The router contract

```python
class RouteDecision(BaseModel):
    router_id: str            # "RT-TRANSFORM"
    router_version: str       # policy version, pinned in the run
    policy_level: Literal["L0", "L1", "L2", "L3", "L4"]
    choice: str               # catalogue ID ("R17") or component ID
    params: dict              # e.g. {"k": 20}
    confidence: float         # 0..1, calibrated
    reason: str               # short human-readable explanation
    features: dict            # inputs used (redacted, no raw content)
    alternatives: list[tuple[str, float]]  # runner-up choices and scores
    fallback_used: bool       # true if confidence < threshold and L0 was applied
    cost: Usage               # tokens / USD / ms spent by the router itself

class Router(Protocol):
    id: str
    def decide(self, ctx: RouteContext) -> RouteDecision: ...
```

Rules:
- **A router only chooses among plugins that are available and compatible.** The candidate list comes from the technique registry plus the provider capability flags, so a router never selects something that will fail.
- **Low confidence falls back to the safe default.** If confidence is below the router's threshold, the decision falls back to L0 and the decision is recorded as a fallback. It is never labelled as the advanced technique.
- **Users can lock any stage.** A user or experiment can force a stage, for example "always use HyDE". The router then records `forced` and is skipped.
- **Router cost counts separately.** Router cost and latency are counted apart from the pipeline (see `.claude/rules/30-evaluation.md`).

## 4. Policy levels (how each router gets smarter)

| Level | Policy | Notes |
|---|---|---|
| L0 | Fixed default | Always exists; the fallback |
| L1 | Rules over cheap features | First version of every router; explainable; zero model cost |
| L2 | Learned classifier | Small model (for example gradient boosting or logistic regression) trained on oracle labels; runs locally in milliseconds |
| L3 | LLM router | Structured-output call to a small local model, only for low-confidence L2 cases |
| L4 | Online bandit | 5–10% exploration with feedback; only after offline gates pass, with a per-tenant opt-in |

A router moves up a level only when the new level beats the current one on the held-out set (gate in §5.4).

## 5. Measuring whether the router is correct

### 5.1 Build "oracle" labels
1. Freeze a dev set: documents plus questions with gold evidence. It must include Arabic, English, mixed-language, table, OCR and unanswerable cases.
2. For each decision point, run every candidate option. Hold the other stages fixed at their defaults, changing one factor at a time.
3. Score each option with that stage's metric (below). The **acceptable set** is every option within ε (for example 2%) of the best score. A choice is "correct" if it lands in the acceptable set, because options often tie.
4. Use only cheap local models (Ollama) for grid runs. Run them in the worker, under a cost cap.

### 5.2 Router metrics

| Metric | Meaning |
|---|---|
| Route accuracy | Share of decisions inside the acceptable set |
| Regret | Oracle score minus routed score, averaged (lower is better) |
| Win vs best fixed | Routed pipeline vs the single best fixed pipeline on the same set |
| Cost / latency overhead | What the router itself costs (tokens, ms) |
| Calibration | Does 0.8 confidence mean right about 80% of the time (ECE) |
| Fallback rate | How often it falls back to L0 |
| Per-cohort breakdown | Arabic / English / mixed / tables / OCR / unanswerable, reported separately |

### 5.3 Stage metrics (what "best" means at each decision)

| Router | Primary metric | Secondary |
|---|---|---|
| RT-EXTRACT | Character error rate and table-cell accuracy vs gold pages | cost per page |
| RT-CHUNK | Downstream Recall@k with the default retriever | chunk count, boundary quality |
| RT-INDEX | Dev-set nDCG@k gain per index | build time, storage |
| RT-QUERY | End-to-end faithfulness × answer relevance; correct abstention | latency, cost |
| RT-TRANSFORM | Recall@k gain over "none" | extra latency and tokens, meaning drift |
| RT-RETRIEVE | Recall@k, nDCG@k | latency per branch |
| RT-CONTEXT | nDCG@5 and evidence-in-context recall | added p95 latency |
| RT-VERIFY | Final faithfulness and citation accuracy | retries per answer, abstention precision |

### 5.4 Promotion gate (per router)
A new router version ships only if, on the **held-out** set:
- it beats the best fixed pipeline, or matches it at lower cost;
- regret does not increase in any cohort;
- it causes no security or scope failures (these fail the release on their own);
- its own cost stays within its budget.

If it fails the gate, the router stays at its previous level. A router that never beats the fixed default is fine: the result is recorded and the default is kept.

### 5.5 Online checks
- **Shadow mode:** the new router decides but does not act. Its decisions are logged next to the live ones.
- **Traces:** every decision is an OpenTelemetry span `router.decision` (and appears in LangSmith when enabled), carrying the fields from §3.
- **Feedback:** user feedback (thumbs, corrections) and sampled LLM-judge scores become new labelled examples. Judges are calibrated against human review first.
- **Drift alerts:** an alert fires when the fallback rate or regret on a weekly re-run moves beyond a threshold.

## 6. Where it fits in the roadmap

The 10×5 structure is unchanged; router work is added inside existing steps.

| Step | Router work |
|---|---|
| P1.5 (plugin contract) | `Router` interface, `RouteDecision`, L0 defaults, `PipelineOrchestrator` producing plans |
| P3.2 (extraction) | RT-EXTRACT L1 rules |
| P3.5 (first vertical slice) | Plans recorded in runs; every decision traced |
| P4.x | RT-CHUNK L1 rules; oracle grid for chunkers |
| P5.x | RT-TRANSFORM, RT-RETRIEVE, RT-CONTEXT L1 rules plus oracle grids |
| P6.5 (evaluation) | Router metrics, acceptable sets, promotion gate, held-out reports |
| P7.x | RT-INDEX gains graph, RAPTOR and CAG options |
| P8.1 (R32 adaptive routing) | RT-QUERY and RT-VERIFY; L2 learned classifiers; L3 LLM fallback |
| P9.x | UI: route plan view, per-stage "why", force/lock controls |
| P10.x | L4 bandit (opt-in), drift alerts, published router benchmark |

## 7. Open questions
- ε for the acceptable set (proposed 2% relative).
- Minimum labelled examples before L2 (proposed 300 per router, across cohorts).
- Whether RT-INDEX may rebuild indexes automatically, or only propose rebuilds (proposed: propose only).
