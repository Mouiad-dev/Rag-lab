# ADR-0007: Router architecture (many stage routers, one plan)

Date: 2026-11-01
Status: proposed (the maintainer decides)

## Context
Every stage of the pipeline has several techniques: extraction tools, 20 chunkers, several embedders and indexes, 7 query transforms, retrievers, fusion methods, rerankers, context builders and verification loops. The app should choose the right one per document and per question, and it must prove the choice is better than a fixed pipeline.

## Options
1. **One global router.** It predicts the whole pipeline at once. There are thousands of combinations, it cannot be trained with realistic data, and its failures are hard to explain.
2. **Independent stage routers.** Simple, but they ignore dependencies: a query cannot use an index that was never built, and a failed verification needs a retry route.
3. **Many stage routers behind one contract, combined by an orchestrator into one plan.**

## Decision
Option 3.
- **Eight routers:** RT-EXTRACT, RT-CHUNK and RT-INDEX run at ingestion; RT-QUERY, RT-TRANSFORM, RT-RETRIEVE, RT-CONTEXT and RT-VERIFY run at query time.
- **One shared contract:** every router implements one `Router` protocol and returns a traced `RouteDecision`.
- **One plan per run:** a `PipelineOrchestrator` validates compatibility and emits one replayable `IngestPlan` or `QueryPlan`.
- **Policies grow by level:** fixed default (L0), rules (L1), learned classifier (L2), LLM fallback (L3), online bandit (L4).
- **Promotion is measured:** a router moves up a level only after passing a held-out promotion gate, with oracle acceptable sets, regret and win against the best fixed pipeline.

Full design: [ROUTER_DESIGN.md](../architecture/ROUTER_DESIGN.md).

## Consequences
- R32 (adaptive routing) becomes the RT-QUERY router, built on the shared contract instead of a one-off.
- Each router is itself a plugin with a version that is pinned in run records. Experiments can lock any stage.
- Evaluation must produce per-stage oracle grids. This adds compute, so grid runs use local models under the worker's cost caps.
- Routers never choose unavailable or incompatible plugins. Low confidence falls back to L0 and is recorded as a fallback, never as the advanced technique.
