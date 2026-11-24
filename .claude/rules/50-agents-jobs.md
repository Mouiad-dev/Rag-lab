---
paths:
  - "packages/agents/**"
  - "packages/sandbox/**"
  - "packages/integrations/**"
  - "apps/worker/**"
---
# Agent and job runtime

- Name the owner of state, retries, cancellation, approvals and resource cleanup for every flow.
- Bound loops by steps, time, tokens and tool calls; detect repeated queries/no new evidence.
- Use Deep Agents for tasks benefiting from files/tools/planning; retain a simple fast Q&A path.
- Agent/subagent tools inherit or reduce authorization and budgets. Tool names/descriptions/results are untrusted external input.
- Keep long-running I/O in durable job steps (Temporal activities if Temporal is adopted); orchestration code must remain replay-safe. Use idempotent side effects and cleanup in failure/cancel paths.
- Use persistent LangGraph checkpoints for recovery features. In-memory demonstrations must be labeled as such.
- Persist essential task events and final output. Redis pub/sub is not durable job dispatch or approval storage.
- Avoid holding an expensive sandbox while waiting indefinitely for approval; define workspace persistence and resumption.
- Trace safe operation summaries, usage and evidence IDs. Never expose hidden reasoning as a user-facing progress feature.
