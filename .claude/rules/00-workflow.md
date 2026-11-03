# Workflow (always loaded; keep short)

- Open source, English repo docs and code. Chat explanations follow the user's language; the product UI is bilingual Arabic/English.
- Work in small scoped tasks from `docs/plan/ROADMAP.md` using the planner → coder → reviewer loop in `docs/plan/AGENT_WORKFLOW.md`. One task spec per task in `docs/tasks/`.
- Token budget: read `docs/architecture/SUMMARY.md`, not the full `PROJECT_ARCH.md`; open only cited sections. Coders read only the files a spec lists. Reviewers read the spec and the diff. Max two review rounds, then escalate to the user.
- Preserve every catalogue entry; scope is controlled by phases, not by deleting techniques.
- Inspect current state before proposing changes. Distinguish planned, implemented, tested and benchmarked; never invent runs, counts or benchmarks.
- Proceed with authorized reversible work without per-file confirmation. Ask before paid model calls without an agreed budget, large downloads, external messages, publishing or destructive actions.
- Keep changes focused: no opportunistic upgrades or unrelated refactors. Verify changing APIs against official docs and lock tested versions.
- Checks proportional to risk. Retrieval semantics, scope isolation, retries, approvals, provider adapters and hooks need meaningful tests; doc typos do not.
- Update `docs/PROGRESS.md` with exact commands/results after each step.
