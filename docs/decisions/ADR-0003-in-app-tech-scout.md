# ADR-0003: In-app Tech Scout

Date: 2026-10-26
Status: accepted (implementation at roadmap P8.5; schedule and budget defaults need user confirmation)

## Context
Retrieval and knowledge techniques change quickly. Users want the app to notice new techniques weekly, compare them with the catalogue and roadmap, and propose additions, without an agent changing the plan or code on its own.

## Decision
1. **Where it runs.** A scheduled job in `apps/worker`, code in `packages/scout`. Default schedule weekly (Monday 06:00 UTC), configurable; manual "run now" and dry-run modes.
2. **Model.** Default profile `anthropic/claude-opus-5-5` (Claude Opus 5.5) through the provider layer (ADR-0001); any profile with `structured_output` can be selected, including local Ollama models. An optional cheaper triage profile can screen candidates before the main model.
3. **Pipeline.**
   - Fetch (deterministic, no LLM): arXiv API (for example cs.IR, cs.CL with retrieval/graph/RAG terms), GitHub search (topics and recent repositories above a star threshold), allowlisted official blogs and RSS feeds. Respect rate limits and terms of each source.
   - Normalize and **dedup**: canonical URL, arXiv ID, DOI, GitHub `owner/repo`, normalized title hash; compare against catalogue `primary_source`, earlier Scout findings and the rejected list. Embedding similarity to catalogue entries flags near-duplicates and variants.
   - Triage with structured output: category, mechanism summary, novelty vs nearest catalogue entries, maturity (paper only / code / production use), suggested roadmap phase, confidence.
   - Report: Markdown + JSON stored in the app database, with every claim tied to a source URL and retrieval date. No proposal without a source.
   - Proposals: new catalogue entries with `catalog_status: proposed` in the app database.
4. **Human approval.** A reviewer accepts, rejects (with reason, used for future dedup) or defers in the UI/API. Acceptance can export a JSON patch for `config/techniques.json` that a maintainer applies through a normal PR. The Scout never edits the repository, roadmap, catalogue file or code, and never opens PRs or issues.
5. **Cost caps.** Per-run hard caps on USD, tokens, fetched items and LLM-triaged items (defaults in `config/scout.example.json`). On reaching a cap the run stops and reports partial results. Usage goes to the provider usage ledger.
6. **Security.** Fetched content is untrusted: it is passed as data, never as instructions; the Scout has no tools beyond allowlisted fetchers; no code execution; outbound domains allowlisted; no private documents are sent in queries.

## Alternatives
- External scheduled agent (CI cron or a Claude Code routine) editing the repo: simpler to start but outside the product, needs repo write access, and does not help users of the app.
- Pure keyword RSS digest without an LLM: cheap and safe, but weak at novelty comparison. Kept as dry-run fallback when no model is configured.

## Consequences
- Requires the scheduler and durable jobs from P8.4.
- Default Opus usage has a real cost; local-model mode is available and the cap is enforced.
- Output quality depends on sources and model; proposals are suggestions, not validated techniques.

## Evidence
Check arXiv API and GitHub search API terms and rate limits at P8.5. Measure dedup precision on a hand-labelled set of past findings before enabling weekly runs by default.
