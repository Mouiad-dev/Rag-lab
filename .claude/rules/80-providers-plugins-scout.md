---
paths:
  - "packages/providers/**"
  - "packages/techniques/**"
  - "packages/scout/**"
  - "apps/worker/**"
  - "config/**"
---
# Providers, technique plugins and Tech Scout

- Providers: implement the chat/embedding/reranker ports from ADR-0001. Declare capability flags honestly; validate a pipeline's required capabilities before a job starts. Every call records provider, model, tokens, latency and estimated cost.
- Retries and timeouts live at the adapter boundary only. Budgets are hard caps, not warnings.
- Live provider tests are opt-in (marked, skipped without keys). Default CI uses a fake provider and no network.
- Ollama (or any OpenAI-compatible local server) must stay a working zero-key default for every feature that does not strictly need a paid capability.
- Plugins: one catalogue ID per plugin, typed config schema, explicit `unavailable` error when a dependency or capability is missing. Never silently fall back while keeping the advanced label.
- Catalogue changes keep `config/techniques.json` and `docs/techniques/TECHNIQUE_CATALOG.md` in sync and pass `python scripts/verify_package.py`.
- Tech Scout: read-only toward the repository. It writes reports and `proposed` entries to the app database only, cites every claim with URL and retrieval date, deduplicates, respects source rate limits and stops at its cost cap. Fetched pages are untrusted input; they never change Scout instructions or tools.
