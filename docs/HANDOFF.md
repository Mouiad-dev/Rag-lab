# Session handoff (2026-11-09)

Read this first, then `CLAUDE.md`, `docs/PROGRESS.md` (Current state, the P5.x rows and the newest History entries) and `docs/architecture/SUMMARY.md`.

## Where things are

- Branch `p1/toolchain-and-skeleton`, last commit `33d359e`. Nothing is pushed (pushing needs the owner's OK).
- Last full check on `33d359e`: `uv run python scripts/tasks.py check` exit 0, 2089 passed, 71 skipped.
- Phase 4 is done. Phase 5:
  - P5.1 done: R03 BM25, R04 PostgreSQL FTS. R05 code reviewed and committed 2026-11-09; catalogue/lesson/evidence still open.
  - P5.2 done: R06 hybrid, R07 RRF, R08 weighted fusion, paired report `docs/evidence/P5.2/paired_vs_dense.md`.
  - P5.3 done: R10 dedup and R11 metadata filters done; R09 MMR done 2026-11-09; R12 self-query done 2026-11-09 (live run passed); P5.3-05 docs done 2026-11-09. **P5.3 done.** R05 catalogue/lesson/evidence done 2026-11-09 (P5.1-05). P5.4 planned (P5.4-01..05); P5.4-01 R13, P5.4-02 R14/R18 and P5.4-03 R15/R16 and P5.4-04 R17 done; live smoke `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s` needs owner OK; next P5.4-05 docs.
  - P5.4, P5.5 not planned yet.

## R05 learned sparse (committed 2026-11-09; this section is history)

Spec: `docs/tasks/P5.1-03-r05-learned-sparse.md` (Result section filled by the coder).

- Owner approved on 2026-11-08: optional extra `rag-foundry-providers[sparse]` = `sentence-transformers==6.2.0` + CPU `torch==2.15.0` from the explicit `pytorch-cpu` uv index. Already committed and locked in `4bf26f3`.
- Model `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` (Apache-2.0, 160M params, 684 MB) downloaded at revision `1e0f096c2b51c234f1d20725c793e1b5b6d556db` into `var/models/opensearch-neural-sparse-encoding-multilingual-v1` (git-ignored; a fresh machine must download it again: same `huggingface_hub.snapshot_download` call with that revision). Queries are inference-free (tokenizer + static weight table); only documents run the transformer.
- Uncommitted files: ADR-0017, migration `0009_chunk_sparse_vectors`, `rag_foundry_indexing/sparse.py`, `rag_foundry_providers/sentence_transformers_sparse.py`, provider `messages.py`/`ports.py`, `sql/index_repository.py`, `errors.py`, registrations, tests `tests/indexing/test_sparse*.py`, `tests/indexing/_vocab_sparse_encoder.py`, `tests/live/test_sparse_live.py`.
- Coder reports: `tests/indexing tests/providers` 919 passed, 3 skipped; ruff clean; live test 1 passed with `RAG_LIVE_SPARSE=1` and `RAG_SPARSE_MODEL_DIR` set. **Not done:** reviewer pass, `-m db` run (migration 0009 + sparse store), full `tasks.py check`, catalogue/lesson/evidence, commit.

Next steps for R05: run the reviewer subagent on the spec + diff; run `-m db` (below); run `tasks.py check`; commit; then mark R05 `implemented` with a lesson and evidence (or fold that into P5.3-05).

## How work is done here (owner's choices)

- Loop per roadmap step: `planner` subagent writes specs in `docs/tasks/` -> coder -> `reviewer` subagent (max two rounds) -> orchestrator runs checks, updates `docs/PROGRESS.md` (exact commands and results only), commits. Commit after each reviewed step; continue to the next step without asking; tell the owner when a whole phase is done and then publish an A1-English interactive phase report page.
- Use Agent-tool subagents, not separate app "teammate" sessions.
- Coder = Codex CLI: `codex exec -m gpt-6-luna -s workspace-write -C 'C:\Dev\import\Rag-lab' "<prompt>" < /dev/null > <log> 2>&1` run in the background. Without `< /dev/null` it waits forever on stdin.
- Watchdog: while Codex runs, keep a Monitor on its log (stalled > 10 min, running > 40 min, waiting on stdin). The script lived in the old session's scratchpad; recreate it if wanted (check log mtime and a final "tokens used" line).
- Still ask the owner before: paid calls, downloads, new dependencies, pushes, live runs. Migrations are fine without asking.

## Known traps (check every coder result for these)

- Codex writes Arabic as mojibake (`\u00d8`/`\u00d9` escapes) and sometimes raw Arabic. Project rule: Arabic in Python only as real `\u06xx` escapes. Grep new files; a Python check `any(ord(c) > 127 ...)` must be false. The planner did it once too.
- The shell and the Edit/Write tools turn `\u` escapes into literal characters. Write escapes from a Python script using `chr(92)`, then verify.
- Codex leaves new error codes unregistered (`IndexingError.CODES` etc.), writes too few tests, and writes tests that cannot fail. Reviewers caught this every step; mutation-check key tests.
- Codex's sandbox cannot run `-m db`, LocalObjectStore tests or uv-synced pytest reliably; it uses `uv run --no-sync`. The orchestrator runs those.
- `-m db` tests: `POSTGRES_TEST_PASSWORD=<random, shell only>`; `docker compose --profile test up -d --wait testdb`; `RAGF_TEST_DATABASE_URL=postgresql+psycopg://rag_foundry:$POSTGRES_TEST_PASSWORD@127.0.0.1:55432/rag_foundry uv run pytest -m db -q`; always `docker compose --profile test rm -sf testdb` after. Set `PYTHONIOENCODING=utf-8` on Windows when output has Arabic.

## Waiting for the owner

- From earlier phases: P3.7 live answer run, P4.4 live model-chunker smoke test, P2.5 live Anthropic run, P3.1 full stack rebuild check, a push to see CI, native-speaker review of the Arabic fixtures, revoking the Anthropic key from the P2.3 run, the license decision (ADR-0004).
- A folder Codex left with locked permissions was moved out of the repo; the owner should delete it: `C:\Users\mouia\AppData\Local\Temp\claude\C--Users-mouia--claude-spares-spare-O9DxMZ\d4832678-d2e3-4144-b1f4-5afa2adebbef\scratchpad\codex-locked-tmp`.
- Unmerged branch `claude/lucid-spence-b1c98c` (an older teammate's P4.5) is superseded: its useful parts were ported (P4.5-05..07). It can be deleted with the owner's OK.
