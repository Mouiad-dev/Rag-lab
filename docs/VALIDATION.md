# Repository validation record

## Version 2.0 restructure, executed 2026-10-26

Environment: Windows 11, Git Bash, Python 3.14.0 (`python`).

- `python scripts/verify_package.py`: PASS. Required files; Python/JSON syntax; 72 catalogue entries in schema v2 with valid category, status, catalog_status, date_added, primary_source and roadmap_step, and Markdown rows in sync; roadmap has 10 phases × 5 steps and PROGRESS tracks the same 50 steps; hook example command shapes; 9 rule files; 4 subagent definitions with expected models; CLAUDE.md under 60 lines; SUMMARY.md under 150 lines; `.gitignore` entries; local Markdown links; no private/proprietary references in public files.
- Negative check: a temporary Markdown file with a broken link and a private marker made the script fail with both errors; the file was removed.
- `python -m unittest discover -s tests/hooks -v`: PASS, 21 tests, 0 skipped (symlink tests ran on this machine). The orientation test now also asserts the 72-entry text.
- Hooks were not activated in a live Claude Code session; `.claude/settings.example.json` is not loaded by Claude Code.
- No product features, model, provider or Tech Scout code exist yet (only the P1 skeleton: settings, /health, worker, compose stack). CI workflow (`.github/workflows/ci.yml`) first ran on [PR #1](https://github.com/Mouiad-dev/Rag-lab/pull/1), [run 41307162676](https://github.com/Mouiad-dev/Rag-lab/actions/runs/41307162676) (commit 3cd2e6e): all three jobs green; 75 passed, 0 skipped on ubuntu and on windows; the image job's stack was healthy. Jobs: `check` (uv sync --locked and `scripts/tasks.py check` on ubuntu and windows, plus a compose config check on ubuntu) and `image` (ubuntu: compose build, health check, teardown).

## Earlier versions (summary)

- v1.1 (2026-10-26): verify script and 21 hook tests passed on Linux with `python3`; R47–R51 appended to the original 67 entries; hooks unchanged from v1.0.
- v1.0 (2026-10-14): 21 hook tests passed; configured hook commands smoke-tested through Bash in a path with spaces; no Windows or live Claude Code test.
