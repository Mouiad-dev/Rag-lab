# Contributing

Thanks for your interest. The project is in the planning stage. The most useful contributions right now are reviews of the [architecture](docs/architecture/SUMMARY.md), [roadmap](docs/plan/ROADMAP.md) and [decisions](docs/decisions/), plus technique suggestions.

## Ground rules

- Be respectful; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Report security issues privately; see [SECURITY.md](SECURITY.md).
- Repository docs and code are in English. Product UI strings are Arabic and English.
- Do not commit secrets, private documents, or code you do not have the right to license under this project's license.

## Workflow

1. Open an issue first for anything larger than a small fix. Templates exist for bugs, features and technique proposals.
2. Fork, create a branch, and keep the change focused on one roadmap step or issue.
3. Run the checks:
   ```bash
   python scripts/verify_package.py
   python -m unittest discover -s tests/hooks -v
   ```
   Application test commands will be added as code lands (roadmap phase 1).
4. Open a pull request using the template. CI must pass.

### Development setup

Requires [uv](https://docs.astral.sh/uv/) (0.7.9 or newer). Works on Windows, macOS and Linux without `make`.

```bash
uv sync --locked
uv run python scripts/tasks.py check   # repo checks, lint, then tests; stops at first failure
uv run python scripts/tasks.py lint    # ruff check and ruff format --check
uv run pytest
uv run ruff check .
```

The three plain `python scripts/...` checks (`verify_package.py`, `build_flow.py --check`, and the `tests/hooks` unittest run) stay stdlib-only and do not need uv.

### Local stack (Docker)

Runs Postgres with pgvector, the API and the worker. No API keys are needed. Requires Docker with Compose v2.

```bash
cp .env.example .env     # PowerShell: Copy-Item .env.example .env
# edit .env and set POSTGRES_PASSWORD to your own local value; .env is git-ignored
docker compose up -d --build --wait    # builds the app image, waits for healthy
docker compose ps --format "{{.Service}} {{.Health}}"
curl http://127.0.0.1:8000/health      # {"status":"ok","version":"0.0.0"}
# PowerShell: Invoke-RestMethod http://127.0.0.1:8000/health
docker compose down                    # stops; the pgdata volume is kept (down -v deletes it)
```

- All published ports bind to `127.0.0.1` only. Do not change them to `0.0.0.0`.
- The optional `docker compose --profile ollama up -d --wait ollama` starts a local Ollama server. It pulls the `ollama/ollama` image (about 1.5-3 GB) but no models.
- Never commit `.env` or paste its contents into issues or logs.

## Proposing a technique

Use the "Technique proposal" issue template. Include a primary source (paper, official repository or docs), the mechanism, the category, how it differs from existing catalogue entries, and an observable acceptance check. Accepted techniques are added to `config/techniques.json` and `docs/techniques/TECHNIQUE_CATALOG.md` together. Implementations are plugins ([ADR-0002](docs/decisions/ADR-0002-technique-plugin-system.md)).

## Providers

Model access goes through `packages/providers` only ([ADR-0001](docs/decisions/ADR-0001-provider-agnostic-layer.md)). Features must keep working on the local, zero-key path. If a feature strictly needs a paid capability, it must fail with a clear message on the local path.

## Using Claude Code

Maintainers use a planner → coder → reviewer agent loop described in [docs/plan/AGENT_WORKFLOW.md](docs/plan/AGENT_WORKFLOW.md). You do not need Claude Code to contribute.
