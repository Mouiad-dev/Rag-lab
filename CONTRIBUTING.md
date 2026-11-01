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

## Proposing a technique

Use the "Technique proposal" issue template. Include a primary source (paper, official repository or docs), the mechanism, the category, how it differs from existing catalogue entries, and an observable acceptance check. Accepted techniques are added to `config/techniques.json` and `docs/techniques/TECHNIQUE_CATALOG.md` together. Implementations are plugins ([ADR-0002](docs/decisions/ADR-0002-technique-plugin-system.md)).

## Providers

Model access goes through `packages/providers` only ([ADR-0001](docs/decisions/ADR-0001-provider-agnostic-layer.md)). Features must keep working on the local, zero-key path. If a feature strictly needs a paid capability, it must fail with a clear message on the local path.

## Using Claude Code

Maintainers use a planner → coder → reviewer agent loop described in [docs/plan/AGENT_WORKFLOW.md](docs/plan/AGENT_WORKFLOW.md). You do not need Claude Code to contribute.
