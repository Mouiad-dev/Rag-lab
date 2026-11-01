# RAG Foundry

An open-source, provider-agnostic lab for retrieval and knowledge techniques: RAG, GraphRAG and knowledge graphs, CAG/KAG, caching, agentic workflows, and whatever comes next. Ingest your documents, compare techniques side by side, inspect every intermediate step, and get answers with exact source citations. The product UI supports Arabic and English.

> **Status: planning.** This repository currently holds the architecture, roadmap, technique catalogue and development tooling. There is no runnable application yet.

## Goals

- **Any model provider.** Local (Ollama, LM Studio, vLLM) or paid APIs (Anthropic, OpenAI, Google, Mistral, Moonshot Kimi and others), chosen per pipeline stage. The local path needs no paid keys.
- **Techniques as plugins.** 72 catalogued techniques to start. A new technique is a plugin plus a catalogue entry, with no core changes.
- **Tech Scout.** A weekly in-app agent proposes new techniques from arXiv, GitHub and official blogs, with citations and a cost cap. A human approves each proposal.
- **Evidence first.** Source anchors, per-case evaluation and honest status for every technique.

## Quickstart

Not available yet. The target (roadmap step P10.2) is a single command along these lines:

```bash
docker compose --profile ollama up   # placeholder: no compose file exists yet
```

## Repository checks that work today

```bash
python scripts/verify_package.py
python -m unittest discover -s tests/hooks -v
```

## Documentation

| Topic | Link |
|---|---|
| Architecture summary | [docs/architecture/SUMMARY.md](docs/architecture/SUMMARY.md) |
| Full architecture | [docs/architecture/PROJECT_ARCH.md](docs/architecture/PROJECT_ARCH.md) |
| Roadmap (10 phases × 5 steps) | [docs/plan/ROADMAP.md](docs/plan/ROADMAP.md) |
| Progress | [docs/PROGRESS.md](docs/PROGRESS.md) |
| Technique catalogue | [docs/techniques/TECHNIQUE_CATALOG.md](docs/techniques/TECHNIQUE_CATALOG.md) |
| Decisions (ADRs) | [docs/decisions/](docs/decisions/) |
| Building with Claude Code agents | [docs/plan/AGENT_WORKFLOW.md](docs/plan/AGENT_WORKFLOW.md), [docs/HOOKS.md](docs/HOOKS.md) |
| Sources | [docs/SOURCES.md](docs/SOURCES.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).

## License

The repository currently contains an MIT [LICENSE](LICENSE). A switch to Apache-2.0 is proposed in [ADR-0004](docs/decisions/ADR-0004-open-source-license.md) and awaits the maintainer's decision.
