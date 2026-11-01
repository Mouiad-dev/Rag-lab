# ADR-0002: Technique plugin system

Date: 2026-10-26
Status: accepted (v0 at roadmap P1.5, v1 at P7.1)

## Context
The project is no longer RAG-only. It must cover RAG, GraphRAG and knowledge graphs, CAG/KAG, caching, agentic workflows and techniques that do not exist yet. The 72-entry catalogue must be kept. Adding a technique should not require core changes.

## Decision
1. **Plugin kinds** with typed protocols: `chunker`, `representation`, `retriever`, `fusion`, `query_transform`, `reranker`, `context_builder`, `workflow`, `graph_builder`, `knowledge_store`, `cache`, `evaluator`. New kinds are added by ADR.
2. **Plugin manifest** (class attributes or `plugin.toml`): `technique_id` (catalogue ID), `kind`, `version`, `fidelity` (`faithful` | `adaptation` | `simulation`), `required_capabilities` (from ADR-0001 flags and infra such as `pgvector`, `graph_store`), a Pydantic config schema, and optional dependency extras.
3. **Discovery.** Built-in plugins register in `packages/techniques`; third-party packages register through the Python entry-point group `rag_foundry.techniques`. The registry refuses duplicate IDs and IDs missing from the catalogue (unless marked `experimental` locally).
4. **Pipelines compose plugins** by ID. The resolver checks capabilities and compatible index types and fails with an explicit `unavailable` error instead of falling back silently.
5. **Catalogue schema v2** (`config/techniques.json`): adds `category`, `catalog_status` (`accepted` | `proposed` | `rejected` | `deprecated`), `primary_source`, `date_added`, `phase`, `roadmap_step`; keeps `status` (implementation state), `family`, `legacy_milestone`, acceptance and evidence fields. The JSON file is the repository's canonical catalogue; the app imports it into its database, where Tech Scout proposals live until approved.
6. **Adding a technique** = plugin + catalogue entry + lesson page from `docs/templates/TECHNIQUE.md` + contract tests. No core edits.

## Alternatives
- Hard-coded strategy enums in core: simple, but every technique touches core and third parties cannot extend.
- Fully dynamic, untyped plugins: flexible, but weak validation and poor error messages.
- One plugin per research paper regardless of kind: hides the fact that most techniques are components, not whole architectures.

## Consequences
- Core stays small; contributors ship techniques as packages.
- Plugin code runs in-process; third-party plugins are trusted code and the docs must say so. Untrusted code execution stays in the sandbox.
- The catalogue grows; the Markdown catalogue must stay in sync (checked by `scripts/verify_package.py`).

## Evidence
Python entry points (`importlib.metadata`) are a standard mechanism. Validate the exact plugin API with two built-in plugins (C05, R02) at P3.4 before freezing v1 at P7.1.
