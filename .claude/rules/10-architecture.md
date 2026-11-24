---
paths:
  - "apps/**/*.py"
  - "packages/**/*.py"
  - "config/**"
---
# Architecture and contracts

- Use FastAPI at the edge and domain/services for behavior. Heavy ingestion/embedding runs in a worker.
- SQLAlchemy owns application persistence; Alembic owns its migrations. Keep LangGraph framework tables separately managed.
- A fresh async database session belongs to each request/task unit. Do not share one AsyncSession between concurrent tasks.
- Inject model/storage/tool adapters. Model calls go only through `packages/providers` ports; never import a vendor SDK (anthropic, openai, litellm, ...) outside provider adapters.
- Techniques are plugins registered by catalogue ID (`packages/techniques`); do not add technique-specific branches to core code.
- Validate config strictly and fail explicitly on unknown or incompatible components. Never silently label a fallback as the selected technique.
- Pin corpus/index/pipeline/model versions in runs. Same vector dimension does not prove embedding-space compatibility.
- Store large blobs in object storage; put scoped metadata/references in relational records and checkpoints.
- Keep deterministic transforms deterministic. LangGraph owns AI workflow decisions; the worker's pluggable durable job engine (PostgreSQL default, Temporal optional; ADR-0005) owns long job lifecycle.
- Changes to retry ownership, schema lineage, tenant scope or paper fidelity require a short ADR.

- Independent-product and public-contract requirements in `70-reuse-contracts.md` apply to all boundaries; no consumer-specific domain fields in the RAG core.
