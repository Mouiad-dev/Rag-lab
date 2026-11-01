# rag-foundry-contracts

Framework-neutral, versioned public contracts for RAG Foundry. Contract version `1.0`.

- Depends only on `pydantic`. No ORM, FastAPI, LangChain/LangGraph, vendor SDK or lab UI dependency, and no import of `rag_foundry_core`.
- Currently contains the source reference and source anchor models (`SourceRef`, `TextRange`, `PageRegion`, `Precision`, `SourceAnchor`).
- Anchors are owned by immutable document/extraction versions. They carry no chunk or index ID.
- The JSON Schema is committed in `schemas/` and a test fails if it drifts. Regenerate with
  `uv run python -m rag_foundry_contracts.export packages/contracts/schemas`.

## v1 rules

- Within v1, only additive optional fields are allowed. Breaking meaning or required-field changes need v2.
- v1 models reject unknown fields (`extra="forbid"`). Consumer adapters may relax this later (REUSE_AND_INTEGRATION section 6).
- `check_contract_version` accepts `1.x` and raises `UnsupportedContractVersionError` for other majors or malformed strings.
- Text ranges are zero-based, half-open Unicode code-point ranges in canonical block text, not UTF-8 bytes or UTF-16 units.
- The JSON is informed by the W3C Web Annotation Data Model but is not claimed to be conformant.
