# Independent products and portable source provenance

- RAG Lab must build, start and test without book2video or any other consumer repository. Consumer integration is optional and replaceable.
- Do not import consumer code, share databases/migrations, add course/video/TTS fields to core RAG entities, or require sibling checkout paths.
- Read `docs/architecture/REUSE_AND_INTEGRATION.md` and `docs/architecture/SOURCE_ANCHORS_AND_MEDIA.md` for public API, source bundle or provenance changes.
- Public contracts are framework-neutral versioned JSON; SDKs do not depend on server ORM, Lang packages or lab UI.
- Stable source anchors are owned by immutable document/extraction versions, not chunk/index IDs. Rechunking must preserve external links.
- Keep document structure distinct from retrieval chunks and downstream lesson boundaries.
- Consumer code owns script-to-source and final-media timing maps. Never invent timestamps, region coordinates or alignment confidence.
- Enrichment sources stay separate from original text. Source-only mode must not make web calls.
- CAG, KAG and caching are separate mechanisms. Read `docs/techniques/CACHE_AND_KNOWLEDGE_LABS.md`; unavailable implementations fail explicitly.
- Cache hits revalidate scope, source versions and freshness. Similarity alone is not answer equivalence.
- Integration and caching changes require contract/version/isolation evidence appropriate to the change. Repository checks do not prove unimplemented services work.
