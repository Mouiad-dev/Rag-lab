# Source map and verification policy

Prepared 2026-10-14. Links below are primary implementation references, not frozen API guarantees. Recheck the relevant API against installed versions before implementation and record that version in the technique page.

## Project context

Early scope was informed by private planning notes and third-party educational material that are not part of this repository. Their claims and sample code were treated as unverified; the official documentation and papers below are the implementation references.

## Primary implementation references

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [Text splitter integrations](https://docs.langchain.com/oss/python/integrations/splitters/index)
- [Retrieval concepts](https://docs.langchain.com/oss/python/deepagents/retrieval)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agents sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- [LangChain MCP integration](https://docs.langchain.com/oss/python/langchain/mcp)
- [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)
- [Temporal workflow model](https://docs.temporal.io/workflows)
- [GraphRAG indexing](https://microsoft.github.io/graphrag/index/overview/)
- [ColPali explanation by a paper author](https://huggingface.co/blog/manu/colpali)
- [Late Chunking paper](https://arxiv.org/abs/2409.04701)
- [Self-RAG paper](https://arxiv.org/abs/2310.11511)
- [RAPTOR paper](https://arxiv.org/abs/2401.18059)

## Claude Code configuration references checked for this package

- [Project memory and path-scoped rules](https://code.claude.com/docs/en/memory)
- [Hooks events, stdin/stdout and decision formats](https://code.claude.com/docs/en/hooks)
- [Settings file locations](https://code.claude.com/docs/en/settings)

Our hook scripts are original implementations using the documented event contract. They were exercised with synthetic hook inputs, not inside an installed Claude Code runtime. See `HOOKS.md` and `VALIDATION.md` for the exact scope of verification.

## Additions checked 2026-10-26

- [CAG authors repository](https://github.com/hhhuang/CAG) and [CAG paper](https://arxiv.org/abs/2412.15605): bounded context and compatible knowledge KV reuse; not saved answer caching.
- [OpenSPG/KAG](https://github.com/OpenSPG/KAG): schema construction, knowledge/chunk mutual links and logical-form-guided reasoning/retrieval; fidelity must be documented.
- [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/): shared-prefix prefill reuse; runtime capabilities must be verified against installed versions.
- [W3C Web Annotation Data Model](https://www.w3.org/TR/annotation-model/): source selectors and resource version/state considerations. The project JSON is a custom contract inspired by these concepts, not a claim of full W3C serialization compliance.

The four user-provided screenshots supply topic names. Performance claims such as instant CAG answers or universally better KAG quality are not accepted without project-specific evidence. Independence, cache safeguards, export and media ownership are original project design decisions.
