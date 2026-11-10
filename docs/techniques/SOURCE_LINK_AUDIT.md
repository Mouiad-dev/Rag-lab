# Source link audit

Date: 2026-10-26. Every `primary_source` in `config/techniques.json` was checked: arXiv IDs against the arXiv API (title, first author, date), web links by HTTP status and page title.

| ID | Technique | Source | Verified title / page | Result |
|---|---|---|---|---|
| C17 | Propositions | arXiv 2312.06648 | Dense X Retrieval: What Retrieval Granularity Should We Use? (Chen et al., 2023) | correct |
| C20 | Late chunking | arXiv 2409.04701 | Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (Günther et al., 2024) | correct |
| R02 | Dense two-step RAG | arXiv 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) | correct; DPR (2004.04906) is the companion dense-retriever paper |
| R05 | Learned sparse retrieval | arXiv 2107.05720 | SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al., 2021) | correct; SPLADE v2 (2109.10086) is the more used version |
| R09 | MMR selection | doi.org/10.1145/290941.291025 | The use of MMR, diversity-based reranking for reordering documents and producing summaries (Carbonell and Goldstein, SIGIR 1998) | correct; checked 2026-11-09 via Crossref metadata; doi.org redirects to dl.acm.org (HTTP 302), ACM page returns 403 to scripts |
| R16 | Step-back retrieval | arXiv 2310.06117 | Take a Step Back: Evoking Reasoning via Abstraction in LLMs (Zheng et al., 2023) | correct |
| R17 | HyDE | arXiv 2212.10496 | Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022) | correct |
| R18 | RAG fusion | arXiv 2402.03367 | RAG-Fusion: a New Take on Retrieval-Augmented Generation (Rackauckas, 2024) | correct |
| R19 | Cross-encoder reranking | arXiv 1901.04085 | Passage Re-ranking with BERT (Nogueira and Cho, 2019) | correct; checked 2026-11-09 via the arXiv API |
| R20 | LLM reranking | arXiv 2304.09542 | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (Sun et al., 2023) | correct; checked 2026-11-09 via the arXiv API |
| R24 | Contextual enrichment | anthropic.com | Contextual Retrieval (Anthropic) | **fixed**: old `/news/` URL redirected; now `/research/contextual-retrieval` |
| R31 | Faithful Self-RAG | arXiv 2310.11511 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023) | correct |
| R37 | RAPTOR | arXiv 2401.18059 | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024) | correct |
| R38 | Entity-relation GraphRAG | microsoft.github.io/graphrag | Microsoft GraphRAG docs | correct (HTTP 200) |
| R39 | Community/global GraphRAG | arXiv 2404.16130 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Edge et al., 2024) | correct |
| R41 | Visual multivector retrieval | arXiv 2407.01449 | ColPali: Efficient Document Retrieval with Vision Language Models (Faysse et al., 2024) | correct |
| R45 | Late-interaction retrieval | arXiv 2004.12832 | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (Khattab et al., 2020) | correct |
| R46 | CRAG reproduction | arXiv 2401.15884 | Corrective Retrieval Augmented Generation (Yan et al., 2024) | correct |
| R47 | CAG | github.com/hhhuang/CAG | Cache-Augmented Generation repo | correct; paper: arXiv 2412.15605 "Don't Do RAG" (Chan et al., 2024) |
| R48 | KAG | github.com/OpenSPG/KAG | KAG repo | correct; paper: arXiv 2409.13731 "KAG: Boosting LLMs in Professional Domains…" |
| R49 | Prefix KV caching | docs.vllm.ai | "Automatic Prefix Caching - vLLM" | correct (HTTP 200) |
| R27 | Context ordering and packing | arxiv.org/abs/2307.03172 | "Lost in the Middle: How Language Models Use Long Contexts", Liu et al., v1 submitted 2023-07-06 | 2026-11-10: verified via arXiv abstract page by orchestrator |

Summary: 19 links checked, 18 correct, 1 redirect fixed. Added 2026-11-09: R09, R19, R20 (all correct). Added 2026-11-10: R27 (correct). The other 54 entries have no single primary source yet (`null`); add one when the technique's lesson is written.
