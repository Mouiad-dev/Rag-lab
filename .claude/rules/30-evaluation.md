---
paths:
  - "packages/evaluation/**"
  - "tests/**"
  - "config/**"
  - "docs/techniques/**"
---
# Evaluation integrity

- Freeze corpus/dataset/model/config/code versions. Change one factor before testing combinations.
- Separate development, validation and held-out data; check document/near-duplicate leakage.
- Define relevance labels and metric denominators. HitRate and Recall are different with multiple relevant items.
- Report per-case outputs and failures, not only averages. Include Arabic, English, mixed-language, table, OCR and unanswerable cohorts.
- Version and human-calibrate LLM judges. Judge scores are not truth guarantees.
- Count query rewriting, grading, reranking, auxiliary agents and indexing separately in usage/cost.
- Separate TTFT from total latency, cold from warm cache, and fast Q&A from long reports.
- Security failures independently fail a release; averaging them with quality scores is prohibited.
- Preserve all curriculum entries even when measured quality is worse. Production eligibility is a separate field.
