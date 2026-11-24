# Citation check

The `lexical-v1` rule segments claims, checks citation markers, compares evidence constraints, and measures content-term overlap (default minimum overlap 0.5). It checks lexical similarity and constraints, not meaning.

False positive: `Tolls apply to bicycles [S1].` is `supported_lexical` with overlap `1.0` because the terms match despite the negation in the evidence. This is not entailment; LLM judges come in P6.5.

Warnings:

- `evidence_packed_dropped`: R27 dropped at least one gated hit (budget or contained).
- `evidence_packing_empty`: nothing fit the budget, so the run abstained with `no_evidence` and made no chat call.
- `claim_uncited`: at least one claim has no marker.
- `claim_unsupported_lexical`: at least one cited claim failed the lexical rule (a missing constraint or low overlap).
- `claim_invalid_citation`: defensive only. It cannot occur after `check_answer`, which abstains first.

The warning code is a free-form `^[a-z_]{1,64}$` string; no contract change is needed. A known limitation is that a claim with no terms (for example `[S1]`) is marked `supported_lexical`, so `claim_*` warnings miss it.
