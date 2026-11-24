# Human labelling guide (answer-support rubric v1)

## Who may label

A human. The Arabic items (`j-03`, `j-04`) need an Arabic native speaker. Do not label for someone else, and do not let a model fill in labels.

## The file

Edit `tests/evaluation/data/human_labels_answer_support_v1.json`. Today it has `status: "placeholder_not_labeled"` and `labels: []`. Keep `rubric_id`, `rubric_version`, `items_id` and `items_sha256` as they are. The items are in `tests/evaluation/data/judge_items_v1.json`, but that file stores Arabic as `\uXXXX` escapes. Read the items with:

```bash
uv run python scripts/human_labels.py items
```

It prints each item's question, answer and evidence, and the allowed labels. It prints no judge output.

## Field rules

Add one entry per item you label:

```json
{"item_id": "j-01", "annotator_id": "human-<name>", "label": "supported", "labeled_on": "YYYY-MM-DD", "source": "human"}
```

- `source` must be `"human"`.
- `annotator_id` matches `human-<name>`: lowercase letters, digits and hyphens, at most 32 characters after `human-`.
- `labeled_on` is the date in `YYYY-MM-DD`.
- `label` is one of `supported`, `partially_supported`, `unsupported`. Meaning: every claim is in the evidence / some claims are / none are or one contradicts it. Judge only against the evidence, not world knowledge; numbers, names, dates and negations must match exactly.
- One entry per (`item_id`, `annotator_id`) pair; duplicates are rejected.
- When you add at least one label, change `status` to `"labeled"` and update `note` (at most 500 characters). A `labeled` file with no labels is rejected.

## How to label

Label without seeing the judge verdicts or the item design comments. Read the question, the answer and the evidence marker text only (from `scripts/human_labels.py items`). Do not open `docs/evidence/P6.5/`, `tests/evaluation/_judge_items.py` or the judge lessons before you finish: they show judge labels or design intent.

## Never paste model output

Do not copy any model output, judge verdict or guess into the file.

## Check the file and compute agreement

1. Check that the file loads strictly (rubric, item set sha, field rules, duplicates):

   ```bash
   uv run python scripts/human_labels.py check
   ```

   It prints `status=labeled labels=<n> annotators=<ids>`. Then run `uv run pytest tests/evaluation/test_human_labels.py -q`. With a `labeled` file the committed-file test only loads it strictly; it does not compare it with the placeholder dump, and `RAG_WRITE_EVIDENCE=1` never overwrites a `labeled` file.
2. Agreement needs judge verdicts from a fresh judge run with the same rubric (same `prompt_sha256`); no verdict file is saved. With Ollama running locally (cost 0, 6 chat calls):

   ```bash
   uv run python scripts/human_labels.py agreement
   ```

   It runs the judge (`gemma4:e4b` by default; `--model` changes it) and prints percent agreement and Cohen's kappa per annotator. With the placeholder file it prints "not yet measured (0 human labels)". With 6 items the numbers are a plumbing check, not a reliable agreement estimate (see `llm-judge.md`). Judge labels are evaluator output, not ground truth.
