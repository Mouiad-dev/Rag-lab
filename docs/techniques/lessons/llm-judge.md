# LLM judge (answer-support rubric)

This is not a catalogue technique. It is an evaluator tool in `packages/evaluation` that grades whether an answer is supported by its evidence.

## Rubric and version

- Rubric id `answer-support`, version `1`. Labels: `supported`, `partially_supported`, `unsupported`.
- The system prompt text is built from the rubric. Its SHA-256 (`prompt_sha256`) is stored on every verdict. In the offline demo it starts with `b6d6c59076e1`.
- Prompt sha rule: any text change (criteria, labels, wording) changes the sha, so it needs a version bump. The pin test `test_version_pin` in `tests/evaluation/test_judge.py` fails on any change that is not accompanied by a new pinned value.
- The judged question, answer and evidence are passed as untrusted data, not as instructions.

## Verdict schema

The judge must reply with exactly `{"label": <one of the labels>, "reason": "<at most 300 characters>"}`. The prompt asks for at most 300 characters; the check accepts a reason of up to 500. The check is strict: exact keys, a known label, valid JSON without duplicate keys. In `complete` mode only: a JSON schema is sent when the model profile supports structured output, and a reply cut off at the length limit is `invalid_output`. Stream mode sends no schema and has no length cut-off check.

## invalid_output

A reply that fails the check is recorded with `status: invalid_output` and no label. It is never given a default label (not `unsupported`, not anything). In the offline demo `j-05` shows this. Invalid verdicts are left out of agreement and counted separately.

## Metering and modes

- Each judge call goes through its own metered chat wrapper, so its tokens and cost are reported under their own stage (`judge`, or `judge_complete` / `judge_stream` in the live run), separate from answering. See `cost-latency-report.md`.
- `complete` mode makes one non-streamed call. `stream` mode streams the reply, so a time to first token can be measured.

## Judge labels are not ground truth

Judge labels are evaluator output, not ground truth. A judge can be wrong, and it can be stable and wrong.

## Agreement with humans

- Percent agreement: the share of compared items where judge and human give the same label.
- Cohen's kappa: agreement corrected for chance. `kappa = (p_o - p_e) / (1 - p_e)`.
- Undefined case: when chance agreement `p_e` is 1 (both sides always use one and the same label), kappa has no value and is reported as `undefined_chance_agreement_1`.
- Hand example: 10 items, the judge says `supported` 5 times and `unsupported` 5 times, the human does the same, and they match on 8 items. Percent agreement is 0.8, `p_e` is 0.5, kappa is (0.8 - 0.5) / (1 - 0.5) = 0.6.

## Current state

Quoted from `docs/evidence/P6.5/judge_and_cost_demo.md`:

> Judge agreement: not yet measured (0 human labels).
>
> Judge labels are evaluator output, not ground truth.

The label file `tests/evaluation/data/human_labels_answer_support_v1.json` is a placeholder with zero labels. See `human-labeling-guide.md`.

## Live run

Quoted from the live section of `docs/evidence/P6.5/judge_and_cost_demo.md`. One run, n=6, not an evaluation; agreement not measured (0 human labels). Local Ollama, `gemma4:e4b`, cost 0, 12 calls (6 complete, 6 stream). Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_judge_live.py -m live -v -s` (1 passed in 24.41s).

| item | mode | status | label | reason |
|---|---|---|---|---|
| j-01 | complete | ok | supported | 'The evidence explicitly states that the tidal observatory at Port Sallow records' |
| j-02 | complete | ok | unsupported | 'The evidence states the observatory records sea levels every 17 minutes, which c' |
| j-03 | complete | ok | supported | 'The evidence S1 explicitly states that the Oasis Library holds three thousand tw' |
| j-04 | complete | ok | unsupported | 'The answer claims the library holds no manuscripts, but the evidence states it h' |
| j-05 | complete | ok | partially_supported | 'The evidence states the reading room opens at 08:30, but it does not mention whe' |
| j-06 | complete | ok | unsupported | 'The answer states that the choir meets on Tuesday evenings, but the evidence sta' |
| j-01 | stream | ok | supported | 'The evidence explicitly states that the tidal observatory at Port Sallow records' |
| j-02 | stream | ok | unsupported | 'The evidence states the observatory records sea levels every 17 minutes, which c' |
| j-03 | stream | ok | supported | 'The evidence S1 explicitly states that the Oasis Library holds three thousand tw' |
| j-04 | stream | ok | unsupported | 'The answer claims the library holds no manuscripts, but the evidence states it h' |
| j-05 | stream | ok | partially_supported | 'The evidence states the reading room opens at 08:30, but it does not mention whe' |
| j-06 | stream | ok | unsupported | 'The answer states that the choir meets on Tuesday evenings, but the evidence sta' |

In this run the stream mode gave the same label as the complete mode for each of the 6 items. These labels have not been compared with any human label.

## Limitations

- One local judge model (`gemma4:e4b`), no ensemble, no second judge.
- n=6 items. Even after human labelling, a kappa from 6 items would be unstable; treat it as a plumbing check, not as a quality figure.
- Item design is small and neutral; it does not cover tables, OCR or long evidence.
- A paid judge needs an agreed budget first.
