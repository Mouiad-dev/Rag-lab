# P7.5 C20 late chunking and R45 late interaction: demo output and live evidence

## Simulation (offline)

Command: `PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/c20_r45_demo.py`. Date: 2026-11-23. Output is byte-identical between runs (two runs compared).

This section is SIMULATION. The encoder is a test-only fake (one token per word, vectors from code points, plus a toy rule that adds half the running mean of earlier words so late and early vectors differ). It has no language understanding. The ranks show the mechanics of the four arms (C20 early, C20 late, R45 single-vector mean, R45 MaxSim), not a model result, and nothing here is a quality or speed claim. Per-chunk encoding is the early baseline, never late chunking. Ranks are among the chunks of the question's own document (each document is its own small index).

```text
SIMULATION: fake token encoder, not a model result
Ties between equal scores are broken by chunk start. Fixture: 3 documents (EN, AR, mixed); ranks are among the chunks of the question's own document. Late chunking = encode the whole document once, then mean-pool the token vectors inside each chunk span. Early = encode each chunk alone (same encoder, same pooling).

## Index sizes
EN: chunks=4 tokens=43 dim=4 estimated_bytes=688 late_windows=1 early_truncated=False
AR: chunks=3 tokens=16 dim=4 estimated_bytes=256 late_windows=1 early_truncated=False
MIXED: chunks=3 tokens=24 dim=4 estimated_bytes=384 late_windows=1 early_truncated=False

## Relevant-chunk rank and score per question
| question | arm | result |
|---|---|---|
| EN1 | c20_early | rank=3 score=0.979 top1_chunk=3 ranking(chunk:score)=3:1.000 2:0.982 1:0.979 0:0.975 |
| EN1 | c20_late | rank=3 score=0.985 top1_chunk=3 ranking(chunk:score)=3:1.000 2:0.991 1:0.985 0:0.975 |
| EN1 | r45_single | rank=2 score=0.983 top1_chunk=3 ranking(chunk:score)=3:1.000 1:0.983 0:0.979 2:0.976 |
| EN1 | r45_maxsim | rank=4 score=6.788 top1_chunk=3 ranking(chunk:score)=3:6.992 2:6.982 0:6.957 1:6.788 |
| EN2 | c20_early | rank=4 score=0.939 top1_chunk=0 ranking(chunk:score)=0:0.991 1:0.984 3:0.973 2:0.939 |
| EN2 | c20_late | rank=4 score=0.963 top1_chunk=0 ranking(chunk:score)=0:0.991 1:0.990 3:0.979 2:0.963 |
| EN2 | r45_single | rank=4 score=0.929 top1_chunk=0 ranking(chunk:score)=0:0.992 1:0.979 3:0.970 2:0.929 |
| EN2 | r45_maxsim | rank=3 score=8.574 top1_chunk=0 ranking(chunk:score)=0:8.914 3:8.645 2:8.574 1:8.554 |
| AR1 | c20_early | rank=3 score=0.926 top1_chunk=0 ranking(chunk:score)=0:0.938 2:0.930 1:0.926 |
| AR1 | c20_late | rank=3 score=0.924 top1_chunk=0 ranking(chunk:score)=0:0.938 2:0.927 1:0.924 |
| AR1 | r45_single | rank=2 score=0.938 top1_chunk=0 ranking(chunk:score)=0:0.947 1:0.938 2:0.935 |
| AR1 | r45_maxsim | rank=1 score=3.940 top1_chunk=1 ranking(chunk:score)=1:3.940 0:3.748 2:3.735 |
| MIX1 | c20_early | rank=3 score=0.933 top1_chunk=0 ranking(chunk:score)=0:0.975 2:0.942 1:0.933 |
| MIX1 | c20_late | rank=3 score=0.956 top1_chunk=0 ranking(chunk:score)=0:0.975 2:0.967 1:0.956 |
| MIX1 | r45_single | rank=3 score=0.937 top1_chunk=0 ranking(chunk:score)=0:0.976 2:0.939 1:0.937 |
| MIX1 | r45_maxsim | rank=2 score=7.894 top1_chunk=0 ranking(chunk:score)=0:7.935 1:7.894 2:7.882 |
| MIX2 | c20_early | rank=2 score=0.973 top1_chunk=0 ranking(chunk:score)=0:0.984 2:0.973 1:0.921 |
| MIX2 | c20_late | rank=2 score=0.980 top1_chunk=0 ranking(chunk:score)=0:0.984 2:0.980 1:0.951 |
| MIX2 | r45_single | rank=2 score=0.975 top1_chunk=0 ranking(chunk:score)=0:0.984 2:0.975 1:0.930 |
| MIX2 | r45_maxsim | rank=3 score=9.844 top1_chunk=0 ranking(chunk:score)=0:9.950 1:9.894 2:9.844 |
| UNANS | c20_early | top1=0.962 top1_chunk=1 ranking(chunk:score)=1:0.962 0:0.955 3:0.947 2:0.943 |
| UNANS | c20_late | top1=0.965 top1_chunk=1 ranking(chunk:score)=1:0.965 2:0.962 3:0.956 0:0.955 |
| UNANS | r45_single | top1=0.963 top1_chunk=1 ranking(chunk:score)=1:0.963 0:0.960 3:0.951 2:0.936 |
| UNANS | r45_maxsim | top1=6.920 top1_chunk=0 ranking(chunk:score)=0:6.920 3:6.662 2:6.614 1:6.591 |

UNANS has no relevant chunk: only the top-1 score is shown. Scores of different arms are not comparable.
Losing case: EN2 has a keyword-heavy distractor chunk (chunk 3); an arm loses on EN2 when top1_chunk is not 2.
```

## Live run (orchestrator)

<orchestrator>
One local CPU run, not an evaluation; no quality or speed claim. Timings are single wall-clock values on this machine; the first early build includes the model load. This is the second live run of the session: the first (3 passed in 29.21s) printed only the relevant chunk's rank, so review asked for top-1 and full rankings; this run replaces it.

- Command: `PYTHONIOENCODING=utf-8 RAG_LIVE_TOKENS=1 RAG_TOKEN_MODEL_DIR=var/models/bge-m3 uv run pytest tests/live/test_late_live.py tests/live/test_token_encoder_live.py -m live -v -s -p no:cacheprovider`
- Date: 2026-11-23. Result: 3 passed in 28.68s (coder run; log checked by the orchestrator). New asserts: all question encodings differ, and no two questions in a document have identical score maps in any arm.
- Model: `BAAI/bge-m3` at commit 5617a9f61b028005a4858fdac845db406aefb181, loaded from local safetensors converted by the orchestrator (`model.safetensors` 2,271,064,536 bytes, sha256 cc473709b9a576b32d81b1a42f79db8f1544425f2486698177e0a0b0ae87e6fa; `colbert_linear.safetensors` 2,099,464 bytes, sha256 37ab8857a5fb7be83d25f349f1a74f7183e2b7262181a680c0cb38b146d81706; originals and their hashes in the P7.5-02 spec). dim 1024.
- Machine: AMD Ryzen 7 5800H, CPU only; torch 2.15.0+cpu, transformers 5.22.2, Python 3.13.5.
- Structure: each document fit in one late window; no truncation in either mode; per-chunk token counts identical for early, late and multivector.
- Ranks are within each small single-document index (3 or 4 chunks), not a corpus-wide ranking. Late cosine scores are higher than early for the answerable questions, but absolute cosine level is not a quality signal (the unanswerable question also scores about 0.63 in both). Between C20 early and C20 late, only AR1 changed rank (3 to 2); R45 single-vector and MaxSim differ on MIX1 and MIX2. R45 MaxSim scores are sums over query tokens, so they are not on the same scale as cosine scores. The unanswerable question still gets a top-1 chunk in every arm; there is no abstention threshold here.
- Losing case (shown): EN2's relevant chunk (2) is rank 3 in all four arms, and the distractor chunk 3 is top-1 in all four. Late chunking and MaxSim do not fix it here.
- Identical relevant-chunk scores across two questions (e.g. EN1 and EN2 late 0.821) come from different (question, chunk) pairs; the full rankings differ per question. bge-m3 mean-pooled cosines sit in a narrow band (about 0.57 to 0.87), so 3-decimal ties occur by chance.

Printed output (progress bars removed):

```text
model=BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181 revision=5617a9f61b028005a4858fdac845db406aefb181
DOC EN chunks=4 mv_tokens=70 dim=1024 estimated_bytes=286720 early_tokens=70 late_tokens=70 late_windows=1 early_truncated=False late_truncated=False build_s early=12.92 late=0.23 multivector=0.54
  chunk 0 tokens early=16 late=16 mv=16
  chunk 1 tokens early=13 late=13 mv=13
  chunk 2 tokens early=16 late=16 mv=16
  chunk 3 tokens early=25 late=25 mv=25
DOC AR chunks=3 mv_tokens=29 dim=1024 estimated_bytes=118784 early_tokens=29 late_tokens=29 late_windows=1 early_truncated=False late_truncated=False build_s early=0.36 late=0.16 multivector=0.33
  chunk 0 tokens early=11 late=11 mv=11
  chunk 1 tokens early=10 late=10 mv=10
  chunk 2 tokens early=8 late=8 mv=8
DOC MIXED chunks=3 mv_tokens=36 dim=1024 estimated_bytes=147456 early_tokens=36 late_tokens=36 late_windows=1 early_truncated=False late_truncated=False build_s early=0.38 late=0.16 multivector=0.37
  chunk 0 tokens early=13 late=13 mv=13
  chunk 1 tokens early=14 late=14 mv=14
  chunk 2 tokens early=9 late=9 mv=9
RANK EN1 c20_early rank=2 score=0.675 top1_chunk=0 ranking(chunk:score)=0:0.870 1:0.675 3:0.666 2:0.650 query_s=0.118
RANK EN1 c20_late rank=2 score=0.821 top1_chunk=0 ranking(chunk:score)=0:0.839 1:0.821 2:0.817 3:0.811 query_s=0.118
RANK EN1 r45_single rank=2 score=0.420 top1_chunk=0 ranking(chunk:score)=0:0.704 1:0.420 2:0.276 3:0.246 query_s=0.118
RANK EN1 r45_maxsim rank=2 score=4.477 top1_chunk=0 ranking(chunk:score)=0:8.344 1:4.477 3:3.050 2:3.036 query_s=0.185
RANK EN2 c20_early rank=3 score=0.783 top1_chunk=3 ranking(chunk:score)=3:0.823 0:0.797 2:0.783 1:0.669 query_s=0.123
RANK EN2 c20_late rank=3 score=0.821 top1_chunk=3 ranking(chunk:score)=3:0.828 0:0.825 2:0.821 1:0.802 query_s=0.154
RANK EN2 r45_single rank=3 score=0.620 top1_chunk=3 ranking(chunk:score)=3:0.684 0:0.637 2:0.620 1:0.453 query_s=0.141
RANK EN2 r45_maxsim rank=3 score=8.398 top1_chunk=3 ranking(chunk:score)=3:9.432 0:9.127 2:8.398 1:6.051 query_s=0.233
RANK AR1 c20_early rank=3 score=0.740 top1_chunk=0 ranking(chunk:score)=0:0.867 2:0.749 1:0.740 query_s=0.117
RANK AR1 c20_late rank=2 score=0.833 top1_chunk=0 ranking(chunk:score)=0:0.855 1:0.833 2:0.829 query_s=0.119
RANK AR1 r45_single rank=2 score=0.506 top1_chunk=0 ranking(chunk:score)=0:0.678 1:0.506 2:0.425 query_s=0.129
RANK AR1 r45_maxsim rank=2 score=4.695 top1_chunk=0 ranking(chunk:score)=0:6.153 1:4.695 2:3.757 query_s=0.123
RANK MIX1 c20_early rank=2 score=0.795 top1_chunk=0 ranking(chunk:score)=0:0.834 1:0.795 2:0.663 query_s=0.117
RANK MIX1 c20_late rank=2 score=0.857 top1_chunk=0 ranking(chunk:score)=0:0.859 1:0.857 2:0.839 query_s=0.115
RANK MIX1 r45_single rank=1 score=0.659 top1_chunk=1 ranking(chunk:score)=1:0.659 0:0.654 2:0.375 query_s=0.119
RANK MIX1 r45_maxsim rank=2 score=6.244 top1_chunk=0 ranking(chunk:score)=0:6.586 1:6.244 2:3.763 query_s=0.156
RANK MIX2 c20_early rank=1 score=0.795 top1_chunk=2 ranking(chunk:score)=2:0.795 1:0.785 0:0.777 query_s=0.138
RANK MIX2 c20_late rank=1 score=0.857 top1_chunk=2 ranking(chunk:score)=2:0.857 1:0.847 0:0.839 query_s=0.124
RANK MIX2 r45_single rank=1 score=0.604 top1_chunk=2 ranking(chunk:score)=2:0.604 1:0.551 0:0.549 query_s=0.127
RANK MIX2 r45_maxsim rank=2 score=6.809 top1_chunk=0 ranking(chunk:score)=0:7.009 2:6.809 1:6.106 query_s=0.186
RANK UNANS c20_early top1=0.635 top1_chunk=2 ranking(chunk:score)=2:0.635 0:0.628 3:0.569 1:0.567 query_s=0.111
RANK UNANS c20_late top1=0.633 top1_chunk=0 ranking(chunk:score)=0:0.633 2:0.626 1:0.612 3:0.608 query_s=0.113
RANK UNANS r45_single top1=0.260 top1_chunk=0 ranking(chunk:score)=0:0.260 2:0.203 1:0.127 3:0.102 query_s=0.111
RANK UNANS r45_maxsim top1=2.176 top1_chunk=0 ranking(chunk:score)=0:2.176 2:1.902 1:1.221 3:1.212 query_s=0.148
PASSED
head=hidden revision=5617a9f61b028005a4858fdac845db406aefb181 dim=1024 tokens=[14, 11, 3] offsets_en=((0, 3), (4, 8), (8, 10), (11, 17))
PASSED
head=colbert revision=5617a9f61b028005a4858fdac845db406aefb181 dim=1024 tokens=[14, 11, 3] offsets_en=((0, 3), (4, 8), (8, 10), (11, 17))
PASSED
```
</orchestrator>
