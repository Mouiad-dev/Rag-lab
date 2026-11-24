# R08: Weighted score fusion

## Intuition

Weighted fusion puts branch scores on a per-list min-max scale before combining them. It is an adaptation whose weights must be chosen on dev only.

## Mechanism

Each list is independently normalized as `(score-min)/(max-min)`, then weighted contributions are summed. Missing candidates contribute zero. Configured weights are used as given; absent weights default to equal weights. If a list is single-hit or all its scores are equal, its present hit(s) normalize to 1.0 and carry `normalization_degenerate`.

Equal fused scores are ordered by best input rank, then extraction view and start/end offsets, then chunk id.

The evidence reports `w*=0.5`, selected by tie-break, not learned. Its dev sweep is flat: every tested weight from 0.0 through 1.0 gives dev MRR 1.0000. The weight is selected on dev only.

## Runnable demo

The success case is the weighted hand calculation in `tests/indexing/test_fusion.py`. The losing case shows a degenerate one-hit branch assigning normalized score 1.0 to its only hit.

```powershell
$env:PYTHONIOENCODING='utf-8'; uv run python -c "from rag_foundry_indexing import WeightedConfig,WeightedFusion; from rag_foundry_techniques import RetrievalResult,RetrievedChunk; hit=lambda cid,r,s: RetrievedChunk(chunk_id=cid,extraction_view_id='v',start=0,end=1,block_ids=('b',),text=cid,score=s,rank=r); result=lambda tid,hs: RetrievalResult(technique_id=tid,grounding='corpus',index_version_id='v1',hits=tuple(hs)); f=WeightedFusion(WeightedConfig()); out=f.fuse([result('R03',[hit('x',1,8),hit('y',2,4),hit('z',3,2)]),result('R02',[hit('y',1,.9),hit('w',2,.5)])]); print('success:',[(h.chunk_id,round(h.score,4)) for h in out.hits]); out=f.fuse([result('R03',[hit('solo',1,9)]),result('R02',[hit('other',1,0)])]); h=next(h for h in out.hits if h.chunk_id=='solo'); print('losing case:',h.chunk_id,[(c.name,c.value) for c in h.score_components if c.name in ('min_max_score','normalization_degenerate')])"
```

Output from running the command with `PYTHONIOENCODING=utf-8`:

```text
success: [('y', 0.6667), ('x', 0.5), ('w', 0.0), ('z', 0.0)]
losing case: solo [('normalization_degenerate', 1.0), ('min_max_score', 1.0)]
```

## Evidence

The per-case report is in [the paired comparison](../../evidence/P5.2/paired_vs_dense.md). Dev and validation overlap, and the small cohort supports no significance claim.

## Status

Implemented; not validated or benchmarked.
