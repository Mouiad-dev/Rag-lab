# R07: Reciprocal rank fusion

## Intuition

RRF combines ordered result lists when their raw scores have different scales. It rewards agreement near the top without comparing those raw scores.

## Mechanism

For each occurrence, add `1/(k+rank)`; the conventional constant is `k=60` (Cormack, Clarke, Buettcher, 2009). Rank is taken from each input list. A missing list contributes zero. Candidates are de-duplicated by `chunk_id`. The implementation breaks equal fused scores by best input rank, then lexicographically by `chunk_id`. Raw scores are ignored.

## Runnable demo

The success case reproduces the hand example in `tests/indexing/test_fusion.py`: y scores `1/62 + 1/61 = 0.0325225...` and ranks first, followed by x, w, z. The losing case demonstrates that RRF ignores score margins: x has a large R03 score advantage at rank 1, but is rank 3 in R02; q at rank 2 in both lists wins.

```powershell
$env:PYTHONIOENCODING='utf-8'; uv run python -c "from rag_foundry_indexing import RrfConfig,RrfFusion; from rag_foundry_techniques import RetrievalResult,RetrievedChunk; hit=lambda cid,r,s: RetrievedChunk(chunk_id=cid,extraction_view_id='v',start=0,end=1,block_ids=('b',),text=cid,score=s,rank=r); result=lambda tid,hs: RetrievalResult(technique_id=tid,grounding='corpus',index_version_id='v1',hits=tuple(hs)); f=RrfFusion(RrfConfig()); a=result('R03',[hit('x',1,99),hit('y',2,1),hit('z',3,0)]); b=result('R02',[hit('y',1,0.1),hit('w',2,0.05)]); out=f.fuse([a,b]); print('success:',[(h.chunk_id,round(h.score,7)) for h in out.hits]); a=result('R03',[hit('x',1,1000),hit('q',2,1)]); b=result('R02',[hit('q',1,10),hit('p',2,9),hit('x',3,0.01)]); out=f.fuse([a,b]); print('losing case:',[(h.chunk_id,round(h.score,7)) for h in out.hits])"
```

Output from running the command with `PYTHONIOENCODING=utf-8`:

```text
success: [('y', 0.0325225), ('x', 0.0163934), ('w', 0.016129), ('z', 0.015873)]
losing case: [('q', 0.0325225), ('x', 0.0322665), ('p', 0.016129)]
```

## Limitations

Ranks retain no information about score gaps. A candidate with an overwhelming score at rank 1 can lose when another candidate is consistently near the top of multiple lists.

## Status

Implemented; not validated or benchmarked.
