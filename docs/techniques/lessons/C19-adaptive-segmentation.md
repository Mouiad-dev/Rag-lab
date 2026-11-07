# C19: Adaptive segmentation

## Decision table

Fixture decision output (`chunk_size=64`):

```text
kind | policy | start | end | chunks | failed
prose | C06 | 0 | 42 | 1 | False
list | C07 | 44 | 70 | 1 | False
table | C07 | 72 | 118 | 1 | False
code | C05 | 124 | 147 | 1 | False
mixed_script | C21 | 149 | 179 | 1 | False
```

## Comparison

The same fixture compared adaptive routing with fixed C05:

```text
adaptive_chunks=5 fixed_chunks=4 fixed_policy=C05 fixed_cross_region=1 failures=0 oversized_adaptive=0 oversized_fixed=0
```

## Limitations

Rule thresholds are hand-set. There is no model router yet. Tables and code use generic policies until the P4.3 (C08-C14) policies can be routed to.
