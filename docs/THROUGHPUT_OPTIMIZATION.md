# Calibration throughput investigation

This work starts from main commit
`2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d`. The requested investigation window is
2026-11-12 10:50–19:50 UTC. Validation below is an intermediate checkpoint;
the paired benchmark is still running.

## Bulk triangle candidates

Profiling saved left- and dual-arm observations identified per-point R-tree
intersections and Python iteration over candidate IDs as the main bottleneck.
The same bounding boxes now use libspatialindex's bulk intersection operation.
Projection and the distance/face-ID tie rule remain unchanged.

The small compatibility module keeps allocation sizes and offsets as Python
integers. Rtree 1.4.1's vector API can otherwise promote unsigned counts to
floating point when its result buffer grows under NumPy 1.x. It uses the ctypes
signature supplied by Rtree; unavailable or unbound native functions and
scalar-only index adapters retain scalar intersections. No dependency upgrade
or global library patch is required. The adapted loop includes Rtree's MIT
notice in its source.

At this checkpoint:

- 116 tests, Ruff, and whitespace checks pass, including dense buffer growth,
  strided boxes, empty groups, duplicate-face ties and scalar fallback paths.
- Six paired synthetic query cases match all projection, distance and face-ID
  arrays exactly. Seven alternating-order repeats show 1.53–3.79 times faster
  median queries on this shared workstation. This is a query measurement, not
  an end-to-end calibration speedup.
- Saved left- and dual-arm observation replay matches every report field
  exactly after excluding timing fields, including diagnostics, decisions,
  retries and full solver histories.

The [query measurements](results/throughput_query.json) include the source
fingerprint. Reproduce with the locked simulation environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_proximity.py \
  --baseline-ref 2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d
```

## Paired regression in progress

The [frozen plan](../configs/benchmark_throughput.json) has 16 new seeds across
14 cases: left/right/dual visible depth, shared encoder noise, depth scale with
outliers, bounded continuation, quadratic axial noise, unanchored degeneracy,
both standalone arms, frame selection, dropout/quantization, large offsets and
sampled matching. Baseline and candidate run in isolated source snapshots.
All non-timing record fields must agree exactly; different source fingerprints
are compared, never merged. The local artifacts are isolated under
`outputs/throughput_20261112/`.

All accuracy checks use synthetic observations. This investigation does not
establish hardware calibration accuracy.
