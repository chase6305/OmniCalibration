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

At the bulk-only checkpoint:

- 116 tests, Ruff, and whitespace checks pass, including dense buffer growth,
  strided boxes, empty groups, duplicate-face ties and scalar fallback paths.
- Six paired synthetic query cases match all projection, distance and face-ID
  arrays exactly. Seven alternating-order repeats show 1.53–3.79 times faster
  median queries on this shared workstation. This is a query measurement, not
  an end-to-end calibration speedup.
- Saved left- and dual-arm observation replay matches every report field
  exactly after excluding timing fields, including diagnostics, decisions,
  retries and full solver histories.

The next candidate projects complete query groups in batches targeting 65,536
candidate pairs. A single dense query may exceed that target. This limits the
large projection temporaries; the candidate-ID buffer itself still scales with
the total number of intersections. Forced budgets of 1, 7, 64 and 513 pairs
preserve exact outputs, including empty groups and per-point distance limits.
The complete local suite now has 124 passing tests. Dependency compatibility,
peak-memory and end-to-end measurements remain pending in the resource queue.

The [query measurements](results/throughput_query.json) include the source
fingerprint. Reproduce with the locked simulation environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_proximity.py \
  --baseline-ref 2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d \
  --memory-points 40000 --output outputs/throughput_query.json
```

The query script extracts the selected numerical package, including its relative
imports, records both source fingerprints and individual timings, and refuses
source changes during measurement. The optional Unix memory measurement uses
fresh processes so each peak RSS is independent. Saved-observation end-to-end
timing is available through `scripts/benchmark_replay.py`; its alternating runs
require exact complete-report agreement before reporting any speedup.

## Paired regression in progress

The [frozen plan](../configs/benchmark_throughput.json) has four new seeds across
14 cases (56 pairs): left/right/dual visible depth, shared encoder noise, depth scale with
outliers, bounded continuation, quadratic axial noise, unanchored degeneracy,
both standalone arms, frame selection, dropout/quantization, large offsets and
sampled matching. Baseline and candidate run in isolated source snapshots.
All non-timing record fields must agree exactly; different source fingerprints
are compared, never merged. The local artifacts are isolated under
`outputs/throughput_20261112/`.

The initial 224-run development plan was reduced after the shared-workstation
resource policy was supplied. The first four seeds by ordinal were selected
uniformly across all scenarios, without using accuracy metrics. Eight unchanged
baseline checkpoints were reused with their original source/input hashes; the
larger incomplete development batch remains separate. Subsequent heavy jobs
hold `/home/ubuntu/.cache/codex-heavy-work.lock`, run serially with one numerical
thread, and defer when available memory or load fails the resource checks.

All accuracy checks use synthetic observations. This investigation does not
establish hardware calibration accuracy.
