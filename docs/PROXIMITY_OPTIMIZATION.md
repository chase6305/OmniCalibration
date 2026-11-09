# Linear triangle candidate selection

Exact triangle matching formerly sorted every candidate by query row, squared
distance, and face ID. Candidates are already contiguous by query row, so grouped
minimum reductions select the same result in linear time in candidate count.
Exact distance ties still choose the smallest face ID. Empty candidate groups
retain NaN projections, infinite distances, and face ID -1. Distance limits
still bound the search; they do not promise to reject all farther candidates.

## Validation

- 110 tests pass, including exhaustive independent triangle queries, reversed
  candidate order with duplicate geometry, empty groups, and zero distance limits.
- Ruff and `git diff --check` pass.
- Full reproduction passes: single and dual arms, exact observation replay,
  standalone arms, retained unobservable case, and unchanged source throughout.
  Outputs are isolated under `outputs/proximity_optimization/`.
- [Nine reproduction reports](results/proximity_reproduction_comparison.json)
  match the existing pre-change reports exactly after excluding recursive
  `elapsed_s`/`render_seconds` fields and normalizing the relocated replay input
  path. All arrays in the three saved NPZ datasets match exactly as well.
- Six paired query cases match the baseline arrays exactly, including face IDs
  and missing projections. The baseline is commit
  `8775c98b71eee92fd0d9bd37831ab6fdf3e1fb15`.

The [recorded query benchmark](results/proximity_optimization.json) uses seed
20261109, seven alternating-order repeats, and median query time. Mesh creation
and index construction are outside the timed region; both indexes are warmed by
the equality check. Results on this shared workstation:

| Faces | Points | Search limit (m) | Baseline (ms) | Optimized (ms) | Speed ratio |
|---|---|---|---|---|---|
| 320 | 2,000 | Unlimited | 36.11 | 29.50 | 1.224 |
| 320 | 2,000 | 0.02 | 10.83 | 10.74 | 1.008 |
| 1,280 | 2,000 | Unlimited | 78.28 | 66.31 | 1.180 |
| 1,280 | 2,000 | 0.02 | 12.45 | 12.28 | 1.014 |
| 1,280 | 10,000 | Unlimited | 453.57 | 466.81 | 0.972 |
| 1,280 | 10,000 | 0.02 | 71.56 | 62.04 | 1.153 |

The ratio is baseline time divided by optimized time. One case is slower in
this measurement; these results do not establish a uniform speedup or full
solver acceleration. R-tree traversal and triangle projection remain part of
each query. This change does not establish improved calibration accuracy.

Reproduce from the repository root with simulation dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_proximity.py \
  --baseline-ref 8775c98b71eee92fd0d9bd37831ab6fdf3e1fb15
```

The script executes the proximity module from the explicitly selected Git
revision; use a trusted revision. Timings vary with machine load. Historical
accuracy studies remain evidence for their original source snapshots.
