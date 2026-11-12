# Calibration throughput investigation

This work starts from main commit
`2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d`. The requested investigation window is
2026-11-12 10:50:40–19:50:40 UTC. Validation below is an intermediate
checkpoint; final query/replay measurements are verified and paired regression
is pending.

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

- 128 tests, Ruff, and whitespace checks pass, including dense buffer growth,
  strided boxes, empty groups, duplicate-face ties and scalar fallback paths.
- Six paired synthetic query cases match all projection, distance and face-ID
  arrays exactly. Seven alternating-order repeats show 1.80–3.76 times faster
  median queries on this shared workstation. This is a query measurement, not
  an end-to-end calibration speedup.
- All three saved-observation replay modes match every non-timing report field
  exactly over three alternating repeats, including diagnostics, decisions,
  retries and full solver histories.
- Rtree 1.3.0 with NumPy 1.26.4 passes 23 related tests; NumPy 2.2.6 with
  Rtree 1.4.1 passes 24. Logs identify the actual imported versions and paths.

The implementation now streams complete query groups in batches targeting
65,536 candidate pairs. Both candidate storage and projection temporaries are
limited by that target; a single dense query may exceed it. Each yielded batch
owns its arrays. The native loop resets its completed-row count before every
call because libspatialindex leaves that count unchanged when the first query
does not fit. Forced budgets of 1, 7, 64 and 513 preserve exact outputs, including
empty groups, dense groups, duplicate-face ties and per-point distance limits.

The full suite also works without trimesh or Rtree: 95 tests pass and 33
simulation tests skip. This includes both new benchmark-tool regression tests.
Both query and replay timings use the final streaming source, one logical CPU
and explicit one-thread numerical settings. Each baseline/candidate pair uses
the same execution conditions.

## Isolated query memory

The [corrected memory measurement](results/throughput_memory.json) compares
40,000 fixed points against 1,280 faces, in independent processes under the same
one-CPU affinity and explicit one-thread numerical settings. All result hashes
agree exactly. Full-process high-water RSS includes imports and mesh/index
construction and is sampled before output hashing.

| Source | Peak RSS (MiB) |
|---|---:|
| Baseline main | 1099.05 |
| Projection batches | 150.18 |
| Streamed candidates and projections | 113.92 |

Streaming reduces this measured peak by 89.6% relative to main and by 24.1%
relative to projection batches. The current numerical source fingerprint is
`6b34f4cb149fd6e9ea1739913612e35746f7c00e461130ff1a7ae91a55e7945f`.

The initial memory figures were withdrawn: directly launching a worker from a
loaded benchmark could count the parent's pre-exec image in `ru_maxrss`.
Raw readings and their original source fingerprints remain marked as superseded
in the memory report.
The corrected launcher starts the measured worker from a small stdlib-only
interpreter. Startup peak was 11.25 MiB in all three workers, and Linux `VmHWM`
matched each reported query peak. A regression test touches a 64 MiB parent-only
allocation and verifies that the worker does not inherit it into its peak.

The [query measurements](results/throughput_query.json) include the source
fingerprint. Reproduce with the locked simulation environment:

```bash
# Check memory, load and competing processes before starting.
flock -n -E 75 /home/ubuntu/.cache/codex-heavy-work.lock env \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python scripts/benchmark_proximity.py \
  --baseline-ref 2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d \
  --memory-points 40000 --output outputs/throughput_query.json
```

The query script extracts the selected numerical package, including its relative
imports, records both source fingerprints and individual timings, and refuses
source changes during measurement. The optional Unix memory measurement uses
workers forked by a lightweight launcher. Linux [preserves resource usage across
exec](https://man7.org/linux/man-pages/man2/getrusage.2.html), so a large parent
must not be the immediate fork parent of a peak-memory worker. Saved-observation end-to-end
timing is available through `scripts/benchmark_replay.py`; its alternating runs
require exact complete-report agreement before reporting any speedup.
Every replay attempt receives a distinct report directory and input manifest.
Interrupted attempts remain separate, and each completed attempt retains its
own summary even when the requested summary filename is reused.

## Saved-observation replay

Three alternating-order repeats for each mode include model loading, index
construction, calibration and report generation. Each pair's complete report
matches exactly; only timing fields are excluded. The [recorded timings and
input hashes](results/throughput_replay.json) identify the same final numerical
source fingerprint as the query measurement.

| Mode | Baseline median (s) | Optimized median (s) | Speed ratio |
|---|---:|---:|---:|
| Left arm | 31.38 | 23.36 | 1.344 |
| Right arm | 21.21 | 16.17 | 1.312 |
| Dual arm | 16.83 | 12.81 | 1.314 |

These saved synthetic datasets are available after the ordinary reproduction
run. For example, under the same resource checks and shared lock:

```bash
flock -n -E 75 /home/ubuntu/.cache/codex-heavy-work.lock env \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 .venv/bin/python scripts/benchmark_replay.py \
  --baseline-ref 2ebf11c9d2c7bc97d39ad706b1a0466e89902d3d \
  --asset-root ../HumanoidAssets/Marvin_M6_S_CCS_696_V4.0 \
  --mode left --output outputs/throughput_left_replay.json
```

The speed ratios apply to these inputs and this workstation. They do not
establish a uniform speedup across all meshes, datasets or environments.

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
larger incomplete development batch remains separate. Reused baseline elapsed
times are not used for speedup claims; the paired regression compares non-timing
fields, while query and replay timings use matched execution conditions.
Subsequent heavy jobs
hold `/home/ubuntu/.cache/codex-heavy-work.lock`, run serially with one numerical
thread, and defer when available memory or load fails the resource checks.
New rendering checkpoints also receive a process-local affinity mask containing
one logical CPU before numerical libraries load. This bounds native concurrency
without changing other processes or global settings; [oneTBB's concurrency
queries respect process affinity](https://oneapi-src.github.io/oneTBB/main/tbb_userguide/Guiding_Task_Scheduler_Execution.html).

All accuracy checks use synthetic observations. This investigation does not
establish hardware calibration accuracy.
