# Development

Run commands from the repository root. The implementation and examples use
meters and radians internally; report fields with `_mm` or `_deg` convert units
for display. Coordinates and dataset requirements are defined in the
[dataset guide](DATA_FORMAT.md).

## Find the relevant code

| Area | Entry point | Responsibility |
|---|---|---|
| CLI | [`omnicalib/__main__.py`](../omnicalib/__main__.py) | Parse arguments, build shared solver options, dispatch commands |
| Robot assets | [`omnicalib/assets.py`](../omnicalib/assets.py) | Read URDF/meshes, generate synthetic observations, fit saved datasets, produce proposals |
| Dataset I/O | [`omnicalib/dataset.py`](../omnicalib/dataset.py) | Validate and serialize named-joint NPZ observations; keep truth optional |
| Upper-body fit | [`omnicalib/upper.py`](../omnicalib/upper.py) | Refresh associations, optimize offsets/extrinsics, continue or retry candidates |
| Parameter layout | [`omnicalib/parameters.py`](../omnicalib/parameters.py) | Joint/camera/scale slices, bounds, and convergence tolerances |
| Diagnostics | [`omnicalib/observability.py`](../omnicalib/observability.py) | Local rank, conditioning, physical-unit covariance, and uncertainty limits |
| Depth and noise | [`omnicalib/render.py`](../omnicalib/render.py), [`omnicalib/noise.py`](../omnicalib/noise.py) | Synthetic first-hit depth and conditional covariance models |
| Evaluation | [`scripts/benchmark.py`](../scripts/benchmark.py) | Frozen source/input provenance, independent seeds, checkpointed records |

The CLI handlers are separate from parser construction. `_solver_options`
combines a named recipe with explicit command-line options for both asset demos
and observation-only calibration. A new shared solver option should have one
translation path, so those entry points do not silently diverge.

## Read the upper-body solver

`calibrate_upper` keeps validation, the outer ICP loop, and final diagnostics
visible together. Three helpers contain distinct phases:

- `_initialize_camera` uses geometry independent of estimated joint offsets. It
  updates only locally constrained camera directions and adds no final prior.
- `_continue_local_fit` refines one eligible terminal estimate. It records failed
  continuations and returns a replacement only when all candidate gates pass.
- `_retry_camera_fit` probes neighboring camera translations and selects from
  measured residuals, convergence, and diagnostics. It does not read truth.

`_UpperObjective` freezes associations and covariance during each inner solve.
`_valid_candidate` applies the same acceptance predicate to continuation and
retry candidates. The final correction proposal additionally records reasons
and units; the raw estimate remains available even when the proposal retains
nominal values.

`UpperResult.success`, `converged`, and `all_inner_success` describe different
things: the final inner solve, outer ICP stability, and the complete selected
solve history. Do not replace them with one flag. Continuation limits apply to
a single continuation, not total runtime; see [budget semantics](BUDGET_v032.md).

## Checks

The full local environment is pinned in `requirements-lock.txt`. After
installing it as described in the [README](../README.md), run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check omnicalib scripts tests
bash -n scripts/reproduce.sh
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS `launch_testing` plugins
in the original workstation shell. Asset integration tests skip when the
external Marvin directory is absent. The full reproduction script requires
those assets and the simulation dependencies:

```bash
bash scripts/reproduce.sh
```

It checks synthetic recovery for both single arms and the dual arm, exact NPZ
replay, standalone arms, retained rank deficiency, and a consistent numerical
source fingerprint throughout the run. Output goes to `outputs/`, which is
excluded from Git.

## Preserve experiment provenance

Every benchmark records a numerical source fingerprint, the plan, random seeds,
configuration snapshots, and referenced asset hashes. Even a behavior-preserving
source refactor changes that fingerprint. Run it into a **new output directory**;
do not resume or merge records from another fingerprint. Historical reports
remain evidence for their own snapshots, not new results for current code.

Rendered batches can retain substantial mesh memory. Increase concurrent
processes gradually while checking available RAM; CPU count alone does not
describe the usable budget. A resource interruption can be resumed from atomic
per-case checkpoints with the same frozen inputs. An unfinished case is rerun,
and is never counted as a completed estimate.

[Experiment commands and frozen plans](EXPERIMENTS.md) explain resume, merge,
and audit. Commit compact summaries and reproducible plans; keep bulky runtime
artifacts under `outputs/`. A passing internal proposal gate is not a claim of
physical accuracy, as the [stress-case results](BUDGET_v032.md) demonstrate.

For a behavior-preserving refactor, audit both complete batches independently,
then compare them without merging their different source fingerprints:

```bash
.venv/bin/python scripts/audit_results.py outputs/before
.venv/bin/python scripts/audit_results.py outputs/after
.venv/bin/python scripts/compare_benchmarks.py outputs/before outputs/after \
  --output outputs/comparison.json
```

The comparison requires the same complete plan, seeds, and environment. It
checks every run field exactly, including the entire solver history and input
hashes, except the source fingerprint and the result's `elapsed_s` and
`render_seconds`. A numerical change, missing record, changed schema, or exception
record prevents a passing comparison. Matching these runs is regression evidence
for the chosen inputs, not a proof for all possible data.

For exact triangle query performance, see the [linear candidate selection
benchmark](PROXIMITY_OPTIMIZATION.md), including baseline revision, exact output
checks, and workload-dependent timing results.
