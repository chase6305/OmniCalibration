# Readability refactor: validation

This refactor keeps version **0.3.2** and reorganizes the implementation and
documentation. Its numerical source fingerprint differs from the earlier
v0.3.2 snapshot because source text is part of the fingerprint. Historical
validation reports still refer to their own frozen implementations.

## What changed

- The README shrinks from 184 to 82 lines. Detailed recipes move to the
  [usage guide](USAGE.md); a [developer guide](DEVELOPMENT.md) maps source files,
  solver phases, checks, and provenance rules.
- CLI parser construction, command handlers, and shared solver-option translation
  are separate. Long adapter calls and result construction use named arguments.
- Camera initialization, one-shot continuation, and camera retries have explicit
  helpers. Continuation and retry selection share one candidate predicate.
- Synthetic and observation-only reports share their solver-evidence mapping.
  Existing field names, values, convergence thresholds, and proposal rules remain
  unchanged. Dictionary display order can differ.
- The budget report presents parameters and per-scene results in tables; the
  previously observed **0.201°** stress-case error remains visible.

## Completed checks

| Check | Result |
|---|---|
| Test suite | 105 passed |
| Ruff | Passed |
| Full reproduction | Passed; exact NPZ replay and retained unobservable case |
| CLI help | Root command and all four subcommands unchanged |
| Original 64-run v0.3.2 plan | Exact replay match; both batches audited |
| Broader 240-pair replay | 239 pairs match; one baseline run unfinished at time budget |
| Extracted source bundle | Full reproduction passed with the same frozen source |
| Minimal wheel installation | Numerical demo passed with only NumPy and SciPy |
| Moving-root wrist example | Both synthetic wrist proposals supported |
| Documentation | Local links/anchors, 15 CLI examples, shell/Python syntax, GitHub rendering checked |

The first replay uses `configs/benchmark_budget_v032.json` and the same eight
seeds, compared against the already frozen outputs. The broader replay runs
`configs/benchmark_final.json` on baseline commit `3933a7d` and the refactor, with
seeds 12001–12010, 12012, and 12014. It includes historical residual-rejection and
continuation cases. These are paired regression checks, **not fresh holdout
accuracy results**, and do not replace the historical 400-run assessment.

Some workers resumed from per-case checkpoints after reducing concurrency to
eight to resolve resource contention. One benchmark worker and a preview workflow had exited with status 137; checkpoints were resumed and the complete workflow rerun successfully, with the interrupted log retained. Source, inputs, environment versions,
seeds, and solver settings stayed frozen; only complete atomic run records are
counted. Wall-clock measurements are excluded from the comparison.

One baseline run had not returned at the one-hour optimization budget. It was stopped without a convergence or accuracy claim; its paired refactor result, all prior completed records, and the original full plan remain intact.

## Exact comparison

[`scripts/compare_benchmarks.py`](../scripts/compare_benchmarks.py) checks every
run field, including estimates, covariance, association counts, complete solver
histories, inputs, quality flags, and proposal decisions. It excludes only:

- The top-level numerical source fingerprint.
- `result.elapsed_s` and `result.render_seconds`.

It uses exact equality, with no rounding or numerical tolerance. Missing or
duplicate cases, incompatible plans/environments, invalid JSON numbers,
changed schema, and exception records prevent a passing comparison. Provenance
and referenced asset hashes are audited separately for each source; different
source snapshots are never merged into one batch.

The comparison is evidence for the exercised inputs, not proof of behavior for
all possible datasets. All calibration runs use synthetic observations and
assumed camera placements. Internal acceptance still does not establish
hardware accuracy; see the [stress-case limitations](BUDGET_v032.md).

## Artifacts

Compact results: [regression JSON](results/readability_regression.json). The complete 64-run replay passed; 239 additional completed pairs matched exactly. The extended 240-pair batch is **incomplete**, not a passed full batch.
Full records and logs are local under `outputs/readability/`:

- `frozen.json`: source identity and baseline commit.
- `budget_complete/`, `budget_comparison.json`: 64-run replay and audit.
- `wide_before_partial/`, `wide_after_partial/`: audited study batches (baseline 239/240, refactor 240/240); missing keys retained in reports.
- `tests_final.log`, `reproduce_final.log`: tests and full reproduction.
- `core_demo.json`, `staged_dual.json`: installed-wheel and wrist examples.

Commands for independent audits and comparisons are in the
[developer guide](DEVELOPMENT.md#preserve-experiment-provenance).
