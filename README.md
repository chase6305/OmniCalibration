# OmniCalibration

Independent reproduction of [OmniCalib](https://arxiv.org/abs/2609.19582v2) for
single- and dual-arm robot kinematic calibration, with synthetic validation,
observability analysis, and reproducible benchmarks.

The current implementation estimates joint zero offsets and camera extrinsics
from robot-surface depth observations using Marvin URDF and mesh assets. It can
also estimate a depth scale and export a calibration proposal for review.

**Status:** algorithm implementation and synthetic validation. Hardware accuracy
has not been validated. Example camera placements are assumed; the code does
not write corrections to the URDF, encoders, or controller.

## Quick start

Tested on Linux x86_64 with Python 3.10; a CPU is sufficient. Run commands from
the repository root.

```bash
uv venv .venv --python python3.10
uv pip install --python .venv/bin/python -r requirements-lock.txt -e .

# Numerical demo; no robot assets required
.venv/bin/python -m omnicalib demo --output outputs/demo.json
```

Without `uv`, create a virtual environment with `python3.10 -m venv .venv` and
install with `.venv/bin/python -m pip install -r requirements-lock.txt -e .`.

## Calibrate one or both arms

The Marvin assets are supplied separately. Their default location is
`../HumanoidAssets/Marvin_M6_S_CCS_696_V4.0`; add `--asset-root /path/to/assets`
to use another location.

```bash
# Left arm: render synthetic depth, fit offsets/extrinsics, and save observations
.venv/bin/python -m omnicalib asset-demo \
  --config configs/marvin_left.json --matching triangle --render-depth \
  --profile robust --frames 8 --iterations 30 --noise-m 0.0002 \
  --camera-position 1.6 0 1.1 --camera-target 0 0 1.0 \
  --save-dataset outputs/left.npz --output outputs/left.json

# Replay saved observations without loading synthetic ground truth
.venv/bin/python -m omnicalib calibrate \
  --config configs/marvin_left.json --input outputs/left.npz \
  --profile robust --iterations 30 --output outputs/left_replayed.json
```

For the right arm or both arms, replace the config in the first command with
`configs/marvin_right.json` or `configs/marvin_dual.json`. Use the same config
when replaying its dataset. Defaults include a fixed body reference; without
one, shoulder offsets can be ambiguous with camera pose—even with both arms.

Reports are written to `outputs/`. Check `proposal.block_supported` and
`proposal.block_reasons` before interpreting a correction: solver `success`
alone does not establish observability, convergence, or accuracy.

## Validation and limits

Run the full local reproduction, including tests, asset demos, exact dataset
replay, and an unobservable case:

```bash
bash scripts/reproduce.sh
```

All reported experiments use synthetic data. A quadratic-noise stress case
reached **0.201°** worst joint error despite passing internal quality checks;
acceptance does not guarantee accuracy. Wrist calibration uses synthetic metric
poses; real image/VO frontends are not implemented. Head and lower-body modules
remain standalone numerical examples.

| Guide | Contents |
|---|---|
| [Usage](docs/USAGE.md) | Standalone arms, noise, depth scale, wrist stage, frame selection |
| [Dataset format](docs/DATA_FORMAT.md) | NPZ schema, units, coordinates, real-data requirements |
| [Method and scope](docs/REPRODUCTION.md) | Paper mapping, assumptions, observability, missing components |
| [Experiments](docs/EXPERIMENTS.md) | Frozen plans, results, audits, and historical comparisons |
| [Continuation validation](docs/BUDGET_v032.md) | Budgets, fallback behavior, 64 synthetic trials |
| [Development](docs/DEVELOPMENT.md) | Source map, provenance, [refactor checks](docs/READABILITY.md) |
