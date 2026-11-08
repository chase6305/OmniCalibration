#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${OMNICALIB_PYTHON:-.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$task_python" - <<'PY'
import hashlib
from pathlib import Path
from omnicalib import __version__
from omnicalib.reporting import write_json
h = hashlib.sha256()
for path in sorted(Path('omnicalib').glob('*.py')):
    h.update(path.name.encode())
    h.update(path.read_bytes())
write_json('outputs/reproduce_manifest.json', {'version': __version__, 'source_sha256': h.hexdigest()})
print(f'Numerical source: v{__version__} / {h.hexdigest()}')
PY
"$task_python" -m pytest -q
"$task_python" -m omnicalib demo --output outputs/synthetic.json
"$task_python" -m omnicalib demo --noisy --output outputs/synthetic_noisy.json
for mode in left right dual; do
  "$task_python" -m omnicalib asset-demo --config "configs/marvin_${mode}.json" \
    --matching triangle --render-depth --profile robust --frames 8 --iterations 30 \
    --noise-m 0.0002 --camera-position 1.6 0 1.1 --camera-target 0 0 1.0 \
    --save-dataset "outputs/marvin_${mode}.npz" --output "outputs/marvin_${mode}_visible.json"
done
"$task_python" -m omnicalib calibrate --config configs/marvin_left.json \
  --input outputs/marvin_left.npz --profile robust --iterations 30 \
  --output outputs/marvin_left_replayed.json
for side in left right; do
  "$task_python" -m omnicalib asset-demo --config "configs/marvin_standalone_${side}.json" \
    --fixed-camera --matching triangle --independent-sampling \
    --output "outputs/marvin_standalone_${side}.json"
done
"$task_python" -m omnicalib asset-demo --config configs/marvin_left.json \
  --matching triangle --render-depth --no-anchor \
  --output outputs/marvin_left_no_anchor.json
"$task_python" scripts/preview_depth.py --random-posture --output outputs/depth_preview.png
"$task_python" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('outputs')
for mode in ('left', 'right', 'dual'):
    result = json.loads((root / f'marvin_{mode}_visible.json').read_text())
    assert result['proposal']['block_supported'], (mode, result['proposal']['block_reasons'])
    assert result['max_joint_error_deg'] < 0.1, (mode, result['max_joint_error_deg'])
replay = json.loads((root / 'marvin_left_replayed.json').read_text())
original = json.loads((root / 'marvin_left_visible.json').read_text())
np.testing.assert_allclose(replay['offsets_deg'], original['estimated_offsets_deg'], atol=1e-9)
for side in ('left', 'right'):
    standalone = json.loads((root / f'marvin_standalone_{side}.json').read_text())
    assert standalone['proposal']['block_supported']
gauge = json.loads((root / 'marvin_left_no_anchor.json').read_text())
assert gauge['diagnostics']['rank'] == 12
assert not gauge['proposal']['block_supported']
print('Artifact checks passed: supported recovery, exact replay, retained gauge.')
import hashlib
h = hashlib.sha256()
for path in sorted(Path('omnicalib').glob('*.py')):
    h.update(path.name.encode())
    h.update(path.read_bytes())
assert h.hexdigest() == json.loads((root / 'reproduce_manifest.json').read_text())['source_sha256']
print('Numerical source unchanged throughout reproduction.')
PY
