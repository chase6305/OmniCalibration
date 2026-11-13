#!/usr/bin/env python3
"""Measure repeated saved-observation calls in isolated Linux processes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from omnicalib.reporting import environment_versions, write_json

if __package__:
    from .benchmark_proximity import (
        _isolated_python,
        baseline_module,
        execution_policy,
        source_fingerprint,
    )
    from .benchmark_replay import _replay_artifacts
else:
    from benchmark_proximity import (
        _isolated_python,
        baseline_module,
        execution_policy,
        source_fingerprint,
    )
    from benchmark_replay import _replay_artifacts


WORKER = r'''
import gc, hashlib, json, resource, sys, time
from pathlib import Path
startup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
from omnicalib.assets import calibrate_dataset
from omnicalib.reporting import json_safe, write_json
import trimesh

config, dataset, assets, output = map(Path, sys.argv[1:5])
collect_each, repeats = sys.argv[5] == '1', int(sys.argv[6])
assert Path(calibrate_dataset.__code__.co_filename).resolve().parent == Path.cwd() / 'omnicalib'

def snapshot():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    count = sum(isinstance(item, trimesh.Trimesh) for item in gc.get_objects())
    return {'rss_bytes': int(fields['VmRSS'].split()[0]) * 1024,
            'live_trimesh_objects': count, 'gc_counts': list(gc.get_count())}

gc.collect()
policy = {'automatic_enabled': gc.isenabled(), 'thresholds': list(gc.get_threshold()),
          'initial_full_collection': True, 'collect_before_each_call': collect_each}
initial = snapshot()
records, reports = [], []
for repeat in range(repeats):
    collected = gc.collect() if collect_each else None
    before = snapshot()
    started = time.perf_counter()
    result = calibrate_dataset(config, dataset, asset_root=assets, outer_iterations=30,
        solver_options={'trim_start_iteration': 2, 'uncertainty_step_fraction': 0.05,
                        'step_relaxation': 0.5})
    elapsed = time.perf_counter() - started
    records.append({'repeat': repeat + 1, 'collected_before_call': collected,
                    'before': before, 'after': snapshot(), 'elapsed_seconds': elapsed})
    reports.append(result)
    print(output.name, 'repeat', repeat + 1, json.dumps(records[-1]), file=sys.stderr, flush=True)
before_cleanup = snapshot()
collected = gc.collect()
after_cleanup = snapshot()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
hashes = []
for repeat, report in enumerate(reports, 1):
    write_json(output / f'repeat_{repeat}.json', report)
    normalized = json_safe(report)
    normalized.pop('elapsed_s', None)
    normalized.pop('render_seconds', None)
    hashes.append(hashlib.sha256(json.dumps(normalized, sort_keys=True, allow_nan=False).encode()).hexdigest())
print(json.dumps({'garbage_collection': policy, 'initial': initial, 'records': records,
                  'before_final_cleanup': before_cleanup, 'collected_final': collected,
                  'after_final_cleanup': after_cleanup, 'peak_rss_bytes': peak,
                  'startup_peak_rss_bytes': startup_peak, 'normalized_report_hashes': hashes}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Trusted local Git revision")
    parser.add_argument("--datasets", type=Path, default=Path("outputs"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("left", "right", "dual"), default="left")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Residency measurements require Linux /proc")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], text=True
    ).strip()
    root = Path(__file__).resolve().parents[1]
    config_name = f"configs/marvin_{args.mode}.json"
    config = root / config_name
    config_bytes = config.read_bytes()
    if config_bytes != subprocess.check_output(["git", "show", f"{revision}:{config_name}"]):
        parser.error("Lifetime comparison requires identical robot configurations")
    dataset = (args.datasets / f"marvin_{args.mode}.npz").resolve()
    inputs = {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    fingerprint = source_fingerprint(root / "omnicalib")
    policy = execution_policy()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    measurements = {}
    with baseline_module(revision) as (baseline, baseline_hash):
        baseline_root = Path(baseline.__file__).resolve().parent.parent
        manifest = {
            "baseline_commit": revision, "baseline_source_sha256": baseline_hash,
            "candidate_source_sha256": fingerprint, **inputs, "mode": args.mode,
            "repeats": args.repeats, "environment": environment_versions(),
            "execution_policy": policy,
            "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "method": "Four independent workers from a lightweight launcher; raw reports retained until after memory snapshots; explicit full collection initially in every worker; optional full collection before each API call",
        }
        directory = _replay_artifacts(args.output, manifest)
        print(f"Lifetime artifacts: {directory}", flush=True)
        for name, source, source_hash in (
            ("baseline", baseline_root, baseline_hash), ("optimized", root, fingerprint),
        ):
            for collect_each in (False, True):
                label = name + ("_gc" if collect_each else "_default")
                destination = directory / label
                arguments = [str(config), str(dataset), str(args.asset_root.resolve()),
                             str(destination.resolve()), str(int(collect_each)), str(args.repeats)]
                row = json.loads(_isolated_python(WORKER, arguments, source, environment))
                row["source_sha256"] = source_hash
                measurements[label] = row
                write_json(destination / "measurement.json", row)
                print(f"Completed {label}", flush=True)
    if source_fingerprint(root / "omnicalib") != fingerprint:
        raise RuntimeError("Numerical source changed during lifetime measurement")
    if config.read_bytes() != config_bytes or hashlib.sha256(dataset.read_bytes()).hexdigest() != inputs["dataset_sha256"]:
        raise RuntimeError("Lifetime inputs changed during measurement")
    hashes = {value for row in measurements.values() for value in row["normalized_report_hashes"]}
    report = {**manifest, "measurements": measurements, "exact_non_timing_reports": len(hashes) == 1,
              "artifact_directory": str(directory),
              "scope": "Residency after a fixed number of API calls for these inputs and GC settings; use benchmark_replay.py for alternating-order speed measurements"}
    write_json(directory / "summary.json", report)
    write_json(args.output, report)
    if not report["exact_non_timing_reports"]:
        raise SystemExit("Reports differ; inspect retained lifetime artifacts")


if __name__ == "__main__":
    main()
