#!/usr/bin/env python3
"""Time exact saved-observation replay against a trusted Git revision."""

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path
from statistics import median

from benchmark_proximity import baseline_module, source_fingerprint
from compare_benchmarks import differences

from omnicalib import assets
from omnicalib.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Trusted local Git revision")
    parser.add_argument("--datasets", type=Path, default=Path("outputs"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("left", "right", "dual"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], text=True
    ).strip()
    root = Path(__file__).resolve().parents[1]
    config_name = f"configs/marvin_{args.mode}.json"
    config = root / config_name
    config_bytes = config.read_bytes()
    original_config = subprocess.check_output(["git", "show", f"{revision}:{config_name}"])
    if config_bytes != original_config:
        parser.error("Replay requires identical baseline/current robot configuration")
    dataset = (args.datasets / f"marvin_{args.mode}.npz").resolve()
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    fingerprint = source_fingerprint(root / "omnicalib")
    artifact_root = args.output.with_suffix("")
    artifact_root.mkdir(parents=True, exist_ok=True)
    options = {"trim_start_iteration": 2, "uncertainty_step_fraction": 0.05, "step_relaxation": 0.5}
    timings = {"baseline": [], "optimized": []}
    with baseline_module(revision) as (baseline, baseline_hash):
        old_assets = importlib.import_module(f"{baseline.__package__}.assets")
        for repeat in range(args.repeats):
            implementations = [("baseline", old_assets), ("optimized", assets)]
            if repeat % 2:
                implementations.reverse()
            reports = {}
            for name, implementation in implementations:
                started = time.perf_counter()
                result = implementation.calibrate_dataset(
                    config, dataset, asset_root=args.asset_root.resolve(),
                    outer_iterations=30, solver_options=options,
                )
                elapsed = time.perf_counter() - started
                timings[name].append(elapsed)
                reports[name] = result
                write_json(artifact_root / f"{name}_{repeat + 1}.json", result)
                print(f"{args.mode} repeat={repeat + 1} {name}: {elapsed:.3f}s", flush=True)
            changed = list(differences({"result": reports["baseline"]}, {"result": reports["optimized"]}))
            if changed:
                write_json(args.output, {"exact_match": False, "changed_paths": changed[:20]})
                raise SystemExit("Replay outputs differ; full reports retained for inspection")
    if source_fingerprint(root / "omnicalib") != fingerprint:
        raise RuntimeError("Numerical source changed during benchmark")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != dataset_hash or config.read_bytes() != config_bytes:
        raise RuntimeError("Replay inputs changed during benchmark")
    before, after = median(timings["baseline"]), median(timings["optimized"])
    report = {
        "baseline_commit": revision, "baseline_source_sha256": baseline_hash,
        "candidate_source_sha256": fingerprint, "mode": args.mode,
        "dataset_sha256": dataset_hash, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "environment": reports["optimized"]["environment"],
        "repeats": args.repeats, "alternating_order": True, "exact_match": True,
        "timings_seconds": timings, "baseline_median_seconds": before,
        "optimized_median_seconds": after, "speedup": before / after,
        "timed_region": "calibrate_dataset including model loading, indexing, solving and reporting",
        "validation": "Saved synthetic observations; not hardware accuracy",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
