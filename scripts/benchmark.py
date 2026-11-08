#!/usr/bin/env python3
"""Checkpointed synthetic benchmark. All estimators see observations only."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# Avoid oversubscribing the shared workstation; set before importing NumPy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from omnicalib.assets import run_asset_demo
from omnicalib.render import DepthNoise
from omnicalib.reporting import write_json

# Load numerical modules before recording a source snapshot. Later edits to
# working files do not alter this benchmark process's imported implementation.
for module_name in ("dataset", "excitation", "proximity"):
    importlib.import_module(f"omnicalib.{module_name}")


def source_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted((root / "omnicalib").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_resume_inputs(
    directory, previous, plan, seeds, fingerprint, environment, root, asset_root_override=None
):
    """Refuse stale caches before changing an existing batch's metadata."""
    for name, old, new in (
        ("source", previous["source_sha256"], fingerprint),
        ("plan", previous["plan"], plan),
        ("seeds", sorted(previous["seeds"]), sorted(seeds)),
        ("environment", previous["environment"], environment),
    ):
        if old != new:
            raise ValueError(f"Resume {name} differs; use a new output directory")
    for name, recorded in previous["input_configs"].items():
        path = (root / name).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
            raise ValueError("Resume config differs; use a new output directory")
        config = json.loads(path.read_text())
        base = (
            Path(asset_root_override).resolve()
            if asset_root_override
            else (path.parent.parent / config["asset_root"]).resolve()
        )
        if base != Path(recorded["asset_root"]).resolve():
            raise ValueError("Resume asset location differs; use a new output directory")
        if (
            hashlib.sha256((base / config["urdf"]).read_bytes()).hexdigest()
            != recorded["urdf_sha256"]
        ):
            raise ValueError("Resume URDF differs; use a new output directory")
    checked = set()
    for path in (directory / "runs").glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("source_sha256") != fingerprint:
            raise ValueError("Existing runs use a different source; use a new output directory")
        for mesh in row.get("result", {}).get("mesh_sources", []):
            key = mesh["path"], mesh["sha256"]
            if key not in checked:
                if hashlib.sha256(Path(mesh["path"]).read_bytes()).hexdigest() != mesh["sha256"]:
                    raise ValueError("Resume mesh differs; use a new output directory")
                checked.add(key)


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["case"], []).append(record)
    result = {}
    for name, rows in groups.items():
        valid = [r for r in rows if "error" not in r]
        summary = {
            "runs": len(rows),
            "completed": len(valid),
            "solver_successes": sum(r["result"]["success"] for r in valid),
            "converged": sum(r["result"]["icp_converged"] for r in valid),
            "full_rank": sum(
                r["result"]["diagnostics"]["rank"] == r["result"]["diagnostics"]["parameters"]
                for r in valid
            ),
            "accepted": sum(
                all(
                    p["decision"] == "propose"
                    for p in r["result"]["proposal"]["parameters"][
                        : len(r["result"]["joint_names"])
                    ]
                )
                for r in valid
            ),
            "within_0p1deg_reference": sum(
                r["result"]["max_joint_error_deg"] <= 0.1 for r in valid
            ),
        }
        standardized = []
        for record in valid:
            result_row = record["result"]
            diagnostics = result_row["diagnostics"]
            if diagnostics["rank"] != diagnostics["parameters"]:
                continue
            count = len(result_row["joint_names"])
            sigma_deg = np.rad2deg(np.asarray(diagnostics["sigma"][:count], dtype=float))
            errors = np.asarray(result_row["joint_error_deg"])
            finite = np.isfinite(sigma_deg) & (sigma_deg > 0)
            standardized.extend((errors[finite] / sigma_deg[finite]).tolist())
        if standardized:
            standardized = np.asarray(standardized)
            summary["marginal_95pct_coverage"] = float(np.mean(np.abs(standardized) <= 1.96))
            summary["rms_standardized_error"] = float(np.sqrt(np.mean(standardized**2)))
        for metric in (
            "max_joint_error_deg",
            "rms_joint_error_deg",
            "heldout_model_point_rms_mm",
            "chest_translation_error_mm",
            "chest_rotation_error_deg",
            "elapsed_s",
            "depth_scale_error_ppm",
            "depth_scale",
            "heldout_model_point_before_rms_mm",
            "heldout_camera_point_rms_mm",
            "heldout_camera_point_before_rms_mm",
        ):
            data = np.array([r["result"][metric] for r in valid if metric in r["result"]])
            if len(data):
                summary[metric] = {
                    "mean": float(data.mean()),
                    "median": float(np.median(data)),
                    "p95": float(np.quantile(data, 0.95)),
                    "max": float(data.max()),
                }
        result[name] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/benchmark_development.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_development"))
    parser.add_argument("--case", action="append", help="run only selected case IDs")
    parser.add_argument("--seeds", type=int, nargs="+", help="override plan seeds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="recompute metrics from the existing source-version runs",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.summarize_only:
        manifest = json.loads((args.output / "manifest.json").read_text())
        records = [json.loads(path.read_text()) for path in (args.output / "runs").glob("*.json")]
        records = [row for row in records if row.get("source_sha256") == manifest["source_sha256"]]
        write_json(
            args.output / "summary.json",
            {
                **manifest,
                "completed_runs": len(records),
                "planned_runs": len(manifest["plan"]["cases"]) * len(manifest["seeds"]),
                "groups": summarize(records),
            },
        )
        print(f"Summary: {args.output / 'summary.json'}")
        return
    plan = json.loads(args.plan.read_text())
    seeds = plan["seeds"] if args.seeds is None else args.seeds
    fingerprint = source_fingerprint(root)
    records = []
    manifest = {
        "source_sha256": fingerprint,
        "plan": plan,
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "trimesh", "rtree", "embreex")
            },
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation": "synthetic modeled geometry; no hardware or control",
    }
    existing_manifest = args.output / "manifest.json"
    if existing_manifest.exists():
        if not args.resume:
            raise SystemExit("Batch already exists; use --resume or a new output directory")
        previous = json.loads(existing_manifest.read_text())
        try:
            verify_resume_inputs(
                args.output,
                previous,
                plan,
                seeds,
                fingerprint,
                manifest["environment"],
                root,
                args.asset_root,
            )
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error)) from None
        manifest["started_utc"] = previous["started_utc"]
        manifest["resumed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.output / "manifest.json", manifest)
    snapshot = args.output / "source_snapshots" / fingerprint
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for folder in ("omnicalib", "configs"):
            shutil.copytree(
                root / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__")
            )
        for filename in ("pyproject.toml", "requirements-lock.txt"):
            shutil.copy2(root / filename, snapshot / filename)
        shutil.copy2(Path(__file__), snapshot / "benchmark.py")
    input_configs = {}
    for case in plan["cases"]:
        name = case["config"]
        if name in input_configs:
            continue
        source_config = (root / name).resolve()
        config_bytes = source_config.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes)
        asset_root = (
            args.asset_root.resolve()
            if args.asset_root
            else (source_config.parent.parent / config["asset_root"]).resolve()
        )
        destination = snapshot / "input_configs" / f"{config_hash}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(config_bytes)
        input_configs[name] = {
            "snapshot": str(destination),
            "sha256": config_hash,
            "asset_root": str(asset_root),
            "urdf_sha256": hashlib.sha256((asset_root / config["urdf"]).read_bytes()).hexdigest(),
        }
    manifest["input_configs"] = input_configs
    write_json(args.output / "manifest.json", manifest)
    cases = [c for c in plan["cases"] if args.case is None or c["id"] in args.case]
    total = len(cases) * len(seeds)
    for index, (case, seed) in enumerate((c, s) for c in cases for s in seeds):
        settings = {**plan.get("defaults", {}), **case.get("settings", {})}
        key = hashlib.sha256(
            json.dumps(
                {
                    "source": fingerprint,
                    "settings": settings,
                    "case": case,
                    "seed": seed,
                    "config_sha256": input_configs[case["config"]]["sha256"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        path = args.output / "runs" / f"{case['id']}__seed_{seed}__{key}.json"
        if args.resume and path.exists():
            record = json.loads(path.read_text())
            print(f"[{index + 1}/{total}] cached {case['id']} seed={seed}", flush=True)
        else:
            print(f"[{index + 1}/{total}] {case['id']} seed={seed}", flush=True)
            record = {
                "case": case["id"],
                "seed": seed,
                "source_sha256": fingerprint,
                "config_sha256": input_configs[case["config"]]["sha256"],
                "settings": settings,
                "expected_observable": case.get("expected_observable", True),
            }
            started = time.perf_counter()
            try:
                call_settings = settings.copy()
                noise = DepthNoise(
                    **call_settings.pop("depth_noise", {"sigma_m": settings.get("noise_m", 0.0002)})
                )
                input_config = input_configs[case["config"]]
                call_settings.setdefault("asset_root", input_config["asset_root"])
                result = run_asset_demo(
                    input_config["snapshot"], seed=seed, depth_noise=noise, **call_settings
                )
                if result["urdf_sha256"] != input_config["urdf_sha256"]:
                    raise RuntimeError("Input URDF changed during benchmark")
                result["elapsed_s"] = time.perf_counter() - started
                result["depth_noise"] = asdict(noise)
                record["result"] = result
                print(
                    f"  max={result['max_joint_error_deg']:.5g}deg rank={result['diagnostics']['rank']}/{result['diagnostics']['parameters']} converged={result['icp_converged']} time={result['elapsed_s']:.1f}s",
                    flush=True,
                )
            except Exception as error:
                record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                print(f"  ERROR: {type(error).__name__}: {error}", flush=True)
            write_json(path, record)
        records.append(record)
        write_json(
            args.output / "summary.json",
            {
                **manifest,
                "completed_runs": len(records),
                "planned_runs": total,
                "groups": summarize(records),
            },
        )
    if any("error" in row for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
