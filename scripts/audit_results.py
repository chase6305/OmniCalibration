#!/usr/bin/env python3
"""Check provenance, artifact consistency, and acceptance invariants of a batch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from omnicalib.reporting import write_json


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON number {value} in {path}")

    return json.loads(path.read_text(), parse_constant=invalid)


def source_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_continuations(value, budget):
    """Check retained traces; copies in nested diagnostics are checked as well."""
    if isinstance(value, list):
        return sum(audit_continuations(item, budget) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    for name, item in value.items():
        if name == "local_continuation" and isinstance(item, dict) and item.get("used"):
            if "error" in item:
                assert not item["converged"] and isinstance(item["error"], str)
            else:
                assert "continuation_history" in item, "Missing continuation history"
                assert len(item["continuation_history"]) <= budget, "Continuation exceeded budget"
            count += 1
        count += audit_continuations(item, budget)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-root", type=Path, help="relocate the original single asset root")
    args = parser.parse_args()
    manifest = read_json(args.directory / "manifest.json")
    fingerprint = manifest["source_sha256"]
    snapshot = args.directory / "source_snapshots" / fingerprint
    assert source_hash(snapshot / "omnicalib") == fingerprint, "Frozen source digest mismatch"
    current_hash = source_hash(Path(__file__).resolve().parents[1] / "omnicalib")
    for config in manifest["input_configs"].values():
        frozen = snapshot / "input_configs" / f"{config['sha256']}.json"
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == config["sha256"], (
            "Frozen configuration digest mismatch"
        )
    cases = {c["id"]: c for c in manifest["plan"]["cases"]}
    expected = {(case, seed) for case in cases for seed in manifest["seeds"]}
    seen, asset_hashes, exceptional, rejected, inaccurate = set(), {}, [], [], []
    completed = accepted = 0
    for path in sorted((args.directory / "runs").glob("*.json")):
        row = read_json(path)
        key = row["case"], row["seed"]
        assert key in expected and key not in seen, f"Unexpected/duplicate key {key}"
        seen.add(key)
        assert row["source_sha256"] == fingerprint, f"Mixed source {key}"
        config = manifest["input_configs"][cases[row["case"]]["config"]]
        assert row["config_sha256"] == config["sha256"], f"Mixed input config {key}"
        if "error" in row:
            exceptional.append({"case": key[0], "seed": key[1], "error": row["error"]})
            continue
        r = row["result"]
        options = row["settings"].get("solver_options", {})
        continuation_budget = options.get("continuation_iterations")
        if continuation_budget is None:
            continuation_budget = row["settings"].get("outer_iterations", 12)
        audit_continuations(r.get("initialization", {}), continuation_budget)
        completed += 1
        assert r["urdf_sha256"] == config["urdf_sha256"], f"Mixed URDF {key}"
        sources = r["mesh_sources"] + [{"path": r["urdf"], "sha256": r["urdf_sha256"]}]
        for source in sources:
            filename, digest = source["path"], source["sha256"]
            assert asset_hashes.get(filename, digest) == digest, (
                f"Asset changed across runs: {filename}"
            )
            asset_hashes[filename] = digest
        diag = r["diagnostics"]
        n = len(r["joint_names"])
        p = n + 6 * bool(r["estimate_camera"]) + bool(r.get("estimate_depth_scale", False))
        assert diag["parameters"] == p and len(diag["sigma"]) == p, f"Parameter layout {key}"
        assert 0 <= diag["rank"] <= p
        if diag["rank"] == p:
            covariance = np.asarray(diag["covariance"])
            assert covariance.shape == (p, p) and np.isfinite(covariance).all()
            np.testing.assert_allclose(
                np.diag(covariance), np.square(diag["sigma"]), rtol=1e-9, atol=1e-18
            )
        else:
            assert diag["covariance"] is None
        profile = r["residual_profile"]
        assert profile["overall"]["count"] == r["correspondences"]
        assert sum(x["count"] for x in profile["frames"]) == r["correspondences"]
        assert sum(x["count"] for x in profile["links"].values()) == r["correspondences"]
        np.testing.assert_allclose(
            profile["overall"]["rms_mm"], r["point_to_plane_rms_mm"], rtol=1e-10
        )
        if r["icp_converged"]:
            assert all(h["solver_success"] for h in r["history"][-2:]), (
                f"Unstable terminal solve {key}"
            )
        proposal = r["proposal"]
        assert proposal["block_supported"] == (not proposal["block_reasons"])
        if proposal["block_supported"]:
            assert (
                r["success"] and r["icp_converged"] and r["quality_passed"] and not r["at_bounds"]
            )
            assert diag["rank"] == p and all(diag["supported"])
            assert all(d["decision"] == "propose" for d in proposal["parameters"][:n])
            accepted += 1
        else:
            assert all(d["decision"] == "retain" for d in proposal["parameters"])
            rejected.append({"case": key[0], "seed": key[1], "reasons": proposal["block_reasons"]})
        if cases[key[0]].get("expected_observable", True) and r["max_joint_error_deg"] > 0.1:
            inaccurate.append(
                {
                    "case": key[0],
                    "seed": key[1],
                    "error_deg": r["max_joint_error_deg"],
                    "accepted": proposal["block_supported"],
                }
            )
    missing = sorted(expected - seen)
    if not args.allow_incomplete:
        assert not missing, f"Missing {len(missing)} expected runs"
    original_roots = {Path(c["asset_root"]).resolve() for c in manifest["input_configs"].values()}
    if args.asset_root and len(original_roots) != 1:
        raise ValueError("A single --asset-root requires one original asset root")
    for filename, digest in asset_hashes.items():
        actual = Path(filename)
        if args.asset_root:
            actual = args.asset_root.resolve() / actual.relative_to(next(iter(original_roots)))
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest, (
            f"Current asset differs: {filename}"
        )
    result = {
        "source_sha256": fingerprint,
        "working_source_matches_snapshot": current_hash == fingerprint,
        "expected": len(expected),
        "recorded": len(seen),
        "completed": completed,
        "accepted": accepted,
        "missing": missing,
        "exceptional": exceptional,
        "rejected": rejected,
        "above_0p1deg_reference": inaccurate,
        "verified_asset_files": asset_hashes,
        "asset_root_override": str(args.asset_root.resolve()) if args.asset_root else None,
        "checked": [
            "strict JSON",
            "unique expected keys",
            "frozen source/config/URDF/mesh hashes",
            "parameter layout",
            "covariance units",
            "residual aggregation",
            "terminal solver stability",
            "retained continuation iteration caps and exception status",
            "whole-block acceptance decisions",
        ],
    }
    output = args.output or args.directory / "audit.json"
    write_json(output, result)
    print(
        f"Audit passed: {len(seen)}/{len(expected)} records, {len(asset_hashes)} unchanged asset files; {output}"
    )


if __name__ == "__main__":
    main()
