#!/usr/bin/env python3
"""Export small, portable aggregate/CSV/figure artifacts for repository review."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from omnicalib.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    portable = {
        key: summary[key]
        for key in (
            "source_sha256",
            "plan",
            "seeds",
            "environment",
            "completed_runs",
            "planned_runs",
            "groups",
        )
    }
    portable["input_hashes"] = {
        name: {"config_sha256": row["sha256"], "urdf_sha256": row["urdf_sha256"]}
        for name, row in summary.get("input_configs", {}).items()
    }
    write_json(args.docs / "results" / "validation_summary.json", portable)
    rows = []
    expected = {
        case["id"]: case.get("expected_observable", True) for case in summary["plan"]["cases"]
    }
    for path in sorted((args.directory / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if "result" not in record:
            continue
        r = record["result"]
        rows.append(
            {
                "scenario": record["case"],
                "seed": record["seed"],
                "expected_observable": expected[record["case"]],
                "accepted": r["proposal"]["block_supported"],
                "rank": r["diagnostics"]["rank"],
                "parameters": r["diagnostics"]["parameters"],
                "max_joint_error_deg": r["max_joint_error_deg"],
                "rms_joint_error_deg": r["rms_joint_error_deg"],
                "heldout_model_point_rms_mm": r["heldout_model_point_rms_mm"],
                "chest_translation_error_mm": r["chest_translation_error_mm"],
                "chest_rotation_error_deg": r["chest_rotation_error_deg"],
                "residual_rms_mm": r["point_to_plane_rms_mm"],
                "normalized_rms": r["quality"]["normalized_rms"],
                "depth_scale": r.get("depth_scale", 1.0),
                "depth_scale_error_ppm": r.get("depth_scale_error_ppm"),
                "point_noise_model": r["quality"].get("point_noise_model", "isotropic"),
                "rejection_fraction": r["rejection_fraction"],
                "iterations": len(r["history"]),
                "elapsed_s": r.get("elapsed_s"),
            }
        )
    output = args.docs / "results" / "metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    figures = args.docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for path in (args.directory / "figures").glob("*.*"):
        if path.suffix in (".png", ".svg"):
            shutil.copy2(path, figures / path.name)
    print(f"Portable results: {args.docs / 'results'}")


if __name__ == "__main__":
    main()
