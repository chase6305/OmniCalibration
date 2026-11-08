#!/usr/bin/env python3
"""Merge disjoint checkpointed batches from one frozen implementation/plan."""

import argparse
import json
import shutil
import time
from pathlib import Path

from benchmark import summarize

from omnicalib.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifests = [
        json.loads((directory / "manifest.json").read_text()) for directory in args.directories
    ]
    first = manifests[0]
    fingerprint = first["source_sha256"]
    canonical_plan = json.dumps(first["plan"], sort_keys=True)
    if any(
        m["source_sha256"] != fingerprint or json.dumps(m["plan"], sort_keys=True) != canonical_plan
        for m in manifests
    ):
        raise SystemExit("Cannot combine different implementations/plans")
    expected_configs = {
        name: (value["sha256"], value["urdf_sha256"])
        for name, value in first.get("input_configs", {}).items()
    }
    for manifest in manifests:
        got = {
            name: (value["sha256"], value["urdf_sha256"])
            for name, value in manifest.get("input_configs", {}).items()
        }
        if got != expected_configs:
            raise SystemExit("Input configurations or URDF hashes differ")
    seeds = sorted({seed for manifest in manifests for seed in manifest["seeds"]})
    cases = {case["id"] for case in first["plan"]["cases"]}
    records = {}
    duplicates = 0
    for directory in args.directories:
        for path in sorted((directory / "runs").glob("*.json")):
            row = json.loads(path.read_text())
            if (
                row.get("source_sha256") != fingerprint
                or row["case"] not in cases
                or row["seed"] not in seeds
            ):
                continue
            key = row["case"], row["seed"]
            if key in records:
                duplicates += 1
                if records[key]["settings"] != row["settings"]:
                    raise SystemExit("Duplicate experiment has different settings")
                continue
            records[key] = row
            write_json(args.output / "runs" / f"{key[0]}__seed_{key[1]}.json", row)
    snapshot = args.output / "source_snapshots" / fingerprint
    if not snapshot.exists():
        shutil.copytree(args.directories[0] / "source_snapshots" / fingerprint, snapshot)
    manifest = {
        **first,
        "seeds": seeds,
        "merged_from": [str(d) for d in args.directories],
        "merged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duplicate_checkpoints_ignored": duplicates,
    }
    for value in manifest.get("input_configs", {}).values():
        value["snapshot"] = str(snapshot / "input_configs" / f"{value['sha256']}.json")
    total = len(seeds) * len(cases)
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "summary.json",
        {
            **manifest,
            "completed_runs": len(records),
            "planned_runs": total,
            "groups": summarize(list(records.values())),
        },
    )
    print(f"Merged {len(records)}/{total} unique experiments -> {args.output}")
    if args.require_complete and len(records) != total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
