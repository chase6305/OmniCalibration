#!/usr/bin/env python3
"""Compare exact query outputs and timings against a trusted Git revision."""

import argparse
import json
import subprocess
import time
from statistics import median

import numpy as np
import trimesh

from omnicalib.proximity import TriangleIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Trusted local Git revision")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], text=True
    ).strip()
    namespace = {}
    # Execute only the explicitly selected, trusted repository revision.
    source = subprocess.check_output(
        ["git", "show", f"{revision}:omnicalib/proximity.py"], text=True
    )
    exec(compile(source, f"{revision}:omnicalib/proximity.py", "exec"), namespace)
    baseline_index = namespace["TriangleIndex"]
    rng = np.random.default_rng(20261109)
    results = []
    for subdivisions, count in [(2, 2000), (3, 2000), (3, 10000)]:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.3)
        points = rng.normal(0, 0.25, (count, 3))
        old, new = baseline_index(mesh), TriangleIndex(mesh)
        for limit in [None, 0.02]:
            for expected, actual in zip(old.query(points, limit), new.query(points, limit), strict=True):
                np.testing.assert_array_equal(actual, expected)
            timings = {"baseline": [], "optimized": []}
            for repeat in range(args.repeats):
                pairs = [("baseline", old), ("optimized", new)]
                if repeat % 2:
                    pairs.reverse()
                for name, index in pairs:
                    started = time.perf_counter()
                    index.query(points, limit)
                    timings[name].append(time.perf_counter() - started)
            baseline = median(timings["baseline"])
            optimized = median(timings["optimized"])
            results.append({
                "faces": len(mesh.faces), "points": count, "max_distance": limit,
                "exact_match": True, "baseline_seconds": baseline,
                "optimized_seconds": optimized, "speedup": baseline / optimized,
            })
    print(json.dumps({"baseline_commit": revision, "seed": 20261109,
                      "repeats": args.repeats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
