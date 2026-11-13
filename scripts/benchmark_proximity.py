#!/usr/bin/env python3
"""Compare exact query outputs and timings against a trusted Git revision."""

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median

import numpy as np

from omnicalib.proximity import TriangleIndex
from omnicalib.reporting import environment_versions, write_json


def source_fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execution_policy():
    from rtree import core

    return {
        "thread_variables": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else None,
        "libspatialindex": core.rt.SIDX_Version().decode(),
    }


@contextmanager
def baseline_module(revision):
    """Load the selected revision's package, including its relative imports."""
    archive = subprocess.check_output(["git", "archive", revision, "omnicalib"])
    name = "_omnicalib_proximity_baseline"
    with tempfile.TemporaryDirectory(prefix="omnicalib-proximity-") as directory:
        # Only the explicitly selected trusted repository revision is executed.
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(directory)
        package = Path(directory) / "omnicalib"
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            proximity = importlib.import_module(f"{name}.proximity")
            yield proximity, source_fingerprint(package)
        finally:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]


def benchmark(baseline_index, repeats):
    import trimesh

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
            for repeat in range(repeats):
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
                "timings_seconds": timings,
            })
    return results


def _isolated_python(code, arguments, source_root, environment):
    """Fork the measured worker from a small process, not the warmed benchmark.

    Linux preserves resource usage across exec. An intermediate interpreter
    avoids counting the large parent's pre-exec resident image as worker peak.
    """
    launcher = "import subprocess, sys\nraise SystemExit(subprocess.call(sys.argv[1:]))"
    return subprocess.check_output(
        [sys.executable, "-c", launcher, sys.executable, "-c", code, *arguments],
        cwd=source_root, env=environment, text=True,
    )


def query_memory(source_root, count):
    """Measure independent query processes launched from a small interpreter."""
    code = """
import hashlib, json, resource, sys, time
from pathlib import Path
units = 1024 if sys.platform != 'darwin' else 1
startup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units
import numpy as np
import trimesh
from omnicalib.proximity import TriangleIndex
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
points = np.random.default_rng(20261109).normal(0, 0.25, (int(sys.argv[1]), 3))
index = TriangleIndex(mesh)
started = time.perf_counter()
result = index.query(points)
elapsed = time.perf_counter() - started
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units
vm_hwm = None
if sys.platform == 'linux':
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            vm_hwm = int(line.split()[1]) * 1024
digest = hashlib.sha256()
for values in result:
    digest.update(values.tobytes())
print(json.dumps({'points': len(points), 'faces': len(mesh.faces),
                 'elapsed_seconds': elapsed, 'result_sha256': digest.hexdigest(),
                 'peak_rss_bytes': peak, 'startup_peak_rss_bytes': startup_peak,
                 'linux_vm_hwm_bytes': vm_hwm,
                 'peak_method': 'getrusage after a lightweight launcher'}))
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = _isolated_python(code, [str(count)], source_root, environment)
    return json.loads(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Trusted local Git revision")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Save a standalone JSON report")
    parser.add_argument(
        "--memory-points", type=int, default=0,
        help="Also measure an unlimited query's peak RSS in fresh Unix processes",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.memory_points < 0:
        parser.error("--memory-points must be nonnegative")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"], text=True
    ).strip()
    package = Path(sys.modules[TriangleIndex.__module__].__file__).parent
    candidate_fingerprint = source_fingerprint(package)
    memory = None
    with baseline_module(revision) as (baseline, fingerprint):
        results = benchmark(baseline.TriangleIndex, args.repeats)
        if args.memory_points:
            memory = {
                "baseline": query_memory(Path(baseline.__file__).parent.parent, args.memory_points),
                "optimized": query_memory(package.parent, args.memory_points),
            }
            if memory["baseline"]["result_sha256"] != memory["optimized"]["result_sha256"]:
                raise RuntimeError("Memory benchmark query outputs differ")
    if source_fingerprint(package) != candidate_fingerprint:
        raise RuntimeError("Numerical source changed during benchmark; repeat from a frozen source")
    report = {
        "baseline_commit": revision,
        "baseline_source_sha256": fingerprint,
        "candidate_source_sha256": candidate_fingerprint,
        "environment": environment_versions(),
        "execution_policy": execution_policy(),
        "seed": 20261109, "repeats": args.repeats, "results": results,
    }
    if memory is not None:
        report["memory"] = memory
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
