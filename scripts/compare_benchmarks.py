#!/usr/bin/env python3
"""Compare complete batches exactly after a behavior-preserving source refactor.

Unlike merge_benchmarks, this permits different source fingerprints. Audit each
batch separately first. Only the run's source hash and two measured wall-clock
fields are excluded; estimates, diagnostics, histories and decisions must match.
"""

import argparse
import json
import math
from pathlib import Path

from omnicalib.reporting import write_json

IGNORED_PATHS = {
    ("source_sha256",),
    ("result", "elapsed_s"),
    ("result", "render_seconds"),
}


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number: {value}")
    return number


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    return json.loads(
        path.read_text(),
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_keys,
    )


def load_batch(directory):
    """Require every planned case/seed once, from its declared source."""
    manifest = _read_json(directory / "manifest.json")
    expected = {
        (case["id"], seed) for case in manifest["plan"]["cases"] for seed in manifest["seeds"]
    }
    if len(expected) != len(manifest["plan"]["cases"]) * len(manifest["seeds"]):
        raise ValueError(f"Duplicate cases or seeds in {directory}'s plan")
    records = {}
    for path in sorted((directory / "runs").glob("*.json")):
        row = _read_json(path)
        key = row["case"], row["seed"]
        if key in records:
            raise ValueError(f"Duplicate case/seed in {directory}: {key}")
        if row["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"Mixed source fingerprints in {directory}: {key}")
        records[key] = row
    if set(records) != expected:
        raise ValueError(f"Incomplete or unexpected cases in {directory}")
    if not expected:
        raise ValueError(f"Empty comparison plan in {directory}")
    return manifest, records


def differences(before, after, path=()):
    """Yield changed paths; do not round away numerical or status changes."""
    if path in IGNORED_PATHS:
        return
    if type(before) is not type(after):
        yield path
    elif isinstance(before, dict):
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                yield (*path, key)
            else:
                yield from differences(before[key], after[key], (*path, key))
    elif isinstance(before, list):
        if len(before) != len(after):
            yield path
        else:
            for index, (left, right) in enumerate(zip(before, after, strict=True)):
                yield from differences(left, right, (*path, index))
    elif before != after:
        yield path


def compare_batches(baseline, candidate):
    before_manifest, before_rows = load_batch(baseline)
    after_manifest, after_rows = load_batch(candidate)
    for key in ("plan", "seeds", "environment"):
        if before_manifest[key] != after_manifest[key]:
            raise ValueError(f"Comparison {key} differs")
    if before_rows.keys() != after_rows.keys():
        raise ValueError("Comparison case/seed sets differ")
    changed = []
    difference_count = 0
    for case, seed in sorted(before_rows):
        for path in differences(before_rows[case, seed], after_rows[case, seed]):
            difference_count += 1
            if len(changed) < 20:
                changed.append({"case": case, "seed": seed, "path": list(path)})
    exception_count = sum("error" in row for row in after_rows.values())
    return {
        "baseline_source_sha256": before_manifest["source_sha256"],
        "candidate_source_sha256": after_manifest["source_sha256"],
        "compared_runs": len(before_rows),
        "exception_records": exception_count,
        "exact_data_match": difference_count == 0,
        "passed": difference_count == 0 and exception_count == 0,
        "difference_count": difference_count,
        "first_differences": changed,
        "excluded_paths": [list(path) for path in sorted(IGNORED_PATHS)],
        "validation": "Synthetic regression comparison; not hardware accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_batches(args.baseline, args.candidate)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.output:
        write_json(args.output, report)
    print(
        f"Compared {report['compared_runs']} runs: "
        f"{report['difference_count']} changed data fields, "
        f"{report['exception_records']} exception records"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
