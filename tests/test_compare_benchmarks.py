import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.compare_benchmarks import compare_batches, differences


@pytest.mark.parametrize(
    "before,after",
    [
        ({"offset": 0.1}, {"offset": 0.10000000000001}),
        ({"success": True}, {"success": False}),
        ({"history": [1, 2]}, {"history": [1]}),
        ({"success": True}, {"success": 1}),
        ({"rank": 12}, {}),
        ({"quality": {"elapsed_s": 1}}, {"quality": {"elapsed_s": 2}}),
    ],
)
def test_comparison_does_not_hide_numbers_status_or_schema_changes(before, after):
    assert list(differences(before, after))


def _batch(directory, source):
    (directory / "runs").mkdir(parents=True)
    manifest = {
        "source_sha256": source,
        "plan": {"cases": [{"id": "left"}]},
        "seeds": [7],
        "environment": {"python": "3.10"},
    }
    row = {
        "case": "left",
        "seed": 7,
        "source_sha256": source,
        "config_sha256": "same input",
        "result": {"offsets": [0.01], "elapsed_s": 1.0, "render_seconds": 0.5},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "runs/left.json").write_text(json.dumps(row))
    return row


def test_only_source_hash_and_known_timing_fields_can_differ(tmp_path):
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    _batch(baseline, "old source")
    row = _batch(candidate, "new source")
    row["result"]["elapsed_s"] = 99.0
    row["result"]["render_seconds"] = 10.0
    (candidate / "runs/left.json").write_text(json.dumps(row))
    report = compare_batches(baseline, candidate)
    assert report["exact_data_match"] and report["compared_runs"] == 1
    row["config_sha256"] = "changed input"
    (candidate / "runs/left.json").write_text(json.dumps(row))
    assert not compare_batches(baseline, candidate)["exact_data_match"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "source",
        "plan",
        "nonfinite",
        "overflow",
        "duplicate_key",
        "duplicate_seed",
    ],
)
def test_incomplete_or_incompatible_batches_cannot_claim_equivalence(tmp_path, problem):
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    _batch(baseline, "old")
    row = _batch(candidate, "new")
    path = candidate / "runs/left.json"
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        (candidate / "runs/copy.json").write_text(path.read_text())
    elif problem == "source":
        row["source_sha256"] = "unexpected"
        path.write_text(json.dumps(row))
    elif problem == "plan":
        manifest_path = candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["plan"]["purpose"] = "changed plan"
        manifest_path.write_text(json.dumps(manifest))
    elif problem == "nonfinite":
        row["result"]["offsets"] = [float("nan")]
        path.write_text(json.dumps(row))
    elif problem == "overflow":
        path.write_text(path.read_text().replace("0.01", "1e999"))
    elif problem == "duplicate_key":
        path.write_text(path.read_text().replace('"seed": 7', '"seed": 7, "seed": 7'))
    else:
        manifest_path = candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["seeds"] = [7, 7]
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compare_batches(baseline, candidate)


def test_identical_exceptions_do_not_count_as_passing_numerical_regression(tmp_path):
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    for directory in (baseline, candidate):
        row = _batch(directory, directory.name)
        del row["result"]
        row["error"] = {"type": "ValueError", "message": "no correspondences"}
        (directory / "runs/left.json").write_text(json.dumps(row))
    report = compare_batches(baseline, candidate)
    assert report["exact_data_match"]
    assert report["exception_records"] == 1
    assert not report["passed"]


@pytest.mark.parametrize("malformed,expected_status", [(False, 1), (True, 2)])
def test_cli_returns_failure_and_keeps_diagnostics_reviewable(tmp_path, malformed, expected_status):
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    _batch(baseline, "old")
    row = _batch(candidate, "new")
    row["result"]["offsets"] = [float("nan") if malformed else 0.02]
    (candidate / "runs/left.json").write_text(json.dumps(row))
    output = tmp_path / "comparison.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/compare_benchmarks.py"),
            str(baseline),
            str(candidate),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    assert process.returncode == expected_status
    assert "Traceback" not in process.stderr
    if not malformed:
        report = json.loads(output.read_text())
        assert not report["passed"]
        assert report["first_differences"][0]["path"] == ["result", "offsets", 0]
