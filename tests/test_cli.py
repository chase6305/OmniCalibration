import json
import os
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "omnicalib", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )


def test_cli_writes_strict_json_and_reports_input_errors(tmp_path):
    path = tmp_path / "head.json"
    process = _run("demo", "--module", "head", "--output", str(path))
    assert process.returncode == 0, process.stderr
    result = json.loads(path.read_text())
    assert result["results"][0]["success"]
    missing = _run(
        "calibrate",
        "--config",
        str(tmp_path / "missing.json"),
        "--input",
        str(tmp_path / "missing.npz"),
    )
    assert missing.returncode == 2
    assert "error:" in missing.stderr
    assert "Traceback" not in missing.stderr
