import json
import os
import subprocess
import sys

import numpy as np
import pytest


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


@pytest.mark.parametrize("step", [None, "0.75"])
def test_asset_and_dataset_commands_preserve_shared_solver_options(tmp_path, monkeypatch, step):
    """Exercise both adapters, including profile defaults and explicit overrides."""
    from omnicalib import __main__ as cli
    from omnicalib import assets

    calls = {}

    def capture(command):
        def run(*args, **kwargs):
            calls[command] = kwargs
            return {
                "success": True,
                "icp_converged": False,
                "diagnostics": {"rank": 12, "parameters": 13},
                "proposal": {"block_supported": False, "block_reasons": ["rank_deficient"]},
                "max_joint_error_deg": 0.01,
                "depth_scale": 1.0,
            }

        return run

    monkeypatch.setattr(assets, "run_asset_demo", capture("asset-demo"))
    monkeypatch.setattr(assets, "calibrate_dataset", capture("calibrate"))
    shared = [
        "--config",
        "model.json",
        "--profile",
        "robust",
        "--loss",
        "linear",
        "--assumed-encoder-sigma-deg",
        "0.01",
        "--point-noise-model",
        "axial",
        "--axial-floor-fraction",
        "0.8",
        "--axial-range-power",
        "2",
        "--max-rejection-fraction",
        "0.1",
        "--depth-scale-log-bound",
        "0.02",
        "--estimate-depth-scale",
    ]
    if step is not None:
        shared += ["--step-relaxation", step]
    for command in ("asset-demo", "calibrate"):
        extra = (
            ["--noise-m", "0.002", "--encoder-noise-deg", "0.03"]
            if command == "asset-demo"
            else ["--input", "observations.npz", "--max-camera-translation-sigma-mm", "2"]
        )
        output = tmp_path / f"{command}.json"
        monkeypatch.setattr(
            sys, "argv", ["omnicalib", command, *shared, *extra, "--output", str(output)]
        )
        cli.main()
        assert not json.loads(output.read_text())["proposal"]["block_supported"]

    expected = {
        "trim_start_iteration": 2,
        "uncertainty_step_fraction": 0.05,
        "step_relaxation": 0.5 if step is None else 0.75,
        "loss": "linear",
        "encoder_sigma_rad": float(np.deg2rad(0.01)),
        "depth_scale_log_bound": 0.02,
        "point_noise_model": "axial",
        "axial_floor_fraction": 0.8,
        "axial_range_power": 2.0,
        "max_rejection_fraction": 0.1,
    }
    assert calls["asset-demo"]["solver_options"] == calls["calibrate"]["solver_options"] == expected
    assert calls["asset-demo"]["noise_m"] == 0.002
    assert calls["asset-demo"]["encoder_noise_deg"] == 0.03
    assert calls["calibrate"]["camera_translation_sigma_limit_m"] == 0.002
    assert all(call["estimate_depth_scale"] for call in calls.values())
