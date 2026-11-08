from pathlib import Path

import numpy as np
import pytest

from omnicalib.assets import load_asset_config
from omnicalib.geometry import pose, so3_exp
from omnicalib.motion import sample_reaching_poses, solve_ik


def test_marvin_dual_reaching_ik_respects_limits_and_targets():
    root = Path(__file__).resolve().parents[1]
    if not (root.parent / "HumanoidAssets/Marvin_M6_S_CCS_696_V4.0").exists():
        pytest.skip("Marvin assets not installed")
    config, _, robot = load_asset_config(root / "configs/marvin_dual.json")
    zero = robot.fk(np.zeros(14))
    targets = {
        entry["link"]: pose(
            so3_exp([0, 0, np.deg2rad(entry["orientation_z_delta_deg"])])
            @ zero[entry["link"]][:3, :3],
            entry["position"],
        )
        for entry in config["reaching_targets"]
    }
    result = solve_ik(robot, targets)
    assert result.success
    assert max(result.position_errors_m.values()) < 1e-5
    by_name = {j.name: j for j in robot.joints}
    assert all(
        by_name[name].lower <= q <= by_name[name].upper
        for name, q in zip(robot.active_names, result.q, strict=True)
    )
    generated = sample_reaching_poses(robot, targets, 3, seed=9)
    assert generated.shape == (3, 14)
    assert np.linalg.norm(generated[0] - generated[-1]) > 0.01
