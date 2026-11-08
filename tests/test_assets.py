"""Optional local-asset integration tests; core suite works without Marvin files."""

from pathlib import Path

import numpy as np
import pytest

from omnicalib.assets import load_asset_config, run_asset_demo
from omnicalib.geometry import transform_points
from omnicalib.kinematics import Robot
from omnicalib.observability import jacobian

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ASSET = CONFIGS.parent.parent / "HumanoidAssets/Marvin_M6_S_CCS_696_V4.0"
pytestmark = pytest.mark.skipif(not ASSET.is_dir(), reason="Marvin assets not installed")


def test_full_and_modular_marvin_frames_agree():
    full = Robot.from_urdf(ASSET / "robot_with_ee.urdf")
    values = dict.fromkeys(full.active_names, 0.0)
    for side in ("left", "right"):
        modular = Robot.from_urdf(ASSET / f"{side}_arm.urdf")
        q = np.linspace(-0.2, 0.3, 7)
        values.update(zip(modular.active_names, q, strict=True))
        full_tf = full.fk(values)
        local_tf = modular.fk(q)
        for link, transform in local_tf.items():
            np.testing.assert_allclose(full_tf[link], full_tf[modular.root] @ transform, atol=1e-12)


@pytest.mark.parametrize("mode,n", [("left", 7), ("right", 7), ("dual", 14)])
def test_marvin_joint_selection(mode, n):
    _, _, robot = load_asset_config(CONFIGS / f"marvin_{mode}.json")
    assert len(robot.active_names) == n
    assert len(robot.fk(np.zeros(n))) == 30


def test_selected_arm_spatial_jacobian_matches_finite_differences():
    _, _, robot = load_asset_config(CONFIGS / "marvin_left.json")
    q = np.linspace(-0.3, 0.4, 7)
    transforms, screws = robot.fk_with_jacobian(q)
    point = np.array([[0.02, -0.03, 0.07]])
    for link in ("left_ee", "base_link", "right_ee", "wrist_pitch_l_j5_link"):
        world = transform_points(transforms[link], point)[0]
        analytic = screws[link][:3] + np.cross(screws[link][3:].T, world).T
        numeric = jacobian(lambda x, link=link: transform_points(robot.fk(x)[link], point)[0], q)
        np.testing.assert_allclose(analytic, numeric, atol=2e-9)


def test_noncalibrated_joint_motion_is_frame_specific():
    _, _, robot = load_asset_config(CONFIGS / "marvin_left.json")
    left = np.zeros(7)
    original = robot.fk(left)
    moved = robot.fk(left, fixed_positions={"SHOULDER_PITCH_R_J1": 0.3})
    np.testing.assert_allclose(original["left_ee"], moved["left_ee"])
    assert np.linalg.norm(original["right_ee"] - moved["right_ee"]) > 0.1
    with pytest.raises(ValueError, match="uncalibrated"):
        robot.fk(left, fixed_positions={"SHOULDER_PITCH_L_J1": 0.3})


@pytest.mark.parametrize("mode,rank", [("left", 12), ("right", 12), ("dual", 19)])
def test_marvin_arm_gauge_is_retained(mode, rank):
    pytest.importorskip("trimesh")
    result = run_asset_demo(
        CONFIGS / f"marvin_{mode}.json", include_anchor=False, frames_count=5, samples_per_link=35
    )
    assert result["diagnostics"]["rank"] == rank
    assert all(p["decision"] == "retain" for p in result["proposal"]["parameters"])


@pytest.mark.parametrize("side", ["left", "right"])
def test_standalone_arm_recovers_with_known_camera(side):
    pytest.importorskip("trimesh")
    result = run_asset_demo(
        CONFIGS / f"marvin_standalone_{side}.json",
        estimate_camera=False,
        frames_count=5,
        samples_per_link=35,
    )
    assert result["success"]
    assert result["diagnostics"]["rank"] == 7
    assert result["max_joint_error_deg"] < 1e-4
    assert all(p["decision"] == "retain" for p in result["proposal"]["parameters"][7:])
