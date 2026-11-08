import numpy as np

from omnicalib.geometry import pose, rotation_error_deg, se3_exp, so3_exp
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame
from omnicalib.wrist import calibrate_eye_in_hand


def test_corrected_arm_and_moving_root_are_projected_before_hand_eye():
    joints = [
        Joint("a", "base", "one", pose(translation=[0.1, 0, 0]), np.array([0.0, 0, 1.0])),
        Joint("b", "one", "two", pose(translation=[0.2, 0, 0]), np.array([0.0, 1.0, 0])),
        Joint("c", "two", "tool", pose(translation=[0.2, 0, 0]), np.array([1.0, 0, 0])),
    ]
    robot = Robot("base", joints)
    rng = np.random.default_rng(84)
    q = rng.uniform(-0.7, 0.7, (16, 3))
    offsets = np.array([0.01, -0.015, 0.02])
    frames = [DepthFrame(row - offsets, np.empty((0, 3))) for row in q]
    world_root = np.array([se3_exp(rng.normal(0, 0.03, 6)) for _ in q])
    truth = se3_exp([0.04, -0.02, 0.03, 0.1, -0.2, 0.05])
    cameras = np.array(
        [base @ robot.fk(row)["tool"] @ truth for base, row in zip(world_root, q, strict=True)]
    )
    calibrated = calibrate_eye_in_hand(
        robot, frames, "tool", cameras, offsets, world_root_poses=world_root
    )
    assert calibrated.success and calibrated.diagnostics.rank == 6
    np.testing.assert_allclose(calibrated.transform, truth, atol=1e-8)
    wrong = calibrate_eye_in_hand(robot, frames, "tool", cameras, offsets)
    assert rotation_error_deg(wrong.transform[:3, :3], truth[:3, :3]) > 0.1


def test_single_axis_wrist_motion_is_rejected_by_observability():
    robot = Robot("base", [Joint("a", "base", "tool", pose(), np.array([0.0, 0, 1.0]))])
    q = np.linspace(-0.6, 0.6, 8)
    frames = [DepthFrame([angle], np.empty((0, 3))) for angle in q]
    camera = pose(so3_exp([0.1, 0.2, 0.3]), [0.02, -0.03, 0.04])
    poses = np.array([robot.fk([angle])["tool"] @ camera for angle in q])
    fit = calibrate_eye_in_hand(robot, frames, "tool", poses, [0.0])
    assert fit.diagnostics.rank < 6
    assert not any(fit.diagnostics.supported)
