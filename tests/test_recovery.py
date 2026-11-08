import numpy as np

from omnicalib.geometry import inverse, se3_exp
from omnicalib.head import calibrate_head
from omnicalib.lower import calibrate_lower
from omnicalib.synthetic import (
    head_problem,
    lower_problem,
    run_head,
    run_lower,
    run_upper,
    upper_problem,
)
from omnicalib.upper import DepthFrame, calibrate_upper, calibrate_wrist


def test_upper_fourteen_joints_and_six_dof_wrist():
    result = run_upper()
    assert result["success"]
    assert result["diagnostics"]["rank"] == 20
    assert result["max_joint_error_deg"] < 1e-4
    assert result["chest_translation_error_mm"] < 1e-3
    for wrist in result["wrist_handeye"]:
        assert wrist["rank"] == 6 and wrist["success"]
        assert wrist["translation_error_mm"] < 1e-3


def test_upper_with_depth_noise():
    result = run_upper(noise_m=0.0002)
    assert result["success"]
    assert result["max_joint_error_deg"] < 0.2
    assert 0.1 < result["point_to_plane_rms_mm"] < 0.4


def test_single_arm_fit_uses_other_arm_states_per_frame():
    full, surfaces, original, nominal, truth, camera = upper_problem()
    selected = full.select(full.active_names[:7])
    # Here the other arm's actual states are independently known fixture data.
    frames = [
        DepthFrame(
            f.q[:7],
            f.points_camera,
            dict(zip(full.active_names[7:], f.q[7:] + truth[7:], strict=True)),
        )
        for f in original
    ]
    fitted = calibrate_upper(selected, surfaces, frames, nominal, allow_restarts=False)
    assert fitted.converged and fitted.success and fitted.quality_passed
    assert fitted.diagnostics.rank == 13
    np.testing.assert_allclose(fitted.offsets, truth[:7], atol=1e-6)
    np.testing.assert_allclose(fitted.chest_transform, camera, atol=1e-6)
    missing = [DepthFrame(f.q, f.points_camera) for f in frames]
    wrong = calibrate_upper(selected, surfaces, missing, nominal, allow_restarts=False)
    assert not wrong.quality_passed


def test_lower_twelve_joint_conditional_recovery():
    result = run_lower()
    assert result["success"]
    assert result["posterior_diagnostics"]["rank"] == 12
    assert result["max_joint_error_deg"] < 1e-4


def test_repeated_stance_cannot_identify_leg_offsets():
    robot, q, _, _ = lower_problem()
    repeated = np.repeat(q[:1], 4, axis=0)
    result = calibrate_lower(robot, repeated, "left_foot", "right_foot")
    assert result.data_diagnostics.rank < 12
    assert not any(result.data_diagnostics.supported)


def test_pure_translation_cannot_identify_wrist_translation():
    x = se3_exp([0.02, -0.01, 0.03, 0.1, 0.2, -0.1])
    a = np.array(
        [
            se3_exp([0.1, 0, 0, 0, 0, 0]),
            se3_exp([0, 0.1, 0, 0, 0, 0]),
            se3_exp([0, 0, 0.1, 0, 0, 0]),
        ]
    )
    b = np.array([inverse(x) @ motion @ x for motion in a])
    _, diag, _ = calibrate_wrist(a, b)
    assert diag.rank < 6
    assert not any(diag.supported)


def test_head_common_rotation_and_retained_translation():
    result = run_head()
    assert result["success"] and result["centers_retained"]
    assert result["diagnostics"]["rank"] == 3
    assert result["max_rotation_error_deg"] < 1e-4


def test_head_single_baseline_leaves_rotation_gauge():
    initial, centers, motions, pairs, _ = head_problem()
    result = calibrate_head(initial, centers, motions, pairs[:1])
    assert result["diagnostics"].rank == 2
    assert not any(result["diagnostics"].supported)
