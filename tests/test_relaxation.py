import numpy as np

from omnicalib.geometry import pose, transform_points
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame, Surface, calibrate_upper


def _problem():
    robot = Robot("base", [Joint("a", "base", "arm", pose(), np.array([0.0, 0, 1.0]))])
    surface = Surface("arm", [[0.3, 0, 0]], [[0, 1.0, 0]])
    frames = [
        DepthFrame([q], transform_points(robot.fk([q + 0.05])["arm"], surface.points))
        for q in (-0.2, 0.2)
    ]
    return robot, [surface], frames


def test_tiny_accepted_step_cannot_fake_convergence():
    robot, surfaces, frames = _problem()
    result = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        step_relaxation=1e-8,
        relaxation_step_ratio=np.inf,
        outer_iterations=2,
    )
    assert not result.converged
    assert abs(result.offsets[0]) < 1e-7
    assert all(row["tolerance_step_ratio"] > 100 for row in result.history)


def test_relaxed_updates_recover_the_same_offset():
    robot, surfaces, frames = _problem()
    result = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        step_relaxation=0.5,
        relaxation_step_ratio=np.inf,
        outer_iterations=30,
    )
    assert result.converged and result.quality_passed
    np.testing.assert_allclose(result.offsets, [0.05], atol=2e-5)


def test_axial_covariance_must_be_active_before_claiming_convergence():
    robot, _, _ = _problem()
    surface = Surface("arm", [[0.3, 0, 1.0]], [[0, 1.0, 0]])
    frames = [
        DepthFrame([q], transform_points(robot.fk([q + 0.01])["arm"], surface.points))
        for q in (-0.2, 0.2)
    ]
    result = calibrate_upper(
        robot,
        [surface],
        frames,
        np.eye(4),
        estimate_camera=False,
        point_noise_model="axial",
        axial_start_iteration=5,
        outer_iterations=12,
    )
    assert result.converged and result.iterations >= 7
    assert result.history[-1]["point_noise_model"] == "axial"


def test_inexact_early_solves_can_recover_but_zero_steps_cannot_fake_success():
    robot, surfaces, frames = _problem()
    fit = calibrate_upper(
        robot, surfaces, frames, np.eye(4), estimate_camera=False, max_nfev=3, outer_iterations=20
    )
    assert fit.success and fit.converged and not fit.all_inner_success
    assert fit.history[0]["solver_status"] == 0
    assert all(h["solver_success"] for h in fit.history[-2:])
    np.testing.assert_allclose(fit.offsets, [0.05], atol=1e-6)
    stalled = calibrate_upper(
        robot, surfaces, frames, np.eye(4), estimate_camera=False, max_nfev=2, outer_iterations=4
    )
    assert not stalled.success and not stalled.converged


def test_noise_consistent_terminal_fit_continues_without_camera_restart():
    robot, surfaces, frames = _problem()
    primary = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        outer_iterations=2,
        allow_restarts=False,
        local_continuation=False,
    )
    assert primary.success and primary.quality_passed and not primary.converged
    continued = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        outer_iterations=2,
        allow_restarts=True,
    )
    assert continued.converged and continued.success
    trace = continued.initialization["local_continuation"]
    assert trace["primary_iterations"] == 2 and len(trace["primary_history"]) == 2
    assert continued.iterations == len(continued.history) == 4
    assert [h["phase"] for h in continued.history] == ["primary"] * 2 + ["local_continuation"] * 2
    np.testing.assert_allclose(continued.offsets, [0.05], atol=1e-6)


def test_unfinished_continuation_is_reported_without_claiming_convergence():
    robot, surfaces, frames = _problem()
    fit = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        outer_iterations=2,
        convergence_patience=3,
        allow_restarts=False,
    )
    assert not fit.converged
    trace = fit.initialization["local_continuation"]
    assert trace["used"] and not trace["converged"]
    assert len(trace["primary_history"]) == len(trace["continuation_history"]) == 2


def test_continuation_budget_is_enforced_without_relaxing_stop_rule():
    robot, surfaces, frames = _problem()
    fit = calibrate_upper(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        outer_iterations=2,
        continuation_iterations=1,
        allow_restarts=False,
    )
    assert not fit.converged
    assert len(fit.initialization["local_continuation"]["continuation_history"]) == 1


def test_continuation_value_error_retains_primary_candidate(monkeypatch):
    import omnicalib.upper as upper

    original = upper.calibrate_upper

    def fail_child(*args, **kwargs):
        if kwargs.get("initial") is not None:
            raise ValueError("synthetic association failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(upper, "calibrate_upper", fail_child)
    robot, surfaces, frames = _problem()
    fit = original(
        robot,
        surfaces,
        frames,
        np.eye(4),
        estimate_camera=False,
        outer_iterations=2,
        allow_restarts=False,
    )
    assert not fit.converged and fit.quality_passed
    assert fit.initialization["local_continuation"]["error"] == "synthetic association failure"
    np.testing.assert_allclose(fit.offsets, [0.05], atol=1e-6)
