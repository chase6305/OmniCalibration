import numpy as np

from omnicalib.assets import correction_proposal
from omnicalib.geometry import pose
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame, Surface, calibrate_upper


def test_converged_full_rank_fit_with_bad_noise_consistency_is_retained():
    robot = Robot("base", [Joint("a", "base", "arm", pose(), np.array([0.0, 0, 1.0]))])
    surface = Surface("arm", [[0.1, 0, 0]], [[0, 1.0, 0]])
    points = np.tile([[0.1, 0, 0]], (100, 1))
    points[:, 1] = np.tile([-0.005, 0.005], 50)
    frames = [DepthFrame([0.0], points), DepthFrame([0.0], points)]
    fit = calibrate_upper(
        robot,
        [surface],
        frames,
        np.eye(4),
        estimate_camera=False,
        sigma=0.001,
        allow_restarts=False,
    )
    assert fit.success and fit.converged
    assert fit.diagnostics.rank == 1
    assert fit.quality["normalized_rms"] > 4.0
    assert not fit.quality_passed
    proposal = correction_proposal(["a"], fit, "base")
    assert proposal["block_reasons"] == ["residual_noise_inconsistent"]
    assert all(p["decision"] == "retain" for p in proposal["parameters"])


def test_low_residual_on_a_small_subset_does_not_certify_the_observation():
    robot = Robot("base", [Joint("a", "base", "arm", pose(), np.array([0.0, 0.0, 1.0]))])
    surface = Surface("arm", [[0.1, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    points = np.concatenate([np.tile([[0.1, 0.0, 0.0]], (100, 1)), np.ones((100, 3))])
    fit = calibrate_upper(
        robot,
        [surface],
        [DepthFrame([0.0], points)] * 2,
        np.eye(4),
        estimate_camera=False,
        allow_restarts=False,
    )
    assert fit.converged and fit.diagnostics.rank == 1 and fit.rms_m < 1e-10
    assert not fit.quality_passed and not fit.quality["coverage_passed"]
    proposal = correction_proposal(["a"], fit, "base")
    assert proposal["block_reasons"] == ["insufficient_correspondence_coverage"]
