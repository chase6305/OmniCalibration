import numpy as np
import pytest

from omnicalib.geometry import inverse, mean_se3, pose, se3_exp, se3_log, so3_exp
from omnicalib.head import project_link_trajectory, relative_motions, sampson_residual
from omnicalib.observability import analyze


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.1, 2.9, np.pi - 1e-7])
def test_se3_round_trip(angle):
    xi = np.r_[0.3, -0.4, 0.1, np.array([1.0, 2.0, -1.0]) / np.sqrt(6) * angle]
    transform = se3_exp(xi)
    np.testing.assert_allclose(se3_exp(se3_log(transform)), transform, atol=1e-9)
    np.testing.assert_allclose(inverse(transform) @ transform, np.eye(4), atol=1e-12)


def test_mean_respects_common_transform():
    common = se3_exp([0.1, 0.2, -0.1, 0.2, -0.1, 0.05])
    transforms = [common @ se3_exp([0.001 * k, 0, 0, 0, 0, 0.001 * k]) for k in (-1, 0, 1)]
    np.testing.assert_allclose(mean_se3(transforms), common, atol=1e-10)


def test_rank_deficiency_is_not_zero_uncertainty():
    diag = analyze(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert diag.rank == 2 and diag.parameters == 3
    assert np.isinf(diag.sigma[2])
    assert not any(diag.supported)
    assert np.isinf(diag.condition)


def test_scaled_observability_is_invariant_to_translation_units():
    j = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 1.0]])
    meters = analyze(j, parameter_scale=[0.01, 0.02])
    millimeters = analyze(j @ np.diag([0.001, 1.0]), parameter_scale=[10.0, 0.02])
    assert meters.rank == millimeters.rank == 2
    assert meters.scaled_condition == pytest.approx(millimeters.scaled_condition)
    np.testing.assert_allclose(meters.sigma, np.array(millimeters.sigma) * [0.001, 1.0])


def test_numerical_noise_is_not_information():
    diag = analyze(np.eye(3) * 1e-12)
    assert diag.rank == 0
    assert all(np.isinf(diag.sigma))


def test_live_tf_compensation_recovers_camera_motion():
    world_base = np.array([pose(so3_exp([0, 0, 0.1 * k]), [0.1 * k, 0, 0]) for k in range(5)])
    base_link = np.array([pose(so3_exp([0.08 * k, 0, 0]), [0, 0, 0.3]) for k in range(5)])
    x = se3_exp([0.01, -0.03, 0.02, 0.2, 0.1, -0.1])
    link = project_link_trajectory(world_base, base_link)
    camera = link @ x
    a, b = relative_motions(link), relative_motions(camera)
    np.testing.assert_allclose(a @ x, x @ b, atol=1e-12)
    frozen = relative_motions(
        project_link_trajectory(world_base, np.repeat(base_link[:1], 5, axis=0))
    )
    assert np.linalg.norm(frozen @ x - x @ b) > 0.1


def test_sampson_rejects_zero_baseline():
    with pytest.raises(ValueError, match="baseline"):
        sampson_residual(
            np.eye(3),
            np.eye(3),
            np.zeros(3),
            np.zeros(3),
            np.array([[0, 0, 1]]),
            np.array([[0, 0, 1]]),
        )
