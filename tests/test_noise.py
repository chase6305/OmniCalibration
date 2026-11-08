import numpy as np

from omnicalib.noise import LowRankWhitener, axial_plane_factors


def test_low_rank_whitener_matches_dense_covariance_inverse():
    rng = np.random.default_rng(24)
    j = rng.normal(size=(30, 3))
    sigma = np.array([0.2, 0.1, 0.3])
    whitener = LowRankWhitener.from_jacobian(j, sigma)
    dense = np.eye(30) + (j * sigma) @ (j * sigma).T
    transform = whitener.apply(np.eye(30))
    np.testing.assert_allclose(transform @ dense @ transform.T, np.eye(30), atol=1e-12)
    point = rng.normal(size=30)
    np.testing.assert_allclose(
        np.dot(whitener.apply(point), whitener.apply(point)),
        point @ np.linalg.solve(dense, point),
        atol=1e-10,
    )


def test_duplicate_points_do_not_erase_shared_encoder_uncertainty():
    sigma_encoder = 0.01
    variances = []
    for count in (10, 10000):
        j = np.ones((count, 1)) / 0.001
        whitened = LowRankWhitener.from_jacobian(j, [sigma_encoder]).apply(j)
        variances.append(float(1 / (whitened.T @ whitened)[0, 0]))
    assert variances[-1] >= sigma_encoder**2
    np.testing.assert_allclose(
        variances, [sigma_encoder**2 + 0.001**2 / n for n in (10, 10000)], rtol=1e-8
    )


def test_axial_plane_covariance_matches_optical_z_perturbations():
    from omnicalib.geometry import so3_exp

    rng = np.random.default_rng(25)
    points = np.array([[0.3, 0.4, 1.5], [-0.2, 0.1, 0.8]])
    normals = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
    rotation = so3_exp([0.1, -0.2, 0.05])
    factors = axial_plane_factors(points, normals, rotation, 0.995, 1e-8)
    rays = points / points[:, 2, None]
    perturbation = rng.normal(size=(100000, 2))
    residual = perturbation * np.sum(normals * (rays @ rotation.T), axis=1) * 0.995
    np.testing.assert_allclose(residual.std(axis=0), factors, rtol=0.01)
    np.testing.assert_allclose(
        axial_plane_factors(points, normals, rotation, 0.995, 1e-8, 2),
        factors * (points[:, 2] * 0.995) ** 2,
    )


def test_diagonal_axial_and_shared_encoder_covariance_whiten_together():
    from omnicalib.geometry import pose
    from omnicalib.kinematics import Joint, Robot
    from omnicalib.upper import DepthFrame, _MatchGroup, _UpperObjective

    rng = np.random.default_rng(7)
    robot = Robot("base", [Joint("a", "base", "arm", pose(), np.array([0.0, 0.0, 1.0]))])
    points = rng.uniform(0.1, 1.0, (30, 3))
    normals = rng.normal(size=(30, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    group = _MatchGroup("arm", points, normals, points + 0.01)
    objective = _UpperObjective(
        robot,
        [(DepthFrame([0.0], points), [group])],
        np.eye(4),
        0.001,
        False,
        encoder_sigma_rad=0.002,
        point_noise_model="axial",
    )
    objective.initialize_whitening(np.zeros(1))
    j = objective._cached_jacobian
    covariance = np.diag(objective._point_factors**2) + (j * 0.002) @ (j * 0.002).T
    w = objective._whiten(np.eye(30))
    np.testing.assert_allclose(w @ covariance @ w.T, np.eye(30), atol=1e-11)
