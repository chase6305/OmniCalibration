import numpy as np

from omnicalib.assets import correction_proposal
from omnicalib.geometry import inverse, pose, se3_exp, transform_points
from omnicalib.kinematics import Robot
from omnicalib.upper import DepthFrame, Surface, calibrate_upper


def _box():
    rng = np.random.default_rng(31)
    points, normals = [], []
    for axis in range(3):
        for sign in (-1, 1):
            p = rng.uniform(-0.2, 0.2, (30, 3))
            p[:, axis] = 0.2 * sign
            n = np.zeros_like(p)
            n[:, axis] = sign
            points.append(p)
            normals.append(n)
    return Surface("base", np.concatenate(points), np.concatenate(normals))


def test_metric_geometry_recovers_camera_and_depth_scale():
    robot = Robot("base", [])
    surface = _box()
    camera = pose(translation=[0, 0, -0.8]) @ se3_exp([0.002, -0.003, 0.001, 0.004, -0.003, 0.002])
    observed = transform_points(inverse(camera), surface.points) * 1.005
    frames = [DepthFrame([], observed), DepthFrame([], observed)]
    fit = calibrate_upper(
        robot,
        [surface],
        frames,
        pose(translation=[0, 0, -0.8]),
        estimate_depth_scale=True,
        outer_iterations=30,
        allow_restarts=False,
    )
    assert fit.success and fit.converged and fit.diagnostics.rank == 7
    np.testing.assert_allclose(fit.depth_scale, 1 / 1.005, atol=1e-7)
    np.testing.assert_allclose(fit.chest_transform, camera, atol=1e-7)


def test_unknown_camera_plane_does_not_observe_metric_depth_scale():
    robot = Robot("base", [])
    rng = np.random.default_rng(82)
    points = np.column_stack((rng.uniform(-0.2, 0.2, (100, 2)), np.ones(100)))
    surface = Surface("base", points, np.tile([0.0, 0, 1.0], (100, 1)))
    frames = [DepthFrame([], points * 1.005), DepthFrame([], points * 1.005)]
    fit = calibrate_upper(
        robot, [surface], frames, np.eye(4), estimate_depth_scale=True, allow_restarts=False
    )
    assert fit.diagnostics.rank < 7
    assert not correction_proposal([], fit, "base")["block_supported"]


def test_known_camera_scale_is_reported_separately_from_fixed_camera():
    surface = _box()
    robot = Robot("base", [])
    observed = surface.points * 1.01
    frames = [DepthFrame([], observed), DepthFrame([], observed)]
    fit = calibrate_upper(
        robot, [surface], frames, np.eye(4), estimate_camera=False, estimate_depth_scale=True
    )
    assert fit.diagnostics.rank == 1 and fit.converged
    proposal = correction_proposal([], fit, "base")
    assert all(p["decision"] == "retain" for p in proposal["parameters"][:6])
    assert proposal["parameters"][-1]["parameter"] == "depth_log_scale"
    assert proposal["parameters"][-1]["decision"] == "propose"
    np.testing.assert_allclose(fit.depth_scale, 1 / 1.01, atol=1e-7)


def test_global_metric_geometry_scale_is_indistinguishable_from_sensor_scale():
    surface = _box()
    robot = Robot("base", [])
    # The sensor is exact; the actual object is 0.5% larger than the CAD.
    # This produces the same observations as a depth bias and a rescaled camera.
    camera = pose(translation=[0, 0, -0.8])
    observed = transform_points(inverse(camera), surface.points * 1.005)
    fit = calibrate_upper(
        robot,
        [surface],
        [DepthFrame([], observed)] * 2,
        camera,
        estimate_depth_scale=True,
        outer_iterations=30,
    )
    assert fit.converged and fit.diagnostics.rank == 7
    np.testing.assert_allclose(fit.depth_scale, 1 / 1.005, atol=1e-7)
    np.testing.assert_allclose(fit.chest_transform[:3, 3], camera[:3, 3] / 1.005, atol=1e-7)
