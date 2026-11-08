import numpy as np
import pytest

from omnicalib.geometry import pose
from omnicalib.kinematics import Robot
from omnicalib.render import (
    DepthImage,
    DepthNoise,
    Pinhole,
    corrupt_depth,
    depth_points,
    look_at,
    render_depth,
)
from omnicalib.upper import Surface


def test_pinhole_unprojects_pixel_centers():
    camera = Pinhole(3, 3, 2.0, 2.0, 1.0, 1.0)
    xyz = camera.unproject(np.full((3, 3), 2.0))
    np.testing.assert_allclose(xyz[1, 1], [0, 0, 2])
    np.testing.assert_allclose(xyz[0, 2], [1, -1, 2])


def test_look_at_camera_convention():
    camera = look_at([2.0, 0, 1.0], [0, 0, 1.0])
    np.testing.assert_allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(camera[:3, :3]) > 0.999999
    np.testing.assert_allclose(camera[:3, 2], [-1, 0, 0])
    np.testing.assert_allclose(camera[:3, 1], [0, 0, -1])


def test_nearest_surface_occludes_rear_plane():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("embreex")
    robot = Robot("base", [])
    surfaces = []
    # Front plane covers center rays; larger rear plane covers the side rays.
    for size, z in ((0.7, 1.0), (4.0, 2.0)):
        mesh = trimesh.Trimesh(
            vertices=[[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        surfaces.append(
            Surface("base", np.asarray(mesh.vertices), np.tile([0, 0, 1.0], (4, 1)), mesh)
        )
    camera = Pinhole(5, 3, 2, 2, 2, 1)
    image = render_depth(robot, [], surfaces, pose(), camera)
    assert image.depth[1, 2] == pytest.approx(1.0)
    assert image.depth[1, 0] == pytest.approx(2.0)
    assert image.link_ids[1, 2] == 0
    assert image.link_ids[1, 0] == 1


def test_axial_noise_and_quantization_are_before_unprojection():
    depth = np.array([[1.0, 1.0], [np.nan, 2.0]])
    noise = DepthNoise(sigma_m=0.0, scale_bias=0.1, quantization_m=0.01)
    changed = corrupt_depth(depth, noise, np.random.default_rng(0))
    np.testing.assert_allclose(changed, [[1.1, 1.1], [np.nan, 2.2]])
    image = DepthImage(depth, np.zeros((2, 2), int), np.zeros((2, 2, 3)), ["base"])
    camera = Pinhole(2, 2, 1, 1, 0, 0)
    points, pixels = depth_points(image, camera, noise=noise, seed=0)
    np.testing.assert_array_equal(pixels, [0, 1, 3])
    np.testing.assert_allclose(points, [[0, 0, 1.1], [1.1, 0, 1.1], [2.2, 2.2, 2.2]])


def test_depth_noise_is_reproducible_and_does_not_mutate_input():
    depth = np.ones((30, 40))
    noise = DepthNoise(dropout=0.2, outlier_fraction=0.1)
    first = corrupt_depth(depth, noise, np.random.default_rng(42))
    second = corrupt_depth(depth, noise, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(depth, np.ones_like(depth))
    assert np.isnan(first).sum() > 100
