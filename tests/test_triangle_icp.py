import numpy as np
import pytest

from omnicalib.geometry import pose, transform_points
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame, Surface, calibrate_upper


def test_independent_box_points_do_not_need_template_sample_coincidence():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    rng = np.random.default_rng(14)
    mesh = trimesh.creation.box(extents=[0.3, 0.15, 0.1])
    mesh.apply_translation([0.2, 0, 0])
    robot = Robot("base", [Joint("joint", "base", "arm", pose(), np.array([0, 0, 1.0]))])
    # Deliberately useless template sample locations: exact matching uses mesh.
    surface = Surface("arm", [[10.0, 10.0, 10.0]], [[1.0, 0, 0]], mesh)
    frames = []
    delta = 0.012
    for angle in (-0.4, 0.0, 0.5):
        triangles = mesh.triangles[rng.integers(0, len(mesh.faces), 200)]
        uv = rng.random((200, 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        local = (
            triangles[:, 0]
            + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
            + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
        )
        points = transform_points(robot.fk([angle + delta])["arm"], local)
        frames.append(DepthFrame(np.array([angle]), points))
    result = calibrate_upper(
        robot, [surface], frames, np.eye(4), matching="triangle", estimate_camera=False
    )
    assert result.success and result.converged
    np.testing.assert_allclose(result.offsets, [delta], atol=1e-7)
    assert result.diagnostics.rank == 1
    assert result.rms_m < 1e-8
    multiscale = calibrate_upper(
        robot,
        [surface],
        frames,
        np.eye(4),
        matching="triangle",
        estimate_camera=False,
        distance_schedule=[0.03, 0.003],
    )
    assert multiscale.converged
    levels = [h["distance_schedule_level"] for h in multiscale.history]
    assert levels.count(0) >= 3  # Correct first; do not discard observations by a timer.
    assert levels[-1] == 1
