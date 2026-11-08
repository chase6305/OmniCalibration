import numpy as np
import pytest

from omnicalib.proximity import TriangleIndex


def test_cached_proximity_matches_independent_trimesh_reference():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    points = np.random.default_rng(19).normal(0, 0.3, (100, 3))
    index = TriangleIndex(mesh)
    expected, distance, _ = trimesh.proximity.closest_point(mesh, points)
    actual, got_distance, ids = index.query(points)
    np.testing.assert_allclose(got_distance, distance, atol=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert (ids >= 0).all()
    limited, limits, limited_ids = index.query(points, 0.01)
    # Candidates may include farther faces; points truly within the bound must
    # still be found exactly, which is the contract used by ICP.
    near = distance < 0.01
    np.testing.assert_allclose(limits[near], distance[near], atol=1e-12)
    assert (limited_ids[near] >= 0).all()


def test_empty_triangle_query():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    index = TriangleIndex(trimesh.creation.box())
    projected, distance, ids = index.query(np.empty((0, 3)))
    assert projected.shape == (0, 3) and distance.size == ids.size == 0
