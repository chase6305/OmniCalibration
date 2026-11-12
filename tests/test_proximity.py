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


def test_distance_ties_choose_lowest_face_with_empty_groups():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    # Duplicate triangles force exact ties; reversed index order must not affect
    # selection. Misses before, between and after hits exercise empty groups.
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 1, 2]],
        process=False,
    )
    index = TriangleIndex(mesh)
    tree = index.triangle_tree

    class ReversedTree:
        def intersection(self, box):
            return reversed(list(tree.intersection(box)))

    index.triangle_tree = ReversedTree()
    points = np.array([[5, 5, 5], [0.2, 0.2, 0.01], [6, 6, 6], [0.3, 0.3, 0], [7, 7, 7]])
    projected, distances, ids = index.query(points, 0.02)
    np.testing.assert_array_equal(ids, [-1, 0, -1, 0, -1])
    np.testing.assert_allclose(projected[[1, 3]], [[0.2, 0.2, 0], [0.3, 0.3, 0]])
    np.testing.assert_allclose(distances[[1, 3]], [0.01, 0])
    assert np.isnan(projected[[0, 2, 4]]).all()
    assert np.isinf(distances[[0, 2, 4]]).all()


@pytest.mark.parametrize("max_distance", [None, 0.0, 0.01, 0.2])
def test_triangle_queries_match_exhaustive_reference(max_distance):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
    points = np.random.default_rng(53).normal(0, 0.25, (80, 3))
    projected, distances, ids = TriangleIndex(mesh).query(points, max_distance)
    expected, expected_distances, _ = trimesh.proximity.closest_point_naive(mesh, points)
    # Limited searches may return a farther candidate or no candidate. Every
    # returned hit within the limit must equal the exhaustive global nearest.
    near = np.ones(len(points), dtype=bool) if max_distance is None else distances <= max_distance
    if max_distance is not None:
        assert (ids[expected_distances <= max_distance] >= 0).all()
    np.testing.assert_allclose(distances[near], expected_distances[near], atol=1e-12)
    np.testing.assert_allclose(projected[near], expected[near], atol=1e-12)


@pytest.mark.parametrize("pair_budget", [1, 7, 64, 513])
@pytest.mark.parametrize("limited", [False, True])
def test_projection_batches_preserve_entire_query_groups(monkeypatch, pair_budget, limited):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    from omnicalib import proximity

    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
    index = TriangleIndex(mesh)
    points = np.random.default_rng(91).normal(0, 0.25, (60, 3))
    limits = np.linspace(0.0, 0.4, len(points)) if limited else None
    monkeypatch.setattr(proximity, "_PROJECTION_BATCH_PAIRS", 10**9)
    expected = index.query(points, limits)
    calls = []
    original = trimesh.triangles.closest_point

    def project(triangles, observed):
        calls.append((len(triangles), len(np.unique(observed, axis=0))))
        return original(triangles, observed)

    monkeypatch.setattr(trimesh.triangles, "closest_point", project)
    monkeypatch.setattr(proximity, "_PROJECTION_BATCH_PAIRS", pair_budget)
    actual = index.query(points, limits)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(after, before)
    if sum(pairs for pairs, _ in calls) > pair_budget:
        assert len(calls) > 1
    # A single query can have more candidates than the target; splitting that
    # group would change the reduction's ownership and tie-breaking semantics.
    assert all(pairs <= pair_budget or unique_points == 1 for pairs, unique_points in calls)
