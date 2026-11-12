import numpy as np
import pytest

from omnicalib._rtree import intersection_many


@pytest.mark.parametrize("backend", ["native", "scalar", "missing", "unbound"])
def test_dense_bulk_intersections_and_scalar_fallbacks(monkeypatch, backend):
    rtree = pytest.importorskip("rtree")
    from rtree import core

    # Both dense hits exceed the initial buffer. Empty queries around them
    # exercise the native row/face offsets across multiple buffer expansions.
    properties = rtree.index.Property(dimension=3)
    tree = rtree.index.Index(
        ((i, (-1, -1, -1, 1, 1, 1), None) for i in range(513)), properties=properties
    )
    storage = np.array([
        [4, 4, 4, 5, 5, 5],
        [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1],
        [2, 2, 2, 3, 3, 3],
        [0, 0, 0, 0, 0, 0],
        [6, 6, 6, 7, 7, 7],
    ])
    # Strided coordinates must be copied before giving pointers to the C API.
    boxes = np.repeat(storage, 2, axis=1)[:, ::2]
    expected = [sorted(tree.intersection(box)) for box in boxes]
    if backend == "scalar":
        class ScalarAdapter:
            def intersection(self, box):
                return reversed(list(tree.intersection(box)))

        actual_tree = ScalarAdapter()
    else:
        actual_tree = tree
        if backend == "missing":
            monkeypatch.setattr(core.rt, "Index_Intersects_id_v", None)
        elif backend == "unbound":
            # A library symbol without Rtree's ctypes signature is unsafe.
            monkeypatch.setattr(core.rt, "Index_Intersects_id_v", lambda *args: None)
    faces, counts = intersection_many(actual_tree, boxes)
    assert counts.dtype.kind == "i"
    np.testing.assert_array_equal(counts, [0, 513, 0, 513, 0])
    starts = np.r_[0, np.cumsum(counts)]
    for i, row in enumerate(expected):
        assert sorted(faces[starts[i] : starts[i + 1]]) == row


def test_empty_bulk_intersections_and_invalid_shape():
    rtree = pytest.importorskip("rtree")
    tree = rtree.index.Index(properties=rtree.index.Property(dimension=3))
    faces, counts = intersection_many(tree, np.empty((0, 6)))
    assert faces.size == counts.size == 0
    with pytest.raises(ValueError, match="shape"):
        intersection_many(tree, np.zeros((2, 4)))


def test_dense_triangle_ties_match_scalar_candidate_order():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    from omnicalib.proximity import TriangleIndex

    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=np.tile([[0, 1, 2]], (513, 1)),
        process=False,
    )
    index = TriangleIndex(mesh)
    points = np.array([[5, 5, 5], [0.2, 0.2, 0.01], [6, 6, 6], [0.3, 0.3, 0]])
    bulk = index.query(points, [0.02, 0.02, 0.02, 0.0])
    np.testing.assert_array_equal(bulk[2], [-1, 0, -1, 0])
    tree = index.triangle_tree

    class ScalarAdapter:
        def intersection(self, box):
            return reversed(list(tree.intersection(box)))

    index.triangle_tree = ScalarAdapter()
    scalar = index.query(points, [0.02, 0.02, 0.02, 0.0])
    for expected, actual in zip(scalar, bulk, strict=True):
        np.testing.assert_array_equal(actual, expected)
