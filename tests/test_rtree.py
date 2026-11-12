import numpy as np
import pytest

from omnicalib._rtree import intersection_batches


def collect_batches(tree, boxes, budget=64):
    batches = list(intersection_batches(tree, boxes, budget))
    next_row = 0
    for first, faces, counts in batches:
        assert first == next_row
        next_row += len(counts)
        assert len(faces) == counts.sum()
        assert len(faces) <= budget or np.count_nonzero(counts) == 1
    assert next_row == len(boxes)
    faces = np.concatenate([batch[1] for batch in batches]) if batches else np.empty(0, int)
    counts = np.concatenate([batch[2] for batch in batches]) if batches else np.empty(0, int)
    return faces, counts, batches


@pytest.mark.parametrize("backend", ["native", "scalar", "missing", "unbound"])
def test_dense_bulk_intersections_and_scalar_fallbacks(monkeypatch, backend):
    rtree = pytest.importorskip("rtree")
    from rtree import core

    # A leading empty group completes before a dense group cannot fit. The
    # next call must reset the native completed-row count before growing.
    # Further misses exercise global row offsets across streamed batches.
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
            monkeypatch.setattr(core.rt, "Index_Intersects_id_v", None, raising=False)
        elif backend == "unbound":
            # A library symbol without Rtree's ctypes signature is unsafe.
            monkeypatch.setattr(core.rt, "Index_Intersects_id_v", lambda *args: None, raising=False)
    faces, counts, batches = collect_batches(actual_tree, boxes)
    assert len(batches) > 1
    assert counts.dtype.kind == "i"
    np.testing.assert_array_equal(counts, [0, 513, 0, 513, 0])
    starts = np.r_[0, np.cumsum(counts)]
    for i, row in enumerate(expected):
        assert sorted(faces[starts[i] : starts[i + 1]]) == row


def test_empty_bulk_intersections_and_invalid_shape():
    rtree = pytest.importorskip("rtree")
    tree = rtree.index.Index(properties=rtree.index.Property(dimension=3))
    faces, counts, _ = collect_batches(tree, np.empty((0, 6)))
    assert faces.size == counts.size == 0
    with pytest.raises(ValueError, match="shape"):
        list(intersection_batches(tree, np.zeros((2, 4))))
    with pytest.raises(ValueError, match="positive"):
        list(intersection_batches(tree, np.zeros((1, 6)), max_pairs=0))


@pytest.mark.parametrize("scalar", [False, True])
def test_streamed_batches_own_arrays_and_keep_distinct_query_results(scalar):
    rtree = pytest.importorskip("rtree")
    tree = rtree.index.Index(properties=rtree.index.Property(dimension=3))
    for i in range(40):
        tree.insert(i, (-1, -1, -1, 1, 1, 1))
        tree.insert(i + 1000, (9, 9, 9, 11, 11, 11))
    boxes = np.array([
        [0, 0, 0, 0, 0, 0], [3, 3, 3, 4, 4, 4],
        [10, 10, 10, 10, 10, 10], [0, 0, 0, 0, 0, 0],
        [3, 3, 3, 4, 4, 4], [10, 10, 10, 10, 10, 10],
    ])
    expected = [sorted(tree.intersection(box)) for box in boxes]

    class ScalarAdapter:
        def intersection(self, box):
            return reversed(list(tree.intersection(box)))

    faces, counts, batches = collect_batches(ScalarAdapter() if scalar else tree, boxes)
    assert len(batches) == 4
    offsets = np.r_[0, np.cumsum(counts)]
    for row, ids in enumerate(expected):
        assert sorted(faces[offsets[row] : offsets[row + 1]]) == ids
    for index, (_, batch_faces, batch_counts) in enumerate(batches):
        for _, other_faces, other_counts in batches[index + 1 :]:
            assert not np.shares_memory(batch_faces, other_faces)
            assert not np.shares_memory(batch_counts, other_counts)


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
