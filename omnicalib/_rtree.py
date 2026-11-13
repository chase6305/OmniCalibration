"""Bulk R-tree intersections, including NumPy 1.x buffer-growth compatibility.

The native buffer loop is adapted from Rtree 1.4.1's Index.intersection_v:
https://github.com/Toblerity/rtree/blob/1.4.1/rtree/index.py

The MIT License (MIT)

Copyright (c) 2018: Sean C. Gillies, Howard Butler and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import ctypes
import operator

import numpy as np


def _pack_candidates(candidates):
    counts = np.fromiter(map(len, candidates), dtype=int, count=len(candidates))
    faces = np.concatenate([ids for ids in candidates if ids]) if counts.any() else []
    return np.asarray(faces, dtype=int), counts


def _scalar_batches(tree, boxes, max_pairs):
    first, pairs, candidates = 0, 0, []
    for row, box in enumerate(boxes):
        ids = list(tree.intersection(box))
        if candidates and pairs + len(ids) > max_pairs:
            yield first, *_pack_candidates(candidates)
            first, pairs, candidates = row, 0, []
        candidates.append(ids)
        pairs += len(ids)
    if candidates:
        yield first, *_pack_candidates(candidates)


def intersection_batches(tree, boxes, max_pairs=65536):
    """Yield (first row, face IDs, counts) for complete 3D query groups.

    Older libspatialindex builds and scalar-only adapters use the same scalar
    intersection API as before. Each batch owns its arrays. The pair target
    bounds candidate storage unless a single query has more intersections.
    Python integer sizes avoid NumPy 1.x unsigned-count promotion on growth.
    """
    from rtree import core
    from rtree.index import Index

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("Expected interleaved 3D query boxes with shape (N,6)")
    max_pairs = operator.index(max_pairs)
    if max_pairs < 1:
        raise ValueError("Candidate pair target must be positive")
    count = len(boxes)
    if count == 0:
        return
    native = getattr(core.rt, "Index_Intersects_id_v", None)
    if not getattr(native, "argtypes", None) or not isinstance(tree, Index):
        yield from _scalar_batches(tree, boxes, max_pairs)
        return

    lower = np.ascontiguousarray(boxes[:, :3], dtype=np.float64)
    upper = np.ascontiguousarray(boxes[:, 3:], dtype=np.float64)
    faces = np.empty(max_pairs, dtype=np.int64)
    counts = np.empty(count, dtype=np.uint64)
    completed = ctypes.c_int64(0)
    first = 0
    while first < count:
        # libspatialindex does not write nr if the first query cannot fit.
        completed.value = 0
        native(
            tree.handle,
            count - first,
            3,
            len(faces),
            3,
            1,
            lower[first:].ctypes.data,
            upper[first:].ctypes.data,
            faces.ctypes.data,
            counts[first:].ctypes.data,
            ctypes.byref(completed),
        )
        if completed.value:
            batch_counts = counts[first : first + completed.value].astype(int)
            yield first, faces[: int(batch_counts.sum())].copy(), batch_counts
            first += completed.value
            if len(faces) > max_pairs:
                faces = np.empty(max_pairs, dtype=np.int64)
        else:
            required = int(counts[first])
            if required <= len(faces):
                raise RuntimeError("Native bulk intersection made no progress")
            faces.resize(required, refcheck=False)
