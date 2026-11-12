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

import numpy as np


def intersection_many(tree, boxes):
    """Return concatenated face IDs and one count per interleaved 3D box.

    Older libspatialindex builds and scalar-only adapters use the same scalar
    intersection API as before. Native offsets and allocation sizes are Python
    integers: mixing int and uint64 otherwise promotes them to float in NumPy
    1.x when a dense query needs to grow its output buffer.
    """
    from rtree import core
    from rtree.index import Index

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("Expected interleaved 3D query boxes with shape (N,6)")
    count = len(boxes)
    if count == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    native = getattr(core.rt, "Index_Intersects_id_v", None)
    if not getattr(native, "argtypes", None) or not isinstance(tree, Index):
        candidates = [list(tree.intersection(box)) for box in boxes]
        counts = np.fromiter(map(len, candidates), dtype=int, count=count)
        faces = np.concatenate([ids for ids in candidates if ids]) if counts.any() else []
        return np.asarray(faces, dtype=int), counts

    lower = np.ascontiguousarray(boxes[:, :3], dtype=np.float64)
    upper = np.ascontiguousarray(boxes[:, 3:], dtype=np.float64)
    faces = np.empty(2 * count, dtype=np.int64)
    counts = np.empty(count, dtype=np.uint64)
    completed = ctypes.c_int64(0)
    row_offset = face_offset = 0
    while True:
        native(
            tree.handle,
            count - row_offset,
            3,
            len(faces) - face_offset,
            3,
            1,
            lower[row_offset:].ctypes.data,
            upper[row_offset:].ctypes.data,
            faces[face_offset:].ctypes.data,
            counts[row_offset:].ctypes.data,
            ctypes.byref(completed),
        )
        if completed.value == count - row_offset:
            return faces[: int(counts.sum())], counts.astype(int)
        face_offset += int(counts[row_offset : row_offset + completed.value].sum())
        row_offset += completed.value
        faces.resize(2 * len(faces) + int(counts[row_offset]), refcheck=False)
