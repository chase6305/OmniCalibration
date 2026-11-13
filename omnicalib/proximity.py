"""Cached, exact unsigned point-to-triangle queries for static link meshes.

An R-tree bounds candidate triangles by the nearest mesh vertex distance.
Any nearer point must lie in that box, so the candidate reduction is exact.
Unlike rebuilding per-query vertex trees, this keeps both indexes per mesh.
"""

import numpy as np
from scipy.spatial import cKDTree

from ._rtree import intersection_batches

_PROJECTION_BATCH_PAIRS = 65536


class TriangleIndex:
    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles)
        self.normals = np.asarray(mesh.face_normals)
        self.bounds = np.asarray(mesh.bounds)
        self.vertex_tree = cKDTree(np.asarray(mesh.vertices)[mesh.referenced_vertices])
        from trimesh.triangles import bounds_tree

        self.triangle_tree = bounds_tree(self.triangles)

    def query(self, points, max_distance=None):
        from trimesh.triangles import closest_point

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Triangle query expects finite (N,3) points")
        result = np.full_like(points, np.nan)
        distances = np.full(len(points), np.inf)
        triangle_ids = np.full(len(points), -1, dtype=int)
        if len(points) == 0:
            return result, distances, triangle_ids
        radii = self.vertex_tree.query(points)[0]
        if max_distance is not None:
            limits = np.broadcast_to(np.asarray(max_distance, dtype=float), (len(points),))
            if np.any(limits < 0) or np.isnan(limits).any():
                raise ValueError("Distance limits must be nonnegative")
            radii = np.minimum(radii, limits)
        radii += 1e-10
        boxes = np.column_stack((points - radii[:, None], points + radii[:, None]))
        for first, faces, counts in intersection_batches(
            self.triangle_tree, boxes, _PROJECTION_BATCH_PAIRS
        ):
            if len(faces) == 0:
                continue
            rows = np.repeat(np.arange(first, first + len(counts)), counts)
            projected = closest_point(self.triangles[faces], points[rows])
            squared = np.sum((projected - points[rows]) ** 2, axis=1)
            # Reduce each group in linear time; exact ties choose the lowest
            # face ID independently of candidate order and batch boundaries.
            batch_counts = counts[counts > 0]
            starts = np.r_[0, np.cumsum(batch_counts)[:-1]]
            minimum_squared = np.minimum.reduceat(squared, starts)
            nearest = squared == np.repeat(minimum_squared, batch_counts)
            minimum_faces = np.minimum.reduceat(
                np.where(nearest, faces, np.iinfo(faces.dtype).max), starts
            )
            winners = nearest & (faces == np.repeat(minimum_faces, batch_counts))
            positions = np.arange(len(faces))
            best = np.minimum.reduceat(np.where(winners, positions, len(faces)), starts)
            result[rows[best]] = projected[best]
            distances[rows[best]] = np.sqrt(squared[best])
            triangle_ids[rows[best]] = faces[best]
        return result, distances, triangle_ids
