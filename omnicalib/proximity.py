"""Cached, exact unsigned point-to-triangle queries for static link meshes.

An R-tree bounds candidate triangles by the nearest mesh vertex distance.
Any nearer point must lie in that box, so the candidate reduction is exact.
Unlike rebuilding per-query vertex trees, this keeps both indexes per mesh.
"""

import numpy as np
from scipy.spatial import cKDTree


class TriangleIndex:
    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles)
        self.normals = np.asarray(mesh.face_normals)
        self.bounds = np.asarray(mesh.bounds)
        self.vertex_tree = cKDTree(np.asarray(mesh.vertices)[mesh.referenced_vertices])
        self.triangle_tree = mesh.triangles_tree

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
        candidates = [list(self.triangle_tree.intersection(box)) for box in boxes]
        counts = np.fromiter(map(len, candidates), dtype=int, count=len(points))
        if counts.sum() == 0:
            return result, distances, triangle_ids
        faces = np.concatenate([a for a in candidates if a]).astype(int)
        rows = np.repeat(np.arange(len(points)), counts)
        projected = closest_point(self.triangles[faces], points[rows])
        squared = np.sum((projected - points[rows]) ** 2, axis=1)
        order = np.lexsort((faces, squared, rows))
        first = np.r_[True, np.diff(rows[order]) != 0]
        best = order[first]
        result[rows[best]] = projected[best]
        distances[rows[best]] = np.sqrt(squared[best])
        triangle_ids[rows[best]] = faces[best]
        return result, distances, triangle_ids
