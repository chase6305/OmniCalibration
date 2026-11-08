"""CPU pinhole depth rendering with nearest ray/triangle hits and self-occlusion.

This is a geometry simulator, not a photometric/physical RGB-D sensor model.
Camera coordinates follow OpenCV: +x right, +y down, +z forward. Rays pass
through integer pixel coordinates (cx/cy follow that same convention).
"""

from dataclasses import dataclass

import numpy as np

from .geometry import inverse, pose, transform_points


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float = 0.1
    far: float = 5.0

    def __post_init__(self):
        if (
            self.width < 1
            or self.height < 1
            or min(self.fx, self.fy, self.near) <= 0
            or self.far <= self.near
        ):
            raise ValueError("Invalid pinhole image size, focal length or depth range")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy, self.near, self.far]).all():
            raise ValueError("Pinhole parameters must be finite")

    def rays(self):
        v, u = np.indices((self.height, self.width))
        return np.column_stack(
            ((u.ravel() - self.cx) / self.fx, (v.ravel() - self.cy) / self.fy, np.ones(u.size))
        )

    def unproject(self, depth):
        depth = np.asarray(depth, dtype=float)
        if depth.shape != (self.height, self.width):
            raise ValueError("Depth shape does not match pinhole image size")
        return (self.rays() * depth.ravel()[:, None]).reshape(self.height, self.width, 3)


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    position, target, up = (
        np.asarray(position, dtype=float),
        np.asarray(target, dtype=float),
        np.asarray(up, dtype=float),
    )
    forward = target - position
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("Camera position and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("View direction must not be parallel to up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return pose(np.column_stack((right, down, forward)), position)


@dataclass
class DepthImage:
    depth: np.ndarray
    link_ids: np.ndarray
    normals_camera: np.ndarray
    link_names: list


def render_depth(robot, q, surfaces, camera_pose, intrinsics, backend="embreex"):
    """Render first intersections of complete triangles, never sampled points.

    Meshes are two-sided; near/far are optical-axis depth, not ray range.
    The model contains only the surfaces explicitly passed by the caller.
    """
    import trimesh

    meshes, face_link = [], []
    transforms = robot.fk(q)
    for i, surface in enumerate(surfaces):
        if surface.mesh is None:
            raise ValueError("Depth rendering requires triangle meshes")
        mesh = surface.mesh.copy()
        mesh.apply_transform(transforms[surface.link])
        meshes.append(mesh)
        face_link.append(np.full(len(mesh.faces), i, dtype=np.int32))
    if not meshes:
        raise ValueError("Render at least one mesh")
    world_mesh = trimesh.util.concatenate(meshes)
    if backend == "embreex":
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    elif backend == "rtree":
        from trimesh.ray.ray_triangle import RayMeshIntersector
    else:
        raise ValueError("Choose embreex or rtree ray backend")
    intersector = RayMeshIntersector(world_mesh)
    ray_camera = intrinsics.rays()
    directions = ray_camera @ camera_pose[:3, :3].T
    origins = np.repeat(camera_pose[None, :3, 3], len(directions), axis=0)
    hits, ray_ids, triangle_ids = intersector.intersects_location(
        origins, directions, multiple_hits=False
    )
    points_camera = transform_points(inverse(camera_pose), hits)
    depth = np.full(len(directions), np.nan)
    labels = np.full(len(directions), -1, dtype=np.int32)
    normals = np.full((len(directions), 3), np.nan)
    valid = (points_camera[:, 2] >= intrinsics.near) & (points_camera[:, 2] <= intrinsics.far)
    ray_ids, triangle_ids = ray_ids[valid], triangle_ids[valid]
    depth[ray_ids] = points_camera[valid, 2]
    labels[ray_ids] = np.concatenate(face_link)[triangle_ids]
    normals[ray_ids] = world_mesh.face_normals[triangle_ids] @ camera_pose[:3, :3]
    shape = (intrinsics.height, intrinsics.width)
    return DepthImage(
        depth.reshape(shape),
        labels.reshape(shape),
        normals.reshape(*shape, 3),
        [s.link for s in surfaces],
    )


@dataclass(frozen=True)
class DepthNoise:
    sigma_m: float = 0.0002
    dropout: float = 0.0
    outlier_fraction: float = 0.0
    outlier_sigma_m: float = 0.03
    quantization_m: float = 0.0
    scale_bias: float = 0.0
    quadratic_range: bool = False

    def __post_init__(self):
        if (
            min(self.sigma_m, self.outlier_sigma_m, self.quantization_m) < 0
            or not 0 <= self.dropout < 1
            or not 0 <= self.outlier_fraction < 1
            or self.scale_bias <= -1
        ):
            raise ValueError("Invalid noise, dropout or outlier settings")
        if not np.isfinite(
            [self.sigma_m, self.outlier_sigma_m, self.quantization_m, self.scale_bias]
        ).all():
            raise ValueError("Depth noise settings must be finite")


def corrupt_depth(depth, noise, rng):
    result = np.asarray(depth, dtype=float).copy()
    valid = np.isfinite(result) & (result > 0)
    z = result[valid]
    sigma = noise.sigma_m * (z**2 if noise.quadratic_range else np.ones_like(z))
    z = z * (1 + noise.scale_bias) + rng.normal(0, sigma)
    if noise.outlier_fraction:
        outliers = rng.random(len(z)) < noise.outlier_fraction
        z[outliers] += rng.normal(0, noise.outlier_sigma_m, outliers.sum())
    if noise.quantization_m:
        z = np.round(z / noise.quantization_m) * noise.quantization_m
    z[rng.random(len(z)) < noise.dropout] = np.nan
    z[z <= 0] = np.nan
    result[valid] = z
    return result


def depth_points(image, intrinsics, *, noise=None, seed=0, max_points=None):
    """Return points and pixel indices; perturb depth before unprojection."""
    rng = np.random.default_rng(seed)
    depth = image.depth if noise is None else corrupt_depth(image.depth, noise, rng)
    cloud = intrinsics.unproject(depth).reshape(-1, 3)
    pixels = np.flatnonzero(
        np.isfinite(cloud).all(axis=1)
        & (cloud[:, 2] >= intrinsics.near)
        & (cloud[:, 2] <= intrinsics.far)
    )
    if max_points is not None:
        if max_points < 1:
            raise ValueError("max_points must be positive")
        if len(pixels) > max_points:
            pixels = np.sort(rng.choice(pixels, max_points, replace=False))
    return cloud[pixels], pixels
