"""T_AB maps coordinates in B to A; twists are [translation, rotation], SI units."""

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def validate_transform(transform, name="transform", tolerance=1e-6):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    r = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=tolerance, rtol=0)
        or not np.allclose(r.T @ r, np.eye(3), atol=tolerance, rtol=0)
        or not np.isclose(np.linalg.det(r), 1.0, atol=tolerance, rtol=0)
    ):
        raise ValueError(f"{name} must be a proper rigid transform in SE(3)")
    return transform


def so3_exp(phi):
    return Rotation.from_rotvec(phi).as_matrix()


def so3_log(rotation):
    return Rotation.from_matrix(rotation).as_rotvec()


def pose(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def inverse(transform):
    r = transform[:3, :3].T
    return pose(r, -r @ transform[:3, 3])


def _left_jacobian(phi):
    angle = np.linalg.norm(phi)
    k = skew(phi)
    if angle < 1e-5:
        a = 0.5 - angle**2 / 24 + angle**4 / 720
        b = 1 / 6 - angle**2 / 120 + angle**4 / 5040
    else:
        a = (1 - np.cos(angle)) / angle**2
        b = (angle - np.sin(angle)) / angle**3
    return np.eye(3) + a * k + b * k @ k


def se3_exp(xi):
    xi = np.asarray(xi, dtype=float)
    return pose(so3_exp(xi[3:]), _left_jacobian(xi[3:]) @ xi[:3])


def se3_log(transform):
    phi = so3_log(transform[:3, :3])
    return np.r_[np.linalg.solve(_left_jacobian(phi), transform[:3, 3]), phi]


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def mean_se3(transforms, tolerance=1e-11, max_iterations=50):
    """Right-invariant log mean, for tightly clustered inter-foot transforms."""
    transforms = np.asarray(transforms)
    if len(transforms) == 0:
        raise ValueError("At least one pose is required")
    mean = transforms[0].copy()
    for _ in range(max_iterations):
        increment = np.mean([se3_log(inverse(mean) @ t) for t in transforms], axis=0)
        mean = mean @ se3_exp(increment)
        if np.linalg.norm(increment) < tolerance:
            return mean
    raise RuntimeError("SE(3) mean did not converge; check frame consistency")


def rotation_error_deg(estimate, reference):
    return float(np.rad2deg(np.linalg.norm(so3_log(reference.T @ estimate))))
