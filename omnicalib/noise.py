"""Low-rank whitening for frame-shared encoder measurement uncertainty."""

from dataclasses import dataclass

import numpy as np


def axial_plane_factors(
    points_camera,
    normals_root,
    camera_rotation,
    depth_scale=1.0,
    floor_fraction=0.25,
    range_power=0.0,
):
    """Plane sigma / optical-Z sigma, with fixed pixel coordinates.

    p=z*[u,v,1], so dr/dz = n.T R [u,v,1]. The floor represents additional
    unmodeled normal error and prevents grazing rays from getting infinite
    weight. Factors are frozen within a least-squares solve.
    """
    points = np.asarray(points_camera, dtype=float)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Axial depth noise requires positive optical camera Z")
    rays = points / points[:, 2, None]
    projection = np.abs(np.sum(normals_root * (rays @ camera_rotation.T), axis=1))
    factors = depth_scale * projection * (points[:, 2] * depth_scale) ** range_power
    return np.maximum(factors, floor_fraction)


@dataclass
class LowRankWhitener:
    basis: np.ndarray
    attenuation: np.ndarray

    @classmethod
    def from_jacobian(cls, joint_jacobian, encoder_sigma):
        """Inverse square root of I + (Jq Sigma_q^1/2)(...)^T.

        Jq is already divided by independent point/plane noise sigma. Encoder
        uncertainty is shared across points within one frame. The resulting
        operator uses O(N * joints) storage instead of an N-by-N covariance.
        """
        joint_jacobian = np.asarray(joint_jacobian, dtype=float)
        sigma = np.broadcast_to(np.asarray(encoder_sigma, dtype=float), (joint_jacobian.shape[1],))
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError("Encoder uncertainty must be finite and nonnegative")
        u, s, _ = np.linalg.svd(joint_jacobian * sigma, full_matrices=False)
        return cls(u, 1 - 1 / np.sqrt(1 + s**2))

    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != self.basis.shape[0]:
            raise ValueError("Whitening operator dimension mismatch")
        projected = self.basis.T @ values
        gain = self.attenuation if values.ndim == 1 else self.attenuation[:, None]
        return values - self.basis @ (gain * projected)
