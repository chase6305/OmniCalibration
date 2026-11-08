"""Rotation-only rig refinement on precomputed motions/bearings: (8)-(12).

No CAD rotations are loaded. An external visual graph must supply R_i^0.
This module does not implement visual odometry, matching or graph initialization.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import inverse, skew, so3_exp, so3_log
from .observability import analyze, jacobian

DEFAULT_ROTATION_SIGMA = np.deg2rad(0.2)


def project_link_trajectory(world_base, base_link):
    """Dynamic articulated compensation, T_WL(t)=T_WB(t)T_BL(t)."""
    world_base, base_link = np.asarray(world_base), np.asarray(base_link)
    if world_base.shape != base_link.shape:
        raise ValueError("Resample tf and odometry at the same timestamps first")
    return world_base @ base_link


def relative_motions(trajectory):
    return np.array([inverse(a) @ b for a, b in zip(trajectory[:-1], trajectory[1:], strict=True)])


@dataclass
class CameraMotion:
    body: np.ndarray
    camera: np.ndarray


@dataclass
class BearingPair:
    i: int
    j: int
    bearings_i: np.ndarray
    bearings_j: np.ndarray
    weight: float = 1.0


def sampson_residual(rotation_i, rotation_j, center_i, center_j, bearings_i, bearings_j):
    """Calibrated projective rays; (x/z,y/z,1) convention, paper equation (11)."""
    baseline_j = rotation_j.T @ (center_i - center_j)
    if np.linalg.norm(baseline_j) < 1e-9:
        raise ValueError("A nonzero structural baseline is required")
    essential = skew(baseline_j) @ rotation_j.T @ rotation_i
    eu = np.asarray(bearings_i) @ essential.T
    etu = np.asarray(bearings_j) @ essential
    numerator = np.sum(np.asarray(bearings_j) * eu, axis=1)
    denominator = np.sum(eu[:, :2] ** 2, axis=1) + np.sum(etu[:, :2] ** 2, axis=1)
    return numerator / np.sqrt(np.maximum(denominator, 1e-20))


def _planar_system(motions, rotation):
    # A X = X B => (R_A-I)t - R_X t_B*s = -t_A.
    # Only the two in-plane link-frame translation components and scale are
    # profiled. z translation is fixed to zero in this baseline approximation.
    matrices, targets = [], []
    for a, b in zip(motions.body, motions.camera, strict=True):
        matrices.append(np.column_stack(((a[:3, :3] - np.eye(3))[:, :2], -rotation @ b[:3, 3])))
        targets.append(-a[:3, 3])
    return np.concatenate(matrices), np.concatenate(targets)


def calibrate_head(
    initial_rotations,
    centers,
    motions,
    pairs,
    *,
    regularization=None,
    rotation_sigma=DEFAULT_ROTATION_SIGMA,
    translation_sigma=0.01,
    bearing_sigma=0.001,
    min_rotation_deg=8.0,
    bound_deg=4.0,
    trim_fraction=0.1,
):
    """Two stages sharing one 3-vector left correction; translations retained.

    Stage 1 profiles a LINEAR quadratic planar objective (not robust IRLS),
    applying Huber to outer rotation residuals only. Stage 2 minimizes robust
    Sampson residuals, trims per-edge residuals, then solves again. D and all
    noise thresholds are explicit reproduction choices, not paper defaults.
    """
    rotations = np.asarray(initial_rotations, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if rotations.shape != (len(motions), 3, 3) or centers.shape != (len(motions), 3):
        raise ValueError("Expected one initial rotation/center per motion stream")
    if min(rotation_sigma, translation_sigma, bearing_sigma, bound_deg) <= 0:
        raise ValueError("Sigmas and correction bound must be positive")
    if not 0 <= trim_fraction < 1 or not pairs:
        raise ValueError("Require bearing pairs and 0 <= trim_fraction < 1")
    for m in motions:
        if np.shape(m.body) != np.shape(m.camera) or np.ndim(m.body) != 3 or len(m.body) == 0:
            raise ValueError("Each camera needs matched nonempty motion arrays")
    for pair in pairs:
        if pair.weight <= 0 or not 0 <= pair.i < len(rotations) or not 0 <= pair.j < len(rotations):
            raise ValueError("Invalid camera edge or weight")
        if np.shape(pair.bearings_i) != np.shape(pair.bearings_j) or len(pair.bearings_i) == 0:
            raise ValueError("Each edge needs matched nonempty bearings")
    d = np.zeros(3) if regularization is None else np.asarray(regularization)
    bound = np.deg2rad(bound_deg)

    def corrected(delta):
        return so3_exp(delta) @ rotations

    def stage1_residual(delta):
        residuals = []
        for rotation, motion in zip(corrected(delta), motions, strict=True):
            for a, b in zip(motion.body, motion.camera, strict=True):
                if np.linalg.norm(so3_log(a[:3, :3])) > np.deg2rad(min_rotation_deg):
                    error = (
                        so3_log(a[:3, :3] @ rotation @ b[:3, :3].T @ rotation.T) / rotation_sigma
                    )
                    # Transform Huber block costs into least-squares residuals.
                    norm = np.linalg.norm(error)
                    weight = np.sqrt((2 * norm - 1) / norm**2) if norm > 1 else 1.0
                    residuals.append(error * weight)
            matrix, target = _planar_system(motion, rotation)
            eta = np.linalg.lstsq(matrix, target, rcond=None)[0]
            residuals.append((matrix @ eta - target) / translation_sigma)
        residuals.append(d * delta)
        return np.concatenate(residuals)

    stage1 = least_squares(stage1_residual, np.zeros(3), bounds=(-bound, bound), jac="3-point")

    def edge_residuals(delta):
        r = corrected(delta)
        return [
            sampson_residual(r[p.i], r[p.j], centers[p.i], centers[p.j], p.bearings_i, p.bearings_j)
            / bearing_sigma
            for p in pairs
        ]

    def bearing_residual(delta, masks=None):
        edges = edge_residuals(delta)
        return np.concatenate(
            [
                np.sqrt(p.weight) * (e if masks is None else e[masks[i]])
                for i, (p, e) in enumerate(zip(pairs, edges, strict=True))
            ]
        )

    first = least_squares(
        bearing_residual, stage1.x, bounds=(-bound, bound), loss="huber", jac="3-point"
    )
    masks = [
        np.abs(e) <= np.quantile(np.abs(e), 1 - trim_fraction) for e in edge_residuals(first.x)
    ]
    final = least_squares(
        lambda x: bearing_residual(x, masks),
        first.x,
        bounds=(-bound, bound),
        loss="huber",
        jac="3-point",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-10,
    )
    r = bearing_residual(final.x, masks)
    weights = np.sqrt(np.minimum(1.0, 1 / np.maximum(np.abs(r), 1e-15)))
    diagnostics = analyze(
        jacobian(lambda x: bearing_residual(x, masks), final.x) * weights[:, None]
    )
    return {
        "rotations": corrected(final.x),
        "centers": centers.copy(),
        "correction": final.x,
        "stage1_correction": stage1.x,
        "diagnostics": diagnostics,
        "success": bool(stage1.success and first.success and final.success),
        "sampson_rms": float(np.sqrt(np.mean(r**2)) * bearing_sigma),
        "retained_matches": sum(int(m.sum()) for m in masks),
    }
