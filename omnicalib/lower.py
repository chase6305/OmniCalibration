"""Double-support joint-offset calibration: paper equations (6) and (7)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import inverse, mean_se3, se3_log
from .observability import analyze, jacobian

DEFAULT_ROTATION_SIGMA = np.deg2rad(0.1)


@dataclass
class LowerResult:
    offsets: np.ndarray
    interfoot: np.ndarray
    data_diagnostics: object
    posterior_diagnostics: object
    success: bool
    cost: float


def calibrate_lower(
    robot,
    q,
    left_foot,
    right_foot,
    *,
    translation_sigma=0.001,
    rotation_sigma=DEFAULT_ROTATION_SIGMA,
    gravity_residual=None,
    prior_indices=(),
    prior_mean=None,
    prior_sigma=None,
    initial=None,
    max_nfev=200,
):
    """All frames must belong to ONE unchanged pair of foot placements.

    gravity_residual(q_corrected, transforms) must return already whitened
    residuals. A callback is intentional: the paper does not specify the exact
    encoder-dependent pelvis-IMU chain R_BI. Do not assume a fixed pelvis IMU
    contains information about leg zeros without a contact/attitude model.
    Priors are independent measurements, not automatically zero-valued truth.
    """
    q = np.asarray(q, dtype=float)
    n = len(robot.active_names)
    if q.ndim != 2 or q.shape[1] != n or len(q) < 2:
        raise ValueError("Expected at least two double-support joint vectors")
    if translation_sigma <= 0 or rotation_sigma <= 0:
        raise ValueError("Noise standard deviations must be positive")
    indices = np.asarray(prior_indices, dtype=int)
    if len(indices):
        if prior_mean is None or prior_sigma is None or np.any(np.asarray(prior_sigma) <= 0):
            raise ValueError("Supply independent prior means and positive sigmas")
        if len(prior_mean) != len(indices) or np.any((indices < 0) | (indices >= n)):
            raise ValueError("Invalid prior dimensions or indices")
    scale = np.r_[np.full(3, translation_sigma), np.full(3, rotation_sigma)]

    def evaluate(offsets, include_prior):
        transforms = [robot.fk(row + offsets) for row in q]
        relatives = [inverse(t[left_foot]) @ t[right_foot] for t in transforms]
        mean = mean_se3(relatives)
        residuals = [se3_log(inverse(mean) @ c) / scale for c in relatives]
        if gravity_residual is not None:
            residuals.append(np.asarray(gravity_residual(q + offsets, transforms)).ravel())
        if include_prior and len(indices):
            residuals.append((offsets[indices] - np.asarray(prior_mean)) / np.asarray(prior_sigma))
        return np.concatenate(residuals)

    x0 = np.zeros(n) if initial is None else np.asarray(initial)
    fit = least_squares(
        lambda x: evaluate(x, True),
        x0,
        jac="3-point",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-9,
    )
    relatives = [
        inverse(t[left_foot]) @ t[right_foot] for t in [robot.fk(row + fit.x) for row in q]
    ]
    return LowerResult(
        fit.x,
        mean_se3(relatives),
        analyze(jacobian(lambda x: evaluate(x, False), fit.x)),
        analyze(jacobian(lambda x: evaluate(x, True), fit.x)),
        bool(fit.success),
        float(fit.cost),
    )
