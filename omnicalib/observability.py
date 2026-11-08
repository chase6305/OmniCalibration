"""Local diagnostics in physical units, keeping data and prior information separate."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Diagnostics:
    rank: int
    parameters: int
    condition: float
    singular_values: list
    sigma: list
    supported: list
    scaled_condition: float | None = None
    parameter_scale: list | None = None
    weak_modes: list | None = None
    rank_threshold: float | None = None
    condition_limit: float = 1e10
    sigma_limits: list | None = None
    covariance: list | None = None
    correlation: list | None = None

    def to_dict(self):
        return asdict(self)


def jacobian(function, x):
    x = np.asarray(x, dtype=float)
    columns = []
    for i in range(len(x)):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x[i]))
        delta = np.zeros_like(x)
        delta[i] = step
        columns.append((function(x + delta) - function(x - delta)) / (2 * step))
    return np.column_stack(columns)


def analyze(
    j, rank_rtol=1e-7, max_condition=1e10, max_sigma=None, parameter_scale=None, rank_atol=1e-9
):
    """Input is a whitened Jacobian. Condition is that of J^T J.

    No residual-variance rescaling: the caller supplied the noise standard
    deviations. Rank-deficient coordinate uncertainties are infinite, never
    silently reported as zero by a pseudoinverse. Thresholds are baseline choices.
    """
    j = np.asarray(j, dtype=float)
    if j.ndim != 2 or not np.isfinite(j).all() or j.shape[1] == 0:
        raise ValueError("Expected a finite, nonempty 2D Jacobian")
    n = j.shape[1]
    scale = np.ones(n) if parameter_scale is None else np.asarray(parameter_scale, dtype=float)
    if scale.shape != (n,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Parameter scales must be finite positive values per column")
    if rank_rtol <= 0 or rank_atol < 0 or max_condition <= 0:
        raise ValueError("Invalid rank or condition thresholds")
    _, s, vt = np.linalg.svd(j * scale, full_matrices=j.shape[0] < n)
    rank_threshold = max(s[0] * rank_rtol if len(s) else 0, rank_atol)
    rank = int(np.count_nonzero(s > rank_threshold))
    covariance = (vt[:rank].T / s[:rank] ** 2) @ vt[:rank]
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0)) * scale
    null_energy = np.sum(vt[rank:] ** 2, axis=0)
    sigma[null_energy > 1e-10] = np.inf
    scaled_condition = float((s[0] / s[-1]) ** 2) if rank == n else float("inf")
    raw_s = np.linalg.svd(j, compute_uv=False) if parameter_scale is not None else s
    condition = float((raw_s[0] / raw_s[-1]) ** 2) if rank == n else float("inf")
    supported = np.isfinite(sigma) & (rank == n) & (scaled_condition <= max_condition)
    if max_sigma is not None:
        thresholds = np.asarray(max_sigma, dtype=float)
        if (
            thresholds.ndim > 1
            or thresholds.size not in (1, n)
            or np.any(thresholds <= 0)
            or np.isnan(thresholds).any()
        ):
            raise ValueError("max_sigma must be positive scalar or per-parameter thresholds")
        supported &= sigma <= thresholds
    weak = (vt[-min(3, n) :] * scale).tolist()
    physical_covariance = covariance * np.outer(scale, scale)
    # A pseudoinverse in a gauge direction is not a finite covariance.
    correlation = physical_covariance / np.outer(sigma, sigma) if rank == n else None
    return Diagnostics(
        rank,
        n,
        condition,
        s.tolist(),
        sigma.tolist(),
        supported.tolist(),
        scaled_condition,
        scale.tolist(),
        weak,
        float(rank_threshold),
        max_condition,
        None if max_sigma is None else np.broadcast_to(np.asarray(max_sigma), (n,)).tolist(),
        physical_covariance.tolist() if rank == n else None,
        correlation.tolist() if correlation is not None else None,
    )
