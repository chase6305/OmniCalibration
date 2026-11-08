"""Articulated point-to-plane ICP and a wrist hand-eye baseline.

Meshes must be sampled by an adapter into link-local points and normals.
Correspondences are refreshed between solves, never differentiated through NN.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import inverse, se3_exp, se3_log, transform_points, validate_transform
from .observability import analyze, jacobian
from .parameters import UpperParameters
from .residuals import residual_profile

DEFAULT_ROTATION_SIGMA = np.deg2rad(0.1)


def _sqrt_robust_weights(residual, loss, scale):
    z = (residual / scale) ** 2
    if loss == "linear":
        return np.ones_like(residual)
    if loss == "huber":
        return np.sqrt(np.minimum(1.0, 1 / np.maximum(np.sqrt(z), 1e-15)))
    if loss == "soft_l1":
        return (1 + z) ** (-0.25)
    if loss == "cauchy":
        return (1 + z) ** (-0.5)
    raise ValueError("Choose linear/huber/soft_l1/cauchy loss")


@dataclass
class Surface:
    link: str
    points: np.ndarray
    normals: np.ndarray
    mesh: object | None = None
    _triangle_index: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        if (
            self.points.shape != self.normals.shape
            or self.points.ndim != 2
            or self.points.shape[1] != 3
            or len(self.points) == 0
        ):
            raise ValueError("Surface points and normals must have shape (N, 3)")
        if not np.isfinite(self.points).all() or not np.isfinite(self.normals).all():
            raise ValueError("Surface coordinates must be finite")
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError("Surface normals must be nonzero")
        self.normals = self.normals / lengths[:, None]


@dataclass
class DepthFrame:
    q: np.ndarray
    points_camera: np.ndarray
    fixed_positions: dict | None = None

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        self.points_camera = np.asarray(self.points_camera, dtype=float)
        if self.q.ndim != 1 or not np.isfinite(self.q).all():
            raise ValueError("Frame joint states must be a finite 1D vector")
        if (
            self.points_camera.ndim != 2
            or self.points_camera.shape[1] != 3
            or not np.isfinite(self.points_camera).all()
        ):
            raise ValueError("Frame points must be finite camera coordinates (N,3)")
        if self.fixed_positions is not None and not all(
            np.isfinite(v) for v in self.fixed_positions.values()
        ):
            raise ValueError("Per-frame fixed joint states must be finite")


def _frame_fk(robot, frame, offsets, with_jacobian=False):
    function = robot.fk_with_jacobian if with_jacobian else robot.fk
    if frame.fixed_positions:
        if not hasattr(robot, "fixed_positions"):
            raise ValueError("Per-frame uncalibrated states require a selected-joint model")
        return function(frame.q + offsets, fixed_positions=frame.fixed_positions)
    return function(frame.q + offsets)


@dataclass
class UpperResult:
    offsets: np.ndarray
    chest_transform: np.ndarray
    correction: np.ndarray
    diagnostics: object
    rms_m: float
    correspondences: int
    success: bool
    iterations: int
    history: list
    converged: bool
    rejection_fraction: float
    at_bounds: bool
    frame_matches: list
    link_matches: dict
    initialization: dict
    quality_passed: bool
    quality: dict
    estimate_camera: bool = True
    estimate_depth_scale: bool = False
    depth_scale: float = 1.0
    residual_profile: dict = field(default_factory=dict)
    all_inner_success: bool = True


@dataclass
class _MatchGroup:
    link: str
    local_points: np.ndarray
    local_normals: np.ndarray
    observed: np.ndarray


class _UpperObjective:
    """Analytic joint Jacobian; only six camera coordinates use tiny FD matrices.

    If a link's spatial twist is [v,w], differentiating n dot (p_camera-p_link)
    gives -n dot v + (n cross p_camera) dot w. Both point and normal rotate;
    dropping the latter term is an incorrect Jacobian away from zero residual.
    """

    def __init__(
        self,
        robot,
        matches,
        nominal,
        sigma,
        estimate_camera,
        encoder_sigma_rad=None,
        estimate_depth_scale=False,
        point_noise_model="isotropic",
        axial_floor_fraction=0.25,
        axial_range_power=0.0,
    ):
        self.robot, self.matches, self.nominal = robot, matches, nominal
        self.sigma, self.estimate_camera = sigma, estimate_camera
        self.n = len(robot.active_names)
        self.layout = UpperParameters(self.n, estimate_camera, estimate_depth_scale)
        self._cached_x = None
        self._whiteners = None
        self._frame_counts = [sum(len(g.observed) for g in groups) for _, groups in matches]
        self.encoder_sigma_rad = encoder_sigma_rad
        self.point_noise_model = point_noise_model
        self.axial_floor_fraction = axial_floor_fraction
        self.axial_range_power = axial_range_power
        self._point_factors = None

    def initialize_whitening(self, x):
        """Freeze diagonal depth and frame-shared encoder covariance."""
        if self.point_noise_model == "axial":
            from .noise import axial_plane_factors

            camera = self.nominal @ se3_exp(self.layout.camera_correction(x))
            factors = []
            for frame, groups in self.matches:
                transforms = _frame_fk(self.robot, frame, x[: self.n])
                for group in groups:
                    normals = group.local_normals @ transforms[group.link][:3, :3].T
                    factors.append(
                        axial_plane_factors(
                            group.observed,
                            normals,
                            camera[:3, :3],
                            self.layout.depth_scale(x),
                            self.axial_floor_fraction,
                            self.axial_range_power,
                        )
                    )
            self._point_factors = np.concatenate(factors)
        if self.encoder_sigma_rad is None or not np.any(self.encoder_sigma_rad):
            return
        from .noise import LowRankWhitener

        self._evaluate(x)
        self._whiteners = None
        diagonal_jacobian = self._whiten(self._cached_jacobian)
        self._whiteners = []
        offset = 0
        for count in self._frame_counts:
            self._whiteners.append(
                LowRankWhitener.from_jacobian(
                    diagonal_jacobian[offset : offset + count, : self.n], self.encoder_sigma_rad
                )
            )
            offset += count

    def _whiten(self, values):
        if self._point_factors is not None:
            divisor = self._point_factors if values.ndim == 1 else self._point_factors[:, None]
            values = values / divisor
        if self._whiteners is None:
            return values.copy()
        rows, offset = [], 0
        for count, whitener in zip(self._frame_counts, self._whiteners, strict=True):
            rows.append(whitener.apply(values[offset : offset + count]))
            offset += count
        return np.concatenate(rows)

    def _evaluate(self, x):
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return
        correction = self.layout.camera_correction(x)
        depth_scale = self.layout.depth_scale(x)
        camera = self.nominal @ se3_exp(correction)
        camera_derivatives = []
        if self.estimate_camera:
            for i in range(6):
                step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(correction[i]))
                delta = np.zeros(6)
                delta[i] = step
                derivative = (
                    self.nominal
                    @ (se3_exp(correction + delta) - se3_exp(correction - delta))
                    / (2 * step)
                )
                camera_derivatives.append(derivative[:3])
        errors, jacobians = [], []
        for frame, groups in self.matches:
            transforms, screws = _frame_fk(self.robot, frame, x[: self.n], with_jacobian=True)
            for group in groups:
                t = transforms[group.link]
                model = transform_points(t, group.local_points)
                normals = group.local_normals @ t[:3, :3].T
                corrected_observed = group.observed * depth_scale
                observed = transform_points(camera, corrected_observed)
                errors.append(np.sum(normals * (observed - model), axis=1) / self.sigma)
                s = screws[group.link]
                joint_jac = -normals @ s[:3] + np.cross(normals, observed) @ s[3:]
                if self.estimate_camera:
                    camera_jac = np.column_stack(
                        [
                            np.sum(normals * (corrected_observed @ d[:, :3].T + d[:, 3]), axis=1)
                            for d in camera_derivatives
                        ]
                    )
                    joint_jac = np.column_stack((joint_jac, camera_jac))
                if self.layout.scale:
                    scale_jac = np.sum(normals * (observed - camera[:3, 3]), axis=1)
                    joint_jac = np.column_stack((joint_jac, scale_jac))
                jacobians.append(joint_jac / self.sigma)
        self._cached_x = np.asarray(x).copy()
        self._cached_residual = np.concatenate(errors)
        self._cached_jacobian = np.concatenate(jacobians)

    def fun(self, x):
        self._evaluate(x)
        # least_squares scales robust-loss Jacobians in place: never expose cache.
        return self._whiten(self._cached_residual)

    def jac(self, x):
        self._evaluate(x)
        return self._whiten(self._cached_jacobian)

    def raw_residual(self, x):
        self._evaluate(x)
        return self._cached_residual.copy()


def _match_frame(
    robot,
    surfaces,
    frame,
    offsets,
    camera,
    distance,
    matching,
    trim_fraction,
    surface_distance_normals=True,
    trim_scope="link",
    depth_scale=1.0,
):
    """Match in link-local coordinates so full meshes stay static and indexed."""
    transforms = _frame_fk(robot, frame, offsets)
    observed = transform_points(camera, frame.points_camera * depth_scale)
    count = len(observed)
    best_distance = np.full(count, distance)
    best_surface = np.full(count, -1, dtype=int)
    best_points, best_normals = np.zeros((count, 3)), np.zeros((count, 3))
    if matching == "sampled":
        model = np.concatenate([transform_points(transforms[s.link], s.points) for s in surfaces])
        distances, indices = cKDTree(model).query(observed)
        starts = np.r_[0, np.cumsum([len(s.points) for s in surfaces])]
        selected = np.flatnonzero(distances < distance)
        sid = np.searchsorted(starts[1:], indices[selected], side="right")
        for i, surface in enumerate(surfaces):
            mask = sid == i
            rows = selected[mask]
            local_indices = indices[rows] - starts[i]
            best_surface[rows], best_distance[rows] = i, distances[rows]
            best_points[rows] = surface.points[local_indices]
            best_normals[rows] = surface.normals[local_indices]
    else:
        from .proximity import TriangleIndex

        for i, surface in enumerate(surfaces):
            if surface._triangle_index is None:
                surface._triangle_index = TriangleIndex(surface.mesh)
            local = transform_points(inverse(transforms[surface.link]), observed)
            bounds = surface.mesh.bounds
            # Conservative distance to an AABB: skip points whose exact closest
            # triangle cannot improve the current best. No approximate k-NN.
            box_delta = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0.0)
            candidate = np.flatnonzero(np.linalg.norm(box_delta, axis=1) < best_distance)
            if len(candidate) == 0:
                continue
            closest, distances, triangle = surface._triangle_index.query(
                local[candidate], best_distance[candidate]
            )
            better = distances < best_distance[candidate]
            rows = candidate[better]
            best_surface[rows], best_distance[rows] = i, distances[better]
            best_points[rows] = closest[better]
            normals = surface.mesh.face_normals[triangle[better]].copy()
            if surface_distance_normals and len(rows):
                delta = local[rows] - closest[better]
                lengths = np.linalg.norm(delta, axis=1)
                valid = lengths > 1e-10
                sign = np.where(np.sum(normals * delta, axis=1) < 0, -1.0, 1.0)
                normals[valid] = delta[valid] / lengths[valid, None] * sign[valid, None]
            best_normals[rows] = normals
    retained = best_surface >= 0
    if trim_fraction and retained.any():
        if trim_scope == "frame":
            retained &= best_distance <= np.quantile(best_distance[retained], 1 - trim_fraction)
        else:
            # A global quantile can erase a small but informative wrist/hand.
            # Preserve each represented link; do not hard-trim tiny groups.
            for i in range(len(surfaces)):
                rows = np.flatnonzero(retained & (best_surface == i))
                if len(rows) >= 10:
                    threshold = np.quantile(best_distance[rows], 1 - trim_fraction)
                    retained[rows] &= best_distance[rows] <= threshold
    groups = []
    for i, surface in enumerate(surfaces):
        rows = np.flatnonzero(retained & (best_surface == i))
        if len(rows):
            groups.append(
                _MatchGroup(
                    surface.link, best_points[rows], best_normals[rows], frame.points_camera[rows]
                )
            )
    return groups


def model_cloud(robot, q, surfaces):
    transforms = robot.fk(q)
    points, normals = [], []
    for surface in surfaces:
        t = transforms[surface.link]
        points.append(transform_points(t, surface.points))
        normals.append(surface.normals @ t[:3, :3].T)
    return np.concatenate(points), np.concatenate(normals)


def calibrate_upper(
    robot,
    surfaces,
    frames,
    chest_nominal,
    *,
    sigma=0.001,
    correspondence_distance=0.03,
    outer_iterations=12,
    robust_scale=2.0,
    estimate_camera=True,
    estimate_depth_scale=False,
    depth_scale_log_bound=0.05,
    max_sigma=None,
    matching="sampled",
    trim_fraction=0.0,
    initial=None,
    max_nfev=80,
    offset_bound_deg=10.0,
    camera_bound=None,
    jacobian_mode="analytic",
    parameter_scale=None,
    convergence_tolerance=None,
    convergence_patience=2,
    trim_start_iteration=0,
    distance_schedule=None,
    distance_schedule_trigger="stable",
    uncertainty_step_fraction=0.0,
    encoder_sigma_rad=None,
    loss="huber",
    initialize_camera=True,
    surface_distance_normals=True,
    max_normalized_rms=2.0,
    allow_restarts=True,
    local_continuation=True,
    restart_translation_m=0.015,
    trim_scope="link",
    step_relaxation=1.0,
    relaxation_step_ratio=20.0,
    point_noise_model="isotropic",
    axial_floor_fraction=0.25,
    axial_range_power=0.0,
    axial_start_iteration=2,
    max_rejection_fraction=0.2,
):
    """Joint zeros and right-composed chest correction, paper (4)-(5).

    Choose sampled points or exact mesh-triangle projections (requires rtree).
    The caller must mask depth to the robot before calling. Full triangle
    matching is independent of the points used to generate observations.
    """
    restart_arguments = locals().copy()
    if not 0 <= max_rejection_fraction < 1:
        raise ValueError("Maximum correspondence rejection fraction must be in [0,1)")
    if point_noise_model not in ("isotropic", "axial"):
        raise ValueError("Choose isotropic or axial point noise")
    if axial_start_iteration < 0:
        raise ValueError("Axial covariance start iteration must be nonnegative")
    if (
        not np.isfinite([axial_floor_fraction, axial_range_power]).all()
        or axial_floor_fraction <= 0
        or not 0 <= axial_range_power <= 2
    ):
        raise ValueError("Axial noise requires a positive floor and range power in [0,2]")
    if not np.isfinite(step_relaxation) or not 0 < step_relaxation <= 1:
        raise ValueError("Step relaxation must be in (0,1]")
    if relaxation_step_ratio <= 0 or np.isnan(relaxation_step_ratio):
        raise ValueError("Relaxation activation ratio must be positive")
    n = len(robot.active_names)
    if (
        max_normalized_rms <= 0
        or np.isnan(max_normalized_rms)
        or not np.isfinite(restart_translation_m)
        or restart_translation_m <= 0
    ):
        raise ValueError("Residual quality limit and restart translation must be positive")
    _sqrt_robust_weights(np.zeros(1), loss, robust_scale)
    joint_types = {j.name: j.kind for j in robot.joints}
    if any(joint_types[name] not in ("revolute", "continuous") for name in robot.active_names):
        raise ValueError(
            "Upper zero-offset calibration supports angular joints; freeze prismatic joints"
        )
    if (
        not np.isfinite([sigma, correspondence_distance, robust_scale, offset_bound_deg]).all()
        or min(sigma, correspondence_distance, robust_scale, offset_bound_deg) <= 0
        or outer_iterations < 1
    ):
        raise ValueError("Invalid ICP noise/distance/iteration parameters")
    if not frames or not surfaces:
        raise ValueError("Frames and surfaces must be nonempty")
    if matching not in ("sampled", "triangle") or not 0 <= trim_fraction < 1:
        raise ValueError("Choose sampled/triangle matching and 0 <= trim_fraction < 1")
    if trim_scope not in ("link", "frame"):
        raise ValueError("Choose per-link or per-frame trimming")
    if trim_start_iteration < 0:
        raise ValueError("Trim start iteration must be nonnegative")
    if not np.isfinite(uncertainty_step_fraction) or uncertainty_step_fraction < 0:
        raise ValueError("Uncertainty stopping fraction must be finite and nonnegative")
    schedule = np.array(
        [correspondence_distance] if distance_schedule is None else distance_schedule, dtype=float
    )
    if (
        schedule.ndim != 1
        or len(schedule) == 0
        or not np.isfinite(schedule).all()
        or np.any(schedule <= 0)
    ):
        raise ValueError("Distance schedule must contain positive finite distances")
    if distance_schedule_trigger not in ("stable", "iteration"):
        raise ValueError("Choose stable or iteration distance schedule trigger")
    if np.any(np.diff(schedule) > 0):
        raise ValueError("Coarse-to-fine distance limits must be nonincreasing")
    if matching == "triangle" and any(s.mesh is None for s in surfaces):
        raise ValueError("Triangle matching requires a mesh for every surface")
    if jacobian_mode not in ("analytic", "numeric"):
        raise ValueError("Choose analytic or numeric Jacobian")
    if any(len(frame.q) != n for frame in frames):
        raise ValueError("Frame joint dimensions do not match selected joints")
    chest_nominal = validate_transform(chest_nominal, "nominal camera transform")
    layout = UpperParameters(n, estimate_camera, estimate_depth_scale)
    parameter_count = layout.size
    if parameter_count == 0:
        raise ValueError("Select at least one parameter to estimate")
    x = np.zeros(parameter_count) if initial is None else np.asarray(initial, dtype=float)
    if x.shape != (parameter_count,) or not np.isfinite(x).all():
        raise ValueError("Initial parameter vector has invalid shape/values")
    bounds = layout.bounds(np.deg2rad(offset_bound_deg), camera_bound, depth_scale_log_bound)
    if bounds.shape != x.shape or np.any(bounds <= 0) or np.isnan(bounds).any():
        raise ValueError("Camera bounds must be six positive values in m/rad")
    if estimate_depth_scale and not np.isfinite(depth_scale_log_bound):
        raise ValueError("Depth scale log bound must be positive and finite")
    tolerance = layout.tolerances()
    if convergence_tolerance is not None:
        tolerance = np.broadcast_to(np.asarray(convergence_tolerance, dtype=float), x.shape)
    if np.any(tolerance <= 0) or not np.isfinite(tolerance).all() or convergence_patience < 1:
        raise ValueError("Require positive convergence tolerances and patience")

    def correction(parameters):
        return layout.camera_correction(parameters)

    initialization = {"used": False, "reason": "disabled or no independent geometry"}
    if initialize_camera and estimate_camera and initial is None:
        independent = robot.calibration_independent_links()
        if any(surface.link in independent for surface in surfaces):
            for camera_iteration in range(4):
                camera = chest_nominal @ se3_exp(correction(x))
                anchor_matches = []
                for frame in frames:
                    groups = _match_frame(
                        robot,
                        surfaces,
                        frame,
                        np.zeros(n),
                        camera,
                        schedule[0],
                        matching,
                        0.0,
                        surface_distance_normals,
                        trim_scope,
                        layout.depth_scale(x),
                    )
                    anchor_matches.append((frame, [g for g in groups if g.link in independent]))
                count = sum(len(g.observed) for _, groups in anchor_matches for g in groups)
                if count < 6:
                    break
                anchor = _UpperObjective(
                    robot,
                    anchor_matches,
                    chest_nominal,
                    sigma,
                    True,
                    estimate_depth_scale=estimate_depth_scale,
                    point_noise_model="isotropic",
                    axial_floor_fraction=axial_floor_fraction,
                    axial_range_power=axial_range_power,
                )
                anchor.initialize_whitening(x)
                j = anchor.jac(x)[:, n:]
                _, s, vt = np.linalg.svd(j, full_matrices=False)
                rank = int(np.count_nonzero(s > max(s[0] * 1e-5, 1e-8)))
                if rank == 0:
                    break
                basis = vt[:rank].T
                start = x[n:].copy()

                def expand(z, start=start, basis=basis):
                    return np.r_[np.zeros(n), start + basis @ z]

                fit_camera = least_squares(
                    lambda z, anchor=anchor, expand=expand: anchor.fun(expand(z)),
                    np.zeros(rank),
                    jac=lambda z, anchor=anchor, expand=expand, basis=basis: (
                        anchor.jac(expand(z))[:, n:] @ basis
                    ),
                    loss="huber",
                    f_scale=robust_scale,
                    max_nfev=40,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-8,
                )
                if not fit_camera.success:
                    break
                limit = np.nextafter(bounds[n:], 0.0)
                x[n:] = np.clip(start + basis @ fit_camera.x, -limit, limit)
                initialization = {
                    "used": True,
                    "reason": "independent model geometry",
                    "anchor_matches": count,
                    "local_rank": rank,
                    "iterations": camera_iteration + 1,
                    "camera_correction": correction(x).tolist(),
                    "nuisance_correction": x[n:].tolist(),
                }
                if np.max(np.abs(basis @ fit_camera.x)) < 1e-6:
                    break

    def correspondences(parameters, iteration, schedule_level):
        camera = chest_nominal @ se3_exp(correction(parameters))
        matches = []
        distance = schedule[schedule_level]
        active_trim = trim_fraction if iteration >= trim_start_iteration else 0.0
        for frame in frames:
            matches.append(
                (
                    frame,
                    _match_frame(
                        robot,
                        surfaces,
                        frame,
                        parameters[:n],
                        camera,
                        distance,
                        matching,
                        active_trim,
                        surface_distance_normals,
                        trim_scope,
                        layout.depth_scale(parameters),
                    ),
                )
            )
        if sum(len(g.observed) for _, groups in matches for g in groups) < len(x):
            raise ValueError("Insufficient ICP correspondences; check initial pose and masks")
        return matches

    all_success = True
    history = []
    converged = False
    stable_iterations = 0
    active_level = 0
    for iteration in range(outer_iterations):
        schedule_level = (
            min(iteration, len(schedule) - 1)
            if distance_schedule_trigger == "iteration"
            else active_level
        )
        matches = correspondences(x, iteration, schedule_level)
        objective = _UpperObjective(
            robot,
            matches,
            chest_nominal,
            sigma,
            estimate_camera,
            encoder_sigma_rad,
            estimate_depth_scale,
            point_noise_model if iteration >= axial_start_iteration else "isotropic",
            axial_floor_fraction,
            axial_range_power,
        )
        objective.initialize_whitening(x)
        initial_residual = objective.fun(x)
        fit = least_squares(
            objective.fun,
            x,
            jac=objective.jac if jacobian_mode == "analytic" else "3-point",
            loss=loss,
            f_scale=robust_scale,
            x_scale="jac",
            max_nfev=max_nfev,
            bounds=(-bounds, bounds),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-8,
        )
        all_success &= bool(fit.success)
        change = np.linalg.norm(fit.x - x)
        effective_tolerance = tolerance
        if uncertainty_step_fraction:
            local_r = objective.fun(fit.x)
            local_weights = _sqrt_robust_weights(local_r, loss, robust_scale)
            local_diagnostics = analyze(objective.jac(fit.x) * local_weights[:, None])
            # Do not let infinite gauge uncertainty make arbitrary steps pass.
            if local_diagnostics.rank == len(x):
                effective_tolerance = np.maximum(
                    tolerance, uncertainty_step_fraction * np.array(local_diagnostics.sigma)
                )
        scaled_step = float(np.max(np.abs(fit.x - x) / effective_tolerance))
        damping = step_relaxation if scaled_step <= relaxation_step_ratio else 1.0
        x = fit.x if damping == 1.0 else x + damping * (fit.x - x)
        fitted_residual = objective.fun(x)
        history.append(
            {
                "iteration": iteration + 1,
                "matches": len(fitted_residual),
                "initial_rms_mm": float(np.sqrt(np.mean(initial_residual**2)) * sigma * 1000),
                "fitted_rms_mm": float(np.sqrt(np.mean(fitted_residual**2)) * sigma * 1000),
                "cost": float(fit.cost),
                "parameter_step_norm": float(change),
                "step_relaxation": damping,
                "accepted_step_norm": float(change * damping),
                "tolerance_step_ratio": scaled_step,
                "uncertainty_step_fraction": uncertainty_step_fraction,
                "point_noise_model": objective.point_noise_model,
                "nfev": fit.nfev,
                "solver_success": bool(fit.success),
                "solver_status": int(fit.status),
                "solver_message": str(fit.message),
                "correspondence_distance_m": float(schedule[schedule_level]),
                "distance_schedule_level": schedule_level,
                "trim_fraction": trim_fraction if iteration >= trim_start_iteration else 0.0,
            }
        )
        final_noise_active = objective.point_noise_model == point_noise_model
        stable_iterations = (
            stable_iterations + 1
            if scaled_step <= 1.0 and final_noise_active and fit.success
            else 0
        )
        if stable_iterations >= convergence_patience:
            if distance_schedule_trigger == "stable" and active_level < len(schedule) - 1:
                active_level += 1
                stable_iterations = 0
                continue
            converged = True
            break
    # Diagnostics describe the last, refreshed, fixed correspondence set.
    matches = correspondences(x, iteration, schedule_level)
    objective = _UpperObjective(
        robot,
        matches,
        chest_nominal,
        sigma,
        estimate_camera,
        encoder_sigma_rad,
        estimate_depth_scale,
        point_noise_model,
        axial_floor_fraction,
        axial_range_power,
    )
    objective.initialize_whitening(x)
    r = objective.fun(x)
    raw_r = objective.raw_residual(x)
    # Huber Gauss-Newton weights; conditional on fixed ICP associations.
    weights = _sqrt_robust_weights(r, loss, robust_scale)
    diagnostics = analyze(
        objective.jac(x) * weights[:, None], max_sigma=max_sigma, parameter_scale=parameter_scale
    )
    frame_matches = [sum(len(g.observed) for g in groups) for _, groups in matches]
    link_matches = {}
    for _, groups in matches:
        for group in groups:
            link_matches[group.link] = link_matches.get(group.link, 0) + len(group.observed)
    normalized_rms = float(np.sqrt(np.mean(r**2)))
    rejection_fraction = 1 - len(r) / sum(len(f.points_camera) for f in frames)
    coverage_passed = rejection_fraction <= max_rejection_fraction
    result = UpperResult(
        x[:n],
        chest_nominal @ se3_exp(correction(x)),
        correction(x),
        diagnostics,
        float(np.sqrt(np.mean(raw_r**2)) * sigma),
        len(r),
        bool(fit.success),
        iteration + 1,
        history,
        converged,
        rejection_fraction,
        bool(np.any(np.abs(x) >= bounds * (1 - 1e-6))),
        frame_matches,
        link_matches,
        initialization,
        normalized_rms <= max_normalized_rms and coverage_passed,
        {
            "normalized_rms": normalized_rms,
            "limit": max_normalized_rms,
            "basis": f"{point_noise_model} point/plane noise plus optional shared encoder covariance",
            "point_noise_model": point_noise_model,
            "coverage_passed": coverage_passed,
            "rejection_fraction": rejection_fraction,
            "max_rejection_fraction": max_rejection_fraction,
        },
        estimate_camera=estimate_camera,
        estimate_depth_scale=estimate_depth_scale,
        depth_scale=layout.depth_scale(x),
        residual_profile=residual_profile(matches, raw_r * sigma),
        all_inner_success=all_success,
    )
    if (
        local_continuation
        and not converged
        and result.success
        and result.quality_passed
        and not result.at_bounds
        and diagnostics.rank == len(x)
        and all(diagnostics.supported)
    ):
        # Continue a noise-consistent local fit before changing its camera basin.
        # The complete proposed step still controls convergence in the child.
        options = {
            **restart_arguments,
            "initial": x.copy(),
            "allow_restarts": False,
            "local_continuation": False,
            "initialize_camera": False,
            "step_relaxation": min(step_relaxation, 0.25),
            "trim_start_iteration": 0,
            "axial_start_iteration": 0,
        }
        trial = calibrate_upper(**options)
        initialization["local_continuation"] = {
            "used": True,
            "converged": trial.converged,
            "primary_iterations": result.iterations,
            "primary_history": result.history,
            "continuation_history": trial.history,
            "quality_passed": trial.quality_passed,
        }
        if (
            trial.success
            and trial.converged
            and trial.quality_passed
            and not trial.at_bounds
            and trial.diagnostics.rank == len(x)
            and all(trial.diagnostics.supported)
        ):
            trial.all_inner_success = result.all_inner_success and trial.all_inner_success
            trial.history = [dict(row, phase="primary") for row in result.history] + [
                dict(row, phase="local_continuation") for row in trial.history
            ]
            trial.iterations += result.iterations
            trial.initialization = dict(initialization)
            return trial
    if (
        allow_restarts
        and estimate_camera
        and diagnostics.rank == len(x)
        and (not converged or not result.quality_passed)
    ):
        # Rigid CAD geometry can have repeated features (e.g. a long body tube).
        # Retry bounded neighboring camera translations, selecting by measured
        # residual consistency only; no injected offsets or truth enter here.
        center = np.r_[np.zeros(n), initialization.get("nuisance_correction", x[n:])]
        attempts = []
        best = result

        def valid_candidate(candidate):
            return (
                candidate.success
                and candidate.converged
                and candidate.quality_passed
                and not candidate.at_bounds
                and candidate.diagnostics.rank == len(x)
                and all(candidate.diagnostics.supported)
            )

        for root_direction in np.eye(3)[[2, 0, 1]]:
            direction = chest_nominal[:3, :3].T @ root_direction
            for sign in (1.0, -1.0):
                candidate = center.copy()
                candidate[n : n + 3] += sign * restart_translation_m * direction
                candidate = np.clip(
                    candidate, -np.nextafter(bounds, 0.0), np.nextafter(bounds, 0.0)
                )
                options = {
                    **restart_arguments,
                    "initial": candidate,
                    "allow_restarts": False,
                    "initialize_camera": False,
                }
                try:
                    trial = calibrate_upper(**options)
                except ValueError as error:
                    attempts.append(
                        {
                            "root_translation_direction": (root_direction * sign).tolist(),
                            "error": str(error),
                        }
                    )
                    continue
                attempts.append(
                    {
                        "root_translation_direction": (root_direction * sign).tolist(),
                        "normalized_rms": trial.quality["normalized_rms"],
                        "converged": trial.converged,
                        "quality_passed": trial.quality_passed,
                    }
                )
                trial_valid = valid_candidate(trial)
                best_valid = valid_candidate(best)
                if (trial_valid and not best_valid) or (
                    trial_valid == best_valid
                    and trial.quality["normalized_rms"] < best.quality["normalized_rms"]
                ):
                    best = trial
                if valid_candidate(best):
                    break
            if valid_candidate(best):
                break
        primary = dict(initialization)
        best.initialization = {
            **best.initialization,
            "primary_initialization": primary,
            "restart_attempts": attempts,
        }
        result = best
    return result


def calibrate_wrist(
    body_motions,
    camera_motions,
    nominal=None,
    translation_sigma=0.001,
    rotation_sigma=DEFAULT_ROTATION_SIGMA,
):
    """Generic AX=XB wrist kernel, NOT the unspecified mutual-view frontend.

    Call after freezing arm/chest estimates. Motions must already be formed in
    the corrected wrist-link frame. Rich excitation is needed for full SE(3).
    """
    a, b = np.asarray(body_motions), np.asarray(camera_motions)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4, 4) or len(a) < 2:
        raise ValueError("Expected matching (N,4,4) motion arrays")
    if translation_sigma <= 0 or rotation_sigma <= 0:
        raise ValueError("Noise standard deviations must be positive")
    nominal = np.eye(4) if nominal is None else np.asarray(nominal)
    sigma = np.r_[np.full(3, translation_sigma), np.full(3, rotation_sigma)]

    def residual(x):
        transform = nominal @ se3_exp(x)
        return np.concatenate(
            [
                se3_log(inverse(ai @ transform) @ transform @ bi) / sigma
                for ai, bi in zip(a, b, strict=True)
            ]
        )

    fit = least_squares(residual, np.zeros(6), jac="3-point")
    return nominal @ se3_exp(fit.x), analyze(jacobian(residual, fit.x)), bool(fit.success)
