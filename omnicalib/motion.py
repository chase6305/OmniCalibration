"""Joint-limit-constrained nominal IK for synthetic reaching excitations.

No collision, dynamics, velocity or hardware execution is implied by an IK
solution. This planner only creates numerical calibration test configurations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import inverse, se3_log, so3_exp, validate_transform


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_errors_m: dict
    rotation_errors_rad: dict


def solve_ik(
    robot,
    targets,
    initial=None,
    *,
    regularization=1e-4,
    max_nfev=200,
    position_tolerance=0.001,
    rotation_tolerance=0.01,
):
    if not targets:
        raise ValueError("IK requires at least one target")
    targets = {link: validate_transform(t, f"IK target {link}") for link, t in targets.items()}
    joints = {j.name: j for j in robot.joints}
    lower = np.array([max(-np.pi, joints[name].lower) for name in robot.active_names])
    upper = np.array([min(np.pi, joints[name].upper) for name in robot.active_names])
    q0 = np.zeros(len(lower)) if initial is None else np.asarray(initial, dtype=float)
    q0 = np.clip(q0, lower + 1e-8, upper - 1e-8)

    def residual(q):
        transforms = robot.fk(q)
        return np.r_[
            np.concatenate(
                [se3_log(inverse(target) @ transforms[link]) for link, target in targets.items()]
            ),
            regularization * (q - q0),
        ]

    fit = least_squares(
        residual,
        q0,
        jac="3-point",
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    transforms = robot.fk(fit.x)
    errors = {link: se3_log(inverse(t) @ transforms[link]) for link, t in targets.items()}
    positions = {link: float(np.linalg.norm(v[:3])) for link, v in errors.items()}
    rotations = {link: float(np.linalg.norm(v[3:])) for link, v in errors.items()}
    success = (
        fit.success
        and max(positions.values()) <= position_tolerance
        and max(rotations.values()) <= rotation_tolerance
    )
    return IKResult(fit.x, bool(success), positions, rotations)


def sample_reaching_poses(
    robot, nominal_targets, count, *, seed=0, position_jitter_m=0.025, rotation_jitter_rad=0.12
):
    rng = np.random.default_rng(seed)
    previous = None
    poses = []
    attempts = 0
    while len(poses) < count and attempts < count * 8:
        attempts += 1
        targets = {}
        for link, nominal in nominal_targets.items():
            target = np.asarray(nominal).copy()
            target[:3, 3] += rng.normal(0, position_jitter_m, 3)
            target[:3, :3] = so3_exp(rng.normal(0, rotation_jitter_rad, 3)) @ target[:3, :3]
            targets[link] = target
        fit = solve_ik(robot, targets, previous)
        if fit.success:
            poses.append(fit.q)
            previous = fit.q
    if len(poses) != count:
        raise RuntimeError(f"Only {len(poses)}/{count} reaching targets were reachable")
    return np.asarray(poses)
