"""Select informative measured frames without accessing synthetic ground truth."""

from dataclasses import dataclass

import numpy as np

from .observability import analyze
from .upper import DepthFrame, _match_frame, _UpperObjective


@dataclass
class FrameSelection:
    indices: list
    gains: list
    diagnostics: object
    candidate_points: list


def select_frames(
    robot,
    surfaces,
    frames,
    camera_nominal,
    count,
    *,
    estimate_camera=True,
    estimate_depth_scale=False,
    sigma=0.001,
    matching="triangle",
    max_points_per_frame=500,
    correspondence_distance=0.03,
    parameter_scale=None,
    regularization=1e-6,
    seed=0,
):
    """Greedy D-optimal selection using nominal-state point/plane Jacobians.

    Regularization stabilizes logdet scores ONLY and is not passed to the final
    calibration or diagnostics. Sampling uses only measured points, joint states
    and the nominal model. No visibility labels or true corrections are inputs.
    """
    if not 1 <= count <= len(frames) or max_points_per_frame < 1 or regularization <= 0:
        raise ValueError("Invalid frame-selection count/point limit/regularizer")
    n = len(robot.active_names) + (6 if estimate_camera else 0) + int(estimate_depth_scale)
    scale = np.ones(n) if parameter_scale is None else np.asarray(parameter_scale, dtype=float)
    if scale.shape != (n,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Selection parameter scales must be finite and positive")
    rng = np.random.default_rng(seed)
    jacobians, information, point_counts = [], [], []
    initial = np.zeros(n)
    for frame in frames:
        points = frame.points_camera
        if len(points) > max_points_per_frame:
            rows = np.sort(rng.choice(len(points), max_points_per_frame, replace=False))
            points = points[rows]
        reduced = DepthFrame(frame.q, points, frame.fixed_positions)
        groups = _match_frame(
            robot,
            surfaces,
            reduced,
            initial[: len(robot.active_names)],
            camera_nominal,
            correspondence_distance,
            matching,
            0.0,
        )
        if not groups:
            jac = np.zeros((0, n))
        else:
            jac = _UpperObjective(
                robot,
                [(reduced, groups)],
                camera_nominal,
                sigma,
                estimate_camera,
                estimate_depth_scale=estimate_depth_scale,
            ).jac(initial)
        jacobians.append(jac)
        scaled = jac * scale
        information.append(scaled.T @ scaled)
        point_counts.append(len(jac))
    total = np.eye(n) * regularization
    selected, gains = [], []
    for _ in range(count):
        previous = np.linalg.slogdet(total)[1]
        choices = [
            (np.linalg.slogdet(total + h)[1], -i)
            for i, h in enumerate(information)
            if i not in selected
        ]
        score, negative_index = max(choices)
        index = -negative_index
        selected.append(index)
        gains.append(float(score - previous))
        total += information[index]
    combined = np.concatenate([jacobians[i] for i in selected])
    return FrameSelection(selected, gains, analyze(combined, parameter_scale=scale), point_counts)
