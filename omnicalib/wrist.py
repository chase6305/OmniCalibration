"""Staged eye-in-hand calibration from synchronized metric camera poses.

The visual odometry/mutual-view frontend is external. Root/base motion is
explicitly projected through the corrected arm chain before forming AX=XB.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import inverse, validate_transform
from .upper import DEFAULT_ROTATION_SIGMA, calibrate_wrist


@dataclass
class WristCalibration:
    transform: np.ndarray
    diagnostics: object
    success: bool
    motions: int


def trajectory_motions(poses, pairs=None):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError("At least three 4x4 poses are required")
    for t in poses:
        validate_transform(t, "trajectory pose")
    if pairs is None:
        pairs = [(i, i + 1) for i in range(len(poses) - 1)]
    if any(not (0 <= i < len(poses) and 0 <= j < len(poses)) or i == j for i, j in pairs):
        raise ValueError("Motion pair indices must name distinct valid poses")
    return np.array([inverse(poses[i]) @ poses[j] for i, j in pairs])


def calibrate_eye_in_hand(
    robot,
    frames,
    mount_link,
    camera_world_poses,
    arm_offsets,
    nominal=None,
    world_root_poses=None,
    pairs=None,
    translation_sigma=0.001,
    rotation_sigma=DEFAULT_ROTATION_SIGMA,
):
    """Freeze arm offsets, then solve a camera-to-mount transform.

    Camera world poses must have METRIC translation. Supplying unscaled
    monocular trajectories is invalid for this full-SE(3) kernel. All streams
    must already be synchronized; this method does not estimate time offsets.
    """
    from .upper import _frame_fk

    offsets = np.asarray(arm_offsets, dtype=float)
    if offsets.shape != (len(robot.active_names),) or not np.isfinite(offsets).all():
        raise ValueError("Invalid frozen arm offsets")
    camera_world_poses = np.asarray(camera_world_poses, dtype=float)
    if camera_world_poses.shape != (len(frames), 4, 4):
        raise ValueError("Provide one metric camera pose per joint/depth frame")
    if world_root_poses is None:
        world_root_poses = np.repeat(np.eye(4)[None], len(frames), axis=0)
    world_root_poses = np.asarray(world_root_poses, dtype=float)
    if world_root_poses.shape != camera_world_poses.shape:
        raise ValueError("Root motion must be synchronized to every camera pose")
    link_poses = []
    for frame, world_root in zip(frames, world_root_poses, strict=True):
        validate_transform(world_root, "world-root pose")
        transforms = _frame_fk(robot, frame, offsets)
        if mount_link not in transforms:
            raise ValueError(f"Unknown wrist mount: {mount_link}")
        link_poses.append(world_root @ transforms[mount_link])
    body = trajectory_motions(link_poses, pairs)
    camera = trajectory_motions(camera_world_poses, pairs)
    transform, diagnostics, success = calibrate_wrist(
        body, camera, nominal, translation_sigma=translation_sigma, rotation_sigma=rotation_sigma
    )
    return WristCalibration(transform, diagnostics, success, len(body))
