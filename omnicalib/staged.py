"""Asset-based synthetic validation of arm-first, wrist-second calibration."""

import numpy as np

from .assets import load_asset_config, run_asset_demo
from .dataset import load_dataset
from .geometry import pose, rotation_error_deg, se3_exp
from .wrist import calibrate_eye_in_hand


def run_staged_demo(
    config_path,
    dataset_path,
    *,
    seed=7,
    asset_root=None,
    pose_translation_noise_m=0.0003,
    pose_rotation_noise_deg=0.01,
    moving_root=False,
    **arm_options,
):
    """Wrist pose streams are synthetic metric measurements, not rendered VO."""
    if (
        arm_options.get("observation_mesh_scale", 1.0) != 1.0
        or arm_options.get("origin_translation_sigma_m", 0.0) != 0.0
    ):
        raise ValueError("Staged demo currently requires matched nominal/physical kinematics")
    if min(pose_translation_noise_m, pose_rotation_noise_deg) < 0:
        raise ValueError("Wrist pose noise must be nonnegative")
    arm = run_asset_demo(
        config_path, asset_root=asset_root, seed=seed, dataset_path=dataset_path, **arm_options
    )
    config, _, robot = load_asset_config(config_path, asset_root)
    # Truth is used only to synthesize observations and score outputs below.
    generation = load_dataset(dataset_path, include_truth=True)
    actual_q = np.asarray(generation.truth["actual_joint_positions"])
    frozen_offsets = np.deg2rad(arm["estimated_offsets_deg"])
    rng = np.random.default_rng(seed + 73009)
    world_root = np.array(
        [se3_exp(rng.normal(0, 0.03, 6)) if moving_root else np.eye(4) for _ in actual_q]
    )
    links = config.get("wrist_links", ["left_ee", "right_ee"])
    zero_transforms, screws = robot.fk_with_jacobian(np.zeros(len(robot.active_names)))
    reports = []
    for link in links:
        if link not in zero_transforms or np.linalg.norm(screws[link]) < 1e-10:
            continue
        nominal = pose(translation=[0.025, 0.01, 0.035])
        truth = nominal @ se3_exp(np.r_[rng.normal(0, 0.004, 3), rng.normal(0, np.deg2rad(0.5), 3)])
        cameras = []
        for q, base in zip(actual_q, world_root, strict=True):
            noise = np.r_[
                rng.normal(0, pose_translation_noise_m, 3),
                rng.normal(0, np.deg2rad(pose_rotation_noise_deg), 3),
            ]
            cameras.append(base @ robot.fk(q)[link] @ truth @ se3_exp(noise))
        fitted = calibrate_eye_in_hand(
            robot,
            generation.frames,
            link,
            cameras,
            frozen_offsets,
            nominal,
            world_root_poses=world_root,
        )
        uninformed = calibrate_eye_in_hand(
            robot,
            generation.frames,
            link,
            cameras,
            np.zeros_like(frozen_offsets),
            nominal,
            world_root_poses=world_root,
        )
        supported = (
            arm["proposal"]["block_supported"]
            and fitted.success
            and all(fitted.diagnostics.supported)
        )
        reports.append(
            {
                "mount_link": link,
                "camera_transform": fitted.transform.tolist(),
                "diagnostics": fitted.diagnostics.to_dict(),
                "success": fitted.success,
                "proposed": bool(supported),
                "motions": fitted.motions,
                "translation_error_mm": float(
                    np.linalg.norm(fitted.transform[:3, 3] - truth[:3, 3]) * 1000
                ),
                "rotation_error_deg": rotation_error_deg(fitted.transform[:3, :3], truth[:3, :3]),
                "without_arm_correction_translation_error_mm": float(
                    np.linalg.norm(uninformed.transform[:3, 3] - truth[:3, 3]) * 1000
                ),
                "without_arm_correction_rotation_error_deg": rotation_error_deg(
                    uninformed.transform[:3, :3], truth[:3, :3]
                ),
            }
        )
    return {
        "module": "staged_upper",
        "synthetic": True,
        "seed": seed,
        "arm": arm,
        "wrists": reports,
        "moving_root": moving_root,
        "pose_translation_noise_m": pose_translation_noise_m,
        "pose_rotation_noise_deg": pose_rotation_noise_deg,
        "assumptions": [
            "Marvin kinematics with hypothetical wrist-camera mounts",
            "synthetic metric camera poses, not a visual odometry implementation",
            "arm corrections frozen before wrist extrinsic estimation",
            "known synchronized world-root motion when moving_root is enabled",
            "source URDF and robot controller are not modified",
        ],
    }
