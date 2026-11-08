"""Seeded toy robots and noiseless/noisy experiments; never the paper's A3 data."""

import numpy as np
from scipy.optimize import least_squares

from .geometry import inverse, pose, rotation_error_deg, se3_exp, se3_log, so3_exp, transform_points
from .head import BearingPair, CameraMotion, calibrate_head
from .kinematics import Joint, Robot
from .lower import calibrate_lower
from .upper import DepthFrame, Surface, calibrate_upper, calibrate_wrist, model_cloud


def _box(rng, link, center, size, per_face=8):
    points, normals = [], []
    for axis in range(3):
        for sign in (-1, 1):
            p = rng.uniform(-0.5, 0.5, (per_face, 3)) * size
            p[:, axis] = sign * size[axis] / 2
            normal = np.zeros_like(p)
            normal[:, axis] = sign
            points.append(p + center)
            normals.append(normal)
    return Surface(link, np.concatenate(points), np.concatenate(normals))


def upper_problem(seed=7, noise_m=0.0):
    rng = np.random.default_rng(seed)
    joints, surfaces = [], []
    # A static torso patch anchors camera pose in this TOY scene. Removing it
    # can introduce camera/shoulder gauges; it is not an A3 identifiability proof.
    surfaces.append(_box(rng, "base", np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.3, 0.3])))
    axes = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    for side, sign in (("left", 1), ("right", -1)):
        parent = "base"
        for k, axis in enumerate(axes):
            child = f"{side}_{k}"
            origin = [0, sign * 0.25, 0.2] if k == 0 else [0.10, 0, 0]
            joints.append(Joint(child, parent, child, pose(translation=origin), np.array(axis)))
            surfaces.append(
                _box(rng, child, np.array([0.045, 0, 0]), np.array([0.08, 0.05, 0.045]))
            )
            parent = child
    robot = Robot("base", joints)
    truth = np.deg2rad(
        [0.55, -0.32, 0.18, -0.47, 0.26, -0.14, 0.39, -0.41, 0.23, -0.58, 0.16, 0.34, -0.27, 0.49]
    )
    nominal = pose(translation=[0.1, 0, 1.0])
    correction = np.r_[0.003, -0.004, 0.002, np.deg2rad([0.3, -0.4, 0.2])]
    camera = nominal @ se3_exp(correction)
    frames = []
    for _ in range(8):
        q = rng.uniform(-0.65, 0.65, 14)
        cloud, _ = model_cloud(robot, q + truth, surfaces)
        observed = transform_points(inverse(camera), cloud)
        observed += rng.normal(0, noise_m, observed.shape)
        frames.append(DepthFrame(q, observed))
    return robot, surfaces, frames, nominal, truth, camera


def run_upper(seed=7, noise_m=0.0):
    robot, surfaces, frames, nominal, truth, camera = upper_problem(seed, noise_m)
    fit = calibrate_upper(robot, surfaces, frames, nominal)
    rng = np.random.default_rng(seed + 1)
    wrist_reports = []
    for _ in range(2):
        x = se3_exp(np.r_[rng.normal(0, 0.02, 3), rng.normal(0, 0.1, 3)])
        a = np.array(
            [se3_exp(np.r_[rng.normal(0, 0.1, 3), rng.normal(0, 0.3, 3)]) for _ in range(18)]
        )
        b = np.array([inverse(x) @ motion @ x for motion in a])
        estimate, diag, success = calibrate_wrist(a, b)
        wrist_reports.append(
            {
                "success": success,
                "rank": diag.rank,
                "translation_error_mm": float(np.linalg.norm(estimate[:3, 3] - x[:3, 3]) * 1000),
                "rotation_error_deg": rotation_error_deg(estimate[:3, :3], x[:3, :3]),
            }
        )
    return {
        "module": "upper",
        "synthetic": True,
        "seed": seed,
        "noise_m": noise_m,
        "success": fit.success,
        "joint_error_deg": np.rad2deg(fit.offsets - truth).tolist(),
        "max_joint_error_deg": float(np.max(np.abs(np.rad2deg(fit.offsets - truth)))),
        "chest_translation_error_mm": float(
            np.linalg.norm(fit.chest_transform[:3, 3] - camera[:3, 3]) * 1000
        ),
        "chest_rotation_error_deg": rotation_error_deg(fit.chest_transform[:3, :3], camera[:3, :3]),
        "point_to_plane_rms_mm": fit.rms_m * 1000,
        "correspondences": fit.correspondences,
        "iterations": fit.iterations,
        "diagnostics": fit.diagnostics.to_dict(),
        "wrist_handeye": wrist_reports,
        "assumptions": [
            "toy sampled surfaces include an anchored torso patch",
            "no visibility renderer or depth occlusion",
            "wrist kernel uses synthetic AX=XB motions, not mutual-view images",
        ],
    }


def leg_robot():
    joints = []
    axes = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]
    for side, sign in (("left", 1), ("right", -1)):
        parent = "base"
        for k, axis in enumerate(axes):
            child = f"{side}_{k}"
            offset = (
                [0, sign * 0.13, 0]
                if k == 0
                else [0, 0, -0.36 if k == 3 else (-0.34 if k == 4 else 0)]
            )
            joints.append(Joint(child, parent, child, pose(translation=offset), np.array(axis)))
            parent = child
        joints.append(
            Joint(
                f"{side}_sole",
                parent,
                f"{side}_foot",
                pose(translation=[0, 0, -0.08]),
                np.array([1.0, 0, 0]),
                "fixed",
            )
        )
    return Robot("base", joints)


def lower_problem(seed=7, encoder_noise_deg=0.0):
    rng = np.random.default_rng(seed)
    robot = leg_robot()
    # Four base poses, exactly the SAME world feet throughout all four stances.
    configurations = [
        ([0, 0, 0.71], [0, 0, 0]),
        ([0.035, 0.025, 0.70], [0.055, -0.06, 0.045]),
        ([-0.03, -0.035, 0.69], [-0.065, 0.075, -0.05]),
        ([0.015, 0.01, 0.66], [0.035, 0.035, 0.08]),
    ]
    actual, gravity = [], []
    previous = np.tile([-0.4, 0, 0, 0.8, -0.4, 0], 2)
    for translation, angles in configurations:
        world_base = pose(so3_exp(angles), translation)
        targets = {
            f"{s}_foot": inverse(world_base) @ pose(translation=[0, sign * 0.13, 0])
            for s, sign in (("left", 1), ("right", -1))
        }

        def error(q, targets=targets):
            tf = robot.fk(q)
            return np.concatenate(
                [se3_log(inverse(target) @ tf[link]) for link, target in targets.items()]
            )

        ik = least_squares(error, previous, jac="3-point", gtol=1e-12, ftol=1e-12, xtol=1e-12)
        if np.linalg.norm(error(ik.x)) > 1e-8:
            raise RuntimeError("Synthetic double-support IK did not close")
        actual.append(ik.x)
        gravity.append(world_base[:3, :3].T @ np.array([0.0, 0, -1]))
        previous = ik.x
    truth = np.deg2rad([2.1, -1.55, 1.35, -2.2, 1.7, -1.25, 1.85, -1.35, 1.55, -2.05, 1.4, -1.65])
    measured = np.array(actual) - truth
    measured += rng.normal(0, np.deg2rad(encoder_noise_deg), measured.shape)
    gravity = np.asarray(gravity)

    def gravity_residual(q_corrected, transforms):
        # Extra explicit assumption: both sole frames are level on the ground.
        # IMU mounting is identity; gravity measured in the base. This contact
        # model is a baseline adapter, not a recovered definition of paper R_BI.
        return np.concatenate(
            [
                (t[link][:3, :3].T @ g)[:2] / np.deg2rad(0.1)
                for t, g in zip(transforms, gravity, strict=True)
                for link in ("left_foot", "right_foot")
            ]
        )

    return robot, measured, truth, gravity_residual


def run_lower(seed=7, encoder_noise_deg=0.0):
    robot, q, truth, gravity = lower_problem(seed, encoder_noise_deg)
    fit = calibrate_lower(
        robot,
        q,
        "left_foot",
        "right_foot",
        gravity_residual=gravity,
        prior_indices=[3, 9],
        prior_mean=truth[[3, 9]],
        prior_sigma=np.deg2rad([0.02, 0.02]),
    )
    # Recovery depends on the two supplied knee references; report this plainly.
    error = np.rad2deg(fit.offsets - truth)
    return {
        "module": "lower",
        "synthetic": True,
        "seed": seed,
        "encoder_noise_deg": encoder_noise_deg,
        "success": fit.success,
        "joint_error_deg": error.tolist(),
        "rms_joint_error_deg": float(np.sqrt(np.mean(error**2))),
        "max_joint_error_deg": float(np.max(np.abs(error))),
        "stances": len(q),
        "data_diagnostics": fit.data_diagnostics.to_dict(),
        "posterior_diagnostics": fit.posterior_diagnostics.to_dict(),
        "assumptions": [
            "toy 12-joint leg geometry, not A3 Ultra",
            "four IK-generated stances with fixed level feet and known base gravity",
            "two synthetic external knee references centered on ground truth",
            "posterior recovery is conditional on these knee references",
        ],
    }


def head_problem(seed=7, bearing_noise=0.0):
    rng = np.random.default_rng(seed)
    # Five overlapping toy cameras, not the panoramic/rear A3 camera topology.
    truth = np.array([so3_exp(rng.normal(0, 0.2, 3)) for _ in range(5)])
    centers = np.array(
        [[-0.12, 0, 0], [0.12, 0, 0], [0, 0.1, 0], [0.06, -0.08, 0], [-0.06, -0.08, 0]]
    )
    common = so3_exp(np.deg2rad([1.2, -0.8, 1.5]))
    initial = common.T @ truth
    motions = []
    for i, rotation in enumerate(truth):
        x = pose(rotation, centers[i])
        a = np.array(
            [
                pose(
                    so3_exp([0, 0, rng.uniform(-0.35, 0.35)]),
                    [rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1), 0],
                )
                for _ in range(20)
            ]
        )
        b = np.array([inverse(x) @ motion @ x for motion in a])
        b[:, :3, 3] /= 1.3 + 0.1 * i  # one monocular scale per stream
        motions.append(CameraMotion(a, b))
    pairs = []
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
        points = rng.uniform([-1, -0.7, 3], [1, 0.7, 6], (100, 3))
        bearings = []
        for k in (i, j):
            ray = (points - centers[k]) @ truth[k]
            ray /= ray[:, 2:3]
            ray[:, :2] += rng.normal(0, bearing_noise, (len(ray), 2))
            bearings.append(ray)
        pairs.append(BearingPair(i, j, *bearings))
    return initial, centers, motions, pairs, truth


def run_head(seed=7, bearing_noise=0.0):
    initial, centers, motions, pairs, truth = head_problem(seed, bearing_noise)
    result = calibrate_head(initial, centers, motions, pairs)
    errors = [rotation_error_deg(a, b) for a, b in zip(result["rotations"], truth, strict=True)]
    return {
        "module": "head",
        "synthetic": True,
        "seed": seed,
        "bearing_noise": bearing_noise,
        "success": result["success"],
        "rotation_error_deg": errors,
        "mean_rotation_error_deg": float(np.mean(errors)),
        "max_rotation_error_deg": float(np.max(errors)),
        "centers_retained": bool(np.array_equal(result["centers"], centers)),
        "diagnostics": result["diagnostics"].to_dict(),
        "sampson_rms": result["sampson_rms"],
        "retained_matches": result["retained_matches"],
        "assumptions": [
            "overlapping five-camera toy rig, not A3 topology",
            "initial rotations are ground truth with a common artificial perturbation",
            "visual graph initialization, time alignment and VO are not evaluated",
            "fixed exact structural baselines and planar body motions",
        ],
    }
