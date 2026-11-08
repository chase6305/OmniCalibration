"""Read existing robot assets without copying or changing the source directory."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inverse, pose, rotation_error_deg, se3_exp, so3_exp, transform_points
from .kinematics import Robot
from .reporting import environment_versions
from .upper import DepthFrame, Surface, calibrate_upper, model_cloud


def sample_urdf_surfaces(urdf_path, links, samples_per_link=100, seed=7, geometry="visual"):
    """Area-weighted triangle samples; honor URDF mesh scales and local origins.

    Relative and absolute mesh paths supported. ROS package:// needs an external
    resolver and is rejected explicitly. Scene-node transforms are loaded by
    trimesh's mesh conversion. No silhouette, occlusion, or sensor simulation.
    """
    import trimesh

    rng = np.random.default_rng(seed)
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    elements = {e.get("name"): e for e in root.findall("link")}
    surfaces, sources = [], []
    for link in links:
        if link not in elements:
            raise ValueError(f"Unknown surface link: {link}")
        meshes = []
        for element in elements[link].findall(geometry):
            mesh_element = element.find("geometry/mesh")
            if mesh_element is None:
                continue
            filename = mesh_element.get("filename")
            if filename.startswith("package://"):
                raise ValueError("Resolve package:// mesh paths before sampling")
            path = (urdf_path.parent / filename).resolve()
            mesh = trimesh.load(path, force="mesh", process=False)
            mesh.apply_scale(np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" "))
            origin = element.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                mesh.apply_transform(pose(Rotation.from_euler("xyz", rpy).as_matrix(), xyz))
            meshes.append(mesh)
            sources.append(
                {
                    "link": link,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if not meshes:
            raise ValueError(f"Link {link} has no {geometry} mesh")
        mesh = trimesh.util.concatenate(meshes)
        areas = np.asarray(mesh.area_faces)
        if not np.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError(f"Invalid mesh area for {link}")
        indices = rng.choice(len(areas), samples_per_link, p=areas / areas.sum())
        triangles = mesh.triangles[indices]
        uv = rng.random((samples_per_link, 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        points = (
            triangles[:, 0]
            + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
            + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
        )
        surfaces.append(Surface(link, points, mesh.face_normals[indices], mesh))
    return surfaces, sources


def load_asset_config(config_path, asset_root=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    # Checked-in configs live in <repo>/configs; asset_root is repo-relative.
    base = (
        Path(asset_root).resolve()
        if asset_root
        else (config_path.parent.parent / config["asset_root"]).resolve()
    )
    urdf = base / config["urdf"]
    model = Robot.from_urdf(urdf)
    selected = model.select(config["joint_names"], config.get("fixed_positions"))
    return config, urdf, selected


def correction_proposal(names, fit, base_frame, camera_frame="synthetic_camera", synthetic=True):
    """Reviewable metadata only; retain the nominal model if a block is weak.

    Rank/conditioning is necessary evidence, not a guarantee about real data.
    This synthetic proposal is never applied to the source URDF or encoders.
    """
    diag = fit.diagnostics
    units = ["rad"] * len(names) + ["m"] * 3 + ["rad"] * 3
    parameters = list(names) + [f"camera_{x}" for x in ("tx", "ty", "tz", "rx", "ry", "rz")]
    estimates = np.r_[fit.offsets, fit.correction]
    camera_estimated = getattr(fit, "estimate_camera", diag.parameters > len(names))
    scale_estimated = getattr(fit, "estimate_depth_scale", False)
    if scale_estimated:
        parameters.append("depth_log_scale")
        units.append("1")
        estimates = np.r_[estimates, np.log(fit.depth_scale)]
    reasons = []
    if not fit.success:
        reasons.append("inner_solver_failed")
    if not fit.converged:
        reasons.append("icp_not_converged")
    if fit.at_bounds:
        reasons.append("parameter_bound_active")
    coverage_passed = getattr(fit, "quality", {}).get("coverage_passed", True)
    if not coverage_passed:
        reasons.append("insufficient_correspondence_coverage")
    if not fit.quality_passed and coverage_passed:
        reasons.append("residual_noise_inconsistent")
    if diag.rank < diag.parameters:
        reasons.append("rank_deficient")
    elif diag.scaled_condition > diag.condition_limit:
        reasons.append("ill_conditioned")
    if diag.rank == diag.parameters and not all(diag.supported) and not reasons:
        reasons.append("uncertainty_limit_exceeded")
    block_supported = not reasons
    decisions = []
    for i, name in enumerate(parameters):
        if i < len(names):
            diagnostic_index = i
        elif i < len(names) + 6:
            diagnostic_index = i if camera_estimated else None
        else:
            diagnostic_index = len(names) + (6 if camera_estimated else 0)
        fixed = diagnostic_index is None
        supported = bool(not fixed and block_supported)
        decisions.append(
            {
                "parameter": name,
                "estimate": float(estimates[i]),
                "proposed_correction": float(estimates[i]) if supported else 0.0,
                "unit": units[i],
                "sigma": None if fixed else diag.sigma[diagnostic_index],
                "frame": base_frame if i < len(names) else camera_frame,
                "composition": (
                    "q_corrected=q_measured+delta"
                    if i < len(names)
                    else (
                        "T_BC_corrected=T_BC_nominal@Exp([v,w])"
                        if i < len(names) + 6
                        else "points_corrected=exp(depth_log_scale)*points_raw"
                    )
                ),
                "decision": "propose" if supported else "retain",
                "reason": "fixed camera extrinsic supplied by caller"
                if fixed
                else (
                    "local rank/conditioning supported"
                    if supported
                    else "solver/convergence/bounds/observability gate failed"
                ),
            }
        )
    return {
        "synthetic_only": synthetic,
        "applied_to_robot": False,
        "block_supported": block_supported,
        "block_reasons": reasons,
        "parameters": decisions,
    }


def run_asset_demo(
    config_path,
    asset_root=None,
    seed=7,
    noise_m=0.0,
    frames_count=10,
    samples_per_link=100,
    include_anchor=True,
    estimate_camera=True,
    matching="sampled",
    independent_sampling=False,
    observation_samples=None,
    rendered=False,
    image_size=(160, 120),
    max_points=2000,
    depth_noise=None,
    trim_fraction=0.0,
    outer_iterations=12,
    dataset_path=None,
    offset_amplitude_deg=0.5,
    residual_sigma_m=0.001,
    solver_options=None,
    selection_count=None,
    selection_strategy="information",
    encoder_noise_deg=0.0,
    camera_position=None,
    camera_target=None,
    observation_mesh_scale=1.0,
    origin_translation_sigma_m=0.0,
    posture_mode="random",
    estimate_depth_scale=False,
    camera_perturbation=None,
):
    if noise_m < 0 or frames_count < 2 or samples_per_link < 3:
        raise ValueError(
            "Require nonnegative noise, at least two frames and three samples per link"
        )
    config, urdf, robot = load_asset_config(config_path, asset_root)
    links = config["surface_links"].copy()
    if include_anchor:
        links += config.get("anchor_links", [])
    surfaces, sources = sample_urdf_surfaces(urdf, links, samples_per_link, seed)
    observation_surfaces = surfaces
    if independent_sampling:
        observation_surfaces, _ = sample_urdf_surfaces(
            urdf, links, observation_samples or samples_per_link, seed + 100003
        )
    if (
        observation_mesh_scale <= 0
        or not np.isfinite(observation_mesh_scale)
        or origin_translation_sigma_m < 0
        or not np.isfinite(origin_translation_sigma_m)
    ):
        raise ValueError("Observation geometry scale/noise must be finite and valid")
    if observation_mesh_scale != 1.0:
        scaled = []
        for surface in observation_surfaces:
            mesh = surface.mesh.copy()
            mesh.apply_scale(observation_mesh_scale)
            scaled.append(
                Surface(
                    surface.link, surface.points * observation_mesh_scale, surface.normals, mesh
                )
            )
        observation_surfaces = scaled
    observation_robot = robot
    origin_perturbations = {}
    if origin_translation_sigma_m:
        from dataclasses import replace

        perturb_rng = np.random.default_rng(seed + 210011)
        full = robot.robot
        joints = []
        for joint in full.joints:
            origin = joint.origin.copy()
            if joint.name in robot.active_names:
                perturbation = perturb_rng.normal(0, origin_translation_sigma_m, 3)
                origin[:3, 3] += perturbation
                origin_perturbations[joint.name] = perturbation.tolist()
            joints.append(replace(joint, origin=origin))
        observation_robot = Robot(full.root, joints).select(
            robot.active_names, robot.fixed_positions
        )
    streams = np.random.SeedSequence(seed).spawn(5)
    rng_truth, rng_pose, rng_points, rng_encoder, rng_heldout = map(np.random.default_rng, streams)
    n = len(robot.active_names)
    if not np.isfinite(offset_amplitude_deg) or offset_amplitude_deg <= 0:
        raise ValueError("Injected joint amplitude must be positive and finite")
    if not np.isfinite(encoder_noise_deg) or encoder_noise_deg < 0:
        raise ValueError("Encoder noise must be finite and nonnegative")
    truth = np.deg2rad(rng_truth.uniform(-offset_amplitude_deg, offset_amplitude_deg, n))
    position = (
        config["synthetic_camera_translation"] if camera_position is None else camera_position
    )
    camera_nominal = pose(translation=position)
    intrinsics = None
    if rendered:
        from .render import DepthNoise, Pinhole, look_at

        camera_nominal = look_at(
            position,
            config.get("synthetic_camera_target", [0, 0, 1.0])
            if camera_target is None
            else camera_target,
        )
        width, height = image_size
        intrinsics = Pinhole(
            width, height, width * 0.65, width * 0.65, (width - 1) / 2, (height - 1) / 2
        )
        depth_noise = DepthNoise(sigma_m=noise_m) if depth_noise is None else depth_noise
    camera_perturbation = np.asarray(
        np.r_[0.003, -0.004, 0.002, np.deg2rad([0.3, -0.4, 0.2])]
        if camera_perturbation is None
        else camera_perturbation,
        dtype=float,
    )
    if camera_perturbation.shape != (6,) or not np.isfinite(camera_perturbation).all():
        raise ValueError("Synthetic camera perturbation must contain six finite m/rad coordinates")
    camera_truth = (
        camera_nominal @ se3_exp(camera_perturbation) if estimate_camera else camera_nominal.copy()
    )
    by_name = {j.name: j for j in robot.joints}
    lower = np.array([max(by_name[name].lower + 0.1, -0.65) for name in robot.active_names])
    upper = np.array([min(by_name[name].upper - 0.1, 0.65) for name in robot.active_names])
    if posture_mode == "random" and np.any(lower >= upper):
        raise ValueError(
            "Synthetic joint sampling interval is empty; configure a valid posture range"
        )
    reaching_targets = None
    if posture_mode == "reaching":
        from .motion import sample_reaching_poses

        specification = config.get("reaching_targets")
        if not specification:
            raise ValueError("Configure reaching_targets for this robot/root frame")
        zero = robot.fk(np.zeros(n))
        reaching_targets = {
            entry["link"]: pose(
                so3_exp([0, 0, np.deg2rad(entry["orientation_z_delta_deg"])])
                @ zero[entry["link"]][:3, :3],
                entry["position"],
            )
            for entry in specification
        }
        reaching_q = sample_reaching_poses(robot, reaching_targets, frames_count, seed=seed + 40009)
    elif posture_mode != "random":
        raise ValueError("Choose random or reaching posture mode")
    frames = []
    actual_q_frames = []
    visibility = []
    render_seconds = 0.0
    for frame_index in range(frames_count):
        actual_q = (
            rng_pose.uniform(lower, upper) if reaching_targets is None else reaching_q[frame_index]
        )
        if rendered:
            import time

            from .render import depth_points, render_depth

            start = time.perf_counter()
            image = render_depth(
                observation_robot, actual_q, observation_surfaces, camera_truth, intrinsics
            )
            render_seconds += time.perf_counter() - start
            observed, pixels = depth_points(
                image,
                intrinsics,
                noise=depth_noise,
                seed=seed * 100003 + frame_index,
                max_points=max_points,
            )
            labels = image.link_ids.ravel()[pixels]
            visibility.append(
                {
                    "visible_pixels": int(np.isfinite(image.depth).sum()),
                    "used_points": len(observed),
                    "per_link": {
                        name: int((labels == i).sum()) for i, name in enumerate(image.link_names)
                    },
                }
            )
        else:
            cloud, _ = model_cloud(observation_robot, actual_q, observation_surfaces)
            observed = transform_points(inverse(camera_truth), cloud)
            observed += rng_points.normal(0, noise_m, observed.shape)
        measured_q = actual_q - truth
        if encoder_noise_deg:
            measured_q += rng_encoder.normal(0, np.deg2rad(encoder_noise_deg), n)
        frames.append(DepthFrame(measured_q, observed))
        actual_q_frames.append(actual_q.copy())
    candidate_count = len(frames)
    selection_report = None
    if selection_count is not None:
        if not 2 <= selection_count <= candidate_count:
            raise ValueError("Select between two and all candidate frames")
        if selection_strategy == "information":
            from .excitation import select_frames

            scales = (
                np.r_[np.full(n, np.deg2rad(1.0)), [0.01] * 3, [np.deg2rad(1.0)] * 3]
                if estimate_camera
                else np.full(n, np.deg2rad(1.0))
            )
            if estimate_depth_scale:
                scales = np.r_[scales, 0.005]
            selected = select_frames(
                robot,
                surfaces,
                frames,
                camera_nominal,
                selection_count,
                estimate_camera=estimate_camera,
                estimate_depth_scale=estimate_depth_scale,
                sigma=residual_sigma_m,
                matching=matching,
                parameter_scale=scales,
                seed=seed,
            )
            indices = selected.indices
            selection_report = {
                "strategy": selection_strategy,
                "indices": indices,
                "gains": selected.gains,
                "diagnostics": selected.diagnostics.to_dict(),
            }
        elif selection_strategy == "first":
            indices = list(range(selection_count))
        elif selection_strategy == "random":
            indices = sorted(
                np.random.default_rng(seed + 170003)
                .choice(candidate_count, selection_count, replace=False)
                .tolist()
            )
        else:
            raise ValueError("Choose information/first/random frame selection")
        frames = [frames[i] for i in indices]
        actual_q_frames = [actual_q_frames[i] for i in indices]
        if selection_report is None:
            selection_report = {"strategy": selection_strategy, "indices": indices}
    if dataset_path is not None:
        from .dataset import CalibrationDataset, save_dataset

        dataset = CalibrationDataset(
            robot.active_names,
            frames,
            camera_nominal,
            robot.root,
            "synthetic_camera",
            {
                "synthetic": True,
                "seed": seed,
                "robot": config["name"],
                "rendered_depth": rendered,
                "independent_sampling": independent_sampling,
                "intrinsics": asdict(intrinsics) if intrinsics is not None else None,
                "depth_noise": asdict(depth_noise) if rendered else None,
            },
            {
                "offsets": truth.tolist(),
                "camera_transform": camera_truth.tolist(),
                "actual_joint_positions": np.asarray(actual_q_frames).tolist(),
                "depth_correction_scale": 1 / (1 + depth_noise.scale_bias) if rendered else 1.0,
            },
        )
        save_dataset(dataset_path, dataset)
    fit = calibrate_upper(
        robot,
        surfaces,
        frames,
        camera_nominal,
        estimate_camera=estimate_camera,
        estimate_depth_scale=estimate_depth_scale,
        matching=matching,
        trim_fraction=trim_fraction,
        outer_iterations=outer_iterations,
        sigma=residual_sigma_m,
        **({} if solver_options is None else solver_options),
    )
    error = np.rad2deg(fit.offsets - truth)
    # Evaluate on configurations not used in ICP, with no noisy point matching.
    # This measures kinematic propagation of offset error under the exact URDF.
    heldout_errors = []
    heldout_before = []
    heldout_camera_errors, heldout_camera_before = [], []
    reference_surfaces = [
        Surface(s.link, s.points * observation_mesh_scale, s.normals) for s in surfaces
    ]
    heldout_q = (
        [rng_heldout.uniform(lower, upper) for _ in range(10)]
        if reaching_targets is None
        else sample_reaching_poses(robot, reaching_targets, 10, seed=seed + 40013)
    )
    for actual_q in heldout_q:
        actual_points, _ = model_cloud(observation_robot, actual_q, reference_surfaces)
        recovered_points, _ = model_cloud(robot, actual_q - truth + fit.offsets, surfaces)
        nominal_points, _ = model_cloud(robot, actual_q - truth, surfaces)
        heldout_errors.extend(np.linalg.norm(actual_points - recovered_points, axis=1))
        heldout_before.extend(np.linalg.norm(actual_points - nominal_points, axis=1))
        actual_c = transform_points(inverse(camera_truth), actual_points)
        recovered_c = transform_points(inverse(fit.chest_transform), recovered_points)
        nominal_c = transform_points(inverse(camera_nominal), nominal_points)
        heldout_camera_errors.extend(np.linalg.norm(actual_c - recovered_c, axis=1))
        heldout_camera_before.extend(np.linalg.norm(actual_c - nominal_c, axis=1))
    return {
        "module": "asset_upper",
        "synthetic": True,
        "robot": config["name"],
        "mode": config["mode"],
        "seed": seed,
        "noise_m": noise_m,
        "estimate_camera": estimate_camera,
        "estimate_depth_scale": estimate_depth_scale,
        "depth_scale": fit.depth_scale,
        "depth_scale_error_ppm": abs(
            fit.depth_scale * (1 + (depth_noise.scale_bias if rendered else 0.0)) - 1
        )
        * 1e6,
        "rng_scheme": "SeedSequence-split-v1",
        "posture_mode": posture_mode,
        "environment": environment_versions(),
        "solver_options": {} if solver_options is None else solver_options,
        "depth_noise_parameters": asdict(depth_noise) if rendered else None,
        "intrinsics": asdict(intrinsics) if intrinsics is not None else None,
        "samples_per_link": samples_per_link,
        "outer_iteration_limit": outer_iterations,
        "trim_fraction": trim_fraction,
        "matching": matching,
        "independent_sampling": independent_sampling,
        "injected_amplitude_deg": offset_amplitude_deg,
        "encoder_noise_deg": encoder_noise_deg,
        "observation_mesh_scale": observation_mesh_scale,
        "origin_translation_sigma_m": origin_translation_sigma_m,
        "origin_perturbations_m": origin_perturbations,
        "camera_nominal": camera_nominal.tolist(),
        "injected_camera_correction": camera_perturbation.tolist()
        if estimate_camera
        else [0.0] * 6,
        "residual_sigma_m": residual_sigma_m,
        "rendered_depth": rendered,
        "visibility": visibility,
        "render_seconds": render_seconds,
        "noise_model": "axial optical-depth perturbation"
        if rendered
        else "isotropic camera point perturbation",
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "mesh_sources": sources,
        "joint_names": robot.active_names,
        "injected_offsets_deg": np.rad2deg(truth).tolist(),
        "estimated_offsets_deg": np.rad2deg(fit.offsets).tolist(),
        "joint_error_deg": error.tolist(),
        "max_joint_error_deg": float(np.max(np.abs(error))),
        "rms_joint_error_deg": float(np.sqrt(np.mean(error**2))),
        "heldout_model_point_rms_mm": float(
            np.sqrt(np.mean(np.asarray(heldout_errors) ** 2)) * 1000
        ),
        "heldout_model_point_before_rms_mm": float(
            np.sqrt(np.mean(np.asarray(heldout_before) ** 2)) * 1000
        ),
        "heldout_camera_point_rms_mm": float(
            np.sqrt(np.mean(np.asarray(heldout_camera_errors) ** 2)) * 1000
        ),
        "heldout_camera_point_before_rms_mm": float(
            np.sqrt(np.mean(np.asarray(heldout_camera_before) ** 2)) * 1000
        ),
        "chest_translation_error_mm": float(
            np.linalg.norm(fit.chest_transform[:3, 3] - camera_truth[:3, 3]) * 1000
        ),
        "chest_rotation_error_deg": rotation_error_deg(
            fit.chest_transform[:3, :3], camera_truth[:3, :3]
        ),
        "point_to_plane_rms_mm": fit.rms_m * 1000,
        "diagnostics": fit.diagnostics.to_dict(),
        "success": fit.success,
        "all_inner_success": fit.all_inner_success,
        "correspondences": fit.correspondences,
        "frames": len(frames),
        "candidate_frames": candidate_count,
        "selection": selection_report,
        "icp_converged": fit.converged,
        "history": fit.history,
        "at_bounds": fit.at_bounds,
        "rejection_fraction": fit.rejection_fraction,
        "frame_match_counts": fit.frame_matches,
        "link_match_counts": fit.link_matches,
        "initialization": fit.initialization,
        "quality_passed": fit.quality_passed,
        "quality": fit.quality,
        "residual_profile": fit.residual_profile,
        "proposal": correction_proposal(robot.active_names, fit, robot.root),
        "assumptions": [
            "real URDF and visual meshes; synthetic observations and camera placement",
            "independent observation samples"
            if independent_sampling
            else "identical surface samples in forward generation and ICP model",
            "pinhole first-hit visibility and self-occlusion"
            if rendered
            else "no camera visibility or occlusion",
            "no collision or motion feasibility filtering",
            "only configured arm joints are estimated; remaining independent joints are frozen",
            f"static base anchor enabled: {include_anchor}",
            "numerical recovery is not physical calibration accuracy",
        ],
    }


def calibrate_dataset(
    config_path,
    dataset_path,
    *,
    asset_root=None,
    matching="triangle",
    samples_per_link=300,
    estimate_camera=True,
    estimate_depth_scale=False,
    include_anchor=True,
    residual_sigma_m=0.001,
    outer_iterations=20,
    trim_fraction=0.0,
    max_sigma=None,
    solver_options=None,
    joint_sigma_limit_deg=None,
    camera_translation_sigma_limit_m=None,
    camera_rotation_sigma_limit_deg=None,
    depth_scale_sigma_limit_pct=None,
):
    """Estimate from observations alone; optional dataset truth is never loaded."""
    from .dataset import load_dataset

    config, urdf, robot = load_asset_config(config_path, asset_root)
    dataset = load_dataset(dataset_path, include_truth=False)
    if dataset.root_frame != robot.root:
        raise ValueError(
            f"Dataset root {dataset.root_frame!r} differs from URDF root {robot.root!r}"
        )
    if set(dataset.joint_names) != set(robot.active_names):
        raise ValueError("Dataset joint names must match configured calibration joints")
    order = [dataset.joint_names.index(name) for name in robot.active_names]
    frames = [
        DepthFrame(frame.q[order], frame.points_camera, frame.fixed_positions)
        for frame in dataset.frames
    ]
    links = config["surface_links"] + (config.get("anchor_links", []) if include_anchor else [])
    surfaces, sources = sample_urdf_surfaces(urdf, links, samples_per_link)
    if any(
        v is not None
        for v in (
            joint_sigma_limit_deg,
            camera_translation_sigma_limit_m,
            camera_rotation_sigma_limit_deg,
            depth_scale_sigma_limit_pct,
        )
    ):
        if max_sigma is not None:
            raise ValueError("Provide either combined or named uncertainty limits")
        max_sigma = [
            np.deg2rad(joint_sigma_limit_deg) if joint_sigma_limit_deg is not None else np.inf
        ] * len(robot.active_names)
        if estimate_camera:
            max_sigma += [
                camera_translation_sigma_limit_m
                if camera_translation_sigma_limit_m is not None
                else np.inf
            ] * 3
            max_sigma += [
                np.deg2rad(camera_rotation_sigma_limit_deg)
                if camera_rotation_sigma_limit_deg is not None
                else np.inf
            ] * 3
        if estimate_depth_scale:
            max_sigma += [
                depth_scale_sigma_limit_pct / 100
                if depth_scale_sigma_limit_pct is not None
                else np.inf
            ]
    fit = calibrate_upper(
        robot,
        surfaces,
        frames,
        dataset.camera_nominal,
        sigma=residual_sigma_m,
        matching=matching,
        estimate_camera=estimate_camera,
        estimate_depth_scale=estimate_depth_scale,
        outer_iterations=outer_iterations,
        trim_fraction=trim_fraction,
        max_sigma=max_sigma,
        **({} if solver_options is None else solver_options),
    )
    synthetic = dataset.metadata.get("synthetic") is True
    return {
        "module": "dataset_upper",
        "environment": environment_versions(),
        "synthetic": synthetic,
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "mesh_sources": sources,
        "matching": matching,
        "solver_options": {} if solver_options is None else solver_options,
        "estimate_camera": estimate_camera,
        "estimate_depth_scale": estimate_depth_scale,
        "depth_scale": fit.depth_scale,
        "joint_names": robot.active_names,
        "offsets_rad": fit.offsets.tolist(),
        "offsets_deg": np.rad2deg(fit.offsets).tolist(),
        "camera_transform": fit.chest_transform.tolist(),
        "camera_correction": fit.correction.tolist(),
        "root_frame": dataset.root_frame,
        "camera_frame": dataset.camera_frame,
        "diagnostics": fit.diagnostics.to_dict(),
        "success": fit.success,
        "all_inner_success": fit.all_inner_success,
        "icp_converged": fit.converged,
        "at_bounds": fit.at_bounds,
        "history": fit.history,
        "frames": len(frames),
        "correspondences": fit.correspondences,
        "frame_match_counts": fit.frame_matches,
        "link_match_counts": fit.link_matches,
        "initialization": fit.initialization,
        "quality_passed": fit.quality_passed,
        "quality": fit.quality,
        "residual_profile": fit.residual_profile,
        "point_to_plane_rms_mm": fit.rms_m * 1000,
        "proposal": correction_proposal(
            robot.active_names, fit, robot.root, dataset.camera_frame, synthetic
        ),
        "metadata": dataset.metadata,
        "truth_used_for_estimation": False,
    }
