"""Run a deterministic synthetic validation: python -m omnicalib demo."""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from .profiles import solver_profile
from .reporting import write_json
from .synthetic import run_head, run_lower, run_upper


def _solver_options(args):
    """Apply a named recipe, then explicit CLI overrides for both entry points."""
    return {
        **solver_profile(args.profile),
        "loss": args.loss,
        "encoder_sigma_rad": float(np.deg2rad(args.assumed_encoder_sigma_deg)),
        **({"step_relaxation": args.step_relaxation} if args.step_relaxation is not None else {}),
        "depth_scale_log_bound": args.depth_scale_log_bound,
        "point_noise_model": args.point_noise_model,
        "axial_floor_fraction": args.axial_floor_fraction,
        "axial_range_power": args.axial_range_power,
        "max_rejection_fraction": args.max_rejection_fraction,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="synthetic numerical validation, not paper metrics")
    demo.add_argument("--module", choices=["all", "upper", "lower", "head"], default="all")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument(
        "--noisy",
        action="store_true",
        help="0.2mm depth / 0.01deg encoders / 0.0002 normalized image noise",
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/synthetic.json"))
    asset = sub.add_parser("asset-demo", help="synthetic recovery with real URDF/visual meshes")
    asset.add_argument("--config", type=Path, required=True)
    asset.add_argument("--asset-root", type=Path)
    asset.add_argument("--seed", type=int, default=7)
    asset.add_argument("--noise-m", type=float, default=0.0)
    asset.add_argument("--frames", type=int, default=10)
    asset.add_argument("--samples-per-link", type=int, default=100)
    asset.add_argument("--matching", choices=["sampled", "triangle"], default="sampled")
    asset.add_argument("--independent-sampling", action="store_true")
    asset.add_argument("--observation-samples", type=int)
    asset.add_argument("--render-depth", action="store_true")
    asset.add_argument("--width", type=int, default=160)
    asset.add_argument("--height", type=int, default=120)
    asset.add_argument("--max-points", type=int, default=2000)
    asset.add_argument("--dropout", type=float, default=0.0)
    asset.add_argument("--outliers", type=float, default=0.0)
    asset.add_argument("--quantization-m", type=float, default=0.0)
    asset.add_argument(
        "--depth-scale-bias",
        type=float,
        default=0.0,
        help="synthetic raw depth multiplier minus one",
    )
    asset.add_argument("--trim-fraction", type=float, default=0.0)
    asset.add_argument("--iterations", type=int, default=12)
    asset.add_argument("--save-dataset", type=Path)
    asset.add_argument("--offset-amplitude-deg", type=float, default=0.5)
    asset.add_argument("--residual-sigma-m", type=float, default=0.001)
    asset.add_argument("--encoder-noise-deg", type=float, default=0.0)
    asset.add_argument("--assumed-encoder-sigma-deg", type=float, default=0.0)
    asset.add_argument("--profile", choices=["basic", "robust"], default="basic")
    asset.add_argument("--loss", choices=["linear", "huber", "soft_l1", "cauchy"], default="huber")
    asset.add_argument("--camera-position", type=float, nargs=3)
    asset.add_argument("--camera-target", type=float, nargs=3)
    asset.add_argument("--observation-mesh-scale", type=float, default=1.0)
    asset.add_argument("--origin-translation-sigma-m", type=float, default=0.0)
    asset.add_argument("--posture-mode", choices=["random", "reaching"], default="random")
    asset.add_argument("--select-frames", type=int)
    asset.add_argument(
        "--selection-strategy", choices=["information", "first", "random"], default="information"
    )
    asset.add_argument(
        "--no-anchor",
        action="store_true",
        help="remove static base surface to test gauge ambiguity",
    )
    asset.add_argument(
        "--fixed-camera",
        action="store_true",
        help="assume known camera extrinsic",
    )
    asset.add_argument("--output", type=Path, default=Path("outputs/asset_synthetic.json"))
    calibration = sub.add_parser("calibrate", help="fit a named-joint, masked-point NPZ dataset")
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--asset-root", type=Path)
    calibration.add_argument("--matching", choices=["sampled", "triangle"], default="triangle")
    calibration.add_argument("--samples-per-link", type=int, default=300)
    calibration.add_argument("--fixed-camera", action="store_true")
    calibration.add_argument("--no-anchor", action="store_true")
    calibration.add_argument("--residual-sigma-m", type=float, default=0.001)
    calibration.add_argument("--iterations", type=int, default=20)
    calibration.add_argument("--trim-fraction", type=float, default=0.0)
    calibration.add_argument("--profile", choices=["basic", "robust"], default="basic")
    calibration.add_argument(
        "--loss", choices=["linear", "huber", "soft_l1", "cauchy"], default="huber"
    )
    calibration.add_argument("--assumed-encoder-sigma-deg", type=float, default=0.0)
    calibration.add_argument("--max-joint-sigma-deg", type=float)
    calibration.add_argument("--max-camera-translation-sigma-mm", type=float)
    calibration.add_argument("--max-camera-rotation-sigma-deg", type=float)
    calibration.add_argument("--max-depth-scale-sigma-pct", type=float)
    calibration.add_argument("--output", type=Path, default=Path("outputs/calibration.json"))
    for command in (asset, calibration):
        command.add_argument(
            "--point-noise-model", choices=["isotropic", "axial"], default="isotropic"
        )
        command.add_argument("--axial-floor-fraction", type=float, default=0.25)
        command.add_argument("--axial-range-power", type=float, default=0.0)
        command.add_argument("--max-rejection-fraction", type=float, default=0.2)
        command.add_argument(
            "--estimate-depth-scale",
            action="store_true",
            help="fit a positive depth multiplier against metric CAD geometry",
        )
        command.add_argument("--depth-scale-log-bound", type=float, default=0.05)
        command.add_argument(
            "--step-relaxation", type=float, default=None, help="local ICP step fraction in (0, 1]"
        )
    staged = sub.add_parser("staged-demo", help="arm-first then wrist-camera synthetic validation")
    staged.add_argument("--config", type=Path, required=True)
    staged.add_argument("--asset-root", type=Path)
    staged.add_argument("--seed", type=int, default=7)
    staged.add_argument("--frames", type=int, default=10)
    staged.add_argument("--noise-m", type=float, default=0.0002)
    staged.add_argument("--wrist-translation-noise-m", type=float, default=0.0003)
    staged.add_argument("--wrist-rotation-noise-deg", type=float, default=0.01)
    staged.add_argument("--moving-root", action="store_true")
    staged.add_argument("--output", type=Path, default=Path("outputs/staged.json"))
    return parser


def _run_staged(args):
    from .staged import run_staged_demo

    started = time.perf_counter()
    result = run_staged_demo(
        args.config,
        args.output.with_suffix(".npz"),
        seed=args.seed,
        asset_root=args.asset_root,
        frames_count=args.frames,
        noise_m=args.noise_m,
        matching="triangle",
        rendered=True,
        outer_iterations=25,
        pose_translation_noise_m=args.wrist_translation_noise_m,
        pose_rotation_noise_deg=args.wrist_rotation_noise_deg,
        moving_root=args.moving_root,
        solver_options={"uncertainty_step_fraction": 0.05},
    )
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.output, result)
    print(f"Arm max error: {result['arm']['max_joint_error_deg']:.6g}deg")
    for wrist in result["wrists"]:
        print(
            f"{wrist['mount_link']}: {wrist['translation_error_mm']:.4g}mm, {wrist['rotation_error_deg']:.4g}deg, rank={wrist['diagnostics']['rank']}/6"
        )
    print(f"Report: {args.output}")
    return


def _run_calibration(args):
    from .assets import calibrate_dataset

    started = time.perf_counter()
    result = calibrate_dataset(
        args.config,
        args.input,
        asset_root=args.asset_root,
        matching=args.matching,
        samples_per_link=args.samples_per_link,
        estimate_camera=not args.fixed_camera,
        estimate_depth_scale=args.estimate_depth_scale,
        include_anchor=not args.no_anchor,
        residual_sigma_m=args.residual_sigma_m,
        outer_iterations=args.iterations,
        trim_fraction=args.trim_fraction,
        solver_options=_solver_options(args),
        joint_sigma_limit_deg=args.max_joint_sigma_deg,
        camera_translation_sigma_limit_m=None
        if args.max_camera_translation_sigma_mm is None
        else args.max_camera_translation_sigma_mm / 1000,
        camera_rotation_sigma_limit_deg=args.max_camera_rotation_sigma_deg,
        depth_scale_sigma_limit_pct=args.max_depth_scale_sigma_pct,
    )
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.output, result)
    diag = result["diagnostics"]
    print(
        f"success={result['success']}, converged={result['icp_converged']}, rank={diag['rank']}/{diag['parameters']}"
    )
    print(f"Report: {args.output}")
    print(
        f"proposal_supported={result['proposal']['block_supported']}, reasons={result['proposal']['block_reasons']}"
    )
    if args.estimate_depth_scale:
        print(f"Depth correction multiplier: {result['depth_scale']:.9g}")
    if not result["success"]:
        raise SystemExit(1)
    return


def _run_asset(args):
    from .assets import run_asset_demo
    from .render import DepthNoise

    started = time.perf_counter()
    print(f"Running asset experiment: {args.config}", flush=True)
    result = run_asset_demo(
        config_path=args.config,
        asset_root=args.asset_root,
        seed=args.seed,
        noise_m=args.noise_m,
        frames_count=args.frames,
        samples_per_link=args.samples_per_link,
        include_anchor=not args.no_anchor,
        estimate_camera=not args.fixed_camera,
        matching=args.matching,
        independent_sampling=args.independent_sampling,
        observation_samples=args.observation_samples,
        rendered=args.render_depth,
        image_size=(args.width, args.height),
        max_points=args.max_points,
        trim_fraction=args.trim_fraction,
        outer_iterations=args.iterations,
        dataset_path=args.save_dataset,
        offset_amplitude_deg=args.offset_amplitude_deg,
        residual_sigma_m=args.residual_sigma_m,
        encoder_noise_deg=args.encoder_noise_deg,
        selection_count=args.select_frames,
        selection_strategy=args.selection_strategy,
        solver_options=_solver_options(args),
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        observation_mesh_scale=args.observation_mesh_scale,
        origin_translation_sigma_m=args.origin_translation_sigma_m,
        posture_mode=args.posture_mode,
        estimate_depth_scale=args.estimate_depth_scale,
        depth_noise=DepthNoise(
            args.noise_m,
            args.dropout,
            args.outliers,
            quantization_m=args.quantization_m,
            scale_bias=args.depth_scale_bias,
        ),
    )
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.output, result)
    diag = result["diagnostics"]
    print(
        f"success={result['success']}, rank={diag['rank']}/{diag['parameters']}, max_joint_error_deg={result['max_joint_error_deg']:.6g}"
    )
    print(f"Report: {args.output}")
    print(
        f"proposal_supported={result['proposal']['block_supported']}, reasons={result['proposal']['block_reasons']}"
    )
    if args.estimate_depth_scale:
        print(f"Depth correction multiplier: {result['depth_scale']:.9g}")
    if not result["success"]:
        raise SystemExit(1)
    return


def _run_demo(args):
    report = {
        "paper": "arXiv:2609.19582v2",
        "validation": "synthetic baseline only",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "results": [],
    }
    runners = {"upper": run_upper, "lower": run_lower, "head": run_head}
    for name, runner in runners.items():
        if args.module not in ("all", name):
            continue
        print(f"Running {name} synthetic experiment...", flush=True)
        started = time.perf_counter()
        noise = {"upper": 0.0002, "lower": 0.01, "head": 0.0002}[name] if args.noisy else 0.0
        result = runner(args.seed, noise)
        result["elapsed_s"] = time.perf_counter() - started
        report["results"].append(result)
        metric = {
            "upper": "max_joint_error_deg",
            "lower": "rms_joint_error_deg",
            "head": "mean_rotation_error_deg",
        }[name]
        print(f"  success={result['success']}, {metric}={result[metric]:.6g}", flush=True)
    write_json(args.output, report)
    print(f"Report: {args.output}")
    if not all(r["success"] for r in report["results"]):
        raise SystemExit(1)


def _main():
    args = _build_parser().parse_args()
    handlers = {
        "demo": _run_demo,
        "asset-demo": _run_asset,
        "calibrate": _run_calibration,
        "staged-demo": _run_staged,
    }
    handlers[args.command](args)


def main():
    try:
        _main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
