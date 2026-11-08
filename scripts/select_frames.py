#!/usr/bin/env python3
"""Select observed frames for offline calibration without reading stored truth."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from omnicalib.assets import load_asset_config, sample_urdf_surfaces
from omnicalib.dataset import load_dataset, save_dataset
from omnicalib.excitation import select_frames
from omnicalib.reporting import json_safe, write_json
from omnicalib.upper import DepthFrame


def select_dataset(
    config_path,
    input_path,
    output_path,
    count,
    *,
    asset_root=None,
    strategy="information",
    seed=7,
    estimate_camera=True,
    estimate_depth_scale=False,
    sigma=0.001,
    samples_per_link=100,
):
    config, urdf, robot = load_asset_config(config_path, asset_root)
    dataset = load_dataset(input_path, include_truth=False)
    if dataset.root_frame != robot.root or set(dataset.joint_names) != set(robot.active_names):
        raise ValueError("Dataset root and joint names must match the configured model")
    if not 2 <= count <= len(dataset.frames):
        raise ValueError("Select between two and all recorded frames")
    order = [dataset.joint_names.index(name) for name in robot.active_names]
    frames = [DepthFrame(f.q[order], f.points_camera, f.fixed_positions) for f in dataset.frames]
    report = {
        "strategy": strategy,
        "candidate_frames": len(frames),
        "truth_used": False,
        "source_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
    }
    if strategy == "information":
        surfaces, _ = sample_urdf_surfaces(
            urdf, config["surface_links"] + config.get("anchor_links", []), samples_per_link
        )
        scale = np.full(len(robot.active_names), np.deg2rad(1.0))
        if estimate_camera:
            scale = np.r_[scale, [0.01] * 3, [np.deg2rad(1.0)] * 3]
        if estimate_depth_scale:
            scale = np.r_[scale, 0.005]
        selection = select_frames(
            robot,
            surfaces,
            frames,
            dataset.camera_nominal,
            count,
            estimate_camera=estimate_camera,
            estimate_depth_scale=estimate_depth_scale,
            sigma=sigma,
            matching="triangle",
            parameter_scale=scale,
            seed=seed,
        )
        indices = selection.indices
        report.update(
            gains=selection.gains,
            diagnostics=selection.diagnostics.to_dict(),
            parameter_joint_order=robot.active_names,
            candidate_points=selection.candidate_points,
            assumptions="Nominal geometric Jacobians and independent isotropic plane noise; ranking regularizer is not a calibration prior",
        )
    elif strategy == "first":
        indices = list(range(count))
    elif strategy == "random":
        indices = sorted(
            np.random.default_rng(seed).choice(len(frames), count, replace=False).tolist()
        )
    else:
        raise ValueError("Choose information, first or random selection")
    report["indices"] = indices
    report = json_safe(report)
    selected = replace(
        dataset,
        frames=[dataset.frames[i] for i in indices],
        truth=None,
        metadata={**dataset.metadata, "frame_selection": report},
    )
    save_dataset(output_path, selected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--strategy", choices=["information", "first", "random"], default="information"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument("--estimate-depth-scale", action="store_true")
    parser.add_argument("--residual-sigma-m", type=float, default=0.001)
    args = parser.parse_args()
    result = select_dataset(
        args.config,
        args.input,
        args.output,
        args.count,
        asset_root=args.asset_root,
        strategy=args.strategy,
        seed=args.seed,
        estimate_camera=not args.fixed_camera,
        estimate_depth_scale=args.estimate_depth_scale,
        sigma=args.residual_sigma_m,
    )
    write_json(args.output.with_suffix(".selection.json"), result)
    print(f"Selected frames {result['indices']}; no stored truth read or copied; {args.output}")


if __name__ == "__main__":
    main()
