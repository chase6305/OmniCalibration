#!/usr/bin/env python3
"""Visualize a frozen observation-only NPZ fit against visible CAD surfaces."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from omnicalib.assets import load_asset_config, sample_urdf_surfaces
from omnicalib.dataset import load_dataset
from omnicalib.render import Pinhole, render_depth
from omnicalib.reporting import write_json


def point_depth(points, intrinsics):
    """Project recorded points without inventing data in masked/empty pixels."""
    points = points[points[:, 2] > 0]
    u = np.rint(intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx).astype(int)
    v = np.rint(intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy).astype(int)
    valid = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    depth = np.full((intrinsics.height, intrinsics.width), np.inf)
    np.minimum.at(depth, (v[valid], u[valid]), points[valid, 2])
    depth[~np.isfinite(depth)] = np.nan
    return depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration_preview.png"))
    parser.add_argument("--animate", action="store_true", help="also export a GIF of all frames")
    parser.add_argument("--residual-limit-mm", type=float, default=3.0)
    args = parser.parse_args()
    dataset = load_dataset(args.input, include_truth=False)
    result = json.loads(args.report.read_text())
    config, urdf, robot = load_asset_config(args.config, args.asset_root)
    if dataset.root_frame != robot.root or result["root_frame"] != robot.root:
        raise ValueError("Dataset, fit, and robot must use the same root frame")
    if set(dataset.joint_names) != set(robot.active_names) or set(result["joint_names"]) != set(
        robot.active_names
    ):
        raise ValueError("Dataset and report must name the configured joints")
    intrinsics = Pinhole(**dataset.metadata["intrinsics"])
    order = [dataset.joint_names.index(name) for name in robot.active_names]
    offsets = np.asarray(result["offsets_rad"])[
        [result["joint_names"].index(name) for name in robot.active_names]
    ]
    surfaces, _ = sample_urdf_surfaces(
        urdf, config["surface_links"] + config.get("anchor_links", []), 20
    )
    frames, summaries = [], []
    count = len(dataset.frames) if args.animate else 1
    for index, frame in enumerate(dataset.frames[:count]):
        frame_robot = robot.robot.select(
            robot.active_names, robot.fixed_positions | (frame.fixed_positions or {})
        )
        observed = point_depth(frame.points_camera, intrinsics)
        q = frame.q[order]
        nominal = render_depth(frame_robot, q, surfaces, dataset.camera_nominal, intrinsics).depth
        candidate = render_depth(
            frame_robot, q + offsets, surfaces, np.asarray(result["camera_transform"]), intrinsics
        ).depth
        before = (observed - nominal) * 1000
        after = (observed * result.get("depth_scale", 1.0) - candidate) * 1000
        # Both panels use the same observed pixels and their own first-hit CAD.
        row = {"frame": index, "observed_pixels": int(np.isfinite(observed).sum())}
        for name, residual in (("nominal", before), ("candidate", after)):
            finite = residual[np.isfinite(residual)]
            row[name] = {
                "overlap_pixels": len(finite),
                "median_abs_axial_mm": float(np.median(np.abs(finite))) if len(finite) else None,
            }
        frames.append((observed, before, after))
        summaries.append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    depth_plot = axes[0].imshow(frames[0][0], cmap="viridis", vmin=intrinsics.near, vmax=2.5)
    fig.colorbar(depth_plot, ax=axes[0], label="Raw optical depth [m]")
    residual_plots = []
    for axis, data, title in zip(
        axes[1:3],
        frames[0][1:],
        ["Nominal CAD: depth error", "Candidate CAD: corrected depth error"],
        strict=True,
    ):
        plot = axis.imshow(
            data, cmap="coolwarm", vmin=-args.residual_limit_mm, vmax=args.residual_limit_mm
        )
        fig.colorbar(plot, ax=axis, label="Observed minus CAD axial Z [mm]")
        axis.set_title(title)
        residual_plots.append(plot)
    names = [name.replace("_joint", "") for name in robot.active_names]
    axes[3].barh(names, np.rad2deg(offsets), color="#3478a8")
    axes[3].axvline(0, color="black", linewidth=0.6)
    axes[3].set_xlabel("Candidate joint zero correction [deg]")
    accepted = result.get("proposal", {}).get("block_supported", False)
    axes[3].set_title(f"Numerical proposal: {'supported' if accepted else 'retained / rejected'}")
    axes[3].tick_params(axis="y", labelsize=7)
    for axis in axes[:3]:
        axis.set_xlabel("pixel u")
        axis.set_ylabel("pixel v")
    title = fig.suptitle("")

    def update(index):
        observed, before, after = frames[index]
        depth_plot.set_data(observed)
        residual_plots[0].set_data(before)
        residual_plots[1].set_data(after)
        axes[0].set_title(f"Recorded robot points, frame {index + 1}/{count}")
        title.set_text(
            "Visible mesh comparison · synthetic data"
            if dataset.metadata.get("synthetic")
            else "Visible mesh comparison · recorded data"
        )
        return depth_plot, *residual_plots, title

    args.output.parent.mkdir(parents=True, exist_ok=True)
    update(0)
    fig.savefig(args.output, dpi=160)
    if args.animate:
        animation = FuncAnimation(fig, update, frames=count, interval=700, blit=False)
        animation.save(args.output.with_suffix(".gif"), writer=PillowWriter(fps=1.4), dpi=100)
    plt.close(fig)
    write_json(
        args.output.with_suffix(".json"),
        {
            "truth_used": False,
            "frames": summaries,
            "note": "Descriptive axial depth differences at overlapping pixels; silhouette changes and occlusion are included. This is not the ICP point/plane residual or a hardware accuracy measure.",
        },
    )
    print(f"Preview: {args.output}")


if __name__ == "__main__":
    main()
