#!/usr/bin/env python3
"""Render and export a depth/visible-link preview from a configured asset."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omnicalib.assets import load_asset_config, sample_urdf_surfaces
from omnicalib.render import Pinhole, look_at, render_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/marvin_dual.json"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_preview.png"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-posture", action="store_true")
    args = parser.parse_args()
    config, urdf, robot = load_asset_config(args.config, args.asset_root)
    surfaces, _ = sample_urdf_surfaces(
        urdf, config["surface_links"] + config.get("anchor_links", []), 20
    )
    q = (
        np.random.default_rng(args.seed).uniform(-0.4, 0.4, len(robot.active_names))
        if args.random_posture
        else np.zeros(len(robot.active_names))
    )
    camera = look_at([1.5, 0, 1.1], [0, 0, 1.0])
    intrinsics = Pinhole(320, 240, 208, 208, 159.5, 119.5)
    image = render_depth(robot, q, surfaces, camera, intrinsics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot = axes[0].imshow(image.depth, cmap="viridis")
    fig.colorbar(plot, ax=axes[0], label="Optical depth [m]")
    axes[0].set_title("Nearest ray/triangle hits")
    axes[1].imshow(np.ma.masked_less(image.link_ids, 0), cmap="tab20", vmin=0, vmax=19)
    axes[1].set_title("Visible links with self-occlusion")
    for ax in axes:
        ax.set_xlabel("pixel u")
        ax.set_ylabel("pixel v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        depth=image.depth,
        link_ids=image.link_ids,
        camera=camera,
        q=q,
    )
    print(f"Preview: {args.output}")


if __name__ == "__main__":
    main()
