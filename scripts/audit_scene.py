#!/usr/bin/env python3
"""Measure occlusion caused by visual meshes omitted from calibration surfaces."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from omnicalib.assets import load_asset_config, sample_urdf_surfaces
from omnicalib.geometry import pose, so3_exp
from omnicalib.motion import sample_reaching_poses
from omnicalib.render import Pinhole, look_at, render_depth
from omnicalib.reporting import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/marvin_dual.json"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--view", choices=["front", "onboard"], default="front")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/scene_audit.json"))
    args = parser.parse_args()
    config, urdf, robot = load_asset_config(args.config, args.asset_root)
    modeled_links = config["surface_links"] + config.get("anchor_links", [])
    full_links = [
        link.get("name")
        for link in ET.parse(urdf).getroot().findall("link")
        if link.find("visual/geometry/mesh") is not None
    ]
    full_scene, sources = sample_urdf_surfaces(urdf, full_links, 20, args.seed)
    modeled = [surface for surface in full_scene if surface.link in modeled_links]
    intrinsics = Pinhole(160, 120, 104, 104, 79.5, 59.5)
    if args.view == "onboard":
        camera = look_at([0.18, 0, 1.23], [0.45, 0, 0.85])
        zero = robot.fk(np.zeros(len(robot.active_names)))
        targets = {
            entry["link"]: pose(
                so3_exp([0, 0, np.deg2rad(entry["orientation_z_delta_deg"])])
                @ zero[entry["link"]][:3, :3],
                entry["position"],
            )
            for entry in config["reaching_targets"]
        }
        configurations = sample_reaching_poses(robot, targets, args.frames, seed=args.seed)
    else:
        camera = look_at([1.6, 0, 1.1], [0, 0, 1.0])
        configurations = np.random.default_rng(args.seed).uniform(
            -0.65, 0.65, (args.frames, len(robot.active_names))
        )
    frames = []
    full_modeled_indices = [i for i, name in enumerate(full_links) if name in modeled_links]
    for i, q in enumerate(configurations):
        restricted = render_depth(robot, q, modeled, camera, intrinsics)
        complete = render_depth(robot, q, full_scene, camera, intrinsics)
        a, b = np.isfinite(restricted.depth), np.isfinite(complete.depth)
        replaced = a & (~b | (complete.depth < restricted.depth - 1e-5))
        extra = b & ~np.isin(complete.link_ids, full_modeled_indices)
        frames.append(
            {
                "frame": i,
                "configured_visible_pixels": int(a.sum()),
                "full_scene_visible_pixels": int(b.sum()),
                "configured_pixels_occluded_by_extra_geometry": int(replaced.sum()),
                "occluded_fraction": float(replaced.sum() / max(1, a.sum())),
                "visible_unmodeled_pixels": int(extra.sum()),
            }
        )
    result = {
        "view": args.view,
        "config": str(args.config),
        "seed": args.seed,
        "modeled_links": modeled_links,
        "omitted_visual_links": sorted(set(full_links) - set(modeled_links)),
        "mesh_sources": sources,
        "frames": frames,
        "max_occluded_fraction": max(row["occluded_fraction"] for row in frames),
        "mean_occluded_fraction": float(np.mean([row["occluded_fraction"] for row in frames])),
        "assumptions": [
            "pure geometric visibility audit; no calibration or sensor noise",
            "all omitted links are held at configured fixed joint positions",
            "no environment or collision feasibility model",
        ],
    }
    write_json(args.output, result)
    print(
        f"Omitted links: {len(result['omitted_visual_links'])}; mean occluded fraction={result['mean_occluded_fraction']:.3%}, max={result['max_occluded_fraction']:.3%}"
    )


if __name__ == "__main__":
    main()
