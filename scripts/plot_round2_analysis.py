#!/usr/bin/env python3
"""Export the second-round development comparisons, including failed estimates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_runs(path):
    return [json.loads(p.read_text()) for p in sorted((path / "runs").glob("*.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/round2"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=180)
        fig.savefig(args.output / f"{name}.svg")
        plt.close(fig)

    scale = read_runs(args.root / "scale_development")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for index, mode in enumerate(["left", "right", "dual"]):
        before = {
            r["seed"]: r["result"]["max_joint_error_deg"]
            for r in scale
            if r["case"] == f"{mode}_scale005_False"
        }
        after = {
            r["seed"]: r["result"]["max_joint_error_deg"]
            for r in scale
            if r["case"] == f"{mode}_scale005_True"
        }
        for seed in before.keys() & after.keys():
            axes[0].plot(
                [index - 0.15, index + 0.15], [before[seed], after[seed]], color="0.75", lw=0.7
            )
        axes[0].scatter(
            np.full(len(before), index - 0.15),
            list(before.values()),
            marker="x",
            c="#c44e52",
            label="Scale disabled" if index == 0 else None,
        )
        axes[0].scatter(
            np.full(len(after), index + 0.15),
            list(after.values()),
            c="#3478a8",
            label="Scale estimated" if index == 0 else None,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(3), ["Left", "Right", "Dual"])
    axes[0].set_ylabel("Maximum joint error [deg]")
    axes[0].set_title("Raw depth bias +0.5% · 4 paired development seeds")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    for row in scale:
        if row["case"].startswith("dual_scale") and row["result"]["estimate_depth_scale"]:
            bias = row["result"]["depth_noise_parameters"]["scale_bias"] * 100
            axes[1].scatter(
                bias, row["result"]["depth_scale_error_ppm"], s=25, c="#3478a8", alpha=0.6
            )
    axes[1].set_xlabel("Injected raw depth bias [%]")
    axes[1].set_ylabel("Absolute corrected scale error [ppm]")
    axes[1].set_title("Dual arm · trusted exact metric geometry")
    axes[1].grid(alpha=0.2)
    save(fig, "scale_development")

    noise = json.loads((args.root / "noise_development_v2" / "summary.json").read_text())["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for offset, model, color in [(-0.18, "isotropic", "#3478a8"), (0.18, "axial", "#e59035")]:
        names = [f"{scenario}_{model}" for scenario in ["depth", "encoder", "quadratic"]]
        coverage = [noise[n]["marginal_95pct_coverage"] for n in names]
        standardized = [noise[n]["rms_standardized_error"] for n in names]
        axes[0].bar(np.arange(3) + offset, coverage, width=0.34, color=color, label=model)
        axes[1].bar(np.arange(3) + offset, standardized, width=0.34, color=color, label=model)
    axes[0].axhline(0.95, c="black", ls="--", lw=1, label="95% reference")
    axes[1].axhline(1.0, c="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Marginal ±1.96σ coverage")
    axes[1].set_ylabel("Standardized joint error RMS")
    for axis in axes:
        axis.set_xticks(
            range(3), ["Constant Z noise", "Shared encoder noise", "Range-squared noise"]
        )
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Conditional uncertainty · all full-rank fits · 4 development seeds")
    save(fig, "noise_development")

    selection_dir = args.root / "selection_complete"
    if selection_dir.exists():
        selection = read_runs(selection_dir)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for axis, mode in zip(axes, ["left", "dual"], strict=True):
            for offset, strategy, color in [
                (-0.22, "first", "#777777"),
                (0.0, "random", "#e59035"),
                (0.22, "information", "#3478a8"),
            ]:
                values = []
                for x, count in enumerate([4, 6]):
                    data = [
                        r["result"]["max_joint_error_deg"]
                        for r in selection
                        if r["case"] == f"{mode}_{count}_{strategy}"
                    ]
                    values.append(np.mean(data))
                    axis.scatter(
                        x + offset + np.linspace(-0.025, 0.025, len(data)),
                        data,
                        color=color,
                        s=14,
                        alpha=0.55,
                    )
                axis.plot(
                    np.arange(2) + offset,
                    values,
                    color=color,
                    marker="_",
                    markersize=14,
                    lw=1,
                    label=strategy,
                )
            axis.set_title(f"{mode.title()} arm · all estimates, no acceptance filtering")
            axis.set_xticks([0, 1], ["4 of 12 frames", "6 of 12 frames"])
            axis.set_ylabel("Maximum joint error [deg]")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle("Offline frame budget · dots = individual runs, lines = means")
        save(fig, "selection_budget")
    print(f"Analysis figures: {args.output}")


if __name__ == "__main__":
    main()
