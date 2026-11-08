#!/usr/bin/env python3
"""Export standalone figures from checkpointed benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_runs(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    fingerprint = manifest["source_sha256"]
    records = []
    for path in sorted((directory / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("source_sha256") == fingerprint and "result" in record:
            records.append(record)
    return manifest, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest, runs = read_runs(args.directory)
    output = args.directory / "figures" if args.output is None else args.output
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for record in runs:
        groups.setdefault(record["case"], []).append(record["result"])
    if not groups:
        raise SystemExit("No completed results found")
    all_order = [case["id"] for case in manifest["plan"]["cases"] if case["id"] in groups]
    order = [
        case["id"]
        for case in manifest["plan"]["cases"]
        if case["id"] in groups and case.get("expected_observable", True)
    ]
    if not order:
        order = all_order
    x = np.arange(len(order))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9, len(order) * 0.8), 8), sharex=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10")
    for ax, metric, unit in (
        (axes[0], "max_joint_error_deg", "Maximum joint error [deg]"),
        (axes[1], "heldout_model_point_rms_mm", "Held-out model-point RMS [mm]"),
    ):
        values = [np.array([r[metric] for r in groups[name]]) for name in order]
        for i, data in enumerate(values):
            ax.scatter(
                np.full(len(data), i) + np.linspace(-0.1, 0.1, len(data)),
                data,
                color=colors(i % 10),
                s=28,
                alpha=0.7,
            )
            ax.plot([i - 0.25, i + 0.25], [np.median(data)] * 2, color="black", lw=2)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
        positive = np.concatenate(values)
        if positive.max() > max(positive.min(), 1e-8) * 30:
            ax.set_yscale("log")
    labels = [name.replace("_", "\n") + f"\n(n={len(groups[name])})" for name in order]
    axes[-1].set_xticks(x, labels, fontsize=8)
    axes[0].set_title("Observable synthetic cases: points = all runs, black bars = medians")
    fig.savefig(output / "recovery.png", dpi=180)
    fig.savefig(output / "recovery.svg")
    plt.close(fig)

    order = all_order
    x = np.arange(len(order))
    labels = [name.replace("_", "\n") + f"\n(n={len(groups[name])})" for name in order]
    fig, ax = plt.subplots(figsize=(max(9, len(order) * 0.8), 4.5), constrained_layout=True)
    for i, name in enumerate(order):
        rows = groups[name]
        rates = [
            np.mean([r["success"] for r in rows]),
            np.mean([r["icp_converged"] for r in rows]),
            np.mean([r["diagnostics"]["rank"] == r["diagnostics"]["parameters"] for r in rows]),
            np.mean(
                [
                    all(
                        p["decision"] == "propose"
                        for p in r["proposal"]["parameters"][: len(r["joint_names"])]
                    )
                    for r in rows
                ]
            ),
        ]
        for j, rate in enumerate(rates):
            ax.bar(
                i + (j - 1.5) * 0.18,
                rate,
                width=0.18,
                color=colors(j),
                label=["Inner solver", "ICP stable", "Full rank", "Proposal accepted"][j]
                if i == 0
                else None,
            )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of completed runs")
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    ax.set_title("Convergence and observability are distinct checks")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "acceptance.png", dpi=180)
    fig.savefig(output / "acceptance.svg")
    print(f"Figures: {output}")


if __name__ == "__main__":
    main()
