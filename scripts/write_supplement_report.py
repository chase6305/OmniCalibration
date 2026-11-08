#!/usr/bin/env python3
"""Export the complete, separate follow-up studies and descriptive figures."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omnicalib.reporting import write_json

STUDIES = [
    (
        "gauges_complete",
        "单臂与双臂的无参考退化",
        "当前 Marvin URDF 的两个肩部第一关节轴线共线且方向相反。10 个新种子、三个配置均不观察固定机身；左右单臂为 12/13 秩，双臂仍为 19/20，全部保留名义修正。",
    ),
    (
        "selection_complete",
        "帧数预算",
        "每个种子先生成同一批 12 帧，分别选择 4 或 6 帧。信息量策略不读取真值，不保证对所有条件都优于随机策略。",
    ),
    (
        "quadratic_holdout_complete",
        "距离平方噪声的后续独立验证",
        "13001–13020 为新种子。三组均为 50 次外迭代上限、局部步长 0.5、不启用相机邻近重试。下限 0.8 在开发种子上选定后注册。原 400 次评估保持不变。",
    ),
    (
        "point_budget_complete",
        "点数与共享编码器噪声",
        "320×240 图像，实际编码器标准差均为 0.01°。每帧最多 500/2000/8000 点，不同点数共享同一帧级编码器扰动。不同帧仍假定独立。耗时包括渲染、加载、关联与拟合。",
    ),
    (
        "right_standalone_complete",
        "独立右臂模块",
        "已知相机时估计 7 个零偏；相机未知且没有固定参考时保留规范自由度，不将该组任意规范解的真值差当作精度。",
    ),
]


def label(name):
    known = {
        "left_unanchored": "左臂，无固定参考",
        "right_unanchored": "右臂，无固定参考",
        "dual_unanchored": "双臂，无固定参考",
        "isotropic": "各向同性控制",
        "axial_floor025": "光轴，下限 0.25",
        "axial_floor08": "光轴，下限 0.8",
        "right_standalone_known_camera": "相机已知",
        "right_standalone_unknown_camera": "相机未知（退化）",
    }
    if name in known:
        return known[name]
    if name.startswith("points"):
        count = name.split("_")[0].replace("points", "")
        return f"至多 {count} 点，{'建模' if name.endswith('True') else '不建模'}共享编码器误差"
    mode, count, strategy = name.split("_", 2)
    return (
        f"{'左臂' if mode == 'left' else '双臂'}，{count}/12，"
        + {"first": "前几帧", "random": "随机", "information": "信息量"}[strategy]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/round2"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    portable, loaded = {}, {}
    lines = [
        "# 补充验证",
        "",
        "这些对照与原 400 次评估分别统计；全部返回估计均纳入误差，不按接受或误差筛选。覆盖率为关节边际 ±1.96σ 的描述性统计，各关节相关，并非独立试验。它条件于关联、噪声与精确 CAD，不保证实机精度。",
        "",
        "生成时间：" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "。",
        "",
    ]
    for directory, title, note in STUDIES:
        base = args.root / directory
        summary = json.loads((base / "summary.json").read_text())
        assert summary["completed_runs"] == summary["planned_runs"], f"Incomplete {directory}"
        records = [json.loads(p.read_text()) for p in sorted((base / "runs").glob("*.json"))]
        assert all("result" in row for row in records), (
            f"Exceptional {directory}; revise the report explicitly"
        )
        loaded[directory] = records
        portable[directory] = {
            k: summary[k]
            for k in ["source_sha256", "plan", "seeds", "completed_runs", "planned_runs", "groups"]
        }
        lines += [
            f"## {title}",
            "",
            note,
            "",
            f"完成 {summary['completed_runs']} 条，数值源 `{summary['source_sha256']}`。",
            "",
            "| 条件 | 接受/返回 | 最大关节误差均值 (°) | P95 (°) | 最坏值 (°) | 边际覆盖率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for case in summary["plan"]["cases"]:
            name = case["id"]
            g = summary["groups"][name]
            if case.get("expected_observable", True):
                e = g["max_joint_error_deg"]
                values = f"{e['mean']:.5f} | {e['p95']:.5f} | {e['max']:.5f} | {g.get('marginal_95pct_coverage', 0):.1%}"
            else:
                values = "— | — | — | —"
            lines.append(f"| {label(name)} | {g['accepted']}/{g['completed']} | {values} |")
        lines += [""]
        if directory == "gauges_complete":
            lines += [
                "![双肩公共规范自由度](figures/dual_gauge.png)",
                "",
                "图中的点变化和雅可比来自命名几何样本的严格 FK 恒等式，展示不可区分性，不是传感器精度。带噪声、可见深度的 30 次对照另外验证上述秩与保留行为。",
                "",
            ]
        elif directory == "selection_complete":
            lines += ["![帧数对照](figures/selection_budget.png)", ""]
        elif directory == "quadratic_holdout_complete":
            lines += [
                "![距离平方噪声后续验证](figures/quadratic_holdout.png)",
                "",
                "原下限 0.25 的一个估计超过 0.1° 参考线；该结果没有删除。候选下限 0.8 的经验覆盖率仍未精确达到名义 95%，不能视为完整置信保证。",
                "",
            ]
        elif directory == "point_budget_complete":
            lines += [
                "![点数对照](figures/point_budget.png)",
                "",
                "| 条件 | 平均实际输入点/帧 | 平均总耗时 (s) |",
                "|---|---:|---:|",
            ]
            for case in summary["plan"]["cases"]:
                rows = [r["result"] for r in records if r["case"] == case["id"]]
                points = np.mean(
                    [np.mean([f["used_points"] for f in r["visibility"]]) for r in rows]
                )
                lines.append(
                    f"| {label(case['id'])} | {points:.1f} | {np.mean([r['elapsed_s'] for r in rows]):.2f} |"
                )
            lines += [
                "",
                "增加点数减少独立深度误差，却不能把同一帧的编码器读数变成许多独立测量。当前模型也没有包含跨帧时间漂移或所有几何失配。",
                "",
            ]
        rows = []
        for record in records:
            r = record["result"]
            rows.append(
                {
                    "case": record["case"],
                    "seed": record["seed"],
                    "expected_observable": record.get("expected_observable", True),
                    "accepted": r["proposal"]["block_supported"],
                    "max_joint_error_deg": r["max_joint_error_deg"],
                    "rms_joint_error_deg": r["rms_joint_error_deg"],
                    "rank": r["diagnostics"]["rank"],
                    "parameters": r["diagnostics"]["parameters"],
                    "iterations": len(r["history"]),
                    "elapsed_s": r["elapsed_s"],
                }
            )
        output = args.docs / "results" / f"{directory}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines += [
        "## 记录与边界",
        "",
        "完整计划、种子与分组指标见 [补充 JSON](results/supplementary_summary.json)，逐案例 CSV 在同目录。审计 JSON 保存在每个结果目录。大初值超时与探索性续算单例见 [第二轮日志](ROUND2_LOG.md)；它们没有替换主评估中的原记录。",
        "",
        "[主验证](VALIDATION.md) · [全部实验入口](EXPERIMENTS.md) · [方法边界](REPRODUCTION.md)",
        "",
    ]
    figures = args.docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    quad = loaded["quadratic_holdout_complete"]
    names = ["isotropic", "axial_floor025", "axial_floor08"]
    colors = ["#777777", "#e59035", "#3478a8"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    summary = portable["quadratic_holdout_complete"]["groups"]
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        rows = [r["result"] for r in quad if r["case"] == name]
        for j, r in enumerate(rows):
            axes[0].scatter(
                i + (j - (len(rows) - 1) / 2) * 0.008,
                r["max_joint_error_deg"],
                s=22,
                c=color,
                marker="o" if r["proposal"]["block_supported"] else "x",
                alpha=0.75,
            )
        axes[0].plot(
            [i - 0.2, i + 0.2], [summary[name]["max_joint_error_deg"]["mean"]] * 2, c="black", lw=2
        )
        axes[1].bar(i, summary[name]["marginal_95pct_coverage"], color=color, width=0.6)
    axes[0].axhline(0.1, c="0.5", ls="--", lw=0.8)
    axes[0].set_ylabel("Maximum joint error [deg]")
    axes[0].set_title("All estimates; x = retained, black = mean")
    axes[1].set_ylabel("Marginal ±1.96σ coverage")
    axes[1].axhline(0.95, c="black", ls="--", lw=1)
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xticks(range(3), ["Isotropic", "Axial floor .25", "Axial floor .8"])
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("New range-squared-noise holdout · 20 paired seeds")
    for suffix in ["png", "svg"]:
        fig.savefig(figures / f"quadratic_holdout.{suffix}", dpi=180)
    plt.close(fig)
    point = loaded["point_budget_complete"]
    summary = portable["point_budget_complete"]["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for modeled, color in [(False, "#e59035"), (True, "#3478a8")]:
        counts = [500, 2000, 8000]
        names = [f"points{n}_shared_encoder{modeled}" for n in counts]
        means = [summary[n]["max_joint_error_deg"]["mean"] for n in names]
        coverage = [summary[n]["marginal_95pct_coverage"] for n in names]
        label_text = "Shared encoder covariance" if modeled else "Independent point covariance"
        axes[0].plot(counts, means, c=color, marker="o", label=label_text)
        axes[1].plot(counts, coverage, c=color, marker="o", label=label_text)
        for n, name in zip(counts, names, strict=True):
            data = [r["result"]["max_joint_error_deg"] for r in point if r["case"] == name]
            axes[0].scatter(
                np.array([n] * len(data)) * np.linspace(0.96, 1.04, len(data)),
                data,
                c=color,
                s=12,
                alpha=0.45,
            )
    axes[0].set_ylabel("Maximum joint error [deg]")
    axes[1].set_ylabel("Marginal ±1.96σ coverage")
    axes[1].axhline(0.95, c="black", ls="--", lw=1)
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks([500, 2000, 8000], ["500", "2000", "8000"])
        axis.set_xlabel("Per-frame point ceiling")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed frame-shared encoder noise · 8 paired seeds · 320×240 rendering")
    for suffix in ["png", "svg"]:
        fig.savefig(figures / f"point_budget.{suffix}", dpi=180)
    plt.close(fig)
    write_json(args.docs / "results" / "supplementary_summary.json", portable)
    (args.docs / "SUPPLEMENTARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Supplementary reports: {args.docs}")


if __name__ == "__main__":
    main()
