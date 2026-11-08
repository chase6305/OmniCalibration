#!/usr/bin/env python3
"""Generate the current validation record from frozen, unfiltered experiments."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LABELS = {
    "left_visible": "左臂可见深度",
    "right_visible": "右臂可见深度",
    "dual_visible": "双臂可见深度",
    "dual_dropout": "双臂，30% 丢点",
    "dual_outliers": "双臂，5% 离群点",
    "dual_quantized": "双臂，1 mm 量化",
    "dual_encoder_noise": "双臂，共享编码器噪声",
    "dual_large_offsets": "双臂，±2° 零偏",
    "standalone_known_camera": "独立单臂，相机已知",
    "left_unanchored": "单臂，无固定参考（退化）",
    "dual_onboard_reaching": "双臂，近机身视角＋伸手动作",
    "left_depth_scale": "左臂，0.5% 深度尺度＋联合估计",
    "right_depth_scale": "右臂，0.5% 深度尺度＋联合估计",
    "dual_depth_scale": "双臂，0.5% 深度尺度＋联合估计",
    "dual_scale_encoder": "双臂，尺度＋编码器噪声",
    "dual_scale_outliers": "双臂，尺度＋离群点",
    "dual_axial_depth": "双臂，光轴噪声权重",
    "dual_axial_encoder": "双臂，光轴权重＋编码器噪声",
    "dual_quadratic_isotropic": "距离平方噪声，各向同性权重",
    "dual_quadratic_axial": "距离平方噪声，光轴/距离权重",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/VALIDATION.md"))
    parser.add_argument("--tests", type=int)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    snapshot = args.directory / "source_snapshots" / summary["source_sha256"]
    match = re.search(r'^version = "([^"]+)"', (snapshot / "pyproject.toml").read_text(), re.M)
    version = match[1] if match else "unknown"
    records = [json.loads(p.read_text()) for p in sorted((args.directory / "runs").glob("*.json"))]
    complete = [row for row in records if "result" in row]
    groups = summary["groups"]
    observable_cases = {
        c["id"] for c in summary["plan"]["cases"] if c.get("expected_observable", True)
    }
    observable = [r for r in complete if r["case"] in observable_cases]
    accepted = sum(r["result"]["proposal"]["block_supported"] for r in observable)
    reference_exceedances = [r for r in observable if r["result"]["max_joint_error_deg"] > 0.1]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 验证记录",
        "",
        f"数值实现 **v{version}**；报告生成于 {now}。这是实际 Marvin URDF/网格上的合成验证，不是实机标定精度。",
        "",
        "第二轮从 2026-11-07 05:39:48 UTC 开始；最终数值实现于 06:13:09 UTC 冻结。第一轮记录与 220 次评估保存在 [v0.2.3 历史报告](history/VALIDATION_v023.md)。",
        "",
        f"自动化测试 **{args.tests} 项通过**。"
        if args.tests is not None
        else "自动化检查结果见复现日志。",
        "",
        "## 全新种子评估",
        "",
        f"完成 **{summary['completed_runs']}/{summary['planned_runs']}** 个唯一试验：{len(summary['plan']['cases'])} 类场景，{len(summary['seeds'])} 个未用于本轮调参的种子。每个进程冻结数值模块、配置与源码哈希；没有依据新种子真值改动该冻结版本。",
        "",
        f"预先声明可观测的 {len(observable)} 次已完成估计中，{accepted} 次通过数值接受条件；{len(reference_exceedances)} 次最大关节误差超过 0.1° 参考线。该参考线只是合成评估指标，不是论文指标或实机接受标准。",
        "",
        "默认每次 8 帧、160×120 针孔图像、每帧至多 2000 点；实际光轴噪声标准差 0.2 mm，零偏注入 ±0.5°。编码器噪声案例为 0.01°，尺度案例为 +0.5%，距离平方案例的光轴标准差为 0.2 mm×z²（z 的数值以米计）。精确配置以 JSON 为准。",
        "",
        "下表包含全部已完成结果，未按恢复误差或接受结果筛选。P95 为这些种子的经验分位数。退化案例的参数不唯一，不将任意规范解与真值之差当作精度。",
        "",
        "| 场景 | 完成 | 接受 | 最大关节误差均值 (°) | P95 (°) | 最坏值 (°) | 留出模型点 RMS 均值 (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["plan"]["cases"]:
        name = case["id"]
        if name not in groups:
            continue
        g = groups[name]
        if case.get("expected_observable", True) and "max_joint_error_deg" in g:
            e = g["max_joint_error_deg"]
            values = f"{e['mean']:.5f} | {e['p95']:.5f} | {e['max']:.5f} | {g['heldout_model_point_rms_mm']['mean']:.5f}"
        else:
            values = "— | — | — | —"
        lines.append(
            f"| {LABELS.get(name, name)} | {g['completed']}/{g['runs']} | {g['accepted']} | {values} |"
        )
    lines += [
        "",
        "留出模型点在另外十组姿态上做无观测噪声的 FK 对比，检查零偏误差传播；仍使用声明的合成几何，不能代替独立实机测量。",
        "",
        "[紧凑 JSON](results/validation_summary.json) · [逐案例 CSV](results/metrics.csv)",
        "",
        "![恢复误差](figures/recovery.png)",
        "",
        "![数值接受](figures/acceptance.png)",
        "",
        "## 尺度与不确定度",
        "",
        "`depth_scale` 修正原始相机点，理论参考为 1/1.005；下表为修正后相对尺度的绝对误差。CAD 的整体米制尺度与传感器尺度存在不可区分性，不能据此诊断真实传感器故障。",
        "",
        "| 场景 | 尺度误差均值 (ppm) | 最坏值 (ppm) |",
        "|---|---:|---:|",
    ]
    for case in summary["plan"]["cases"]:
        if not case.get("settings", {}).get("estimate_depth_scale"):
            continue
        g = groups.get(case["id"], {}).get("depth_scale_error_ppm")
        if g:
            lines.append(
                f"| {LABELS.get(case['id'], case['id'])} | {g['mean']:.3f} | {g['max']:.3f} |"
            )
    lines += [
        "",
        "下表统计关节边际 ±1.96σ 区间覆盖率。各关节相关，样本不是独立伯努利试验；协方差条件于最终关联、噪声模型和固定 CAD。它不包含对应点选择、时间失配或全部系统误差。",
        "",
        "| 场景 | 边际覆盖率 | 标准化误差 RMS |",
        "|---|---:|---:|",
    ]
    for name in [
        "dual_visible",
        "dual_encoder_noise",
        "dual_axial_depth",
        "dual_axial_encoder",
        "dual_quadratic_isotropic",
        "dual_quadratic_axial",
    ]:
        g = groups.get(name, {})
        if "marginal_95pct_coverage" in g:
            lines.append(
                f"| {LABELS[name]} | {g['marginal_95pct_coverage']:.1%} | {g['rms_standardized_error']:.3f} |"
            )
    failures = [
        r
        for r in complete
        if not r["result"]["proposal"]["block_supported"] and r["case"] in observable_cases
    ]
    lines += [
        "",
        "## 未接受结果与局限",
        "",
        f"预期可观测场景有 {len(failures)} 次未通过接受条件，另有 {len(records) - len(complete)} 次异常。退化单臂对照应给出 12/13 秩并保留名义修正。",
    ]
    if failures:
        lines.append("")
    for row in failures[:20]:
        r = row["result"]
        lines.append(
            f"- `{row['case']}` / seed {row['seed']}：{', '.join(r['proposal']['block_reasons'])}；最大关节误差 {r['max_joint_error_deg']:.5f}°。"
        )
    if reference_exceedances:
        lines += ["", "超过 0.1° 参考线的案例也完整保留；低残差或局部满秩不保证正确恢复。"]
        for row in reference_exceedances[:20]:
            r = row["result"]
            lines.append(
                f"- `{row['case']}` / seed {row['seed']}：{r['max_joint_error_deg']:.5f}°，接受={r['proposal']['block_supported']}。"
            )
    lines += [
        "",
        "历史失败包括固定迭代收紧距离导致丢失腕部观测、周期几何错误分支、整帧裁剪掩盖稀疏腕部、过小光轴噪声下限导致迭代不稳定。本轮保留开发与失败记录，没有并入全新种子统计，见 [第二轮日志](ROUND2_LOG.md)。",
        "",
        "原始资产未修改。资产没有已核验的实际相机挂载；渲染只包含配置中的表面，没有环境、完整传感器或硬件执行模型。腕部阶段使用合成米制位姿；真实 VO/互视前端尚未接入。下肢和头部仍是独立玩具数值核心。",
        "",
        "## 复查",
        "",
        f"- 冻结数值源 SHA-256：`{summary['source_sha256']}`。",
        f"- 完整运行记录、配置与源码快照：`{args.directory}`。",
        "- 复现命令：`bash scripts/reproduce.sh`；多种子计划：`configs/benchmark_final.json`。",
        "- 每帧/连杆残差、协方差、拒绝原因保留在逐案例 JSON 中。",
        "- [方法与论文差异](REPRODUCTION.md) · [数据接口](DATA_FORMAT.md) · [尺度对照动画](figures/scale_calibration.gif)。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation record: {args.output}")


if __name__ == "__main__":
    main()
