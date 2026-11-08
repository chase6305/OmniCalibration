#!/usr/bin/env python3
"""Show the Marvin dual-shoulder gauge using an exact FK identity, not sensor accuracy."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omnicalib.assets import load_asset_config, sample_urdf_surfaces
from omnicalib.geometry import inverse, se3_exp, transform_points
from omnicalib.observability import analyze
from omnicalib.render import look_at
from omnicalib.reporting import write_json
from omnicalib.upper import DepthFrame, _MatchGroup, _UpperObjective, model_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/marvin_dual.json"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/figures/dual_gauge.png"))
    args = parser.parse_args()
    config, urdf, robot = load_asset_config(args.config, args.asset_root)
    surfaces, _ = sample_urdf_surfaces(
        urdf, config["surface_links"] + config["anchor_links"], 60, seed=7
    )
    q = np.zeros(len(robot.active_names))
    transforms, screws = robot.fk_with_jacobian(q)
    left = robot.active_names.index("SHOULDER_PITCH_L_J1")
    right = robot.active_names.index("SHOULDER_PITCH_R_J1")
    screw = screws["shoulder_pitch_l_j1_link"][:, left]
    opposite = screws["shoulder_pitch_r_j1_link"][:, right]
    np.testing.assert_allclose(screw, -opposite, atol=1e-12)
    camera = look_at([1.6, 0, 1.1], [0, 0, 1])
    arms = [s for s in surfaces if s.link not in config["anchor_links"]]
    body = [s for s in surfaces if s.link in config["anchor_links"]]
    original_arms = transform_points(inverse(camera), model_cloud(robot, q, arms)[0])
    original_body = transform_points(inverse(camera), model_cloud(robot, q, body)[0])
    deltas = np.linspace(-2, 2, 81)
    arm_rms = []
    body_rms = []
    for deg in deltas:
        delta = np.deg2rad(deg)
        shifted = q.copy()
        shifted[left] += delta
        shifted[right] -= delta
        shifted_camera = se3_exp(screw * delta) @ camera
        a = transform_points(inverse(shifted_camera), model_cloud(robot, shifted, arms)[0])
        b = transform_points(inverse(shifted_camera), model_cloud(robot, shifted, body)[0])
        arm_rms.append(float(np.sqrt(np.mean(np.sum((a - original_arms) ** 2, axis=1))) * 1000))
        body_rms.append(float(np.sqrt(np.mean(np.sum((b - original_body) ** 2, axis=1))) * 1000))
    groups = []
    for surface in surfaces:
        observed = transform_points(
            inverse(camera), transform_points(transforms[surface.link], surface.points)
        )
        groups.append(_MatchGroup(surface.link, surface.points, surface.normals, observed))
    diagnostics = {}
    for anchored in [False, True]:
        chosen = groups if anchored else [g for g in groups if g.link not in config["anchor_links"]]
        frame = DepthFrame(q, np.concatenate([g.observed for g in chosen]))
        objective = _UpperObjective(robot, [(frame, chosen)], camera, 0.0002, True)
        diagnostics[anchored] = analyze(objective.jac(np.zeros(len(q) + 6)))
    assert diagnostics[False].rank == 19 and diagnostics[True].rank == 20
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(deltas, body_rms, label="Fixed body surface", c="#e59035")
    axes[0].plot(deltas, arm_rms, label="Both arm surfaces", c="#3478a8", lw=2)
    axes[0].set_xlabel("Left shoulder +δ, right shoulder −δ [deg]")
    axes[0].set_ylabel("Camera-frame geometric point-change RMS [mm]")
    axes[0].set_title("Camera transformed by the same global screw")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    for anchored, color in [(False, "#e59035"), (True, "#3478a8")]:
        d = diagnostics[anchored]
        s = np.array(d.singular_values)
        axes[1].semilogy(
            range(1, len(s) + 1),
            np.maximum(s / s[0], 1e-16),
            marker="o",
            c=color,
            label=f"{'With body' if anchored else 'Arms only'}: rank {d.rank}/{d.parameters}",
        )
    axes[1].axhline(1e-7, ls="--", c="black", lw=0.8, label="Relative rank threshold")
    axes[1].set_xlabel("Singular-value index")
    axes[1].set_ylabel("Normalized geometric Jacobian singular value")
    axes[1].set_title("Named geometry samples, no sensor-noise claim")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Exact Marvin FK gauge identity · not a calibration-accuracy benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    write_json(
        args.output.with_suffix(".json"),
        {
            "classification": "structural geometric identity using named model samples; not recovery or hardware accuracy",
            "left_shoulder_screw": screw,
            "right_shoulder_screw": opposite,
            "opposite_screw_residual": float(np.linalg.norm(screw + opposite)),
            "shift_deg": deltas,
            "arm_point_change_rms_mm": arm_rms,
            "body_point_change_rms_mm": body_rms,
            "max_arm_change_mm": max(arm_rms),
            "unanchored": diagnostics[False].to_dict(),
            "anchored": diagnostics[True].to_dict(),
        },
    )
    print(
        f"Gauge confirmed: 19/20 without body, 20/20 with body; max arm roundoff {max(arm_rms):.3g}mm; {args.output}"
    )


if __name__ == "__main__":
    main()
