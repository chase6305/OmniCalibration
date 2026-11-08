from pathlib import Path

import pytest

from omnicalib.staged import run_staged_demo


def test_marvin_staged_dual_arm_and_two_wrists_without_noise(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root.parent / "HumanoidAssets/Marvin_M6_S_CCS_696_V4.0").exists():
        pytest.skip("Marvin assets not installed")
    pytest.importorskip("trimesh")
    result = run_staged_demo(
        root / "configs/marvin_dual.json",
        tmp_path / "staged.npz",
        frames_count=6,
        samples_per_link=30,
        pose_translation_noise_m=0.0,
        pose_rotation_noise_deg=0.0,
        moving_root=True,
    )
    assert result["arm"]["proposal"]["block_supported"]
    assert len(result["wrists"]) == 2
    for wrist in result["wrists"]:
        assert wrist["success"] and wrist["proposed"]
        assert wrist["diagnostics"]["rank"] == 6
        assert wrist["translation_error_mm"] < 1e-4
        assert wrist["rotation_error_deg"] < 1e-4
