import json

import numpy as np
import pytest

from omnicalib.dataset import CalibrationDataset, load_dataset, save_dataset
from omnicalib.upper import DepthFrame


def _dataset():
    return CalibrationDataset(
        ["joint_a"],
        [
            DepthFrame([0.1], [[0, 0, 1]], {"other": 0.2}),
            DepthFrame([0.2], [[0, 0, 1], [1, 0, 2]], {"other": 0.3}),
        ],
        np.eye(4),
        "base",
        metadata={"synthetic": True},
        truth={"offsets": [0.005]},
    )


def test_variable_length_point_frames_and_truth_separation(tmp_path):
    path = tmp_path / "dataset.npz"
    expected = _dataset()
    save_dataset(path, expected)
    observed = load_dataset(path)
    assert observed.truth is None
    assert load_dataset(path, include_truth=True).truth == expected.truth
    assert observed.joint_names == expected.joint_names
    for a, b in zip(observed.frames, expected.frames, strict=True):
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.points_camera, b.points_camera)
        assert a.fixed_positions == b.fixed_positions


def test_invalid_rigid_camera_transform_is_rejected(tmp_path):
    dataset = _dataset()
    dataset.camera_nominal[0, 0] = 1.1
    with pytest.raises(ValueError, match="SE"):
        save_dataset(tmp_path / "bad.npz", dataset)


def test_corrupt_point_offsets_are_rejected(tmp_path):
    path = tmp_path / "dataset.npz"
    save_dataset(path, _dataset())
    with np.load(path, allow_pickle=False) as original:
        arrays = dict(original)
    arrays["point_offsets"] = np.array([0, 2, 1])
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="offsets"):
        load_dataset(path)


def test_nonfinite_points_fail_early():
    with pytest.raises(ValueError, match="finite"):
        DepthFrame([0.0], [[0, np.nan, 1.0]])


@pytest.mark.parametrize("estimate_scale", [False, True])
def test_estimator_ignores_deliberately_wrong_stored_truth(tmp_path, estimate_scale):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    from omnicalib.assets import calibrate_dataset, load_asset_config, sample_urdf_surfaces
    from omnicalib.geometry import transform_points

    mesh = trimesh.creation.box(extents=[0.3, 0.1, 0.2])
    mesh.apply_translation([0.3, 0, 0])
    mesh.export(tmp_path / "arm.stl")
    (tmp_path / "arm.urdf").write_text("""<robot name="one">
      <link name="base"/><link name="arm"><visual><geometry><mesh filename="arm.stl"/></geometry></visual></link>
      <joint name="joint_a" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
    </robot>""")
    config = {
        "name": "test",
        "mode": "single",
        "asset_root": str(tmp_path),
        "urdf": "arm.urdf",
        "joint_names": ["joint_a"],
        "surface_links": ["arm"],
        "anchor_links": [],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    _, urdf, robot = load_asset_config(path)
    surfaces, _ = sample_urdf_surfaces(urdf, ["arm"], 150)
    frames = [
        DepthFrame(
            [a],
            transform_points(robot.fk([a + 0.01])["arm"], surfaces[0].points)
            * (1.005 if estimate_scale else 1.0),
        )
        for a in [-0.4, 0.1, 0.6]
    ]
    dataset = CalibrationDataset(["joint_a"], frames, np.eye(4), "base", truth={"offsets": [100.0]})
    data_path = tmp_path / "input.npz"
    save_dataset(data_path, dataset)
    result = calibrate_dataset(
        path,
        data_path,
        estimate_camera=False,
        estimate_depth_scale=estimate_scale,
        joint_sigma_limit_deg=0.1,
        depth_scale_sigma_limit_pct=0.1 if estimate_scale else None,
    )
    assert result["truth_used_for_estimation"] is False
    np.testing.assert_allclose(result["offsets_rad"], [0.01], atol=1e-6)
    np.testing.assert_allclose(
        result["depth_scale"], 1 / 1.005 if estimate_scale else 1.0, atol=1e-6
    )
    assert "truth" not in result
    if estimate_scale:
        from scripts.select_frames import select_dataset

        selected_path = tmp_path / "selected.npz"
        selection = select_dataset(path, data_path, selected_path, 2, estimate_depth_scale=True)
        selected = load_dataset(selected_path, include_truth=True)
        assert selected.truth is None and len(selected.frames) == 2
        assert selection["diagnostics"]["rank"] < selection["diagnostics"]["parameters"]
        assert not any(selection["diagnostics"]["supported"])
        replay = calibrate_dataset(
            path, selected_path, estimate_camera=False, estimate_depth_scale=True
        )
        np.testing.assert_allclose(replay["offsets_rad"], [0.01], atol=1e-6)
        np.testing.assert_allclose(replay["depth_scale"], 1 / 1.005, atol=1e-6)
