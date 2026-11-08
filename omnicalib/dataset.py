"""Versioned, pickle-free NPZ interchange for synchronized masked depth frames."""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import validate_transform
from .upper import DepthFrame


@dataclass
class CalibrationDataset:
    joint_names: list
    frames: list
    camera_nominal: np.ndarray
    root_frame: str
    camera_frame: str = "depth_camera"
    metadata: dict = field(default_factory=dict)
    truth: dict | None = None

    def validate(self):
        n = len(self.joint_names)
        if (
            n == 0
            or len(set(self.joint_names)) != n
            or not all(isinstance(name, str) and name for name in self.joint_names)
        ):
            raise ValueError("Dataset needs unique nonempty joint names")
        if len(self.frames) < 2 or any(frame.q.shape != (n,) for frame in self.frames):
            raise ValueError("Dataset needs at least two frames in the named joint order")
        if not self.root_frame or not self.camera_frame:
            raise ValueError("Dataset frames must be named")
        if not isinstance(self.metadata, dict) or (
            self.truth is not None and not isinstance(self.truth, dict)
        ):
            raise ValueError("Dataset metadata and optional truth must be JSON objects")
        validate_transform(self.camera_nominal, "camera_nominal")
        json.dumps(self.metadata, allow_nan=False)
        if self.truth is not None:
            json.dumps(self.truth, allow_nan=False)
        return self


def save_dataset(path, dataset):
    dataset.validate()
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Dataset filename must end in .npz")
    fixed_names = sorted(set().union(*(set(f.fixed_positions or {}) for f in dataset.frames)))
    if any(set(f.fixed_positions or {}) != set(fixed_names) for f in dataset.frames):
        raise ValueError("All frames must specify the same uncalibrated joint names")
    if set(fixed_names) & set(dataset.joint_names):
        raise ValueError("Fixed and calibrated joint names must not overlap")
    counts = np.array([len(frame.points_camera) for frame in dataset.frames], dtype=np.int64)
    arrays = {
        "format_version": np.array(1),
        "joint_names": np.array(dataset.joint_names, dtype=str),
        "q": np.array([f.q for f in dataset.frames]),
        "points_camera": np.concatenate([f.points_camera for f in dataset.frames]),
        "point_offsets": np.r_[0, np.cumsum(counts)],
        "camera_nominal": np.asarray(dataset.camera_nominal),
        "root_frame": np.array(dataset.root_frame),
        "camera_frame": np.array(dataset.camera_frame),
        "fixed_joint_names": np.array(fixed_names, dtype=str),
        "fixed_positions": np.array(
            [[f.fixed_positions[name] for name in fixed_names] for f in dataset.frames]
        ),
        "metadata_json": np.array(json.dumps(dataset.metadata, allow_nan=False)),
        "truth_json": np.array(json.dumps(dataset.truth, allow_nan=False)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_dataset(path, include_truth=False):
    with np.load(Path(path), allow_pickle=False) as source:
        if source["format_version"].shape != () or int(source["format_version"]) != 1:
            raise ValueError("Unsupported dataset format version")
        joint_names = source["joint_names"].tolist()
        q = source["q"]
        points = source["points_camera"]
        offsets = source["point_offsets"]
        fixed_names = source["fixed_joint_names"].tolist()
        fixed = source["fixed_positions"]
        if (
            q.ndim != 2
            or points.ndim != 2
            or points.shape[1] != 3
            or offsets.shape != (len(q) + 1,)
            or offsets.dtype.kind not in "iu"
        ):
            raise ValueError("Malformed dataset array shapes")
        if offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid point offsets")
        if fixed.shape != (len(q), len(fixed_names)) or len(set(fixed_names)) != len(fixed_names):
            raise ValueError("Invalid per-frame uncalibrated joint states")
        if set(fixed_names) & set(joint_names):
            raise ValueError("Fixed and calibrated joint names overlap")
        frames = [
            DepthFrame(row, points[a:b].copy(), dict(zip(fixed_names, fixed[i], strict=True)))
            for i, (row, a, b) in enumerate(zip(q, offsets[:-1], offsets[1:], strict=True))
        ]
        dataset = CalibrationDataset(
            joint_names,
            frames,
            source["camera_nominal"].copy(),
            str(source["root_frame"].item()),
            str(source["camera_frame"].item()),
            json.loads(source["metadata_json"].item()),
            json.loads(source["truth_json"].item()) if include_truth else None,
        )
    return dataset.validate()
