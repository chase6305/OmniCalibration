"""Explicit layout for joint zeros, optional camera pose and metric depth scale."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UpperParameters:
    joints: int
    camera: bool
    scale: bool = False

    @property
    def camera_slice(self):
        return slice(self.joints, self.joints + (6 if self.camera else 0))

    @property
    def scale_index(self):
        return self.joints + (6 if self.camera else 0) if self.scale else None

    @property
    def size(self):
        return self.joints + (6 if self.camera else 0) + int(self.scale)

    def camera_correction(self, x):
        return x[self.camera_slice] if self.camera else np.zeros(6)

    def depth_scale(self, x):
        return float(np.exp(x[self.scale_index])) if self.scale else 1.0

    def bounds(self, angular_bound, camera_bound=None, scale_log_bound=0.05):
        values = [np.full(self.joints, angular_bound)]
        if self.camera:
            camera_bound = (
                np.full(6, np.inf)
                if camera_bound is None
                else np.asarray(camera_bound, dtype=float)
            )
            if camera_bound.shape != (6,):
                raise ValueError("Camera bounds must contain six m/rad values")
            values.append(camera_bound)
        if self.scale:
            values.append(np.array([scale_log_bound]))
        return np.concatenate(values)

    def tolerances(self):
        values = [np.full(self.joints, 1e-5)]
        if self.camera:
            values.append(np.array([1e-6] * 3 + [1e-5] * 3))
        if self.scale:
            values.append(np.array([1e-6]))
        return np.concatenate(values)
