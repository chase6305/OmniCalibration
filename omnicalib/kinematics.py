"""URDF FK with fixed, revolute, continuous, prismatic and mimic joints."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import pose, so3_exp


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    kind: str = "revolute"
    lower: float = -np.inf
    upper: float = np.inf
    mimic: str | None = None
    multiplier: float = 1.0
    offset: float = 0.0


class Robot:
    def __init__(self, root, joints):
        self.root = root
        self.joints = list(joints)
        self._by_name = {j.name: j for j in self.joints}
        if len(self._by_name) != len(self.joints):
            raise ValueError("Joint names must be unique")
        self.active_names = [j.name for j in self.joints if j.kind != "fixed" and j.mimic is None]
        children = [j.child for j in self.joints]
        if len(set(children)) != len(children) or root in children:
            raise ValueError("Each non-root link must have one parent")
        for j in self.joints:
            if j.kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported joint type: {j.kind}")
            if np.shape(j.origin) != (4, 4) or not np.isfinite(j.origin).all():
                raise ValueError("Joint origins must be finite 4x4 transforms")
            if j.kind != "fixed" and (
                np.shape(j.axis) != (3,)
                or not np.isfinite(j.axis).all()
                or not np.isclose(np.linalg.norm(j.axis), 1.0)
            ):
                raise ValueError("Active joint axes must be finite unit vectors")
        known, pending = {root}, self.joints.copy()
        self._ordered = []
        while pending:
            ready = [j for j in pending if j.parent in known]
            if not ready:
                raise ValueError("Disconnected or cyclic kinematic tree")
            self._ordered.extend(ready)
            known.update(j.child for j in ready)
            ready_names = {j.name for j in ready}
            pending = [j for j in pending if j.name not in ready_names]
        self._coefficients, self._constant = {}, {}

        def resolve(name, visited=()):
            if name in self._coefficients:
                return self._coefficients[name], self._constant[name]
            if name in visited or name not in self._by_name:
                raise ValueError("Invalid or cyclic mimic dependency")
            j = self._by_name[name]
            coeff = np.zeros(len(self.active_names))
            constant = 0.0
            if j.kind != "fixed":
                if j.mimic is None:
                    coeff[self.active_names.index(name)] = 1.0
                else:
                    source, offset = resolve(j.mimic, visited + (name,))
                    coeff = j.multiplier * source
                    constant = j.multiplier * offset + j.offset
            self._coefficients[name], self._constant[name] = coeff, constant
            return coeff, constant

        for j in self.joints:
            resolve(j.name)

    def _positions(self, q, names=None):
        if isinstance(q, dict):
            try:
                positions = np.array([q[name] for name in self.active_names], dtype=float)
            except KeyError as error:
                raise ValueError(f"Missing independent joint state: {error.args[0]}") from error
        elif names is not None:
            if len(q) != len(names) or len(set(names)) != len(names):
                raise ValueError("Joint vector length/name mismatch")
            return self._positions(dict(zip(names, q, strict=True)))
        else:
            positions = np.asarray(q, dtype=float)
        if positions.shape != (len(self.active_names),) or not np.isfinite(positions).all():
            raise ValueError("Expected one finite value for each independent joint")
        return positions

    def _forward(self, q, with_jacobians=False):
        transforms = {self.root: np.eye(4)}
        screws = {self.root: np.zeros((6, len(self.active_names)))} if with_jacobians else None
        for joint in self._ordered:
            before = transforms[joint.parent] @ joint.origin
            motion = np.eye(4)
            coeff = self._coefficients[joint.name]
            value = coeff @ q + self._constant[joint.name]
            if joint.kind in ("revolute", "continuous"):
                motion[:3, :3] = so3_exp(joint.axis * value)
            elif joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * value
            transforms[joint.child] = before @ motion
            if with_jacobians:
                jac = screws[joint.parent].copy()
                axis = before[:3, :3] @ joint.axis
                if joint.kind in ("revolute", "continuous"):
                    screw = np.r_[-np.cross(axis, before[:3, 3]), axis]
                    jac += screw[:, None] * coeff
                elif joint.kind == "prismatic":
                    jac[:3] += axis[:, None] * coeff
                screws[joint.child] = jac
        return (transforms, screws) if with_jacobians else transforms

    def fk(self, q, names=None):
        return self._forward(self._positions(q, names))

    def fk_with_jacobian(self, q):
        """Return FK and spatial screws [v,w] in root coordinates per link.

        Each column gives dp/dq = v + w cross p for ANY root-coordinate point
        rigidly attached to that link. Mimic multipliers propagate by chain rule.
        """
        return self._forward(self._positions(q), with_jacobians=True)

    def calibration_independent_links(self, names=None):
        names = self.active_names if names is None else names
        columns = [self.active_names.index(name) for name in names]
        moving = {self.root: False}
        for joint in self._ordered:
            moving[joint.child] = moving[joint.parent] or bool(
                np.any(self._coefficients[joint.name][columns])
            )
        return {link for link, depends in moving.items() if not depends}

    def select(self, names, fixed_positions=None):
        return SelectedRobot(self, names, fixed_positions)

    @classmethod
    def from_urdf(cls, path):
        root = ET.parse(Path(path)).getroot()
        joints = []
        links = {e.attrib["name"] for e in root.findall("link")}
        children = set()
        for e in root.findall("joint"):
            mimic = e.find("mimic")
            origin = e.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            axis_element = e.find("axis")
            axis = (
                np.fromstring(axis_element.get("xyz", "1 0 0"), sep=" ")
                if axis_element is not None
                else np.array([1.0, 0, 0])
            )
            if np.linalg.norm(axis) == 0 and e.get("type") != "fixed":
                raise ValueError("Joint axis must be nonzero")
            if e.get("type") == "fixed":
                axis = np.array([1.0, 0, 0])
            parent, child = e.find("parent").get("link"), e.find("child").get("link")
            if parent not in links or child not in links or child in children:
                raise ValueError("Invalid parent/child links in URDF")
            children.add(child)
            limit = e.find("limit")
            joints.append(
                Joint(
                    e.get("name"),
                    parent,
                    child,
                    pose(Rotation.from_euler("xyz", rpy).as_matrix(), xyz),
                    axis / np.linalg.norm(axis),
                    e.get("type"),
                    float(limit.get("lower", "-inf")) if limit is not None else -np.inf,
                    float(limit.get("upper", "inf")) if limit is not None else np.inf,
                    mimic.get("joint") if mimic is not None else None,
                    float(mimic.get("multiplier", "1")) if mimic is not None else 1.0,
                    float(mimic.get("offset", "0")) if mimic is not None else 0.0,
                )
            )
        roots = links - children
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        model = cls(roots.pop(), joints)
        model.fk(np.zeros(len(model.active_names)))
        return model


class SelectedRobot:
    """Calibrate any joint subset; other joints are fixed at explicitly given states.

    Defaults are zero. For real data, freeze only stationary joints; per-frame
    uncalibrated joint motion needs a separate full-state adapter.
    """

    def __init__(self, robot, names, fixed_positions=None):
        self.robot = robot
        self.root, self.joints = robot.root, robot.joints
        self.active_names = list(names)
        if not names or len(set(names)) != len(names) or not set(names) <= set(robot.active_names):
            raise ValueError("Select unique independent joint names from the URDF")
        self.fixed_positions = dict.fromkeys(robot.active_names, 0.0)
        if fixed_positions:
            if not set(fixed_positions) <= set(robot.active_names):
                raise ValueError("Unknown fixed joint")
            self.fixed_positions.update(fixed_positions)

    def _full_positions(self, q, fixed_positions):
        if len(q) != len(self.active_names):
            raise ValueError("Selected joint vector length mismatch")
        fixed_positions = {} if fixed_positions is None else fixed_positions
        if not set(fixed_positions) <= set(self.robot.active_names) - set(self.active_names):
            raise ValueError(
                "Per-frame fixed positions must name only uncalibrated independent joints"
            )
        return self.fixed_positions | fixed_positions | dict(zip(self.active_names, q, strict=True))

    def fk(self, q, fixed_positions=None):
        return self.robot.fk(self._full_positions(q, fixed_positions))

    def fk_with_jacobian(self, q, fixed_positions=None):
        transforms, jacobians = self.robot.fk_with_jacobian(
            self._full_positions(q, fixed_positions)
        )
        columns = [self.robot.active_names.index(name) for name in self.active_names]
        return transforms, {link: jac[:, columns] for link, jac in jacobians.items()}

    def calibration_independent_links(self):
        return self.robot.calibration_independent_links(self.active_names)
