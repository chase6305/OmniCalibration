import numpy as np
import pytest

from omnicalib.geometry import transform_points
from omnicalib.kinematics import Robot
from omnicalib.observability import jacobian


def test_urdf_origin_axis_and_mimic(tmp_path):
    path = tmp_path / "test.urdf"
    path.write_text("""<robot name="test">
      <link name="root"/><link name="arm"/><link name="tool"/>
      <joint name="rotation" type="revolute">
        <parent link="root"/><child link="arm"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 2"/>
      </joint>
      <joint name="slide" type="prismatic">
        <parent link="arm"/><child link="tool"/><axis xyz="1 0 0"/>
        <mimic joint="rotation" multiplier="2" offset="0.1"/>
      </joint>
    </robot>""")
    robot = Robot.from_urdf(path)
    assert robot.active_names == ["rotation"]
    transforms = robot.fk([0.0])
    np.testing.assert_allclose(transforms["tool"][:3, 3], [1, 0.1, 0], atol=1e-12)
    with pytest.raises(ValueError):
        robot.select(["slide"])
    point = np.array([[0.3, -0.1, 0.2]])
    q = np.array([0.2])
    transforms, screws = robot.fk_with_jacobian(q)
    world = transform_points(transforms["tool"], point)[0]
    analytic = screws["tool"][:3] + np.cross(screws["tool"][3:].T, world).T
    numeric = jacobian(lambda x: transform_points(robot.fk(x)["tool"], point)[0], q)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
