import numpy as np

from omnicalib.geometry import pose
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame, Surface, _match_frame


def test_linkwise_trimming_preserves_rare_informative_link():
    robot = Robot(
        "base", [Joint("q", "base", "hand", pose(translation=[1.0, 0, 0]), np.array([0.0, 0, 1.0]))]
    )
    base = Surface("base", [[0.0, 0, 0]], [[0.0, 1.0, 0]])
    hand = Surface("hand", [[0.0, 0, 0]], [[0.0, 1.0, 0]])
    points = np.r_[np.tile([[0.0, 0.0001, 0]], (100, 1)), np.tile([[1.0, 0.005, 0]], (5, 1))]
    frame = DepthFrame([0.0], points)
    protected = _match_frame(
        robot, [base, hand], frame, np.zeros(1), np.eye(4), 0.03, "sampled", 0.1, trim_scope="link"
    )
    global_trim = _match_frame(
        robot, [base, hand], frame, np.zeros(1), np.eye(4), 0.03, "sampled", 0.1, trim_scope="frame"
    )
    assert any(group.link == "hand" and len(group.observed) == 5 for group in protected)
    assert not any(group.link == "hand" for group in global_trim)
