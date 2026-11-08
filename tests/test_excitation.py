import numpy as np

from omnicalib.excitation import select_frames
from omnicalib.geometry import pose
from omnicalib.kinematics import Joint, Robot
from omnicalib.upper import DepthFrame, Surface


def test_selection_prefers_lever_arm_information_and_keeps_regularizer_out_of_rank():
    robot = Robot("base", [Joint("joint", "base", "arm", pose(), np.array([0.0, 0, 1.0]))])
    # p=(x,0,0), n=(0,1,0) -> rotational information is proportional to x^2.
    surface = Surface("arm", [[0.01, 0, 0], [0.5, 0, 0]], [[0, 1.0, 0], [0, 1.0, 0]])
    frames = [DepthFrame([0.0], [[0.01, 0, 0]]), DepthFrame([0.0], [[0.5, 0, 0]])]
    result = select_frames(
        robot, [surface], frames, np.eye(4), 1, matching="sampled", estimate_camera=False
    )
    assert result.indices == [1]
    assert result.diagnostics.rank == 1
    # With free camera pose, one point cannot identify all seven parameters.
    weak = select_frames(
        robot, [surface], frames, np.eye(4), 1, matching="sampled", estimate_camera=True
    )
    assert weak.diagnostics.rank < 7
    assert not any(weak.diagnostics.supported)


def test_selection_is_seed_reproducible_and_indices_are_unique():
    robot = Robot("base", [Joint("joint", "base", "arm", pose(), np.array([0.0, 0, 1.0]))])
    surface = Surface("arm", [[0.1, 0, 0]], [[0, 1.0, 0]])
    frames = [DepthFrame([0.0], np.tile([[0.1, 0, 0]], (100, 1))) for _ in range(4)]
    first = select_frames(
        robot,
        [surface],
        frames,
        np.eye(4),
        3,
        matching="sampled",
        estimate_camera=False,
        max_points_per_frame=20,
        seed=13,
    )
    second = select_frames(
        robot,
        [surface],
        frames,
        np.eye(4),
        3,
        matching="sampled",
        estimate_camera=False,
        max_points_per_frame=20,
        seed=13,
    )
    assert first.indices == second.indices
    assert len(set(first.indices)) == 3
