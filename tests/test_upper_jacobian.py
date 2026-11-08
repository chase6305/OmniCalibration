import numpy as np
import pytest

from omnicalib.observability import jacobian
from omnicalib.synthetic import upper_problem
from omnicalib.upper import _MatchGroup, _UpperObjective


@pytest.mark.parametrize("estimate_camera", [True, False])
@pytest.mark.parametrize("estimate_scale", [True, False])
def test_articulated_point_plane_jacobian_at_nonzero_residual(estimate_camera, estimate_scale):
    robot, surfaces, frames, nominal, _, _ = upper_problem()
    rng = np.random.default_rng(10)
    groups = []
    # Keep fixed, arbitrary associations, testing derivatives off the solution.
    for surface in surfaces:
        groups.append(
            _MatchGroup(
                surface.link, surface.points[:12], surface.normals[:12], rng.normal(size=(12, 3))
            )
        )
    objective = _UpperObjective(
        robot,
        [(frames[0], groups)],
        nominal,
        0.001,
        estimate_camera,
        estimate_depth_scale=estimate_scale,
    )
    x = rng.normal(0, 0.02, 14 + (6 if estimate_camera else 0) + int(estimate_scale))
    analytic = objective.jac(x)
    numeric = jacobian(objective.fun, x)
    np.testing.assert_allclose(analytic, numeric, atol=2e-5, rtol=1e-6)
    # Callers may mutate returned arrays (SciPy's robust loss does this).
    changed = objective.jac(x)
    changed[:] = 0.0
    np.testing.assert_array_equal(objective.jac(x), analytic)
