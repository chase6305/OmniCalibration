from types import SimpleNamespace

import numpy as np

from omnicalib.assets import correction_proposal
from omnicalib.observability import analyze


def test_weak_coordinate_prevents_partial_commit_of_coupled_solution():
    diagnostics = analyze(np.diag([100.0, 1.0]), max_sigma=[0.1, 0.1])
    assert diagnostics.supported == [True, False]
    fit = SimpleNamespace(
        diagnostics=diagnostics,
        offsets=np.array([0.01, 0.02]),
        correction=np.zeros(6),
        success=True,
        converged=True,
        at_bounds=False,
        quality_passed=True,
    )
    proposal = correction_proposal(["a", "b"], fit, "base")
    assert proposal["block_supported"] is False
    assert proposal["block_reasons"] == ["uncertainty_limit_exceeded"]
    assert all(p["decision"] == "retain" for p in proposal["parameters"])
