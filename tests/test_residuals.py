from types import SimpleNamespace

import numpy as np
import pytest

from omnicalib.observability import analyze
from omnicalib.residuals import residual_profile


def test_covariance_has_physical_units_and_preserves_coupling():
    j = np.array([[10.0, 4.0], [0.0, 2.0], [1.0, 1.0]])
    diagnostics = analyze(j, parameter_scale=[0.01, 2.0])
    expected = np.linalg.inv(j.T @ j)
    np.testing.assert_allclose(diagnostics.covariance, expected)
    np.testing.assert_allclose(np.diag(diagnostics.covariance), np.square(diagnostics.sigma))
    np.testing.assert_allclose(np.diag(diagnostics.correlation), [1, 1])
    assert diagnostics.correlation[0][1] < -0.8
    singular = analyze([[1.0, 1.0]])
    assert singular.covariance is None and singular.correlation is None


def test_residual_profile_keeps_frame_and_link_groups_in_physical_units():
    def group(link, n):
        return SimpleNamespace(link=link, observed=np.zeros((n, 3)))

    matches = [(None, [group("arm", 2), group("body", 1)]), (None, [group("arm", 1)]), (None, [])]
    result = residual_profile(matches, np.array([1.0, -1.0, 4.0, 2.0]) / 1000)
    assert result["overall"]["count"] == 4
    np.testing.assert_allclose(result["overall"]["rms_mm"], np.sqrt(5.5))
    assert result["links"]["body"]["mean_mm"] == 4.0
    assert result["links"]["arm"]["count"] == 3
    assert result["frames"][2]["rms_mm"] is None
    with pytest.raises(ValueError, match="align"):
        residual_profile(matches, [0.0])
