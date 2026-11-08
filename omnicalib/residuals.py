"""Descriptive signed point/plane errors in physical units, without ground truth."""

import numpy as np


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"count": 0, "mean_mm": None, "rms_mm": None, "p95_abs_mm": None}
    return {
        "count": len(values),
        "mean_mm": float(values.mean() * 1000),
        "rms_mm": float(np.sqrt(np.mean(values**2)) * 1000),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95) * 1000),
    }


def residual_profile(matches, residual_m):
    """Residual order must match frame/group order in _UpperObjective.

    These are retained, unwhitened association errors. Their quantiles are
    descriptive: nearest-surface selection, trimming, and shared encoder noise
    prevent interpreting the point count as independent statistical evidence.
    """
    residual_m = np.asarray(residual_m, dtype=float)
    expected = sum(len(g.observed) for _, groups in matches for g in groups)
    if residual_m.shape != (expected,) or not np.isfinite(residual_m).all():
        raise ValueError("Residuals must be finite and align with matched groups")
    frame_rows, link_values = [], {}
    offset = 0
    for index, (_, groups) in enumerate(matches):
        start = offset
        for group in groups:
            end = offset + len(group.observed)
            link_values.setdefault(group.link, []).append(residual_m[offset:end])
            offset = end
        frame_rows.append({"frame": index, **_statistics(residual_m[start:offset])})
    return {
        "basis": "retained signed point/plane errors before covariance whitening",
        "overall": _statistics(residual_m),
        "frames": frame_rows,
        "links": {name: _statistics(np.concatenate(rows)) for name, rows in link_values.items()},
    }
