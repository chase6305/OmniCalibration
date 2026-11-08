"""Explicit solver recipes validated by development ablations, not paper defaults."""


def solver_profile(name):
    if name == "basic":
        return {}
    if name == "robust":
        return {
            "trim_start_iteration": 2,
            "uncertainty_step_fraction": 0.05,
            "step_relaxation": 0.5,
        }
    raise ValueError("Choose basic or robust solver profile")
