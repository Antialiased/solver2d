"""Shared helpers used by multiple test modules."""

import numpy as np

from .solver import RunLog


def descend_then_hold(v: float, x0: float, xf: float):
    """Ceiling at x0 descending at speed v until it reaches xf, then held."""
    t_stop = max((x0 - xf) / v, 0.0)

    def fn(t):
        if t < t_stop:
            return x0 - v * t
        return xf

    return fn


def sweep_mean(log: RunLog) -> float:
    valid = log.sweeps[log.sweeps > 0]
    return float(np.mean(valid)) if len(valid) else float("nan")
