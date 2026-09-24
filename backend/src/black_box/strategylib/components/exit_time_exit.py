"""Time-based exit: flags "time to exit" from bar `bars` onward (1 = exit)."""

from __future__ import annotations

import numpy as np


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Returns a step signal: 0 until bar `bars`-1, then 1."""
    bars = max(int(params.get("bars", 20)), 1)
    n = len(next(iter(data.values())))
    signal = np.zeros(n)
    signal[bars - 1 :] = 1.0
    return signal
