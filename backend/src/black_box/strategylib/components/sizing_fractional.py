"""Fractional position sizing: constant account fraction per bar."""

from __future__ import annotations

import numpy as np


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Returns a constant fraction array (feed-independent)."""
    fraction = float(params.get("fraction", 0.01))
    n = len(next(iter(data.values())))
    return np.full(n, fraction)
