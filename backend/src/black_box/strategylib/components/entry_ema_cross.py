"""EMA cross entry: +1 where fast EMA crosses above slow, -1 below, else 0."""

from __future__ import annotations

import numpy as np

from black_box.strategylib.components._math import ema


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Signal: fast-cross-above-slow = +1, fast-cross-below-slow = -1."""
    close = data["close"].astype(float)
    fast = ema(close, int(params.get("fast", 10)))
    slow = ema(close, int(params.get("slow", 30)))
    diff = fast - slow
    signal = np.zeros(len(close))
    signal[1:] = np.where(diff[1:] > 0, 1.0, 0.0) * (diff[:-1] <= 0)
    signal[1:] -= np.where(diff[1:] < 0, 1.0, 0.0) * (diff[:-1] >= 0)
    return signal
