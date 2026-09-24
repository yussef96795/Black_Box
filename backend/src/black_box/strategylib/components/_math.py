"""Shared vector helpers for strategylib components (DRY — Rules.md §1).

All helpers are plain numpy over 1-D float arrays; correctness over speed
for now — Block C (vectorbt) may replace hot paths later (ponytail: the
recursive EMA is O(n) Python-loop; fine below ~1M bars, upgrade path is a
numpy signal.convolve formulation when sweeps multiply).
"""

from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average; constant input maps to that constant."""
    span = max(int(span), 1)
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = float(x[0])
    for i in range(1, len(x)):
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered-free trailing rolling mean; first `window-1` bars = NaN."""
    window = max(int(window), 1)
    if window == 1:
        return x.astype(float)
    out = np.full(len(x), np.nan)
    csum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling std (population); NaN until `window` bars exist."""
    window = max(int(window), 2)
    mean = rolling_mean(x, window)
    sq = rolling_mean(x * x, window)
    var = sq - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """ATR's true range; NaN on the first bar (no prior close)."""
    tr = np.empty_like(close, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ),
    )
    return tr
