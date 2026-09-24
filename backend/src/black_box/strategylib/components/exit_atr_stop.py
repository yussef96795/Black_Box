"""ATR stop-loss exit: returns a protective stop-distance series (k * ATR)."""

from __future__ import annotations

import numpy as np

from black_box.strategylib.components._math import ema, true_range


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Stop distance per bar = k * EMA(TR, atr_period). Flat ATR -> flat stop."""
    high, low, close = (
        data["high"].astype(float),
        data["low"].astype(float),
        data["close"].astype(float),
    )
    period = int(params.get("atr_period", 14))
    k = float(params.get("k", 2.0))
    atr = ema(true_range(high, low, close), span=period)
    atr = np.nan_to_num(atr, nan=np.nanmean(atr) if np.any(np.isfinite(atr)) else 0.0)
    return k * atr
