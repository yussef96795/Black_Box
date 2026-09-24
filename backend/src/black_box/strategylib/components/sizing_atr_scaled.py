"""ATR-scaled risk sizing: position fraction scales inversely with ATR."""

from __future__ import annotations

import numpy as np

from black_box.strategylib.components._math import ema, true_range


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """fraction_t = risk_pct * mean(ATR) / ATR_t, clipped to [risk_pct, 0.5]."""
    high, low, close = (
        data["high"].astype(float),
        data["low"].astype(float),
        data["close"].astype(float),
    )
    period = int(params.get("atr_period", 14))
    risk = float(params.get("risk_pct", 0.01))
    atr = ema(true_range(high, low, close), span=period)
    base = np.nanmean(atr)
    if base <= 0:
        return np.full(len(close), risk)
    pos = risk * base / atr
    return np.clip(np.nan_to_num(pos, nan=risk), risk, 0.5)
