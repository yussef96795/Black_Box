"""VWAP band breakout entry: +1 on close above upper band, -1 below lower."""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._math import rolling_std


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Band = forward-looking-free VWAP ± k * trailing std; breakout = signal."""
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    lookback = int(params.get("lookback", 20))
    k = float(params.get("k", 2.0))

    cum_v = np.cumsum(volume)
    cum_pv = np.cumsum(close * volume)
    # ponytail: cupy ufuncs reject numpy's `where=` kwarg — guarded
    # denominator + explicit mask is identical where valid, 0 where not.
    mask = cum_v > 0
    vwap = np.where(mask, cum_pv / np.where(mask, cum_v, 1.0), 0.0)
    width = k * rolling_std(close, window=lookback)

    signal = np.zeros(len(close))
    signal[close > (vwap + width)] = 1.0
    signal[close < (vwap - width)] = -1.0
    return signal
