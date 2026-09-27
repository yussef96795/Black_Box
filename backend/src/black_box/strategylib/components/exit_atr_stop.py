"""ATR stop-loss exit: returns a protective stop-distance series (k * ATR)."""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._math import rma, true_range


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """Stop distance per bar = k * Wilder ATR. Flat ATR -> flat stop.

    ponytail: Wilder's RMA, not an EMA. At period 14 the two are alpha 0.071 and
    0.133, and this series *is* the stop distance a position is sized and exited
    against, so the choice is a behaviour change rather than a detail — hence the
    2.0.0 version bump in the manifest. It now matches TradingView's `ta.atr` and
    the registered IND_ATR instead of disagreeing with both.
    """
    high, low, close = (
        data["high"].astype(float),
        data["low"].astype(float),
        data["close"].astype(float),
    )
    period = int(params.get("atr_period", 14))
    k = float(params.get("k", 2.0))
    atr = rma(true_range(high, low, close), period)
    atr = np.nan_to_num(atr, nan=np.nanmean(atr) if np.any(np.isfinite(atr)) else 0.0)
    return k * atr
