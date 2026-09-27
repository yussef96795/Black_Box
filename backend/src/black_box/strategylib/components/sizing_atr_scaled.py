"""ATR-scaled risk sizing: position fraction scales inversely with ATR."""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._math import rma, true_range


def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    """fraction_t = risk_pct * mean(ATR) / ATR_t, clipped to [risk_pct, 0.5].

    Inverse in ATR on purpose: a wider average bar gets a smaller position, so
    the risk taken per trade is roughly constant in *price* terms instead of
    scaling with whatever the instrument's recent volatility happened to be.

    ponytail: Wilder's RMA, not an EMA. At period 14 the two are alpha 0.071 and
    0.133, and this fraction is the actual position size, so the choice is a
    behaviour change rather than a detail — hence the 2.0.0 version bump in the
    manifest. It now matches TradingView's `ta.atr` and the registered IND_ATR
    instead of disagreeing with both.
    """
    high, low, close = (
        data["high"].astype(float),
        data["low"].astype(float),
        data["close"].astype(float),
    )
    period = int(params.get("atr_period", 14))
    risk = float(params.get("risk_pct", 0.01))
    atr = rma(true_range(high, low, close), period)
    base = np.nanmean(atr)
    if base <= 0:
        return np.full(len(close), risk)
    pos = risk * base / atr
    return np.clip(np.nan_to_num(pos, nan=risk), risk, 0.5)
