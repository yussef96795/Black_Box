"""Trend and moving-average indicators.

The moving averages come first (they are the primitives the rest of the library
is built from), then the trend systems that combine them.

Two things are true of everything in this module:

* **Recursions carry no NaN prefix** — they seed at bar 0, per the conventions in
  ``_math.py``. The composite *windowed* indicators (HMA, Donchian, Keltner,
  Bollinger) do carry one, because no value exists yet; it is a prefix and never
  an interior hole, which is the property the contract test pins.
* **Nothing reads a future bar.** Ichimoku's spans are the interesting case: they
  are conventionally *drawn* 26 bars forward, which is a plotting offset rather
  than data. Drawn literally they would let a backtest read the cloud at a bar
  using information from 26 bars later, so they are reported undisplaced and
  ``chikou`` is reported at the bar whose close it actually is.
"""

from __future__ import annotations

from typing import Any

from black_box.strategylib._backend import np
from black_box.strategylib._kernels import recursive
from black_box.strategylib._math import (
    as_float,
    change,
    ema,
    highest,
    lowest,
    rma,
    rolling_mean,
    rolling_sum,
    safe_div,
    shift,
    to_device,
    to_host,
    true_range,
    wma,
)
from black_box.strategylib.indicators._registry import Data, P, indicator

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


@indicator(
    "IND_SMA",
    group="trend",
    params={"period": P(20, 1, 500)},
    synonyms=("sma", "simple moving average", "moving average"),
)
def sma(data: Data, *, period: int = 20) -> dict:
    """Unweighted mean of the last ``period`` closes."""
    return rolling_mean(data["close"], period)


@indicator(
    "IND_EMA",
    group="trend",
    params={"span": P(20, 1, 500)},
    synonyms=("ema", "exponential moving average"),
)
def ema_ind(data: Data, *, span: int = 20) -> dict:
    """Exponentially weighted mean, alpha = 2/(span+1)."""
    return ema(data["close"], span)


@indicator(
    "IND_WMA",
    group="trend",
    params={"period": P(20, 1, 500)},
    synonyms=("wma", "weighted moving average", "linearly weighted average"),
)
def wma_ind(data: Data, *, period: int = 20) -> dict:
    """Linearly weighted mean — weights 1..period, most recent heaviest."""
    return wma(data["close"], period)


@indicator(
    "IND_HMA",
    group="trend",
    params={"period": P(20, 2, 500)},
    synonyms=("hma", "hull moving average"),
)
def hma(data: Data, *, period: int = 20) -> dict:
    """Hull: ``WMA(2*WMA(n/2) - WMA(n), sqrt(n))`` — near-zero lag, still smooth.

    Windowed rather than recursive, so the NaN prefix of the inner WMAs simply
    propagates and no warmup padding is needed.
    """
    half = max(period // 2, 1)
    root = max(int(period**0.5), 1)
    close = data["close"]
    return wma(2.0 * wma(close, half) - wma(close, period), root)


@indicator(
    "IND_DEMA",
    group="trend",
    params={"period": P(20, 1, 500)},
    synonyms=("dema", "double exponential moving average"),
)
def dema(data: Data, *, period: int = 20) -> dict:
    """``2*EMA1 - EMA2(EMA1)`` — cancels the EMA's own lag once."""
    first = ema(data["close"], period)
    return 2.0 * first - ema(first, period)


@indicator(
    "IND_TEMA",
    group="trend",
    params={"period": P(20, 1, 500)},
    synonyms=("tema", "triple exponential moving average"),
)
def tema(data: Data, *, period: int = 20) -> dict:
    """``3*EMA1 - 3*EMA2 + EMA3`` — a cubic correction to the lag, not a smoothing."""
    first = ema(data["close"], period)
    second = ema(first, period)
    return 3.0 * first - 3.0 * second + ema(second, period)


@indicator(
    "IND_MCGINLEY",
    group="trend",
    params={"period": P(10, 2, 200)},
    synonyms=("mcginley dynamic", "mcginley"),
)
def mcginley(data: Data, *, period: int = 10) -> dict:
    """Self-adjusting MA: ``y += k*(x - y)`` with ``k = 2/(period+1)``.

    Algebraically identical to the fixed-alpha scan, so it rides the same kernel.
    What distinguishes it from :func:`ema` is the intent, not the arithmetic:
    the period is quoted against price itself, and the line is meant to be read
    as a dynamic support level.
    """
    return recursive(as_float(data["close"]), 2.0 / (period + 1))


@indicator(
    "IND_KAMA",
    group="trend",
    params={
        "period": P(10, 1, 200),
        "fast": P(2, 1, 50),
        "slow": P(30, 2, 200),
    },
    synonyms=("kama", "kaufman adaptive moving average", "adaptive moving average"),
)
def kama(data: Data, *, period: int = 10, fast: int = 2, slow: int = 30) -> dict:
    """Kaufman's Adaptive MA — the smoothing constant tracks the efficiency ratio.

    ``ER = |close - close[n]| / sum(|close - close[1]|, n)`` is the fraction of
    recent movement that was directional: a trend gives a high ER and a fast
    constant, a chop gives a low ER and a slow one. The constant is squared
    (Kaufman's smoothing-constant formula) to damp the transition.

    The ER is NaN for the first ``period-1`` bars, where the volatility sum does
    not exist yet. Those map to ER 0 — "no measured direction" — rather than
    propagating, because a NaN coefficient would poison the scan permanently.
    """
    close = as_float(data["close"])
    span = max(int(period), 1)
    direction = np.abs(close - shift(close, span, fill=close[0]))
    volatility = rolling_sum(np.abs(change(close)), span)
    er = np.nan_to_num(safe_div(direction, volatility, fill=0.0), nan=0.0)
    fastest, slowest = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = np.clip((er * (fastest - slowest) + slowest) ** 2, 0.0, 1.0)
    from black_box.strategylib._kernels import recursive_step

    return recursive_step(close, sc)


@indicator(
    "IND_TRIX",
    group="trend",
    params={"period": P(15, 1, 200)},
    synonyms=("trix", "triple exponential average", "triple smoothed rate of change"),
)
def trix(data: Data, *, period: int = 15) -> dict:
    """Percent rate of change of a triple-smoothed EMA — the one crossing that counts."""
    close = as_float(data["close"])
    smooth = ema(ema(ema(close, period), period), period)
    return 100.0 * safe_div(change(smooth), shift(smooth, 1, fill=smooth[0]))


@indicator(
    "IND_GMMA",
    group="trend",
    lines=("short1", "short2", "short3", "long1", "long2", "long3", "long4", "long5"),
    params={"short": P(3, 1, 100), "long": P(30, 2, 400)},
    synonyms=("gmma", "guppy multiple moving average"),
)
def gmma(data: Data, *, short: int = 3, long: int = 30) -> dict:
    """Two EMA bands — 3 short and 5 long — for retail/institutional separation.

    Spans are 1, 2, 3 times ``short`` and 1..5 times ``long``, the standard Guppy
    layout. The signal people actually trade is the *spread* between the bands,
    which a consumer builds from any two of these lines.
    """
    close = as_float(data["close"])
    out: dict[str, Any] = {}
    for offset in (1, 2, 3):
        out[f"short{offset}"] = ema(close, short * offset)
    for offset in (1, 2, 3, 4, 5):
        out[f"long{offset}"] = ema(close, long * offset)
    return out


# ---------------------------------------------------------------------------
# Trend systems
# ---------------------------------------------------------------------------


@indicator(
    "IND_MACD",
    group="trend",
    lines=("macd", "signal", "hist"),
    params={
        "fast": P(12, 1, 200),
        "slow": P(26, 2, 400),
        "signal": P(9, 1, 100),
    },
    synonyms=("macd", "moving average convergence divergence"),
)
def macd(data: Data, *, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line, its signal EMA, and the histogram between them.

    Reported as a difference rather than a ratio, so the units stay in price —
    the convention every charting platform uses.
    """
    close = as_float(data["close"])
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return {"macd": line, "signal": sig, "hist": line - sig}


@indicator(
    "IND_ADX",
    group="trend",
    lines=("adx", "plus_di", "minus_di"),
    params={"period": P(14, 2, 200)},
    requires=("high", "low", "close"),
    synonyms=("adx", "average directional index", "directional index", "plus di"),
)
def adx(data: Data, *, period: int = 14) -> dict:
    """Trend *strength* (0-100) plus the two directional indexes that make it.

    Direction-free by construction: a strong downtrend and a strong uptrend both
    print a high ADX, which is why +DI/-DI ship beside it. All three are
    Wilder-smoothed — building this on an EMA instead is the classic way to end
    up with an ADX that disagrees with every platform by ~20 points.

    Bar 0 has no predecessor, so directional movement and true range are seeded
    at zero there; the effect is a +DI/-DI of 0 on the first bar rather than a
    value invented from half a bar of data.
    """
    high, low, close = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
    )
    span = max(int(period), 2)
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    # cupy's concatenate rejects a bare Python list, so the seed bar is a real
    # one-element array.
    seed = np.zeros(1, dtype=float)
    plus_dm = np.concatenate(
        (seed, np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    )
    minus_dm = np.concatenate(
        (seed, np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    )
    tr = np.concatenate((seed, true_range(high, low, close)[1:]))
    atr = rma(tr, span)
    safe_atr = np.where(atr > 0, atr, 1.0)
    plus_di = 100.0 * rma(plus_dm, span) / safe_atr
    minus_di = 100.0 * rma(minus_dm, span) / safe_atr
    dx = 100.0 * safe_div(np.abs(plus_di - minus_di), plus_di + minus_di, fill=0.0)
    return {"adx": rma(dx, span), "plus_di": plus_di, "minus_di": minus_di}


@indicator(
    "IND_DONCHIAN",
    group="trend",
    lines=("middle", "upper", "lower"),
    params={"period": P(20, 1, 500)},
    requires=("high", "low"),
    synonyms=("donchian", "donchian channel", "price channel"),
)
def donchian(data: Data, *, period: int = 20) -> dict:
    """Highest high / lowest low over the window — the Turtle breakout channel."""
    hi, lo = highest(data["high"], period), lowest(data["low"], period)
    return {"upper": hi, "middle": (hi + lo) / 2.0, "lower": lo}


@indicator(
    "IND_KELTNER",
    group="trend",
    lines=("middle", "upper", "lower"),
    params={
        "period": P(20, 2, 400),
        "atr_period": P(10, 1, 200),
        "mult": P(2.0, 0.1, 10.0),
    },
    requires=("high", "low", "close"),
    synonyms=("keltner", "keltner channel"),
)
def keltner(
    data: Data, *, period: int = 20, atr_period: int = 10, mult: float = 2.0
) -> dict:
    """An EMA centre with Wilder-ATR bands — volatility-scaled, unlike Bollinger."""
    close = as_float(data["close"])
    mid = ema(close, period)
    width = mult * rma(true_range(data["high"], data["low"], close), atr_period)
    return {"upper": mid + width, "middle": mid, "lower": mid - width}


@indicator(
    "IND_ICHIMOKU",
    group="trend",
    lines=("tenkan", "kijun", "senkou_a", "senkou_b", "chikou"),
    params={
        "conversion": P(9, 1, 200),
        "base": P(26, 1, 400),
        "span_b": P(52, 1, 600),
    },
    requires=("high", "low", "close"),
    synonyms=("ichimoku", "ichimoku kinko hyo", "ichimoku cloud", "cloud"),
)
def ichimoku(
    data: Data, *, conversion: int = 9, base: int = 26, span_b: int = 52
) -> dict:
    """Ichimoku Kinko Hyo: two conversion lines, a projected cloud, and Chikou."""

    def midpoint(span: int) -> Any:
        return (highest(high, span) + lowest(low, span)) / 2.0

    high, low, close = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
    )
    tenkan = midpoint(conversion)
    kijun = midpoint(base)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": (tenkan + kijun) / 2.0,
        "senkou_b": midpoint(span_b),
        "chikou": close,
    }


@indicator(
    "IND_SUPERTREND",
    group="trend",
    lines=("line", "direction"),
    params={"period": P(10, 1, 200), "mult": P(3.0, 0.1, 20.0)},
    requires=("high", "low", "close"),
    synonyms=("supertrend", "super trend"),
)
def supertrend(data: Data, *, period: int = 10, mult: float = 3.0) -> dict:
    """ATR-banded trend line plus its direction (+1 up, -1 down).

    The band is ``(high+low)/2 +/- mult*ATR`` and only ratchets in the direction
    the trend allows; the flip is decided by the current bar's close and applied
    to the next bar, so no bar is ever revised.

    ponytail: a per-bar state machine, so it runs on the host (two transfers, not
    two per element — see ``to_host``) rather than as array ops. Same reasoning as
    PSAR below: the branch depends on the previous bar's outcome, so there is
    nothing to vectorise. The upgrade path, if this ever lands in a hot sweep, is
    the branch-per-thread form of the ``_kernels`` scan.
    """
    high, low, close = map(
        to_host,
        (as_float(data["high"]), as_float(data["low"]), as_float(data["close"])),
    )
    n = len(close)
    if n == 0:
        empty = to_device(np.zeros(0, dtype=float))
        return {"line": empty, "direction": empty}

    atr = to_host(rma(true_range(data["high"], data["low"], data["close"]), period))
    midpoint = (high + low) / 2.0
    basic_up = midpoint + mult * atr
    basic_dn = midpoint - mult * atr

    final_up = np.empty(n, dtype=float)
    final_dn = np.empty(n, dtype=float)
    line = np.empty(n, dtype=float)
    direction = np.empty(n, dtype=float)
    # Seed long with no prior flip: there is no trend to inherit on bar 0.
    final_up[0], final_dn[0] = basic_up[0], basic_dn[0]
    direction[0] = 1.0
    line[0] = final_dn[0]
    for i in range(1, n):
        prev_close = close[i - 1]
        final_up[i] = (
            basic_up[i]
            if (basic_up[i] < final_up[i - 1] or prev_close > final_up[i - 1])
            else final_up[i - 1]
        )
        final_dn[i] = (
            basic_dn[i]
            if (basic_dn[i] > final_dn[i - 1] or prev_close < final_dn[i - 1])
            else final_dn[i - 1]
        )
        if direction[i - 1] > 0:
            direction[i] = -1.0 if close[i] < final_dn[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_up[i] else -1.0
        line[i] = final_dn[i] if direction[i] > 0 else final_up[i]
    return {"line": to_device(line), "direction": to_device(direction)}


@indicator(
    "IND_PSAR",
    group="trend",
    lines=("sar", "reverse"),
    params={
        "start": P(0.02, 0.001, 0.2),
        "increment": P(0.02, 0.0, 0.2),
        "maximum": P(0.2, 0.01, 1.0),
    },
    requires=("high", "low"),
    synonyms=("psar", "parabolic sar", "stop and reverse", "parabolic"),
)
def psar(
    data: Data, *, start: float = 0.02, increment: float = 0.02, maximum: float = 0.2
) -> dict:
    """Parabolic SAR: a trailing dot that flips side when price crosses it.

    ``reverse`` is +1 while the trend is up and -1 while down, so the flip is
    readable without comparing ``sar`` to price.

    ponytail: the one indicator in the library that cannot be batched or
    vectorised — a per-bar state machine whose branch depends on the previous
    bar's outcome, so it runs on the host (two transfers) rather than as array
    ops. Acceptable because it is one call per strategy, not one per sweep point.
    The upgrade path, if PSAR lands in a hot sweep, is a branch-per-thread kernel
    in the shape of the ``_kernels`` scan.
    """
    high, low = map(to_host, (as_float(data["high"]), as_float(data["low"])))
    n = len(high)
    sar = np.empty(n, dtype=float)
    reverse = np.empty(n, dtype=float)
    if n == 0:
        return {"sar": to_device(sar), "reverse": to_device(reverse)}

    long = True  # seed long, matching every platform
    af = float(start)
    ep = float(low[0])
    sar[0] = float(low[0])
    reverse[0] = 1.0
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if long:
            # The dot may not enter the range of the last two bars.
            sar[i] = min(sar[i], low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar[i]:
                long, sar[i], ep, af = False, ep, low[i], float(start)
            elif high[i] > ep:
                ep, af = high[i], min(af + float(increment), float(maximum))
        else:
            sar[i] = max(sar[i], high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar[i]:
                long, sar[i], ep, af = True, ep, high[i], float(start)
            elif low[i] < ep:
                ep, af = low[i], min(af + float(increment), float(maximum))
        reverse[i] = 1.0 if long else -1.0
    return {"sar": to_device(sar), "reverse": to_device(reverse)}


__all__ = [
    "adx",
    "dema",
    "donchian",
    "ema_ind",
    "gmma",
    "hma",
    "ichimoku",
    "kama",
    "keltner",
    "macd",
    "mcginley",
    "psar",
    "sma",
    "supertrend",
    "tema",
    "trix",
    "wma_ind",
]
