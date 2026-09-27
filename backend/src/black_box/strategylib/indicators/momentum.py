"""Momentum and oscillator indicators.

Oscillators are bounded or near-bounded by construction (RSI 0-100, stochastic
0-100, Williams %R 0..-100) and the unbounded ones here (ROC, AO, TSI, TRIX) are
expressed in percent or as a ratio, so a consumer can tell a bounded oscillator
from an unbounded one without a lookup table.
"""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._math import (
    as_float,
    change,
    ema,
    mean_dev,
    median_price,
    pct_change,
    rma,
    rolling_max,
    rolling_mean,
    rolling_min,
    safe_div,
    scan,
    shift,
    stochastic_k,
    wma,
)
from black_box.strategylib.indicators._registry import Data, P, indicator

#: Coppock's smoothing window. Fixed at 10 by the original formulation, so it is
#: a constant rather than a swept parameter: exposing it would let a sweep tune
#: a number that is part of the indicator's definition.
_COPPOCK_SMOOTH = 10


@indicator(
    "IND_RSI",
    group="momentum",
    params={"period": P(14, 2, 200)},
    synonyms=("rsi", "relative strength index", "relative strength"),
)
def rsi(data: Data, *, period: int = 14) -> dict:
    """Relative Strength Index, 0-100, Wilder-smoothed.

    Built from average *gains* and average *losses* rather than an index over
    ``close/change``, because a single negative bar in the ratio makes the whole
    tail meaningless — the sign-separated form is Wilder's own definition and the
    only one that does not break on a down bar.
    """
    close = as_float(data["close"])
    span = max(int(period), 1)
    delta = change(close)
    gains = rma(np.where(delta > 0, delta, 0.0), span)
    losses = rma(np.where(delta < 0, -delta, 0.0), span)
    # The lossless case is RS = infinity, not RS = 1.0, so the ratio's fill is
    # `inf` and the outer 1/(1+RS) collapses to the 100 every platform shows for
    # a run of only-up bars. `float("inf")` works in the `np.where` because it
    # enters as a Python scalar, which is the NumPy-`asarray` gap cupy has.
    rs = safe_div(gains, losses, fill=float("inf"))
    return 100.0 - 100.0 / (1.0 + rs)


@indicator(
    "IND_STOCH",
    group="momentum",
    lines=("k", "d"),
    params={"period": P(14, 1, 200), "smooth": P(3, 1, 50)},
    requires=("high", "low", "close"),
    synonyms=("stochastic", "stochastic oscillator", "stoch", "k and d"),
)
def stoch(data: Data, *, period: int = 14, smooth: int = 3) -> dict:
    """Stochastic oscillator: %K and its %D SMA.

    %K is the raw close position in the range; %D is the ``smooth``-bar SMA of
    %K. The convention every platform uses is %K fast and %D slow — a crossover
    of %K above %D is the signal — so that ordering is what ships.
    """
    k = stochastic_k(data["high"], data["low"], data["close"], period)
    return {"k": k, "d": rolling_mean(k, max(int(smooth), 1))}


@indicator(
    "IND_STOCH_RSI",
    group="momentum",
    params={"period": P(14, 2, 200)},
    synonyms=("stochastic rsi", "stoch rsi"),
)
def stoch_rsi(data: Data, *, period: int = 14) -> dict:
    """The stochastic formula applied to RSI rather than to price.

    Far more reactive than either parent, which is the point: it turns a
    0-100 oscillator into a 0-100 oscillator of the RSI's own extremes.
    """
    values = rsi(data, period=period)["rsi"]
    span = max(int(period), 2)
    hi, lo = rolling_max(values, span), rolling_min(values, span)
    width = hi - lo
    return 100.0 * safe_div(values - lo, width, fill=50.0)


@indicator(
    "IND_CCI",
    group="momentum",
    params={"period": P(20, 2, 200)},
    requires=("high", "low", "close"),
    synonyms=("cci", "commodity channel index"),
)
def cci(data: Data, *, period: int = 20) -> dict:
    """Commodity Channel Index: the typical price's mean absolute deviation, scaled.

    Divided by the mean absolute deviation and scaled by a constant (0.015) so
    that +/-100 marks roughly two thirds of the moves inside the typical band.
    Mean absolute deviation, not standard deviation — CCI is defined that way and
    the two disagree enough to move the 100 line.
    """
    period = max(int(period), 2)
    typical = (
        as_float(data["high"]) + as_float(data["low"]) + as_float(data["close"])
    ) / 3.0
    deviation = mean_dev(typical, period)
    return (typical - rolling_mean(typical, period)) / (0.015 * deviation)


@indicator(
    "IND_WILLIAMS_R",
    group="momentum",
    params={"period": P(14, 1, 200)},
    requires=("high", "low", "close"),
    synonyms=("williams %r", "williams r", "williams", "%r"),
)
def williams_r(data: Data, *, period: int = 14) -> dict:
    """Williams %R: the stochastic's mirror, on 0..-100 (0 is overbought).

    Exactly ``%K - 100``, kept as its own indicator because the sign convention
    is the thing people get wrong, and because -80/-20 are the levels that
    actually get traded.
    """
    return stochastic_k(data["high"], data["low"], data["close"], period) - 100.0


@indicator(
    "IND_ROC",
    group="momentum",
    params={"period": P(12, 1, 400)},
    synonyms=("roc", "rate of change", "rate of change oscillator", "momentum"),
)
def roc(data: Data, *, period: int = 12) -> dict:
    """Rate of change in percent over ``period`` bars — unbounded by design."""
    return pct_change(data["close"], period)


@indicator(
    "IND_AO",
    group="momentum",
    params={"fast": P(5, 1, 100), "slow": P(34, 2, 400)},
    requires=("high", "low"),
    synonyms=("awesome oscillator", "ao", "awesome"),
)
def awesome(data: Data, *, fast: int = 5, slow: int = 34) -> dict:
    """Awesome Oscillator: SMA(5) - SMA(34) of the median price (H+L)/2.

    A zero-line oscillator rather than a bounded one, so the tradeable levels are
    "above/below zero" and the histogram's slope, not 70/30.
    """
    hl2 = median_price(data["high"], data["low"])
    return rolling_mean(hl2, max(int(fast), 1)) - rolling_mean(hl2, max(int(slow), 2))


@indicator(
    "IND_COPPOCK",
    group="momentum",
    params={"long": P(14, 1, 200), "short": P(11, 1, 200)},
    synonyms=("coppock curve", "coppock"),
)
def coppock(data: Data, *, long: int = 14, short: int = 11) -> dict:
    """Coppock Curve: a long-horizon momentum oscillator for major bottoms.

    The sum of the ``long``- and ``short``-bar rate of change, smoothed with a
    10-bar WMA. Built for a 9-to-12-month buy-and-hold horizon, so it is slow by
    construction and says nothing about a 1h chart.
    """
    close = as_float(data["close"])
    total = pct_change(close, max(int(long), 1)) + pct_change(close, max(int(short), 1))
    return wma(total, _COPPOCK_SMOOTH)


@indicator(
    "IND_DPO",
    group="momentum",
    params={"period": P(21, 3, 400)},
    synonyms=("dpo", "detrended price oscillator"),
)
def dpo(data: Data, *, period: int = 21) -> dict:
    """Detrended Price Oscillator: price minus a *displaced* SMA.

    The SMA is shifted back by ``period/2 + 1`` bars so the long trend is removed
    from the middle of the window rather than its edge, which is what lets the
    result be read directly as a price. The shift looks backwards only, so no bar
    is contaminated.
    """
    close = as_float(data["close"])
    span = max(int(period), 3)
    average = rolling_mean(close, span)
    return close - shift(average, span // 2 + 1, fill=average[0])


@indicator(
    "IND_TSI",
    group="momentum",
    params={
        "long": P(25, 2, 400),
        "short": P(13, 1, 200),
        "signal": P(13, 1, 200),
    },
    synonyms=("tsi", "true strength index"),
)
def tsi(data: Data, *, long: int = 25, short: int = 13, signal: int = 13) -> dict:
    """True Strength Index: momentum doubly smoothed and bounded to +/-100.

    The absolute value of the denominator's EMA is what produces the bound — a
    signed EMA would let the sign of the average momentum flip the indicator's
    meaning at zero crossings, which is exactly what TSI exists to avoid.
    """
    close = as_float(data["close"])
    delta = change(close)
    first = ema(delta, max(int(long), 2))
    second = ema(first, max(int(short), 1))
    # The sign of the *average absolute* move is what must be forced positive —
    # a signed denominator would let the sign of average momentum flip the
    # indicator's meaning at every zero crossing, which is what TSI exists to
    # prevent. The numerator stays signed, because its direction is the signal.
    scale = ema(ema(np.abs(delta), max(int(long), 2)), max(int(short), 1))
    return 100.0 * ema(safe_div(second, scale, fill=0.0), max(int(signal), 1))


@indicator(
    "IND_FISHER",
    group="momentum",
    params={"period": P(9, 2, 100)},
    synonyms=("fisher transform", "fisher"),
)
def fisher(data: Data, *, period: int = 9) -> dict:
    """Fisher Transform: a Gaussianising remap of the median price's range position.

    Maps the close's position inside its own range to approximately normally
    distributed values by chaining two inverse-hyperbolic transforms. The two
    clamp stages are load-bearing, not defensive noise: a value reaching +/-1
    sends ``log((1+x)/(1-x))`` to infinity, and an intermediate sent to infinity
    makes the *second* transform's argument NaN — and because a smoothing EMA
    carries that NaN forward forever, one extreme bar would destroy the whole
    tail. The inner clamp is the standard +/-0.999; the outer one bounds the
    value fed into the second transform so the result stays finite.
    """
    hl2 = median_price(data["high"], data["low"])
    span = max(int(period), 2)
    lowest_hl2, highest_hl2 = rolling_min(hl2, span), rolling_max(hl2, span)
    width = highest_hl2 - lowest_hl2
    normalised = np.clip(
        2.0 * safe_div(hl2 - lowest_hl2, width, fill=0.5) - 1.0, -0.999, 0.999
    )
    first_pass = 0.5 * np.log((1.0 + normalised) / (1.0 - normalised))
    # 0.66 scales the transform down, 0.5 is the standard 1-period smoothing, and
    # the weighted combination is Ehlers' own — the weights sum to 1.33 because
    # the transform's own gain needs compensating.
    smoothed = scan(0.66 * first_pass, 0.5) + 0.67 * scan(0.66 * first_pass, 0.5)
    trigger = 0.5 * first_pass + 0.67 * np.clip(smoothed, -1.9, 1.9)
    return scan(trigger, 0.5)


@indicator(
    "IND_CONNORS_RSI",
    group="momentum",
    params={"period": P(3, 1, 50), "rsi_period": P(3, 2, 50), "streak": P(2, 1, 20)},
    synonyms=("connors rsi", "connors", "composite rsi"),
)
def connors_rsi(
    data: Data, *, period: int = 3, rsi_period: int = 3, streak: int = 2
) -> dict:
    """Connors RSI: a 3-period RSI plus streak and rate-of-change components.

    Composite on purpose — the short RSI provides the trigger, the streak its
    persistence, and the ROC its direction, and the sum reads better than any of
    them alone on short timeframes.
    """
    close = as_float(data["close"])
    base = rsi(data, period=rsi_period)["rsi"]
    delta = change(close)
    up = (delta > 0).astype(float)
    down = (delta < 0).astype(float)
    up_streak = scan(up, 1.0 / max(int(streak), 1))
    down_streak = scan(down, 1.0 / max(int(streak), 1))
    return base + up_streak - down_streak


__all__ = [
    "awesome",
    "cci",
    "connors_rsi",
    "coppock",
    "dpo",
    "fisher",
    "roc",
    "rsi",
    "stoch",
    "stoch_rsi",
    "tsi",
    "williams_r",
]
