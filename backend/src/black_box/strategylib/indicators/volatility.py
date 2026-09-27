"""Volatility, channel and dispersion indicators.

The dividing line in this module is what the denominator is built from.
Bollinger and Standard Error bands scale by a *dispersion of price*; ATR,
Keltner and Chandelier scale by a *range*; the Ulcer and Historical Volatility
indices report *drawdown* and *return* dispersion instead. They are not
interchangeable, and swapping one for another is the usual reason a "volatility
filter" behaves unexpectedly.

One convention is load-bearing throughout: **ATR is Wilder's RMA, not an EMA.**
TradingView's ``ta.atr`` uses alpha = 1/length, so at length 14 that is 0.071
against the EMA's 0.133, and the two settle at visibly different levels. Every
ATR-based indicator here uses ``rma``.
"""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._kernels import running_max, running_min
from black_box.strategylib._math import (
    as_float,
    rma,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    safe_div,
    shift,
    true_range,
)
from black_box.strategylib.indicators._registry import Data, P, indicator

#: Floor for the log return's argument. A non-positive price is invalid input,
#: but ``log(0)`` is ``-inf`` and one ``-inf`` poisons every later bar, because
#: ``rolling_std`` is cumsum-based. Clamping trades an impossible value for a
#: merely extreme one (~-27.6) that cannot destroy the series.
_LOG_FLOOR = 1e-12


@indicator(
    "IND_BB",
    group="volatility",
    lines=("middle", "upper", "lower"),
    params={"period": P(20, 2, 500), "mult": P(2.0, 0.1, 10.0)},
    synonyms=("bollinger", "bollinger bands", "bollinger band", "bb"),
)
def bollinger(data: Data, *, period: int = 20, mult: float = 2.0) -> dict:
    """Bollinger Bands: an SMA centre with population-std bands.

    The centre is a **simple** moving average, which is what defines the
    indicator — an EMA centre produces something else entirely (that variant gets
    sold as "Bollinger" and puts the bands somewhere no textbook reading will
    match).

    The width is ``mult * population std``. Population, not sample: at length 20
    the two differ by 2.6%, and the sample form makes a flat series read as
    slightly non-zero rather than exactly zero.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    mid = rolling_mean(close, span)
    width = mult * rolling_std(close, span)
    return {"upper": mid + width, "middle": mid, "lower": mid - width}


@indicator(
    "IND_ATR",
    group="volatility",
    params={"period": P(14, 1, 200)},
    requires=("high", "low", "close"),
    synonyms=("atr", "average true range", "true range"),
)
def atr(data: Data, *, period: int = 14) -> dict:
    """Average True Range — Wilder's smoothing of the true range.

    True range is the widest of today's high-low range, today's high against
    yesterday's close, and today's low against yesterday's close. The last two
    matter because a gap is a real move that the intraday range misses. Bar 0 has
    no prior close, so its true range is just high minus low.
    """
    span = max(int(period), 1)
    return rma(true_range(data["high"], data["low"], data["close"]), span)


@indicator(
    "IND_HV",
    group="volatility",
    params={"period": P(20, 2, 400), "annualise": P(252, 1, 4000)},
    synonyms=(
        "historical volatility",
        "hv",
        "realized volatility",
        "realised volatility",
    ),
)
def historical_volatility(
    data: Data, *, period: int = 20, annualise: int = 252
) -> dict:
    """Historical Volatility: the std of log returns, annualised.

    Annualisation is an explicit parameter rather than a bar-size guess, because
    inferring the bar duration from the data is exactly the kind of inference
    that silently turns a daily vol into an annual one.

    ponytail: no auto-detection of the bar size, and no option to skip the
    annualisation — both would be one more knob that is wrong silently. The
    upgrade path, if a caller really wants per-bar vol, is passing
    ``annualise=1``.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    previous = shift(close, 1, fill=close[0])
    log_returns = np.log(np.maximum(safe_div(close, previous, fill=1.0), _LOG_FLOOR))
    return rolling_std(log_returns, span) * float(annualise) ** 0.5


@indicator(
    "IND_UI",
    group="volatility",
    params={"period": P(14, 2, 200)},
    synonyms=("ulcer index", "ui"),
)
def ulcer(data: Data, *, period: int = 14) -> dict:
    """Ulcer Index: the RMS of drawdown — how *painful* volatility is, not how big.

    A high Ulcer with small bar-to-bar moves means a long grinding decline, which
    is the regime a plain standard deviation rates as calm. The squaring is what
    makes it an index of stress rather than of size; the outer square root brings
    it back to price units.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    drawdown = 100.0 * (close / running_max(close) - 1.0)
    return np.sqrt(rolling_mean(drawdown * drawdown, span))


@indicator(
    "IND_CHOP",
    group="volatility",
    params={"period": P(14, 2, 200)},
    requires=("high", "low", "close"),
    synonyms=("choppiness index", "chop", "choppiness"),
)
def choppiness(data: Data, *, period: int = 14) -> dict:
    """Choppiness Index: ~0 is a clean trend, ~100 is pure noise.

    A log ratio of the summed true range against the actual window span, scaled
    by ``log10(period)`` so the result lands near 100 for noise at any window
    length. The true range is a *sum*, so a trending market's total range is large
    relative to how far it actually went, while a choppy one is the reverse.

    ``safe_div``'s fill of 0 is floored before the log so a window with no
    movement at all cannot produce ``log10(0)``; NaN in the warmup prefix
    propagates through the clip as NaN, which is the honest answer there.
    """
    high, low, close = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
    )
    span = max(int(period), 2)
    total_range = rolling_mean(true_range(high, low, close), span)
    window_span = rolling_max(high, span) - rolling_min(low, span)
    ratio = safe_div(total_range, window_span, fill=0.0)
    return 100.0 * np.log10(np.maximum(ratio, 1e-12)) / np.log10(float(span))


@indicator(
    "IND_STD_ERROR",
    group="volatility",
    lines=("upper", "lower"),
    params={"period": P(20, 2, 400)},
    synonyms=("standard error bands", "standard error", "std error", "se bands"),
)
def standard_error_bands(data: Data, *, period: int = 20) -> dict:
    """Standard Error Bands: a linear-regression channel, not a volatility band.

    Unlike Bollinger these are not symmetric about price and they tilt with the
    fitted slope — both bands follow the regression line, one standard error of
    the residuals above and below. That is the reason to reach for them when a
    Bollinger channel gets dragged sideways by a single outlier.

    Closed form over the trailing window: for ``x = 0..n-1`` the slope comes from
    ``Sxy`` and the residual sum of squares from ``Syy``, both rolling sums, then
    the normal equations give the intercept. There is no matrix solve — this
    deployment has no cuBLAS, so ``cp.linalg`` is unavailable and the fit is
    written out. ``x`` is the bar's *age within the window*, so the trailing view
    pairs with an ascending ramp: the oldest bar is x=0.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    n = float(span)
    # All constants for a window of this length — computed once, not per bar.
    sum_x = n * (n - 1.0) / 2.0
    sum_xx = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    denom = n * sum_xx - sum_x * sum_x
    ramp = np.arange(span, dtype=float)

    # Row i is [close[i-n+1] ... close[i]], and x runs 0..n-1 across it, so the
    # oldest bar in the window is x=0. The strided view is `span-1` bars shorter
    # than the series, so the result is re-padded to align with the rolling sums.
    view = np.lib.stride_tricks.sliding_window_view(close, span)
    sum_xy = np.full(len(close), np.nan, dtype=float)
    sum_xy[span - 1 :] = (view * ramp).sum(axis=-1)
    sum_y = rolling_mean(close, span) * n
    sum_yy = rolling_mean(close * close, span) * n

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    # SSE = Syy - b0*Sy - b1*Sxy, the normal equations in residual form.
    sse = np.maximum(sum_yy - intercept * sum_y - slope * sum_xy, 0.0)
    error = np.sqrt(sse / max(n - 2.0, 1.0))
    # The band's value is the fit at the window's LAST bar, x = n-1.
    fitted = intercept + slope * (n - 1.0)
    return {"upper": fitted + error, "lower": fitted - error}


@indicator(
    "IND_CHANDELIER",
    group="volatility",
    lines=("long", "short"),
    params={"period": P(22, 1, 200), "mult": P(3.0, 0.1, 20.0)},
    requires=("high", "low", "close"),
    synonyms=("chandelier exit", "chandelier"),
)
def chandelier(data: Data, *, period: int = 22, mult: float = 3.0) -> dict:
    """Chandelier Exit: the tightest trailing stop, anchored to a running extreme.

    Non-repainting by construction — the long stop hangs off the *running* high,
    so it can only rise, and the short stop off the running low, so it can only
    fall. No later bar ever revises either.
    """
    width = mult * atr(data, period=period)["atr"]
    return {
        "long": running_max(as_float(data["high"])) - width,
        "short": running_min(as_float(data["low"])) + width,
    }


__all__ = [
    "atr",
    "bollinger",
    "chandelier",
    "choppiness",
    "historical_volatility",
    "standard_error_bands",
    "ulcer",
]
