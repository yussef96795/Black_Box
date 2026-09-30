"""Volatility, channel and dispersion indicators.

The dividing line in this module is what the denominator is built from.
Bollinger and Standard Error bands scale by a *dispersion of price*; ATR,
Keltner and Chandelier scale by a *range*; the Ulcer and Historical Volatility
indices report *drawdown* and *return* dispersion instead. They are not
interchangeable, and swapping one for another is the usual reason a "volatility
filter" behaves unexpectedly.

**The realized volatility family** — ``IND_HV``, ``IND_VOL_PARKINSON``,
``IND_VOL_GARMAN_KLASS``, ``IND_VOL_YANG_ZHANG`` — lives here together for a
reason beyond tidiness: the interesting property of that family is only visible
in the comparison. Each estimates the same quantity, the annualized standard
deviation of log returns, from progressively more of the bar, and each is a
different bet about where the information lives. Close-to-close uses two prices
and discards the rest; Parkinson uses the range; Garman-Klass adds the open so
the open-to-close move is separated from the extremes; Yang-Zhang adds the prior
close and weights the three variance components against each other by their
sampling errors. Read one alone and it is a number; read them side by side and
the choice becomes visible.

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
    log_ratio,
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

#: ``2 ln 2 - 1`` — Garman-Klass's open/close de-biasing constant, and the term
#: Yang-Zhang's overnight component is the other half of. Precomputed because it
#: appears in two formulas and is a constant, not a parameter.
_GK_COEF = 2.0 * float(np.log(2.0)) - 1.0


def _sqrt_variance(variance, annualise: float):
    """``sqrt(variance) * sqrt(annualise)``, refusing to invent a complex number.

    The clamp is the point of this helper. A negative variance is not a rounding
    artefact here, it is reachable on real data, and ``sqrt`` of it is a NaN that
    appears in the *middle* of the series — which ``indicators/__init__.py`` calls
    a contract violation rather than a warmup prefix, because every composition
    built on the line is poisoned from that bar onward. Flooring at zero reports
    "no variance in this window", which is what a negative estimate means once
    you have already subtracted the open/close drift from the range.
    """
    return np.sqrt(np.maximum(variance, 0.0)) * float(annualise) ** 0.5


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
    return rolling_std(log_ratio(close, previous), span) * float(annualise) ** 0.5


@indicator(
    "IND_VOL_PARKINSON",
    group="volatility",
    lines=("volatility",),
    params={"period": P(20, 2, 500), "annualise": P(252, 1, 4000)},
    requires=("high", "low"),
    synonyms=(
        "parkinson",
        "parkinson volatility",
        "high-low estimator",
        "high low volatility",
    ),
)
def parkinson_volatility(data: Data, *, period: int = 20, annualise: int = 252) -> dict:
    """Parkinson high-low volatility: the range is the estimator.

    ``sigma^2 = (1 / (4 ln2 N)) * sum( ln(H/L)^2 )``. The whole argument is that
    over a bar the extremes are hit by a Brownian path, and the expected squared
    log-range of such a path is proportional to the variance — so the range
    carries more information about the bar's volatility than its two endpoints do.
    At small bar counts that is a large enough gain to be worth the assumption.

    The ``1 / (4 ln 2)`` is not a fudge factor: it is what makes the estimator
    consistent with the variance of a driftless Brownian motion, and it is why
    Parkinson needs no open and no close at all.

    ponytail: no overnight correction, so a gap is read as volatility exactly
    like a real move. Adding Yang-Zhang's overnight term would make this a
    different estimator wearing this one's name — the gap handling is the reason
    to reach for ``IND_VOL_YANG_ZHANG`` instead. The upgrade path is a new id, not
    a new parameter here.
    """
    high, low = as_float(data["high"]), as_float(data["low"])
    span = max(int(period), 2)
    log_range = log_ratio(high, low)
    return {
        "volatility": _sqrt_variance(
            rolling_mean(log_range * log_range, span) / (4.0 * float(np.log(2.0))),
            annualise,
        )
    }


@indicator(
    "IND_VOL_GARMAN_KLASS",
    group="volatility",
    lines=("volatility",),
    params={"period": P(20, 2, 500), "annualise": P(252, 1, 4000)},
    requires=("high", "low", "open", "close"),
    synonyms=(
        "garman-klass",
        "garman klass",
        "garman-klass volatility",
        "garman",
    ),
)
def garman_klass_volatility(
    data: Data, *, period: int = 20, annualise: int = 252
) -> dict:
    """Garman-Klass: the range, minus the part the open-to-close move explains.

    Per bar, ``0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2``, averaged over the
    window. The subtraction is the estimator's entire contribution: a bar can have
    a wide range purely because it *trended* open-to-close, and that trend is
    known rather than uncertain, so counting it as variance double-counts it. The
    ``(2 ln 2 - 1)`` coefficient de-biases what remains.

    **The per-bar term is signed, and legitimately goes negative.** Whenever the
    open-to-close move exceeds ``0.879 * ln(H/L)`` — a gap that dwarfs the
    session's range is the ordinary way this happens on an equity index — the
    subtraction outweighs the range and the term is negative. Averaged over a
    window the sum can still be negative, and the square root of a negative
    variance is a NaN sitting in the interior of the series. ``_sqrt_variance``
    floors it at zero, so a window whose net estimate is "less than no variance"
    reports zero rather than poisoning everything downstream. The estimator is
    the more accurate one *because* it subtracts the drift; that same accuracy is
    what lets the intermediate quantity go negative.

    ponytail: the clamp reports 0.0 rather than dropping the bar or widening the
    window. Both alternatives were worse for the reason ``indicators/__init__.py``
    gives for the NaN rule — they either invent information or break the
    same-length contract. The honest upgrade path is a drift term in the
    numerator, which is a different estimator with a different name.
    """
    high, low = as_float(data["high"]), as_float(data["low"])
    open_, close = as_float(data["open"]), as_float(data["close"])
    span = max(int(period), 2)
    log_range = log_ratio(high, low)
    log_body = log_ratio(close, open_)
    per_bar = 0.5 * log_range * log_range - _GK_COEF * log_body * log_body
    return {
        "volatility": _sqrt_variance(rolling_mean(per_bar, span), annualise),
    }


@indicator(
    "IND_VOL_YANG_ZHANG",
    group="volatility",
    lines=("volatility",),
    params={"period": P(20, 2, 500), "annualise": P(252, 1, 4000)},
    requires=("high", "low", "open", "close"),
    synonyms=(
        "yang-zhang",
        "yang zhang",
        "yang-zhang volatility",
        "yangzhang",
    ),
)
def yang_zhang_volatility(
    data: Data, *, period: int = 20, annualise: int = 252
) -> dict:
    """Yang-Zhang: three variance components, weighted by their own sampling error.

    The overnight gap, the open-to-close move and the intraday range are separate
    sources of information with separate noise, and summing them unweighted
    double-counts the intraday range — which contains both the body and the
    extremes. So the three are combined as::

        k * var(open->close) + (1 - k) * rogers_satchell
        + var(overnight)

    with ``k = 0.34 / (1.34 + (N+1)/(N-1))``, which rises with the window length
    because a longer window shrinks the open-to-close variance's sampling error
    and makes it the more trustworthy of the two intraday terms. Rogers-Satchell
    (``ln(H/C)ln(H/O) + ln(L/C)ln(L/O)``) is the component that needs no open-to-
    close correction, because it uses the close as the pivot rather than as a
    measured move.

    Yang & Zhang's result is that this is the **upper bound** on the other
    estimators — it is never expected to read below Garman-Klass on the same
    window, and the test suite asserts exactly that.

    Two things follow from the definition and are easy to get wrong:

    **The warmup is one bar longer than the others.** A window of ``N`` overnight
    returns spans ``N+1`` closes, so the first finite bar is ``N`` and not
    ``N-1``. Bar 0 gets a genuine NaN rather than a fill of ``close[0]``: that
    fill would inject a fake ``ln(O_0/C_0) = 0`` overnight return, silently
    shortening the warmup by a bar and biasing the first real window low.

    **``period`` cannot be 1.** The ``k`` formula divides by ``N - 1``, so the
    published parameter floor of 2 is load-bearing, not cosmetic.

    ponytail: no drift/mean-adjustment on the open-to-close term. Yang-Zhang
    without a drift estimate is the standard published form and is what the
    literature's upper-bound result is stated for; adding one would make the
    bound untestable against the reference. The upgrade path is a mean term
    parameterised by the window's own return.
    """
    high, low = as_float(data["high"]), as_float(data["low"])
    open_, close = as_float(data["open"]), as_float(data["close"])
    span = max(int(period), 2)
    n = float(span)

    # `shift` with no fill leaves bar 0 NaN: it has no prior close, so it has no
    # overnight return. See the warmup note above.
    previous_close = shift(close, 1)
    overnight = log_ratio(open_, previous_close)
    open_to_close = log_ratio(close, open_)

    # Rogers-Satchell: the high and low each measured against both the close and
    # the open, so the body is the pivot and nothing is double-counted.
    high_close, high_open = log_ratio(high, close), log_ratio(high, open_)
    low_close, low_open = log_ratio(low, close), log_ratio(low, open_)
    rogers_satchell = high_close * high_open + low_close * low_open

    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    variance = (
        rolling_std(overnight, span) ** 2
        + k * rolling_std(open_to_close, span) ** 2
        + (1.0 - k) * rolling_mean(rogers_satchell, span)
    )
    return {"volatility": _sqrt_variance(variance, annualise)}


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
    "garman_klass_volatility",
    "historical_volatility",
    "parkinson_volatility",
    "yang_zhang_volatility",
]
