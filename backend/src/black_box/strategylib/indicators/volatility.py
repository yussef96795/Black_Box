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

**The conditional volatility family** (``IND_GARCH_11``, ``IND_GJR_GARCH_11``,
``IND_EGARCH_11``) is a different kind of object from everything else in this
file, and the file says so rather than blurring it. Every other indicator is a
pure trailing transform: fixed parameters, no state, a value at bar *t* that
depends on bars up to *t* and nothing else. An ARCH model is a *fitted* object —
someone had to estimate five numbers from data before any volatility can be
reported — so the split is:

* The three **indicators** are the recursion only, with the coefficients as
  declared parameters. They are causal, vectorised and free of any optimiser, and
  they pass the same generic contract as the other 52 indicators.
* **:func:`fit_garch`** does the estimation and is deliberately **not
  registered** — not an indicator, no feed id, absent from
  ``primitives_registry.json``. It returns a :class:`GarchFit` carrying the
  parameters and the persistence, and :func:`forecast` turns that into a
  projection. Fitted coefficients are an *input* to the indicators, not an output
  of them, which is what keeps the causality guarantee in ``indicators/__init__``
  true for everything in the registry.

One convention is load-bearing throughout: **ATR is Wilder's RMA, not an EMA.**
TradingView's ``ta.atr`` uses alpha = 1/length, so at length 14 that is 0.071
against the EMA's 0.133, and the two settle at visibly different levels. Every
ATR-based indicator here uses ``rma``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as host_np  # see the note below — the ARCH half is host-side

from black_box.strategylib._backend import np
from black_box.strategylib._kernels import running_max, running_min
from black_box.strategylib._math import (
    as_float,
    log_ratio,
    rma,
    rolling_mean,
    rolling_std,
    shift,
    to_device,
    to_host,
    true_range,
)
from black_box.strategylib.indicators._registry import Data, P, indicator

#: ``host_np`` is NumPy, unconditionally, while the module-level ``np`` is the
#: backend handle (CuPy on GPU, NumPy on the CPU fallback). Both names exist in
#: this one file because the ARCH recursions are *measurably* faster on the host:
#: a single conditional-variance recursion is a serial dependency chain, so a
#: one-thread GPU scan cannot hide memory latency and loses to ``lfilter`` by
#: 12.6x over 5000 bars. Every other indicator in this file is device-side and
#: uses ``np``. ponytail: two array names in one module is a wart bought with a
#: 12.6x measured win, and the upgrade path is a batched ``ar1`` scan in
#: ``_kernels.py`` — which only pays once a Block C sweep exists to fill its
#: second dimension, which is not the case today.

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


# ---------------------------------------------------------------------------
# Conditional volatility — the ARCH family
#
# Three registered indicators and one unregistered fitter. The split is the whole
# design: the recursions are pure trailing transforms with fixed coefficients, so
# they are causal and pass the generic contract; the estimation is a separate
# object that carries its own convergence reporting and publishes nothing to the
# registry. See `fit_garch` for the fitting half.
#
# The recursions run on the host and come back as a device array. That is the
# measured choice, not a default: a single ARCH recursion is a serial dependency
# chain, so a one-thread GPU scan cannot hide memory latency and loses to
# `scipy.signal.lfilter` by 12.6x (0.594 ms vs 0.047 ms over 5000 bars, measured).
# The two device transfers cost 0.139 ms against a 0.047 ms recursion, so the
# round trip alone already exceeds the whole computation.
# ---------------------------------------------------------------------------


@indicator(
    "IND_GARCH_11",
    group="volatility",
    lines=("conditional_volatility",),
    params={
        "omega": P(1e-6, 0.0, 1.0),
        "alpha": P(0.1, 0.0, 1.0),
        "beta": P(0.88, 0.0, 1.0),
        "annualise": P(252, 1, 4000),
        "burn_in": P(500, 100, 20000),
    },
    synonyms=("garch", "garch(1,1)", "garch 1,1", "arch model"),
)
def garch_11(
    data: Data,
    *,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.88,
    annualise: int = 252,
    burn_in: int = 500,
) -> dict:
    """GARCH(1,1): today's variance is a fixed share of yesterday's shock and state.

    ``sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}``. The two
    numbers that matter are ``alpha + beta``, the **persistence**, and what
    ``omega / (1 - beta)`` implies: unconditional variance grows without limit as
    persistence approaches one, so a near-unit-root fit is a statement that the
    instrument has become more volatile than any finite number, not a small
    parameter tweak.

    Symmetric by construction — a 5% loss and a 5% gain move the forecast
    identically. That is the assumption GJR-GARCH below exists to relax, and on
    crypto it is the assumption most likely to be wrong: liquidation cascades are
    one-directional, and a symmetric model reads that asymmetry as ordinary noise.

    The recursion is run in the deviation form ``h - E[h] = a*(e^2) + b*(h - E[h])``
    where ``E[h] = omega/(1-beta)``, which is what makes it a plain first-order
    linear recurrence that ``lfilter`` solves in C.

    ``burn_in`` is a seed-and-settle length, not a window: the recursion starts
    from the sample variance of the first ``burn_in`` returns and the first
    ``burn_in + 1`` bars are NaN. It must comfortably exceed ``1/(1 - beta)`` or
    the early output is mostly the seed talking.

    ponytail: ``alpha`` and ``beta`` are caller-supplied rather than fitted here,
    which is what keeps this indicator causal and lets it share the generic
    contract with the other 54. Estimate them with :func:`fit_garch`. The
    upgrade path is not a `fit=True` flag — that would make the causality
    guarantee conditional on a parameter, and a guarantee that can be switched off
    is not one.
    """
    _check_stationarity("garch", omega, alpha, beta)
    returns, start, seed = _arch_returns(data, burn_in)
    variance = _garch_variance(returns, omega, alpha, beta, seed)
    return {"conditional_volatility": _finish_arch(variance, start, annualise)}


@indicator(
    "IND_GJR_GARCH_11",
    group="volatility",
    lines=("conditional_volatility",),
    params={
        "omega": P(1e-6, 0.0, 1.0),
        "alpha": P(0.05, 0.0, 1.0),
        "beta": P(0.85, 0.0, 1.0),
        "gamma": P(0.1, 0.0, 1.0),
        "annualise": P(252, 1, 4000),
        "burn_in": P(500, 100, 20000),
    },
    synonyms=("gjr-garch", "gjr garch", "gjr-garch(1,1)", "leverage garch"),
)
def gjr_garch_11(
    data: Data,
    *,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.85,
    gamma: float = 0.1,
    annualise: int = 252,
    burn_in: int = 500,
) -> dict:
    """GJR-GARCH: the shock coefficient switches sign of the return, not just its size.

    ``sigma^2_t = omega + (alpha + gamma * 1{eps_{t-1} < 0}) * eps^2_{t-1}
    + beta * sigma^2_{t-1}``. Squared returns destroy the sign, so GARCH cannot see
    that a 10% drop and a 10% gain are different events; ``gamma`` restores it and
    is the **leverage effect** — the reason a negative shock is expected to move
    variance further than a positive one of the same magnitude.

    Two persistence numbers, and they are not interchangeable. ``alpha + beta`` is
    what a *positive* shock implies; ``alpha + beta + gamma/2`` is the
    unconditional one, and the stationarity bound is on the second, because a
    symmetric innovation puts probability 1/2 on the branch that carries ``gamma``.
    :func:`fit_garch` reports both. Using ``alpha + beta`` as a stationarity check
    is a real and common error: it understates persistence by exactly ``gamma/2``.

    The recursion is genuinely sequential — the coefficient on each bar's own
    squared return depends on that bar's sign — so unlike GARCH it cannot be
    handed to ``lfilter``.

    ponytail: the switch is a hard threshold on the sign, not a smoothed function
    of the return. The threshold is what makes it non-linear and un-vectorisable;
    a smooth alternative (Glosten-Jagannathan-Runkle with a linear ramp, or the
    threshold ARCH of Zakoian) is a different model that also happens to lose the
    lfilter shortcut. The upgrade path is a new id.
    """
    _check_stationarity("gjr", omega, alpha, beta, gamma)
    returns, start, seed = _arch_returns(data, burn_in)
    variance = _gjr_variance(returns, omega, alpha, beta, gamma, seed)
    return {"conditional_volatility": _finish_arch(variance, start, annualise)}


@indicator(
    "IND_EGARCH_11",
    group="volatility",
    lines=("conditional_volatility",),
    params={
        "omega": P(0.0, -500.0, 100.0),
        "alpha": P(0.05, -5.0, 5.0),
        "gamma": P(0.1, -5.0, 5.0),
        "beta": P(-0.6, -0.999, 0.0),
        "annualise": P(252, 1, 4000),
        "burn_in": P(500, 100, 20000),
    },
    synonyms=("egarch", "egarch(1,1)", "egarch 1,1", "exponential garch"),
)
def egarch_11(
    data: Data,
    *,
    omega: float = 0.0,
    alpha: float = 0.05,
    gamma: float = 0.1,
    beta: float = -0.6,
    annualise: int = 252,
    burn_in: int = 500,
) -> dict:
    """EGARCH: model the log of the variance, so non-negativity is structural.

    ``log(sigma^2_t) = omega + alpha*z_{t-1} + gamma*(|z_{t-1}| - E|z|)
    + beta*log(sigma^2_{t-1})`` with ``z = eps / sigma``.

    Three things change at once against GARCH, and all three are consequences of
    taking the log:

    * **The variance cannot go negative**, not because of a clamp bolted on
      afterwards but because the state *is* a log. This is the model to reach for
      when a fit on a liquidation cascade drives a linear variance recursion
      through zero; the log has no such floor to fall through.
    * **Coefficients are sensitivities, not shares.** ``|beta| < 1`` is all the
      persistence claim there is — the familiar ``alpha + beta < 1`` test does not
      apply and is not checked here.
    * **The stationarity condition is ``beta < 0``, a sign constraint** (Nelson
      1991), which is why ``beta``'s published bounds are
      ``P(-0.9, -0.999, 0.0)`` rather than a non-negative box. A positive ``beta``
      makes ``E[log sigma^2]`` diverge, so the parameter is excluded rather than
      merely discouraged.

    ``E[z] = 0`` and ``E|z| = sqrt(2/pi)`` under the Gaussian likelihood the fitter
    assumes, so the centring constants are analytic rather than sampled.

    The recursion is sequential even though the *state* is linear in logs, because
    ``z_{t-1}`` divides by the conditional sigma that the same recursion is
    producing. That closes the loop and defeats ``lfilter``.

    **``omega = 0`` means "take the level from the data", not "the level is 0".**
    EGARCH's ``omega`` is a log-variance intercept, so it is inseparable from the
    return scale: ``omega = -18`` is a sensible 13% annualized volatility on daily
    data and an absurd one on monthly, and a published constant cannot be right for
    both. Zero is reserved as the sentinel because it is the one value that cannot
    be a real intercept — it implies an unconditional *variance* of ``exp(0) = 1``,
    a per-bar sigma of 100% — so nothing legitimate is lost by claiming it. Any
    other value is used as given, which is what makes ``{**fit.params}`` from
    :func:`fit_garch` work unchanged.

    **The published ``alpha`` and ``gamma`` are deliberately unequal, and the reason
    is arithmetic rather than taste.** For a negative shock the whole innovation
    term collapses to ``(gamma - alpha)*|z| - gamma*E|z|``, so when ``alpha``
    equals ``gamma`` the ``|z|`` coefficient is *exactly zero* and EGARCH stops
    responding to the size of a negative return — it reads only its sign. That is
    not a weakly-parameterised EGARCH, it is a sign-switching model wearing
    EGARCH's name, and it is invisible in the output: the line still looks like a
    conditional volatility, just one blind to leverage. ``alpha = 0.05`` against
    ``gamma = 0.10`` keeps a real asymmetry while staying stable.

    **EGARCH has a numerical cliff, and these defaults stand well back from it.**
    The log state feeds back into ``z`` through ``1/sigma``, so a state that has
    drifted from the data's own volatility produces a residual several times
    normal, which overshoots the state the other way, and the pair diverges. What
    governs it is roughly ``|beta| + (alpha + gamma) * E|z| / 2`` — a
    *responsiveness* budget, since a model that cannot react to a volatility change
    is not doing its job, but every unit spent on reacting is a unit closer to the
    cliff. Triggering it takes a volatility jump of roughly 4x or more in one bar
    from a calm seed; a gradual regime drift, natural clustering and fat tails all
    pass comfortably. Measured over 30 seeds each of GARCH-simulated returns,
    Student-t(3) tails, gradual 4x ramps and instantaneous 2x/3x/4x steps, the
    published triple escapes on none of them, across volatility scales from 0.05x
    to 20x. The first values tried, ``(0.15, 0.20, -0.88)``, escaped on ordinary
    GARCH data with no regime change at all.

    Note this is *not* the same failure as non-stationarity. ``beta = -0.95`` passes
    ``beta < 0`` comfortably and still escapes, because the ``1/sigma`` feedback is
    a second, independent instability. Crossing the cliff raises rather than
    returning a large finite number, because a clamped escape reads as a result —
    see ``_LOG_STATE_LIMIT``.

    ponytail: unconditional variance is ``exp(omega / (1 - beta))``, so ``omega``
    is an intercept in log space and a handful of percent in variance space — it is
    not the ``omega`` of GARCH and the two must not be compared directly. The
    defaults here are deliberately *less* persistent than a fitted EGARCH usually
    is, for the reason above: :func:`fit_garch` is the way to get a responsive
    model, and it starts from these same conservative coefficients and walks out
    toward the data. A caller wanting a fast-responding EGARCH against live
    defaults needs a floored sigma, which is a different model and gets a new id.
    The upgrade path is the Nelson (1993) asymmetric specification with a second
    standardized-return term.
    """
    _check_stationarity("egarch", omega, alpha, gamma, beta)
    returns, start, seed = _arch_returns(data, burn_in)
    # Same derivation as `fit_garch`'s start point, for the same reason: the one
    # coefficient with no scale-free default. Uses the burn-in variance, which is
    # already computed and is strictly causal.
    if omega == 0.0:
        omega = math.log(max(seed, _SEED_FLOOR)) * (1.0 - beta)
    log_variance = _egarch_log_variance(returns, omega, alpha, gamma, beta, seed)
    # host_np, not the backend np: the recursion above returned a host array, and
    # cupy's exp rejects one. `_finish_arch` is what moves it back to the device.
    return {
        "conditional_volatility": _finish_arch(
            host_np.exp(log_variance), start, annualise
        )
    }


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


# ---------------------------------------------------------------------------
# ARCH engine — recursions, stationarity, and the fitter
#
# Everything below is host-side NumPy. The recursions are the *same functions*
# the indicators and the fitter both call, which is the point: a fitting routine
# with its own private copy of the GARCH recursion is how a library ends up
# reporting a conditional volatility the indicator never reproduces.
# ---------------------------------------------------------------------------

#: ``E[|z|]`` for a standard normal, the EGARCH centring constant. Used instead
#: of a sample moment because the likelihood is Gaussian and the analytic value
#: is the one the model is defined against.
_STD_ABS = math.sqrt(2.0 / math.pi)

#: Floor on a seed variance before it is logged by EGARCH. Not a clamp on the
#: output — EGARCH's state is a log so it cannot go negative — only a guard on the
#: ``log(seed)`` on the very first bar, where a degenerate input (a flat series)
#: would otherwise produce ``-inf`` and seed the whole recursion with it.
_SEED_FLOOR = 1e-18

#: Bound on EGARCH's log-variance state, checked on every bar. Past it the
#: recursion has numerically escaped rather than merely become extreme: the state
#: feeds ``z = eps / exp(state/2)``, so once the state is far enough from zero the
#: residual explodes, and the resulting jump overshoots into a two-cycle that
#: grows without bound. ``+/-350`` is sigma ~ e^+-175, some seventy orders of
#: magnitude beyond any real volatility, so nothing legitimate is caught while
#: leaving room for the widest published ``omega`` (-100, which equilibrates near
#: -53) to sit well inside the bound.
#:
#: ponytail: checked and raised rather than clamped, because a clamped escape
#: returns a large finite number that reads as a result. Raising says the truth.
#: The upgrade path, if a caller genuinely wants a floored sigma, is an explicit
#: ``sigma_floor`` parameter — which is a different model and gets a new id.
_LOG_STATE_LIMIT = 350.0


def _LOG_STATE_OK(state: float) -> bool:
    """True while an EGARCH log-variance state is inside its usable range."""
    return -_LOG_STATE_LIMIT < state < _LOG_STATE_LIMIT


#: Objective value returned for a parameter set that cannot be evaluated. Large
#: enough that no finite likelihood beats it, finite so L-BFGS-B keeps a descent
#: direction instead of seeing a NaN and bailing.
_PENALTY = 1e10


@cache
def _scipy() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """``(lfilter, minimize)``, imported on first use and cached for the process.

    ponytail: lazy, because ``import scipy.signal`` costs 1.54 s on this box and
    ``indicators/__init__`` imports every indicator module to build the registry —
    a top-level import would tax every consumer and every test run to serve one
    estimator in 55. Only GARCH reaches for ``lfilter``; GJR and EGARCH are
    sequential and loop in Python instead, and only the fitter wants ``minimize``.
    The upgrade path, if the import ever does become the bottleneck, is a compiled
    scan in ``_kernels.py`` rather than an eager import.
    """
    from scipy.optimize import minimize
    from scipy.signal import lfilter

    return lfilter, minimize


def _check_stationarity(kind: str, *coeffs: float) -> None:
    """Reject a coefficient set whose variance does not come back to earth.

    ``bind_params`` bounds parameters to a box, and every stationarity condition
    in this family is *not* a box — it is a half-plane the box cannot express. So
    it is checked here, and it raises rather than warns: an explosive parameter
    set produces ``inf`` partway down the series, and ``indicators/__init__.py``
    calls that a contract violation. A caller who wants an explosive model is
    doing something they can do knowingly, in their own code.

    Coefficients are positional in each model's own formula order, so the call
    sites read like the equation rather than like a dictionary lookup.
    """
    if kind == "egarch":
        _, _, _, beta = coeffs
        if not beta < 0.0:
            raise ValueError(
                f"EGARCH needs beta < 0 for E[log sigma^2] to be finite "
                f"(Nelson 1991), got beta={beta}. A non-negative beta makes the "
                f"log-variance level drift upward without limit."
            )
        return
    omega, alpha, beta = coeffs[:3]
    gamma = coeffs[3] if kind == "gjr" else 0.0
    if omega <= 0.0:
        raise ValueError(f"{kind}: omega must be > 0, got {omega}")
    if min(alpha, beta, gamma) < 0.0:
        raise ValueError(
            f"{kind}: alpha, beta and gamma must be >= 0, got "
            f"alpha={alpha}, beta={beta}, gamma={gamma}"
        )
    # alpha + beta + gamma/2, not alpha + beta: a symmetric innovation puts half
    # its probability on the branch that carries gamma, so the unconditional
    # persistence is the larger number and that is the one that must stay below 1.
    persistence = alpha + beta + 0.5 * gamma
    if persistence >= 1.0:
        raise ValueError(
            f"{kind}: persistence alpha + beta + gamma/2 = {persistence:.6f} must be "
            f"< 1 for the variance to be covariance-stationary. omega/(1-persistence) "
            f"is the unconditional variance and it diverges as persistence -> 1."
        )


def _arch_returns(data: Data, burn_in: int) -> tuple[Any, int, float]:
    """``(returns, start_index, seed_variance)`` for the ARCH recursions.

    The seed is the sample variance of the first ``burn_in`` returns, which is
    why this is causal: it looks at nothing but the bars it is about to report as
    NaN. Returns an empty tail when the input is shorter than the burn-in, which
    yields an all-NaN line — the same thing a ``period=500`` window does on 400
    bars, and the contract permits it.
    """
    close = as_float(data["close"])
    returns = host_np.asarray(to_host(log_ratio(close, shift(close, 1))), dtype=float)
    burn_in = max(int(burn_in), 2)
    start = burn_in + 1
    if start >= len(returns):
        return returns[:0], len(returns), float("nan")
    seed = float(host_np.var(returns[1 : burn_in + 1]))
    return returns[start:], start, seed


def _finish_arch(values: Any, start: int, annualise: int) -> Any:
    """Prepend the NaN warmup, take the square root, annualise, move to device.

    The two device transfers are the entire cost of the host recursion and are
    cheaper than the recursion itself (0.139 ms against 0.047 ms for 5000 bars,
    measured), which is the whole justification for the host/device split.
    """
    out = host_np.full(start + len(values), host_np.nan, dtype=float)
    if len(values):
        out[start:] = values
    out = host_np.sqrt(host_np.maximum(out, 0.0)) * float(annualise) ** 0.5
    return to_device(out)


def _ar1(shocks: Any, beta: float, initial: float) -> Any:
    """``y[0] = initial; y[n] = shocks[n-1] + beta*y[n-1]``, solved in C.

    ``lfilter`` gives the recursion seeded at zero, so the initial value is added
    back as an explicitly decayed term: ``y[n] = z[n] + beta**n * initial``. That
    is the exact solution of the same difference equation, not an approximation,
    and it is what lets the GARCH variance recursion be a one-line filter instead
    of a Python loop.
    """
    m = len(shocks) + 1
    y = host_np.empty(m, dtype=float)
    y[0] = initial
    if m > 1:
        lfilter, _ = _scipy()
        y[1:] = lfilter([1.0], [1.0, -beta], shocks) + (
            host_np.power(beta, host_np.arange(1, m, dtype=float)) * initial
        )
    return y


def _garch_variance(returns: Any, omega: float, alpha: float, beta: float, seed: float):
    """``sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}``.

    Run in the deviation form ``h - m = alpha*e^2 + beta*(h - m)`` with
    ``m = omega/(1-beta)``, which is what makes the shock exogenous and the whole
    thing a first-order linear recurrence.

    Contract shared by all three recursions: **one variance per return, no more and
    no fewer.** The empty case needs stating because ``_ar1`` is off by one against
    it — an empty shock list still has a seed bar, so it hands back one value where
    this function owes zero. Without the guard the emitted line is one bar longer
    than its input, which the generic length contract catches.
    """
    if not len(returns):
        return host_np.empty(0, dtype=float)
    unconditional = omega / max(1.0 - beta, 1e-12)
    squares = returns * returns
    return _ar1(alpha * squares[:-1], beta, seed - unconditional) + unconditional


def _gjr_variance(
    returns: Any,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    seed: float,
):
    """GJR-GARCH: the shock coefficient switches on the *sign* of the prior return.

    Squared returns are sign-blind, so this is the one recursion in the family
    that cannot be handed to a linear filter — the coefficient on each bar's own
    term is data. A Python loop over the tail is the honest cost; on 5000 bars it
    is ~1.9 ms, against 0.047 ms for the linear case.
    """
    variance = host_np.empty(len(returns), dtype=float)
    if not len(returns):
        return variance
    variance[0] = seed
    state = seed
    for t in range(1, len(returns)):
        shock = returns[t - 1]
        coefficient = alpha + gamma if shock < 0.0 else alpha
        state = omega + coefficient * shock * shock + beta * state
        variance[t] = state
    return variance


def _egarch_log_variance(
    returns: Any,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: float,
    mean_abs: float = _STD_ABS,
):
    """EGARCH's log-variance recursion — the *state* is a log, hence always finite.

    Sequential despite the state being linear in logs, because ``z_{t-1}`` divides
    by the conditional sigma that this same recursion produces. That closes the
    loop and rules out ``lfilter``.
    """
    out = host_np.empty(len(returns), dtype=float)
    if not len(returns):
        return out
    state = math.log(max(seed, _SEED_FLOOR))
    if not _LOG_STATE_OK(state):
        raise ValueError(
            f"EGARCH seed variance {seed!r} puts ln sigma^2 at {state:.3g}, outside "
            f"+/-{_LOG_STATE_LIMIT:.0f}. The recursion starts from an unusable state."
        )
    out[0] = state
    for t in range(1, len(returns)):
        sigma = math.exp(0.5 * state)
        z = returns[t - 1] / sigma
        state = omega + beta * state + alpha * z + gamma * (abs(z) - mean_abs)
        if not _LOG_STATE_OK(state):
            raise ValueError(
                f"EGARCH recursion diverged at bar {t}: ln sigma^2 left "
                f"+/-{_LOG_STATE_LIMIT:.0f} with omega={omega}, alpha={alpha}, "
                f"gamma={gamma}, beta={beta}. This is EGARCH's known numerical "
                f"instability at high |beta|, not a stationarity violation — see "
                f"the module note in `egarch_11`."
            )
        out[t] = state
    return out


@dataclass(frozen=True)
class GarchFit:
    """One estimated ARCH model, and everything needed to judge the estimate.

    Frozen because a fit that can be edited after the fact is a fit nobody can
    cite: the parameters, the log-likelihood and the convergence flag all describe
    the *same* optimisation run and have to move together or not at all.

    ``converged`` is the field to check first. It is False for a failed
    optimisation, a non-finite likelihood, **and** a parameter sitting on its
    bound — the last one is not a failure the message would otherwise admit to,
    but a coefficient pinned at 0 or 1 is the optimiser declining to move rather
    than a genuine estimate.
    """

    model: str
    params: dict[str, float]
    persistence: float
    unconditional_vol: float
    loglik: float
    converged: bool
    message: str
    n_obs: int
    last_return: float
    last_variance: float
    annualise: int


def fit_garch(
    returns: Any,
    model: str = "garch",
    *,
    burn_in: int = 500,
    maxiter: int = 500,
    annualise: int = 252,
    x0: tuple[float, ...] | None = None,
) -> GarchFit:
    """Maximum-likelihood ARCH parameters by Gaussian quasi-likelihood.

    **Not an indicator, and deliberately not registered.** No feed id, no entry in
    ``primitives_registry.json``, nothing the AST interpreter can address. The
    reason is causality: parameters estimated from the whole sample make bar 5's
    value depend on bar 5000's return, and a full-sample fit republished as a
    feed id would put a lookahead inside the registry's central guarantee. The
    coefficients are an *input* to the indicators — ``fit_garch`` hands you a dict,
    you pass it to ``garch_11`` — which keeps the estimation and the recursion
    separable and each one honest about what it is.

    Solved with L-BFGS-B on the Gaussian negative log-likelihood
    ``0.5 * sum(log h + e^2/h)``, which is also a quasi-likelihood when the
    innovations are not actually Gaussian — the usual case for crypto returns, and
    the reason ``gamma``/``alpha`` in the EGARCH fit are not literally the
    kurtosis and skewness they resemble.

    Args:
        returns: 1-D log returns, CuPy or NumPy. Non-finite values are dropped
            rather than propagated, since one NaN would poison the whole recursion.
        model: ``"garch"``, ``"gjr"`` or ``"egarch"``.
        burn_in: bars reserved for the recursion seed. Must be at least 100.
        maxiter: optimiser iteration cap.
        annualise: carried onto the result so :func:`forecast` needs no extra
            argument to report a number in the units a reader expects.
        x0: starting point, in the model's own coefficient order.

    Returns:
        A :class:`GarchFit`. **Check ``converged`` before using it** — a
        non-converged fit is returned rather than raised, because "the optimiser
        stopped early" is a result to report, not an exception to throw at a
        caller who may want to inspect it.

    Raises:
        ValueError: on an unknown model, or fewer than ``burn_in + 50`` usable
            observations. A sample that short cannot identify five coefficients
            and silently returning a fit would be worse than refusing.
    """
    minimize = _scipy()[1]
    if model not in ("garch", "gjr", "egarch"):
        raise ValueError(f"unknown ARCH model {model!r}, expected garch, gjr or egarch")

    # Accept a host array without paying a device round trip for it, and a device
    # array without the caller having to remember which they hold. `as_float`
    # insists on a backend array, so the host case has to branch ahead of it.
    r = (
        host_np.asarray(returns, dtype=float)
        if isinstance(returns, host_np.ndarray)
        else host_np.asarray(to_host(as_float(returns)), dtype=float)
    ).ravel()
    r = r[host_np.isfinite(r)]
    burn_in = max(int(burn_in), 100)
    if len(r) < burn_in + 50:
        raise ValueError(
            f"fit_garch needs at least burn_in + 50 = {burn_in + 50} observations to "
            f"identify the model, got {len(r)}"
        )
    seed, tail = float(host_np.var(r[:burn_in])), r[burn_in:]
    sample_var = float(host_np.var(tail))

    # omega carries the units of a variance, and for log returns that is ~1e-4 —
    # an order of magnitude L-BFGS-B cannot take a sensible first step in. It is
    # therefore estimated in units of the sample variance and converted back.
    # EGARCH's omega is a log-variance intercept and needs no such help.
    scaled = model != "egarch"

    def objective(theta: Any) -> float:
        omega, rest = (float(theta[0]) * sample_var, host_np.asarray(theta[1:]))
        if model == "garch":
            alpha, beta = rest
            if alpha + beta >= 1.0:
                return _PENALTY
            variance = _garch_variance(tail, omega, alpha, beta, seed)
        elif model == "gjr":
            alpha, beta, gamma = rest
            if alpha + beta + 0.5 * gamma >= 1.0:
                return _PENALTY
            variance = _gjr_variance(tail, omega, alpha, beta, gamma, seed)
        else:
            alpha, gamma, beta = rest
            if beta >= 0.0:
                return _PENALTY
            try:
                variance = host_np.exp(
                    _egarch_log_variance(tail, omega, alpha, gamma, beta, seed)
                )
            except ValueError:
                # A parameter set whose EGARCH recursion escapes has no likelihood
                # at all, which makes it inadmissible rather than merely bad.
                # Reporting that as the penalty is what lets the optimiser walk
                # back out of the divergent region instead of dying on a
                # traceback raised from inside scipy.
                return _PENALTY
        if not host_np.all(host_np.isfinite(variance)) or host_np.any(variance <= 0.0):
            return _PENALTY
        nll = 0.5 * float(
            host_np.sum(host_np.log(variance) + tail * tail / variance)
        ) + 0.5 * len(tail) * math.log(2.0 * math.pi)
        return nll if math.isfinite(nll) else _PENALTY

    # Coefficient names in optimiser order, used for the pinned-at-a-bound report
    # below. Note EGARCH's order is (omega, alpha, gamma, beta) — gamma before
    # beta, because in that model's formula gamma is the asymmetry term and beta
    # the persistence term, and the two must not be transposed by muscle memory
    # from the GARCH block.
    names = {
        "garch": ("omega", "alpha", "beta"),
        "gjr": ("omega", "alpha", "beta", "gamma"),
        "egarch": ("omega", "alpha", "gamma", "beta"),
    }
    bounds = {
        "garch": [(1e-8, 1.0), (0.0, 0.999), (0.0, 0.999)],
        "gjr": [(1e-8, 1.0), (0.0, 0.999), (0.0, 0.999), (0.0, 0.999)],
        "egarch": [(-500.0, 100.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, -1e-8)],
    }
    defaults = {
        # omega is in units of the sample variance for garch/gjr (see above), and
        # 0.1 is the value that makes omega = 0.1*s2 consistent with a persistence
        # of 0.9 — so the start point is already a coherent model, not a guess.
        "garch": (0.1, 0.10, 0.88),
        "gjr": (0.1, 0.05, 0.85, 0.10),
        # omega is 0 here and overwritten below — the same sentinel `egarch_11`
        # uses, for the same reason. alpha/gamma/beta sit well inside EGARCH's
        # numerical stability region rather than at the edge of it: a start point
        # that escapes the recursion evaluates to the penalty along with all its
        # neighbours, and the optimiser then terminates where it started without
        # ever moving. alpha and gamma are unequal for the reason given in
        # `egarch_11` — equal values make the model blind to the size of a
        # negative return — and all three are more conservative than a converged
        # fit usually is, deliberately: the optimiser walks out from here toward
        # the data's own regime.
        "egarch": (0.0, 0.05, 0.10, -0.60),
    }
    start = host_np.asarray(
        x0 if x0 is not None else defaults[model], dtype=float
    ).copy()
    if x0 is None and model == "egarch":
        # omega = ln(sample_var) * (1 - beta) is the one value that makes the
        # unconditional variance `exp(omega/(1-beta))` equal the sample variance,
        # so the recursion starts level with the data instead of having to walk to
        # it. Deriving it beats publishing a constant: EGARCH's omega is a
        # log-variance intercept, so a fixed number is calibrated to one
        # instrument's return scale and meaningless on another's. A caller who
        # passed x0 means it, so this only fills in a default.
        start[0] = math.log(max(sample_var, 1e-18)) * (1.0 - start[3])
    elif scaled:
        start[0] = start[0] / sample_var if sample_var > 0.0 else start[0]

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds[model],
        options={"maxiter": int(maxiter), "maxfun": int(maxiter) * 2},
    )

    theta = host_np.asarray(result.x, dtype=float)
    omega = float(theta[0]) * sample_var if scaled else float(theta[0])
    rest = [float(v) for v in theta[1:]]
    if model == "garch":
        alpha, beta = rest
        params = {"omega": omega, "alpha": alpha, "beta": beta}
        variance = _garch_variance(tail, omega, alpha, beta, seed)
        persistence = alpha + beta
        unconditional_var = omega / max(1.0 - beta, 1e-12)
    elif model == "gjr":
        alpha, beta, gamma = rest
        params = {"omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma}
        variance = _gjr_variance(tail, omega, alpha, beta, gamma, seed)
        persistence = alpha + beta + 0.5 * gamma
        unconditional_var = omega / max(1.0 - persistence, 1e-12)
    else:
        alpha, gamma, beta = rest
        params = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}
        # The optimiser can *return* a divergent EGARCH set — it ran out of
        # iterations rather than finding the escape. Report that as a fit with no
        # usable variance rather than raising out of a function whose whole
        # contract is to return a GarchFit.
        try:
            variance = host_np.exp(
                _egarch_log_variance(tail, omega, alpha, gamma, beta, seed)
            )
        except ValueError:
            variance = host_np.empty(0)
        # Nelson (1991): for log-variance, the persistence is |beta| alone. There
        # is no alpha + beta test here — the model has no such sum.
        persistence = abs(beta)
        unconditional_var = math.exp(omega / max(1.0 - beta, 1e-12))

    # A parameter resting on its bound is the optimiser declining to move, not an
    # estimate. Flagged rather than accepted, because a "fitted" alpha of exactly
    # 0.0 reads as a finding and is really a failure to find one. The tolerance is
    # relative to the width of the box it was allowed to move in, so a coefficient
    # that is legitimately small but nowhere near its bound is not caught.
    pinned = [
        name
        for name, value, (low, high) in zip(names[model], theta, bounds[model])
        if min(value - low, high - value) <= 1e-6 * (high - low)
    ]
    message = str(result.message)
    converged = bool(result.success) and math.isfinite(float(result.fun))
    if pinned:
        converged = False
        message = f"{message} (pinned at a bound: {', '.join(pinned)})"
    if float(result.fun) >= _PENALTY:
        converged = False
        message = f"{message} (no admissible parameter set found)"

    return GarchFit(
        model=model,
        params=params,
        persistence=float(persistence),
        unconditional_vol=math.sqrt(max(unconditional_var, 0.0)) * annualise**0.5,
        loglik=-float(result.fun),
        converged=converged,
        message=message,
        n_obs=len(tail),
        last_return=float(tail[-1]),
        last_variance=float(variance[-1]) if len(variance) else float("nan"),
        annualise=int(annualise),
    )


def forecast(fit: GarchFit, h: int = 1) -> Any:
    """Conditional-volatility forecast ``h`` steps ahead, annualised.

    Returns ``h + 1`` values, index 0 being the last *observed* conditional
    volatility. Including it is not decoration: a forecast is only interpretable
    against the level it steps away from, and a reader given a bare
    ``sigma_{T+1}`` has no way to tell a calm regime from a wild one.

    Beyond one step each model decays geometrically toward its unconditional
    variance, and the rate is model-specific in a way that matters: GARCH decays
    at ``alpha + beta``, GJR at ``alpha + beta + gamma/2`` (the *larger* number, so
    a leveraged model forgets its shock more slowly), and EGARCH at ``|beta|``.

    Not a feed id and not a line — see :func:`fit_garch` for why nothing fitted
    enters the registry.

    Raises:
        ValueError: if the fit carries no usable final variance, which is what an
            EGARCH estimate whose recursion escaped looks like. Raising beats
            returning an array of NaNs, because a NaN forecast and a NaN
            measurement look identical on a chart and mean opposite things.
    """
    h = max(int(h), 1)
    params = fit.params
    root = fit.annualise**0.5
    last = fit.last_variance
    if not math.isfinite(last) or last <= 0.0:
        raise ValueError(
            f"cannot forecast from a {fit.model!r} fit with no usable final "
            f"variance (last_variance={last!r}): {fit.message}"
        )

    if fit.model == "egarch":
        sigma = math.sqrt(max(last, 0.0))
        z = fit.last_return / sigma if sigma > 0.0 else 0.0
        first = math.exp(
            params["omega"]
            + params["beta"] * math.log(max(last, _SEED_FLOOR))
            + params["alpha"] * z
            + params["gamma"] * (abs(z) - _STD_ABS)
        )
        # Decays in LOG space, unlike the GARCH branch below. EGARCH's state is
        # ln(sigma^2), so the multi-step solution is
        # ln h_{T+j} = mu + beta**j * (ln h_{T+1} - mu). Applying `beta**j` to the
        # *variance* instead — the obvious transcription — is wrong twice over:
        # the level is wrong, and because beta is negative, beta**j alternates
        # sign and the forecast oscillates around the unconditional level forever
        # instead of converging to it.
        mu = params["omega"] / max(1.0 - params["beta"], 1e-12)
        gap = math.log(first) - mu
        path = [math.exp(mu + params["beta"] ** (j + 1) * gap) for j in range(h)]
    else:
        omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
        shock = fit.last_return * fit.last_return
        if fit.model == "gjr":
            gamma = params["gamma"]
            coefficient = alpha + gamma if fit.last_return < 0.0 else alpha
            decay = alpha + beta + 0.5 * gamma
        else:
            gamma = 0.0
            coefficient = alpha
            decay = alpha + beta
        first = omega + coefficient * shock + beta * last
        level = omega / max(1.0 - decay, 1e-12)
        path = [level + decay**j * (first - level) for j in range(h)]

    return host_np.sqrt(host_np.maximum(host_np.asarray([last, *path]), 0.0)) * root


__all__ = [
    "GarchFit",
    "atr",
    "bollinger",
    "chandelier",
    "egarch_11",
    "fit_garch",
    "forecast",
    "garch_11",
    "garman_klass_volatility",
    "gjr_garch_11",
    "historical_volatility",
    "parkinson_volatility",
    "yang_zhang_volatility",
]
