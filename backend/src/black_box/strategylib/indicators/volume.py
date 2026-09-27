"""Volume-weighted and volume-derived indicators.

Two families live here and they fail differently. The *cumulative* ones
(VWAP, OBV, the Accumulation/Distribution line, the Klinger and Money Flow
oscillators) are a running sum that only ever changes on the bar that moves it,
so their warmup is free and their value depends on the whole history. The
*windowed* ones (MFI, CMF, VWMA, the volume oscillators) rescale to the recent
bar, so they are responsive but say nothing about the longer accumulation.

The recurring trap in this family is the division by cumulative volume. VWAP is
``sum(price*volume) / sum(volume)``, and the denominator is a sum that is zero
until the first bar and can be zero on a series with no volume at all. Every such
division goes through ``safe_div`` — a raw divide turns a quiet instrument into
an infinite line and then a NaN line, and one NaN reaches every consumer of it.
"""

from __future__ import annotations

from black_box.strategylib._backend import np
from black_box.strategylib._math import (
    as_float,
    change,
    ema,
    median_price,
    rolling_mean,
    rolling_sum,
    safe_div,
    shift,
    typical_price,
)
from black_box.strategylib.indicators._registry import Data, P, indicator


@indicator(
    "IND_VWAP",
    group="volume",
    params={},
    synonyms=("vwap", "volume weighted average price", "volume weighted average"),
)
def vwap(data: Data) -> dict:
    """Volume Weighted Average Price — the day's fair price, volume-adjusted.

    Anchored to the first bar of the array. On an intraday chart that is the
    session open, which is the intended reading; across a multi-day array it is
    the start of the whole window, so a caller wanting per-session VWAP must pass
    one session at a time. There is deliberately no session-detection parameter:
    inferring session boundaries from bar count is a guess, and a wrong guess
    silently rescales every price on the chart.
    """
    close, volume = as_float(data["close"]), as_float(data["volume"])
    typical = (as_float(data["high"]) + as_float(data["low"]) + close) / 3.0
    return safe_div(np.cumsum(typical * volume), np.cumsum(volume), fill=typical[0])


@indicator(
    "IND_OBV",
    group="volume",
    params={},
    # ponytail: the trailing four are legacy names, not synonyms in the ordinary
    # sense. This indicator was called IND_VOLUME_DELTA in the pre-registry
    # Block A, where the name was misleading — it has always been OBV, i.e.
    # cumsum(volume * sign(change(close))), not a delta of volume. The rename
    # kept `_LEGACY_FEEDS` in `strategylib.synthesize` so checked-in AST
    # fixtures still resolve, but that alias is on the *interpreter* side: a
    # paper whose prose says "volume delta" is parsed by `detect_indicators`
    # against this table, and with these rows absent it silently matched nothing
    # and lost the indicator entirely. Synonyms are the one place a renamed
    # indicator can keep its old vocabulary, so the old phrases live here.
    synonyms=(
        "obv",
        "on balance volume",
        "on-balance volume",
        "volume delta",
        "delta volume",
        "cumulative delta",
        "order flow delta",
    ),
)
def obv(data: Data) -> dict:
    """On Balance Volume — volume accumulated by bar direction.

    Adds the volume on an up close, subtracts it on a down close, and ignores a
    flat one. Note what this is *not*: it is not a measure of buying pressure,
    because a large down bar on an exchange is as much a seller hitting a bid as
    a buyer lifting an offer. Direction here is inferred from price alone.
    """
    close, volume = as_float(data["close"]), as_float(data["volume"])
    return np.cumsum(np.sign(change(close)) * volume)


@indicator(
    "IND_AD",
    group="volume",
    params={},
    requires=("high", "low", "close", "volume"),
    synonyms=(
        "accumulation distribution",
        "accumulation/distribution",
        "a/d",
        "ad line",
    ),
)
def accumulation_distribution(data: Data) -> dict:
    """Accumulation/Distribution: a cumulative Chaikin-style money flow.

    Each bar contributes its money-flow volume ``CLV * volume``, where CLV is
    where the close sits inside that bar's own range. Cumulatively, the total
    measures the money that *could* have been accumulated or distributed without
    price moving — which is the claim, and also the limitation: a close near the
    high with heavy volume may equally be a short covering.
    """
    high, low, close = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
    )
    volume = as_float(data["volume"])
    span = high - low
    # A zero-range bar has no position within itself; the midpoint is the neutral
    # reading and keeps the bar from contributing a fabricated ±1.
    clv = np.where(
        span > 0, ((close - low) - (high - close)) / np.where(span > 0, span, 1.0), 0.0
    )
    return np.cumsum(clv * volume)


@indicator(
    "IND_VWMA",
    group="volume",
    params={"period": P(20, 1, 400)},
    synonyms=(
        "vwma",
        "volume weighted moving average",
        "volume weighted average price",
    ),
)
def vwma(data: Data, *, period: int = 20) -> dict:
    """Volume Weighted Moving Average — VWAP, but windowed.

    The windowed counterpart to :func:`vwap`: it answers "what did the average
    bar cost over the last N bars" rather than "over everything so far", which is
    the form that can cross another average.
    """
    close, volume = as_float(data["close"]), as_float(data["volume"])
    span = max(int(period), 1)
    return safe_div(
        rolling_sum(close * volume, span), rolling_sum(volume, span), fill=close[0]
    )


@indicator(
    "IND_MFI",
    group="volume",
    params={"period": P(14, 2, 200)},
    requires=("high", "low", "close", "volume"),
    synonyms=("mfi", "money flow index", "money flow"),
)
def money_flow_index(data: Data, *, period: int = 14) -> dict:
    """Money Flow Index — RSI's volume-weighted twin, 0-100.

    The flow is the typical price times volume, and the up/down split is on the
    *typical price* rather than the close (RSI splits on close changes). An
    unchanged typical price contributes to neither side, so the two sums can both
    be zero on a completely flat series — which maps to 50, the neutral reading.
    """
    high, low, close, volume = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
        as_float(data["volume"]),
    )
    span = max(int(period), 2)
    flow = typical_price(high, low, close) * volume
    delta = change(typical_price(high, low, close))
    positive = rolling_sum(np.where(delta > 0, flow, 0.0), span)
    negative = rolling_sum(np.where(delta < 0, flow, 0.0), span)
    # A neutral window (no up flow, no down flow) is 50, not 0 or 100.
    ratio = safe_div(positive, negative, fill=1.0)
    return np.where(positive + negative > 0, 100.0 - 100.0 / (1.0 + ratio), 50.0)


@indicator(
    "IND_CMF",
    group="volume",
    params={"period": P(20, 2, 200)},
    requires=("high", "low", "close", "volume"),
    synonyms=("cmf", "chaikin money flow", "money flow measure"),
)
def chaikin_money_flow(data: Data, *, period: int = 20) -> dict:
    """Chaikin Money Flow: the windowed A/D line, normalised to roughly -1..1.

    The sum of the money-flow volume over the window divided by the sum of the
    volume over the same window, so the line is scale-free and comparable across
    instruments. Unlike MFI this one keeps its sign — negative really does mean
    distribution, which is information MFI's 0-100 bound throws away.
    """
    high, low, close, volume = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
        as_float(data["volume"]),
    )
    span = max(int(period), 2)
    span_range = high - low
    clv = np.where(
        span_range > 0,
        ((close - low) - (high - close)) / np.where(span_range > 0, span_range, 1.0),
        0.0,
    )
    return safe_div(rolling_sum(clv * volume, span), rolling_sum(volume, span))


@indicator(
    "IND_KVO",
    group="volume",
    params={"fast": P(34, 1, 200), "slow": P(55, 2, 400)},
    requires=("high", "low", "close", "volume"),
    synonyms=("klinger", "klinger volume oscillator", "kvo"),
)
def klinger(data: Data, *, fast: int = 34, slow: int = 55) -> dict:
    """Klinger Volume Oscillator — trend force, measured through volume.

    The trend direction comes from the typical price's sign over a long window,
    the volume force from how far volume departs from its own long average, and
    the two are multiplied and smoothed. The sign convention is what matters:
    above zero is accumulation, below is distribution.

    ponytail: the "volume force" uses a simple ratio rather than a fitted trend
    amplitude, which is the textbook approximation rather than the original
    (volume-only) regression. The upgrade path, if the distinction ever shows up
    in a backtest, is fitting the force against the price change over the same
    window.
    """
    high, low, close, volume = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
        as_float(data["volume"]),
    )
    typical = typical_price(high, low, close)
    direction = np.sign(change(typical))
    baseline = rolling_mean(volume, max(int(slow), 2))
    force = volume - baseline
    return ema(direction * force, max(int(fast), 1))


@indicator(
    "IND_EOM",
    group="volume",
    params={"length": P(14, 2, 200), "divisor": P(100.0, 1.0, 1000.0)},
    requires=("high", "low", "volume"),
    synonyms=("ease of movement", "eom", "ease of movement index"),
)
def ease_of_movement(data: Data, *, length: int = 14, divisor: float = 100.0) -> dict:
    """Ease of Movement: how easily price moved, per unit of volume.

    One bar's contribution is the midpoint's change divided by the volume, so a
    large move on light volume scores high. The magnitude is a squared distance,
    which is what makes the sign carry the direction.

    The reported line is the **one-bar** value scaled by the divisor. Rolling it
    is a consumer's choice, and the raw form is what the indicator is defined as:
    smoothing it changes what the number means, and a one-bar series is trivially
    smoothable after the fact.
    """
    high, low, volume = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["volume"]),
    )
    span = max(int(length), 1)
    midpoint_change = change(median_price(high, low))
    # Volume is scaled by the window so the number is comparable across
    # instruments — the raw box volume depends on where the venue splits shares.
    scaled_volume = rolling_mean(volume, span)
    one_bar = safe_div(midpoint_change, scaled_volume, fill=0.0) * float(divisor)
    return np.sign(midpoint_change) * one_bar * one_bar


@indicator(
    "IND_VOLUME_OSC",
    group="volume",
    lines=("osc", "signal"),
    params={"fast": P(5, 1, 100), "slow": P(10, 2, 200)},
    synonyms=("volume oscillator", "volume osc", "volume oscillator signal"),
)
def volume_oscillator(data: Data, *, fast: int = 5, slow: int = 10) -> dict:
    """Volume Oscillator: the short and long volume averages, differenced.

    Reported as a *difference* of two raw volume sums rather than the more common
    percent form, because the percent form divides by a volume average that can
    be near zero on a thin instrument — the difference has no such denominator.
    """
    volume = as_float(data["volume"])
    quick = rolling_sum(volume, max(int(fast), 1))
    slow = rolling_sum(volume, max(int(slow), 2))
    osc = quick - slow
    return {"osc": osc, "signal": ema(osc, max(int(fast), 1))}


@indicator(
    "IND_ASI",
    group="volume",
    lines=("line", "signal", "osc"),
    params={"period": P(14, 1, 100)},
    requires=("high", "low", "close"),
    synonyms=("accumulation swing index", "asi", "swing index"),
)
def accumulation_swing_index(data: Data, *, period: int = 14) -> dict:
    """Accumulation Swing Index — Wilder's true-range oscillator, needs no volume.

    The raw swing index per bar is the two-bar span in the direction of the move:
    ``high - high[t-1]`` on an up close, ``low[t-1] - low`` on a down close, and 0
    on an unchanged one. Both branches are **positive** — SI measures how far price
    swung, not which way — so the direction lives in the ``osc`` line, which is
    this line minus its own signal. Reading the sign of ``line`` as a trend
    signal is the standard mistake.

    ponytail: this is the classic single-parameter ASI, not the multi-weight
    variant on the charting platforms (``short``/``long`` windows with weights,
    built on a true range divided by ``1 + close/period``). The classic form is
    citable and has no free parameters to disagree about; the upgrade path, if a
    backtest ever wants the platform's exact reading, is that variant's extra
    window and weight parameters added here.
    """
    high, low, close = (
        as_float(data["high"]),
        as_float(data["low"]),
        as_float(data["close"]),
    )
    span = max(int(period), 1)
    previous_close = shift(close, 1, fill=close[0])
    swing = np.where(
        close > previous_close,
        high - shift(high, 1, fill=high[0]),
        np.where(close < previous_close, shift(low, 1, fill=low[0]) - low, 0.0),
    )
    line = ema(swing, span)
    signal = ema(line, span)
    return {"line": line, "signal": signal, "osc": line - signal}


__all__ = [
    "accumulation_distribution",
    "accumulation_swing_index",
    "chaikin_money_flow",
    "ease_of_movement",
    "klinger",
    "money_flow_index",
    "obv",
    "volume_oscillator",
    "vwap",
    "vwma",
]
