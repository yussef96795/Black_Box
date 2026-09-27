"""Statistical indicators — the ones that answer "is this series well behaved?".

Z-score says how extreme the current price is against its own history. Hurst
exponent characterises the *series* rather than the current bar. Rank Correlation
indexes where the current close sits in the ordered window, which is a different
question from "how far from the mean" — a series can sit exactly on its mean while
monotonically drifting.

All three are windowed, so each carries a NaN prefix and none carries a NaN
hole; ``tests/strategylib/test_indicators.py`` pins that across the whole
registry rather than relying on each one being individually correct.
"""

from __future__ import annotations

import numpy

from black_box.strategylib._backend import np
from black_box.strategylib._math import (
    as_float,
    rolling_mean,
    rolling_std,
    safe_div,
    to_device,
    to_host,
)
from black_box.strategylib.indicators._registry import Data, P, indicator


@indicator(
    "IND_ZSCORE",
    group="stats",
    params={"period": P(20, 2, 400)},
    synonyms=(
        "z score",
        "z-score",
        "zscore",
        "standard score",
        "standard deviation score",
    ),
)
def zscore(data: Data, *, period: int = 20) -> dict:
    """Standard score of the close against its own trailing window.

    Signed distance in standard deviations. A flat window has a zero standard
    deviation, so the division would be infinite; ``safe_div`` maps that to 0.0 —
    "on the mean, with no scale to measure against" is the honest reading, and
    returning 0 keeps a flat instrument from producing a line of infinities.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    return safe_div(
        close - rolling_mean(close, span), rolling_std(close, span), fill=0.0
    )


@indicator(
    "IND_HURST",
    group="stats",
    params={"period": P(100, 10, 1000)},
    synonyms=("hurst exponent", "hurst", "hurst exponent ratio"),
)
def hurst(data: Data, *, period: int = 100) -> dict:
    """Hurst exponent from the rescaled range of the mean-adjusted cumulative sum.

    R/S grows like ``period**H``: 0.5 is a random walk, above 0.5 is trending or
    persistent, below 0.5 is mean-reverting. Estimated as ``log(R/S)/log(period)``.

    This is the *structural* constant of the window, not a per-bar reading.
    Consecutive values share almost their whole window and differ in the third
    decimal, which is expected rather than a bug. It needs hundreds of bars to
    carry statistical weight — at period 100 the value is indicative only, and the
    estimate is not clamped, so a noisy window can print outside 0-1.

    ponytail: the R/S estimator is a Python loop on the host rather than a
    cumulative max/min (CuPy has no ``accumulate`` for either, which is why
    ``_kernels`` exists). It is one call per strategy, not one per sweep point, so
    the two-transfer cost is irrelevant next to a per-element device loop. The
    upgrade path, if Hurst ever lands in a hot sweep, is a running-extreme kernel
    over the cumulative series.

    The loop body uses ``numpy`` directly rather than the ``np`` handle: it
    operates on the host array ``to_host`` returned, and the backend handle would
    try to upload a NumPy slice to the device on every one of the ``n - span``
    iterations. NumPy is always installed — it is what ``_backend`` falls back to,
    and CuPy requires it — so importing it here costs nothing.
    """
    close = as_float(data["close"])
    span = max(int(period), 4)
    host = to_host(close)
    out = numpy.full(len(host), numpy.nan, dtype=float)
    for end in range(span, len(host) + 1):
        window = host[end - span : end]
        deviation = window - window.mean()
        cumulative = numpy.cumsum(deviation)
        spread = cumulative.max() - cumulative.min()
        if spread <= 0.0:
            # No movement at all in the window: a random walk by definition.
            out[end - 1] = 0.5
            continue
        out[end - 1] = numpy.log(spread / deviation.std()) / numpy.log(float(span))
    return to_device(out)


@indicator(
    "IND_RCI",
    group="stats",
    params={"period": P(9, 2, 200)},
    synonyms=("rank correlation index", "rci", "rank correlation", "percent rank"),
)
def rank_correlation_index(data: Data, *, period: int = 9) -> dict:
    """Rank Correlation Index: the percentile rank of the close in its own window.

    0-100, where 100 means the current close is the window's highest. Distinct
    from a z-score, which measures distance from the mean: a monotonically rising
    series prints RCI 100 on every bar while its z-score stays near zero, and the
    first says "persistent" where the second says "ordinary".

    The rank is computed with ``argsort`` plus a comparison count rather than a
    comparison sort, so it costs one pass and no ties handling. Ties take the
    lowest rank, which is arbitrary but consistent — a market that closes flat for
    nine bars has no meaningful rank, and picking one deterministically is enough.
    """
    close = as_float(data["close"])
    span = max(int(period), 2)
    view = np.lib.stride_tricks.sliding_window_view(close, span)
    ordered = np.argsort(view, axis=-1)
    last = view[:, -1][:, None]
    rank = (ordered > last).sum(axis=-1) + 1.0
    out = np.full(len(close), np.nan, dtype=float)
    out[span - 1 :] = 100.0 * (rank - 1.0) / float(span - 1)
    return out


__all__ = ["hurst", "rank_correlation_index", "zscore"]
