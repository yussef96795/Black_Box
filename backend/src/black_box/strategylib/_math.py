"""Shared vector primitives for strategylib (DRY — Rules.md §1).

One module, two consumers: the hand-written ``components/`` and the generated
ones, plus the ``indicators/`` package layered on top. It lives at the package
root rather than under ``components/`` because indicators are not components —
keeping the primitives here means the dependency runs one way
(``components -> _math``, ``indicators -> _math``) instead of indicators
importing from their own consumer.

All helpers run over 1-D float arrays on the CuPy GPU backend (NumPy CPU
fallback, see ``_backend``) and every one of them returns an array the same
length as its input — nothing here changes shape, because the compute contract
in ``manifest.json`` is ``-> ndarray`` of equal length.

Two conventions run through the whole module:

* **Recursions seed at the first bar and carry no NaN prefix** (``ema``,
  ``rma``). ponytail: a deliberate divergence from TradingView's SMA-seeded
  warmup. The reason is arithmetic, not taste — the rolling helpers below are
  cumsum-based, and a single NaN anywhere poisons every later bar. Making the
  warmup explicit with ``shift(..., fill=...)`` at the call site keeps the hot
  path NaN-free without teaching every rolling helper to mask.
* **Rolling helpers are trailing and uncentered.** ``out[i]`` summarises
  ``x[i-window+1 : i+1]``; the first ``window-1`` bars are NaN because no value
  exists yet. A centered window would need ``i+window//2`` bars of *future*
  data, which is look-ahead — banned library-wide (see the causality test in
  ``tests/strategylib/test_indicators.py``).
"""

from __future__ import annotations

from typing import Any

from black_box.strategylib._backend import IS_CUPY, cp, np
from black_box.strategylib._kernels import recursive, running_max, running_min

# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------


def as_float(x: Any) -> Any:
    """Contiguous float64 view of ``x`` — the only dtype the contract emits."""
    return np.ascontiguousarray(x, dtype=float)


def to_host(x: Any) -> Any:
    """Backend array -> host (NumPy) array.

    For the handful of indicators that are inherently sequential — PSAR and
    Supertrend are per-bar state machines whose branch depends on the previous
    bar's outcome, so they cannot be expressed as array ops. Running those loops
    on device costs two transfers *per element*; running them here costs two
    transfers per indicator, which is the difference between a 30-second call
    and a 30-millisecond one.
    """
    return cp.asnumpy(x) if IS_CUPY else np.asarray(x)


def to_device(x: Any) -> Any:
    """Host (NumPy) array -> backend array."""
    return cp.asarray(x) if IS_CUPY else np.asarray(x)


def safe_div(a: Any, b: Any, fill: float = 0.0) -> Any:
    """``a / b`` with zeros in ``b`` mapped to ``fill``.

    ponytail: cupy's ufuncs reject NumPy's ``where=`` kwarg, so this is the
    guarded-denominator idiom (``synthesize.py`` open-codes the same thing).
    Upgrading to a real masked divide means a CuPy that supports ``where=``.
    """
    mask = b != 0
    return np.where(mask, a / np.where(mask, b, 1.0), fill)


def shift(x: Any, n: int, fill: Any = None) -> Any:
    """The value ``n`` bars ago, same length; never reads forward.

    ``fill`` replaces the leading ``n`` bars, which have no predecessor. Pass
    ``fill=x[0]`` (the usual choice) when the result feeds a cumsum-based
    helper, so the warmup cannot poison the tail; leave it ``None`` for a NaN
    prefix when NaN is the honest answer.
    """
    n = max(int(n), 0)
    x = as_float(x)
    if n == 0:
        return x.copy()
    pad = np.nan if fill is None else fill
    out = np.full(len(x), pad, dtype=float)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def change(x: Any, fill: float = 0.0) -> Any:
    """First difference, same length; bar 0 has no predecessor so it is ``fill``."""
    x = as_float(x)
    out = np.full(len(x), fill, dtype=float)
    if len(x) > 1:
        out[1:] = x[1:] - x[:-1]
    return out


def pct_change(x: Any, n: int = 1) -> Any:
    """``n``-bar percent change, same length; the first ``n`` bars are 0.0.

    Bar 0 has no predecessor, so it reports no change. A zero prior price maps
    to 0.0 as well rather than -100% — a flat-lining instrument, not a total
    loss, and the distinction matters when an indicator sits on top.
    """
    n = max(int(n), 0)
    if n == 0:
        return np.zeros(len(x), dtype=float)
    prev = shift(x, n, fill=x[0])
    return 100.0 * (safe_div(x, prev, fill=1.0) - 1.0)


# ---------------------------------------------------------------------------
# Rolling helpers — trailing, NaN-prefixed
# ---------------------------------------------------------------------------


def _window(x: Any, window: int) -> Any:
    """``(n - window + 1, window)`` strided view of the trailing windows.

    A view, not a copy: the reduction that follows is the only pass over the
    data. ponytail: rolling max/min and WMA are O(n * window) in bandwidth
    where a true O(n) deque algorithm exists. At window 200 over 1M bars that is
    200M strided reads, which a 4050 absorbs in single-digit milliseconds. The
    upgrade path, if a profile ever disagrees, is a monotonic-deque kernel —
    blocked behind the same reasoning that produced ``_kernels.py``.
    """
    return np.lib.stride_tricks.sliding_window_view(as_float(x), window)


def rolling_sum(x: Any, window: int) -> Any:
    """Trailing rolling sum; NaN until ``window`` consecutive real values exist.

    **NaN-aware, all-or-nothing.** Two cumulative sums are kept: one of the values
    and one of their validity. A window reports its sum only when all ``window``
    of its entries are real, and NaN otherwise.

    This is not a refinement, it is load-bearing. A single cumsum of the raw
    values is NaN from the first NaN onward *forever*, so any composite that
    rolls something which itself has a warmup prefix — ``rolling_mean(|x -
    rolling_mean(x, w)|, w)`` for CCI is the shortest example — would be NaN over
    its entire length. The validity counter is what makes composition work. It
    costs one extra cumsum and changes nothing for a series with no NaN in it.
    """
    window = max(int(window), 1)
    x = as_float(x)
    if window == 1:
        return x
    out = np.full(len(x), np.nan, dtype=float)
    if window > len(x):
        return out
    # cupy has no np.insert — a prepended zero is equivalent (numpy-compatible).
    values = np.nan_to_num(x, nan=0.0)
    counts = np.where(np.isnan(x), 0.0, 1.0)
    # cupy has no np.insert — a prepended zero is equivalent (numpy-compatible).
    value_sum = np.cumsum(np.concatenate((np.zeros(1, dtype=float), values)))
    value_count = np.cumsum(np.concatenate((np.zeros(1, dtype=float), counts)))
    total = value_sum[window:] - value_sum[:-window]
    seen = value_count[window:] - value_count[:-window]
    out[window - 1 :] = np.where(seen >= window, total, np.nan)
    return out


def rolling_mean(x: Any, window: int) -> Any:
    """Trailing rolling mean; first ``window-1`` bars are NaN."""
    window = max(int(window), 1)
    if window == 1:
        return as_float(x)
    return rolling_sum(x, window) / window


def rolling_std(x: Any, window: int) -> Any:
    """Trailing rolling standard deviation (population, ddof=0).

    Population, not sample: matches ``ta.stdev`` and keeps a flat series at
    exactly 0.0 instead of a denormal.
    """
    window = max(int(window), 2)
    mean = rolling_mean(x, window)
    sq = rolling_mean(x * x, window)
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def rolling_max(x: Any, window: int) -> Any:
    """Trailing rolling maximum; first ``window-1`` bars are NaN."""
    window = max(int(window), 1)
    out = np.full(len(x), np.nan, dtype=float)
    if window > len(x):
        return out
    out[window - 1 :] = _window(x, window).max(axis=-1)
    return out


def rolling_min(x: Any, window: int) -> Any:
    """Trailing rolling minimum; first ``window-1`` bars are NaN."""
    window = max(int(window), 1)
    out = np.full(len(x), np.nan, dtype=float)
    if window > len(x):
        return out
    out[window - 1 :] = _window(x, window).min(axis=-1)
    return out


def mean_dev(x: Any, window: int) -> Any:
    """Trailing mean absolute deviation — CCI's denominator, not a std.

    Deliberately not ``rolling_std``: CCI is defined against the mean absolute
    deviation, and the two disagree enough to move the 100/-100 lines.
    """
    return rolling_mean(np.abs(as_float(x) - rolling_mean(x, window)), window)


def highest(x: Any, window: int) -> Any:
    """Alias for :func:`rolling_max`, named for how indicators read."""
    return rolling_max(x, window)


def lowest(x: Any, window: int) -> Any:
    """Alias for :func:`rolling_min`, named for how indicators read."""
    return rolling_min(x, window)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def scan(x: Any, alpha: float) -> Any:
    """Fixed-alpha scan that starts at the first real value, not at bar 0.

    A recursion seeds at ``x[0]``, so a NaN in that first bar propagates to every
    later bar and the whole line is lost. That is not hypothetical: a windowed
    indicator such as the Fisher transform or the Klinger oscillator has a NaN
    warmup prefix, and smoothing it with a plain ``ema`` used to return nothing
    but NaN. Slicing past the prefix and re-attaching it costs one contiguity
    check in the common case and one concatenation in the rare one.

    Interior NaNs are *not* handled: a NaN after the first real bar would still
    poison the tail. That is a contract violation, not a case to paper over — the
    indicator tests assert no indicator produces an interior NaN, so a NaN here
    means a bug upstream and is left visible.
    """
    x = as_float(x)
    real = np.isfinite(x)
    if bool(real.all()):
        return recursive(x, alpha)
    start = int(np.argmax(real)) if bool(real.any()) else 0
    if start == 0:
        return recursive(x, alpha)
    return np.concatenate((x[:start], recursive(x[start:], alpha)))


def ema(x: Any, span: int) -> Any:
    """Exponential moving average, alpha = 2/(span+1); flat input stays flat.

    Skips a NaN warmup prefix — see :func:`scan`.
    """
    span = max(int(span), 1)
    return scan(x, 2.0 / (span + 1))


def rma(x: Any, length: int) -> Any:
    """Wilder's smoothing (alpha = 1/length) — RSI, ATR, ADX, CMF, MFI.

    Distinct from :func:`ema` and not interchangeable: at length 14 Wilder's
    alpha is 0.071 against the EMA's 0.133, so an "ATR" built on ``ema`` settles
    at a materially different level. Skips a NaN warmup prefix — see
    :func:`_scan`.
    """
    length = max(int(length), 1)
    return scan(x, 1.0 / length)


def wma(x: Any, window: int) -> Any:
    """Linearly weighted moving average — weights 1..window, recent bar heaviest."""
    window = max(int(window), 1)
    if window == 1:
        return as_float(x)
    out = np.full(len(x), np.nan, dtype=float)
    if window > len(x):
        return out
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    out[window - 1 :] = (_window(x, window) * weights).sum(axis=-1)
    return out


# ---------------------------------------------------------------------------
# Price transforms
# ---------------------------------------------------------------------------


def true_range(high: Any, low: Any, close: Any) -> Any:
    """ATR's true range; NaN on the first bar (no prior close)."""
    high, low, close = as_float(high), as_float(low), as_float(close)
    tr = np.empty_like(close, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ),
    )
    return tr


def median_price(high: Any, low: Any) -> Any:
    """``(high + low) / 2`` — Williams' HL2, the AO and Ichimoku input."""
    return (as_float(high) + as_float(low)) / 2.0


def typical_price(high: Any, low: Any, close: Any) -> Any:
    """``(high + low + close) / 3`` — the input to CCI, MFI and CMF."""
    return (as_float(high) + as_float(low) + as_float(close)) / 3.0


def stochastic_k(
    high: Any, low: Any, close: Any, window: int, flat: float = 50.0
) -> Any:
    """%K — where the close sits in the trailing high/low range, 0-100.

    A flat range (high == low for the whole window) has no position to report,
    so it maps to ``flat`` — 50.0, the midpoint — rather than NaN. NaN here
    would poison every composition built on it, and "exactly mid-range" is the
    honest reading of a window that never moved.
    """
    window = max(int(window), 1)
    hi = rolling_max(high, window)
    lo = rolling_min(low, window)
    # A zero span has no position to report, so safe_div's fill is mid-range in
    # ratio units and the degenerate window lands on `flat` after scaling. A NaN
    # span still divides through to NaN — that is a missing bar, not a flat one.
    return 100.0 * safe_div(as_float(close) - lo, hi - lo, fill=flat / 100.0)


__all__ = [
    "as_float",
    "change",
    "ema",
    "highest",
    "lowest",
    "mean_dev",
    "median_price",
    "pct_change",
    "recursive",
    "rma",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_std",
    "rolling_sum",
    "running_max",
    "running_min",
    "safe_div",
    "shift",
    "stochastic_k",
    "to_device",
    "to_host",
    "true_range",
    "typical_price",
    "wma",
]
