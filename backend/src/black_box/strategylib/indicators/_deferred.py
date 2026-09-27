"""Six indicators that do not fit the system, and why.

The other fifty are registered in this package. These six are not, and that is a
deliberate decision rather than an oversight — each one violates a rule the
library holds to. They are collected here with the specific rule each breaks, so
the decision is reviewable and reversible once the rule changes.

The two rules they break are:

**Causality.** The value reported at bar *t* must depend only on bars up to *t*.
An indicator that needs to know whether a move has been *confirmed* before it can
report it has to either revise a bar it already published (repainting) or stop
publishing for a variable number of bars. Both are unacceptable in a backtest
whose whole claim is that a signal existed when it was traded.

**A single well-defined output.** Every registered indicator is a pure function
of OHLCV plus its parameters, with one unambiguous reading. Where two platforms
disagree on the formula and neither can be shown canonical, "a defensible
choice" is a coin flip dressed as an implementation.

`deferred_report()` returns these as plain data, so the reason each one is absent
shows up in a test or a manifest rather than only in a comment someone has to
remember to read.
"""

from __future__ import annotations

from typing import NamedTuple


class Deferred(NamedTuple):
    """One indicator that was considered and not registered."""

    id: str
    name: str
    breaks: str
    reason: str
    primitive: str | None
    """The causal building block that *would* be registered instead, if any."""


DEFERRED: tuple[Deferred, ...] = (
    Deferred(
        id="IND_FRVP",
        name="Fractal Volume Profile",
        breaks="not a time series",
        reason=(
            "It is a horizontal histogram of volume-by-price — a distribution "
            "over the *price* axis, not one value per bar. The compute contract "
            "in manifest.json is `dict[str, ndarray] -> ndarray` of equal length, "
            "and the interpreter has nowhere to put a second axis. A cut version "
            "would work: the point of control (highest-volume price) can be "
            "extracted per bar, but that is a different indicator wearing this "
            "one's name, and shipping it under the same id would be worse than "
            "shipping nothing."
        ),
        primitive=None,
    ),
    Deferred(
        id="IND_ZIGZAG",
        name="ZigZag",
        breaks="variable-length confirmation",
        reason=(
            "A pivot is only confirmed once price has moved far enough *after* "
            "it, by an amount that is not known when the pivot happens. The bar "
            "where the pivot becomes known is not the bar where the pivot "
            "happened, and which one to report is the entire question. Reporting "
            "at confirmation introduces a look-back of unbounded length; "
            "reporting at the pivot is repainting. A ZigZag line is a chart "
            "drawing that resolves history, and the fix is the same as FRVP's: it "
            "is not a function of the bar it is drawn at."
        ),
        primitive="swing_highs / swing_lows (causal, no confirmation lag)",
    ),
    Deferred(
        id="IND_SCHIFF",
        name="Schiff Pitchfork",
        breaks="manual anchors",
        reason=(
            "The pitchfork needs three anchor points chosen by eye — a recent "
            "swing high, a recent swing low, and the midpoint between them. There "
            "is no rule for picking them, so the indicator is a function of a "
            "person's judgement and is not reproducible from OHLCV. The compute "
            "contract takes `params` as numbers, not as analyst-chosen points, "
            "and Block C's parameter sweep has nothing to sweep."
        ),
        primitive=None,
    ),
    Deferred(
        id="IND_PITCHFORK",
        name="Pitchfork",
        breaks="manual anchors",
        reason=(
            "Same failure as Schiff, one level down: without the subjective "
            "anchors there is nothing to draw the median lines from. Both are "
            "listed separately because they are named separately in every "
            "indicator reference, but one decision covers them."
        ),
        primitive=None,
    ),
    Deferred(
        id="IND_GANN",
        name="Gann Angles / Fan",
        breaks="no canonical parameterization",
        reason=(
            "A Gann fan needs a chosen anchor, a price per bar, and a set of "
            "angles, and none of the three has a defensible default — the angles "
            "in particular are 1x1, 2x1, 4x1 in some references and 1x1, 1x2, "
            "1x4 in others, with no source canonical enough to pick between them. "
            "More importantly there is no oracle: a golden CSV can only assert "
            "that the code agrees with itself. The same is true of the well-known "
            "Gann retracement levels, which is where most of the actual signal "
            "is; those would be a lookup table, not a computation."
        ),
        primitive=None,
    ),
    Deferred(
        id="IND_MAMA",
        name="MESA Adaptive Moving Average (MAMA/FAMA)",
        breaks="IIR filter is not causal in the required sense",
        reason=(
            "The Hilbert transform that drives MAMA is an IIR band-pass filter "
            "designed to have *zero phase shift* — it is explicitly built to "
            "report each bar's value using the whole surrounding series. That is "
            "the opposite of what a backtest needs: an indicator that already "
            "knows the future by construction. Its own author's warning against "
            "using it on historical data is the reason it is not here. The causal "
            "part — the alpha that tracks the dominant cycle period — is already "
            "registered as IND_KAMA, which computes the same idea from a "
            "lookback window instead of a whole-series filter."
        ),
        primitive="IND_KAMA (causal efficiency-ratio adaptation)",
    ),
    Deferred(
        id="IND_FDI",
        name="Fractal Dimension Index",
        breaks="redundant and needs a window choice",
        reason=(
            "It estimates the fractal dimension of the price path over a window "
            "and reports it as a market-efficiency score — which is what Hurst's "
            "exponent (IND_HURST, registered) measures, on the same data, with "
            "the same interpretation and a better-specified estimator. The CHOP "
            "index (IND_CHOP, registered) already answers the practical version "
            "of the question — is this series trending or ranging — in one line. "
            "Registering a third estimator of the same quantity would give the "
            "library three ids that a consumer cannot choose between."
        ),
        primitive="IND_HURST and IND_CHOP (registered)",
    ),
)


def deferred_report() -> dict[str, dict[str, str | None]]:
    """``id -> {name, breaks, reason, primitive}`` for manifest/test consumption."""
    return {
        item.id: {
            "name": item.name,
            "breaks": item.breaks,
            "reason": item.reason,
            "primitive": item.primitive,
        }
        for item in DEFERRED
    }


__all__ = ["DEFERRED", "Deferred", "deferred_report"]
