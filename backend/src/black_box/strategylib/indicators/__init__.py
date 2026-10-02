"""The indicator library — 53 indicators behind one decorator and two indices.

Importing this package is what populates the registry: each module below declares
its indicators with ``@indicator``, and the decorator records them in
``INDICATORS``. :data:`FEEDS` is then derived from that, and it is the flat
addressable surface the AST interpreter consumes.

A single-line indicator is addressed by its bare id (``IND_RSI``); a multi-line
one gets a suffixed id per line (``IND_MACD``, ``IND_MACD_SIGNAL``,
``IND_MACD_HIST``). That is how multi-output indicators fit an interpreter whose
``interpret()`` contract returns exactly one array per node — solved by
addressing rather than by changing the contract.

Two properties are guaranteed for every indicator in here, and both are tested
generically over the whole registry rather than per indicator:

**Strictly causal.** The value reported at bar *t* depends only on bars up to and
including *t*. Displaced series — Ichimoku's forward-drawn spans, DPO's shifted
SMA, the Chandelier's running extremes — are reported at the bar that *knows*
the value, never the bar a plotting convention would draw them on. A repainting
indicator would be worse than a missing one.

**NaN only in the warmup prefix.** Once a line is finite it stays finite. A NaN
in the middle of a series would poison every composition built on it, because
most of the rolling helpers are cumsum-based.
"""

from __future__ import annotations

from black_box.strategylib.indicators import (
    momentum,
    stats,
    trend,
    volatility,
    volume,
)
from black_box.strategylib.indicators._deferred import DEFERRED, deferred_report
from black_box.strategylib.indicators._registry import (
    INDICATORS,
    Data,
    IndicatorSpec,
    P,
    Series,
    bind_params,
    build_feed_index,
    feed_ids,
    indicator,
    phrase_table,
    resolve_feed,
)

#: ``feed_id -> (indicator_id, line)`` — the interpreter's addressable surface.
#: Derived from ``INDICATORS`` after the imports above have registered everything,
#: so adding an indicator to any module is enough to make it resolvable here.
FEEDS: dict[str, tuple[str, str]] = build_feed_index()

#: Indicators with more than one output line, keyed by id. Block C and the parser
#: use this to know which ids are single-series and safe to use bare.
MULTILINE: dict[str, IndicatorSpec] = {
    spec.id: spec for spec in INDICATORS.values() if spec.is_multiline
}

__all__ = [
    "DEFERRED",
    "FEEDS",
    "INDICATORS",
    "MULTILINE",
    "Data",
    "IndicatorSpec",
    "P",
    "Series",
    "bind_params",
    "build_feed_index",
    "deferred_report",
    "feed_ids",
    "indicator",
    "momentum",
    "phrase_table",
    "resolve_feed",
    "stats",
    "trend",
    "volatility",
    "volume",
]
