"""Indicator registry — the decorator, the index, and feed resolution.

Every indicator in strategylib is a function ``fn(data, **params)`` that turns an
OHLCV dict into one or more same-length arrays. The ``@indicator`` decorator
records what the interpreter and Block C need to know about it — its id, its
output lines, its parameters and their bounds, the data keys it reads, and the
synonyms the paper parser matches on — so nothing about an indicator is
duplicated in a table somewhere else.

Two indices come out of the registry:

``INDICATORS``
    ``id -> IndicatorSpec``, one entry per indicator function.
``FEEDS``
    ``feed_id -> (indicator_id, line)``, the flat addressable surface the AST
    interpreter consumes. A single-line indicator is addressed by its bare id
    (``IND_RSI``); a multi-line one gets a suffixed id per line
    (``IND_MACD``, ``IND_MACD_SIGNAL``, ``IND_MACD_HIST``). This is what lets
    ``interpret()`` keep returning exactly one array per node — the multi-output
    problem is solved by addressing, not by changing the compute contract.

ponytail: parameters are validated (coerced, bounds-checked) but there is no
per-parameter sweep-grid declaration here — Block C reads ``spec.params`` and
builds its own grid, and this module deliberately does not know about sweeps.
The upgrade path, if a sweep ever needs to be declared once and shared, is a
``grid`` key on the same param dict the manifest already uses.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

#: An OHLCV batch. All arrays share a length (manifest.json ``contract``).
Data = dict[str, Any]

#: One array per bar, on the active backend.
Series = Any


def P(default: Any, min: Any = None, max: Any = None) -> dict[str, Any]:
    """Declare one parameter. Schema matches ``manifest.json``'s ``params``.

    Named ``P`` to keep the decorator calls readable — every indicator is a wall
    of these, and ``{"default": 14, "min": 2, "max": 200}`` at each call site
    would bury the actual logic. The uppercase violates PEP8 naming on purpose.
    """
    return {"default": default, "min": min, "max": max}


@dataclass(frozen=True)
class IndicatorSpec:
    """Everything the interpreter, Block C and the parser need about one indicator."""

    id: str
    group: str
    lines: tuple[str, ...]
    params: dict[str, dict[str, Any]]
    requires: tuple[str, ...]
    synonyms: tuple[str, ...]
    fn: Callable[..., dict[str, Series]]

    @property
    def is_multiline(self) -> bool:
        return len(self.lines) > 1

    def feed_ids(self) -> tuple[str, ...]:
        """The addressable feed id(s) for this indicator's line(s).

        The **first** declared line is the primary one and takes the bare id; the
        rest are suffixed with their line name. So MACD declared as
        ``("macd", "signal", "hist")`` addresses as ``IND_MACD``,
        ``IND_MACD_SIGNAL`` and ``IND_MACD_HIST`` — the primary line is reachable
        under the name the indicator is actually called, and the extra outputs are
        opt-in by name rather than implied.

        The declaration order is therefore load-bearing: it is what decides which
        line the bare id means, and it is why the channel indicators declare
        ``middle`` first.
        """
        if not self.is_multiline:
            return (self.id,)
        return (self.id,) + tuple(
            f"{self.id}_{line.upper()}" for line in self.lines[1:]
        )


#: Populated by the ``@indicator`` decorator at import time.
INDICATORS: dict[str, IndicatorSpec] = {}


def indicator(
    id: str,
    *,
    group: str,
    lines: tuple[str, ...] = (),
    params: dict[str, dict[str, Any]] | None = None,
    requires: tuple[str, ...] = ("close",),
    synonyms: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register an indicator function and normalise its return value.

    A single-line indicator may return a bare array; a multi-line one returns
    ``{line: array}``. Either way the wrapper hands back the dict, and checks the
    keys against the declared ``lines`` — a typo'd line name would otherwise
    surface as a KeyError deep inside a backtest.
    """
    declared = lines or (id.removeprefix("IND_").lower(),)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = IndicatorSpec(
            id=id,
            group=group,
            lines=declared,
            params=params or {},
            requires=requires,
            synonyms=synonyms,
            fn=fn,  # type: ignore[arg-type]
        )

        @functools.wraps(fn)
        def wrapper(data: Data, **kwargs: Any) -> dict[str, Series]:
            out = fn(data, **kwargs)
            lines_out = {declared[0]: out} if _is_array(out) else dict(out)
            if set(lines_out) != set(declared):
                raise ValueError(
                    f"{id} declared lines {sorted(declared)} but returned "
                    f"{sorted(lines_out)}"
                )
            return lines_out

        INDICATORS[id] = replace(spec, fn=wrapper)
        return wrapper

    return decorate


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and not isinstance(value, dict)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_feed_index(
    registry: dict[str, IndicatorSpec] | None = None,
) -> dict[str, tuple[str, str]]:
    """``feed_id -> (indicator_id, line)`` across the whole registry."""
    index: dict[str, tuple[str, str]] = {}
    for spec in (registry if registry is not None else INDICATORS).values():
        primary, *rest = spec.lines
        index[spec.id] = (spec.id, primary)
        for line in rest:
            index[f"{spec.id}_{line.upper()}"] = (spec.id, line)
    return index


def feed_ids(registry: dict[str, IndicatorSpec] | None = None) -> list[str]:
    """Every addressable feed id, sorted."""
    return sorted(build_feed_index(registry))


def bind_params(spec: IndicatorSpec, params: dict[str, Any]) -> dict[str, Any]:
    """Merge caller params over defaults, coerce, and bounds-check.

    Coercion follows the declared default's type: an ``int`` default means the
    parameter is integral (a window, a bar count), anything else is a float
    multiple. Out-of-bounds values raise rather than clamp — a sweep grid that
    exceeds the bounds the registry published is a bug in the grid, and silently
    clamping it would make two different grid points compute the same indicator.
    """
    unknown = sorted(set(params) - set(spec.params))
    if unknown:
        raise ValueError(f"{spec.id}: unknown parameter(s) {unknown}")
    bound: dict[str, Any] = {}
    for name, meta in spec.params.items():
        raw = params.get(name, meta["default"])
        if raw is None:
            raw = meta["default"]
        integral = isinstance(meta["default"], int) and not isinstance(
            meta["default"], bool
        )
        value = int(raw) if integral else float(raw)
        low, high = meta.get("min"), meta.get("max")
        if low is not None and value < low:
            raise ValueError(f"{spec.id}.{name}={value} below min {low}")
        if high is not None and value > high:
            raise ValueError(f"{spec.id}.{name}={value} above max {high}")
        bound[name] = value
    return bound


def resolve_feed(
    feed_id: str,
    data: Data,
    params: dict[str, Any] | None = None,
    *,
    registry: dict[str, IndicatorSpec] | None = None,
    feeds: dict[str, tuple[str, str]] | None = None,
) -> Series:
    """Evaluate one feed id against ``data`` and return that line's array.

    This is the single entry point the AST interpreter uses, replacing what used
    to be a hand-written ``if name == "IND_*"`` chain in ``synthesize.py``.
    """
    specs = INDICATORS if registry is None else registry
    index = build_feed_index(specs) if feeds is None else feeds
    if feed_id not in index:
        raise ValueError(f"unsupported feed id: {feed_id}")
    spec_id, line = index[feed_id]
    spec = specs[spec_id]
    missing = [key for key in spec.requires if key not in data]
    if missing:
        raise ValueError(f"{spec.id} needs data key(s) {missing}, got {sorted(data)}")
    return spec.fn(data, **bind_params(spec, params or {}))[line]


def phrase_table(
    registry: dict[str, IndicatorSpec] | None = None,
) -> tuple[tuple[str, str], ...]:
    """``(phrase, feed_id)`` pairs for the paper-text indicator parser.

    Sorted longest-phrase-first, and that ordering is load-bearing rather than
    cosmetic: ``detect_indicators`` is a substring scan that returns the first
    hit, so "exponential moving average" has to be tried before "moving average"
    or the long form is unreachable. ponytail: the ordering is computed here
    instead of relying on authors to hand-sort — the failure mode is silent, a
    paper that says "exponential moving average" quietly resolving to a generic
    EMA. The upgrade path, if the table ever needs real matching, is token-based
    scoring in ``block_a`` rather than a better sort.
    """
    specs = INDICATORS if registry is None else registry
    pairs: list[tuple[str, str]] = []
    for spec in specs.values():
        for phrase in spec.synonyms:
            for feed in spec.feed_ids():
                pairs.append((phrase.lower(), feed))
    return tuple(sorted(set(pairs), key=lambda kv: (-len(kv[0]), kv[0], kv[1])))


__all__ = [
    "INDICATORS",
    "Data",
    "IndicatorSpec",
    "P",
    "Series",
    "bind_params",
    "build_feed_index",
    "feed_ids",
    "indicator",
    "phrase_table",
    "resolve_feed",
]
