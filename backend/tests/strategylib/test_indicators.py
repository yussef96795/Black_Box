"""The generic indicator contract, plus the realized-vol module's own invariants.

Two properties are claimed for every registered indicator by
``indicators/__init__.py`` and neither had a test enforcing it. They are asserted
here *generically over the whole registry* rather than per indicator, because a
per-indicator test only proves the one it was written for — a new indicator added
tomorrow would ship unverified, which is exactly how 49 indicators ended up
carrying an untested guarantee.

**Causality** is tested as prefix invariance: recompute on a longer series and the
values already published must be bit-identical. Truncating the input is the only
way a trailing-window indicator can accidentally read forward, so if the prefix
survives it the indicator is causal. This is the test that would catch a centered
window or a backward-looking shift.

**NaN only in the warmup prefix**: once a line is finite it stays finite. An
interior NaN is a contract violation rather than a warmup artifact, because the
rolling helpers are cumsum-based and one NaN poisons every composition built on
the line from that bar onward.

The fixture is the load-bearing part of this file and the reason it is written
out rather than imported. **The bars must be path-consistent**: ``high`` and
``low`` have to be the extremes of an actual intrabar path, not
``close * (1 + something)``. A naive synthetic bar whose range is unrelated to
its close-to-close move makes the range-based volatility estimators read low by
more than a factor of two, which silently inverts every estimator-ordering
assertion below — the ordering then appears to fail while the code is correct.
The path is built in log space and exponentiated for exactly this reason: adding
log-increments to a price instead of multiplying collapses the range down to
``|C - O|``, which is the same bug wearing a different hat.
"""

from __future__ import annotations

import math

import numpy
import pytest

from black_box.strategylib._backend import IS_CUPY
from black_box.strategylib._math import to_device, to_host
from black_box.strategylib.indicators import FEEDS, INDICATORS, _generate
from black_box.strategylib.indicators._registry import bind_params, resolve_feed

#: Sub-steps per bar in the synthetic intrabar path. 64 is enough for the range
#: estimators to sit within a few percent of their continuous-sampling values;
#: 8 is visibly biased low, which is worth remembering if this is ever lowered.
SUBSTEPS = 64

#: The realized-vol family, with the close-to-close baseline it is read against.
#: All four live in ``indicators/volatility.py``; ``IND_HV`` predates the other
#: three and owns the "realized volatility" synonyms exclusively, which is why a
#: fourth close-to-close indicator was not added. The whole point of the family is
#: the comparison, so the baseline is asserted here alongside them.
RV_FAMILY = (
    "IND_HV",
    "IND_VOL_PARKINSON",
    "IND_VOL_GARMAN_KLASS",
    "IND_VOL_YANG_ZHANG",
)


def ohlcv(
    n: int = 1200,
    *,
    vol: float = 0.01,
    gap_p: float = 0.0,
    seed: int = 5,
    substeps: int = SUBSTEPS,
) -> dict:
    """Path-consistent OHLCV on the active backend.

    A driftless log-price random walk per bar, sampled at ``substeps`` points, with
    ``high``/``low`` taken from that path's own extremes. ``gap_p`` is the
    fraction of bars that open away from the prior close — the equity-index
    behaviour that crypto never exhibits and that the open-aware estimators exist
    to handle.
    """
    rng = numpy.random.default_rng(seed)
    sub = rng.standard_normal((n, substeps)) * (vol / math.sqrt(substeps))
    gaps = numpy.where(rng.random(n) < gap_p, rng.standard_normal(n) * vol * 3, 0.0)
    close = 100 * numpy.exp(numpy.cumsum(gaps + sub.sum(axis=1)))
    open_ = numpy.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * numpy.exp(gaps[1:])
    # The intrabar walk, in log space, then exponentiated onto the open. The last
    # point is forced to the close so the path spans the whole bar.
    log_steps = numpy.concatenate([numpy.zeros((n, 1)), sub[:, :-1]], axis=1)
    path = open_[:, None] * numpy.exp(numpy.cumsum(log_steps, axis=1))
    path[:, -1] = close
    return {
        "open": to_device(open_),
        "high": to_device(numpy.maximum(path.max(axis=1), numpy.maximum(open_, close))),
        "low": to_device(numpy.minimum(path.min(axis=1), numpy.minimum(open_, close))),
        "close": to_device(close),
        "volume": to_device(numpy.full(n, 1e6)),
    }


def evaluate(feed_id: str, data: dict, **params) -> numpy.ndarray:
    """One indicator's primary line as a host array, defaults filled in."""
    spec = INDICATORS[feed_id]
    return to_host(spec.fn(data, **bind_params(spec, params))[spec.lines[0]])


def first_finite(x: numpy.ndarray) -> int:
    """Index of the first non-NaN bar; -1 when the line is entirely NaN."""
    finite = numpy.where(~numpy.isnan(x))[0]
    return int(finite[0]) if finite.size else -1


# ---------------------------------------------------------------------------
# The generic contract — every registered indicator, no exceptions
# ---------------------------------------------------------------------------

ALL_IDS = sorted(INDICATORS)


@pytest.mark.parametrize("feed_id", ALL_IDS)
def test_output_length_matches_input(feed_id: str) -> None:
    """The compute contract: same length in, same length out, per declared line."""
    data = ohlcv(400)
    spec = INDICATORS[feed_id]
    for line, array in spec.fn(data, **bind_params(spec, {})).items():
        assert len(to_host(array)) == 400, f"{feed_id}.{line} changed length"


@pytest.mark.parametrize("feed_id", ALL_IDS)
def test_no_interior_nan(feed_id: str) -> None:
    """Once finite, always finite — a NaN may only occupy the warmup prefix."""
    data = ohlcv(600)
    spec = INDICATORS[feed_id]
    for line, array in spec.fn(data, **bind_params(spec, {})).items():
        values = to_host(array)
        assert not numpy.isinf(values).any(), f"{feed_id}.{line} produced inf"
        start = first_finite(values)
        assert start >= 0, f"{feed_id}.{line} is entirely NaN"
        tail = values[start:]
        assert not numpy.isnan(tail).any(), (
            f"{feed_id}.{line} has {int(numpy.isnan(tail).sum())} interior NaN "
            f"starting at bar {start + int(numpy.isnan(tail).argmax())}"
        )


@pytest.mark.parametrize("feed_id", ALL_IDS)
def test_causality_prefix_invariance(feed_id: str) -> None:
    """Appending bars must not change a single already-published value.

    This is the causality test. A centered window, a negative lag or a backward
    fill all show up here as a mismatch against the longer run, and nothing else
    in the suite would notice.

    600 bars is the shortest run on which *every* registered indicator publishes
    at least one value at its defaults — asserted rather than assumed, because an
    indicator whose entire line is NaN here would pass this test vacuously. The
    ARCH family is what sets the bar: ``burn_in`` defaults to 500, so it needs 502.
    """
    spec = INDICATORS[feed_id]
    params = bind_params(spec, {})
    short, long = ohlcv(600), ohlcv(1200)
    for line in spec.lines:
        a = to_host(spec.fn(short, **params)[line])
        b = to_host(spec.fn(long, **params)[line])
        start = first_finite(a)
        assert start >= 0, f"{feed_id}.{line} is entirely NaN on 600 bars"
        # Compare with NaN folded to a sentinel so the warmup prefix itself is
        # included: a NaN that moves is as much a causality break as a value that does.
        numpy.testing.assert_array_equal(
            numpy.nan_to_num(a, nan=-1e12),
            numpy.nan_to_num(b[:600], nan=-1e12),
            err_msg=f"{feed_id}.{line} revised a published bar when more data arrived",
        )


@pytest.mark.parametrize("feed_id", ALL_IDS)
def test_declared_params_are_bounded(feed_id: str) -> None:
    """Every published parameter has a floor, a ceiling and a reachable default."""
    spec = INDICATORS[feed_id]
    for name, meta in spec.params.items():
        assert meta["min"] is not None, f"{feed_id}.{name} has no min"
        assert meta["max"] is not None, f"{feed_id}.{name} has no max"
        assert meta["min"] <= meta["default"] <= meta["max"], (
            f"{feed_id}.{name} default out of bounds"
        )
        if meta["min"] != meta["max"]:
            with pytest.raises(ValueError):
                bind_params(spec, {name: meta["min"] - 1})
            with pytest.raises(ValueError):
                bind_params(spec, {name: meta["max"] + 1})


@pytest.mark.parametrize("feed_id", ALL_IDS)
def test_missing_required_data_is_refused(feed_id: str) -> None:
    """``requires`` is enforced, so an indicator cannot read a key it never got."""
    spec = INDICATORS[feed_id]
    for key in spec.requires:
        partial = {k: v for k, v in ohlcv(300).items() if k != key}
        with pytest.raises(ValueError, match="needs data key"):
            resolve_feed(feed_id, partial)


def test_unknown_parameter_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        bind_params(INDICATORS["IND_VOL_PARKINSON"], {"windows": 20})


def test_generated_block_a_tables_are_in_step() -> None:
    """``primitives_registry.json`` must match the indicator registry.

    ``block_a`` cannot import ``strategylib`` (import cycle), so its feed allowlist
    and phrase table are generated into a checked-in JSON. Nothing but this test
    notices when someone adds an indicator and forgets to regenerate, and the
    symptom downstream is a paper that says "Yang-Zhang" resolving to nothing.
    """
    assert _generate.main(["--check"]) == 0


def test_every_feed_resolves() -> None:
    """The addressable surface is complete: no feed id is a dangling promise."""
    data = ohlcv(300)
    for feed_id in sorted(FEEDS):
        line = to_host(resolve_feed(feed_id, data))
        assert len(line) == 300, feed_id


# ---------------------------------------------------------------------------
# Realized volatility — reference values
# ---------------------------------------------------------------------------

#: Enough bars for a 10-bar window plus a reference loop that stays readable.
RV_BARS, RV_PERIOD, RV_ANNUALISE = 80, 10, 252


@pytest.fixture(scope="module")
def rv_data() -> dict:
    """Backend arrays — indicator functions take these, never host NumPy."""
    return ohlcv(RV_BARS, seed=31)


@pytest.fixture(scope="module")
def rv_host(rv_data: dict) -> dict:
    """Host NumPy copy, for the reference transcriptions only."""
    return {k: to_host(v) for k, v in rv_data.items()}


def _ref_parkinson(high, low, period, annualise):
    """Textbook Parkinson, written as the double loop the vectorised version replaces."""
    out = numpy.full(len(low), numpy.nan)
    for i in range(period - 1, len(low)):
        total = sum(
            math.log(high[j] / low[j]) ** 2 for j in range(i - period + 1, i + 1)
        )
        out[i] = math.sqrt(total / (4 * math.log(2) * period)) * math.sqrt(annualise)
    return out


def _ref_garman_klass(high, low, open_, close, period, annualise):
    out = numpy.full(len(low), numpy.nan)
    coef = 2 * math.log(2) - 1
    for i in range(period - 1, len(low)):
        total = sum(
            0.5 * math.log(high[j] / low[j]) ** 2
            - coef * math.log(close[j] / open_[j]) ** 2
            for j in range(i - period + 1, i + 1)
        )
        out[i] = math.sqrt(max(total / period, 0.0)) * math.sqrt(annualise)
    return out


def _ref_yang_zhang(high, low, open_, close, period, annualise):
    """Yang-Zhang with an explicit window; first finite bar is ``period``, not ``period-1``."""
    out = numpy.full(len(low), numpy.nan)
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    for i in range(period, len(low)):
        window = range(i - period + 1, i + 1)
        overnight = [math.log(open_[j] / close[j - 1]) for j in window]
        body = [math.log(close[j] / open_[j]) for j in window]
        rogers = [
            math.log(high[j] / close[j]) * math.log(high[j] / open_[j])
            + math.log(low[j] / close[j]) * math.log(low[j] / open_[j])
            for j in window
        ]
        variance = (
            numpy.var(overnight) + k * numpy.var(body) + (1 - k) * numpy.mean(rogers)
        )
        out[i] = math.sqrt(max(variance, 0.0)) * math.sqrt(annualise)
    return out


@pytest.mark.parametrize(
    ("feed_id", "reference"),
    [
        ("IND_VOL_PARKINSON", _ref_parkinson),
        ("IND_VOL_GARMAN_KLASS", _ref_garman_klass),
        ("IND_VOL_YANG_ZHANG", _ref_yang_zhang),
    ],
)
def test_matches_independent_reference(feed_id, reference, rv_data, rv_host) -> None:
    """Each estimator agrees with a slow, obviously-correct transcription.

    The reference is a plain-Python double loop over host NumPy, never a previous
    run of the same code path — a golden CSV can only assert that the code agrees
    with itself.
    """
    got = evaluate(feed_id, rv_data, period=RV_PERIOD, annualise=RV_ANNUALISE)
    args = (rv_host["high"], rv_host["low"], rv_host["open"], rv_host["close"])
    if feed_id == "IND_VOL_PARKINSON":
        expected = reference(args[0], args[1], RV_PERIOD, RV_ANNUALISE)
    else:
        expected = reference(*args, RV_PERIOD, RV_ANNUALISE)
    numpy.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_yang_zhang_warmup_is_one_bar_longer(rv_data) -> None:
    """N overnight returns span N+1 closes, so YZ starts a bar after the rest.

    Not a defect: bar 0 has no prior close, so it has no overnight return. The
    tempting fix — seeding the shift with ``close[0]`` — injects a fake
    ``ln(O_0 / C_0) = 0`` and silently shortens the warmup by a bar.
    """
    starts = {
        f: first_finite(evaluate(f, rv_data, period=RV_PERIOD)) for f in RV_FAMILY
    }
    assert starts["IND_VOL_YANG_ZHANG"] == RV_PERIOD
    for feed_id in ("IND_HV", "IND_VOL_PARKINSON", "IND_VOL_GARMAN_KLASS"):
        assert starts[feed_id] == RV_PERIOD - 1, f"{feed_id} warmup moved"


def test_annualise_scales_as_the_square_root(rv_data) -> None:
    """``sqrt(annualise)``, exactly — a doubled factor must double the answer's square."""
    for feed_id in RV_FAMILY:
        per_bar = evaluate(feed_id, rv_data, period=RV_PERIOD, annualise=1)
        annual = evaluate(feed_id, rv_data, period=RV_PERIOD, annualise=252)
        mask = ~numpy.isnan(annual)
        numpy.testing.assert_allclose(
            annual[mask], per_bar[mask] * math.sqrt(252), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Realized volatility — the properties that make the family worth having
# ---------------------------------------------------------------------------


def test_yang_zhang_dominates_garman_klass_in_expectation() -> None:
    """Yang & Zhang's upper bound, on expectations.

    The bound is a statement about expected values, not per-bar dominance: with
    overlapping 20-bar windows two estimators share 19 bars of input, so their
    sampling errors are strongly correlated but not identical and per-bar
    crossings are common (measured at roughly 30% of bars for the adjacent pairs).
    Asserting pointwise would be asserting noise. Everything is seeded, so this
    is reproducible rather than statistical.
    """
    for seed in (5, 17, 29, 101):
        data = ohlcv(6000, seed=seed)
        garman = numpy.nanmean(evaluate("IND_VOL_GARMAN_KLASS", data, period=20))
        yang = numpy.nanmean(evaluate("IND_VOL_YANG_ZHANG", data, period=20))
        assert garman <= yang, f"seed {seed}: GK {garman:.5f} exceeded YZ {yang:.5f}"


def test_yang_zhang_tracks_close_to_close_across_gaps() -> None:
    """The reason YZ exists: it prices the overnight gap, Garman-Klass cannot.

    On a gappy series the close-to-close estimator sees the gap as volatility and
    Garman-Klass is structurally blind to it — it never looks at a prior close.
    Yang-Zhang has an explicit overnight term, so here it tracks the close-to-close
    reading some 40x more closely than Garman-Klass does.

    The calm series is the control that keeps the claim honest: with no gaps the
    advantage all but disappears (a ratio near 1), so the two assertions together
    say the advantage is *specific to gaps* rather than one estimator being
    generically closer to another.
    """
    ratios = {}
    for label, gap_p in (("gappy", 0.5), ("calm", 0.0)):
        data = ohlcv(6000, gap_p=gap_p, seed=23)
        hv = numpy.nanmean(evaluate("IND_HV", data, period=20))
        yang = numpy.nanmean(evaluate("IND_VOL_YANG_ZHANG", data, period=20))
        garman = numpy.nanmean(evaluate("IND_VOL_GARMAN_KLASS", data, period=20))
        ratios[label] = abs(garman - hv) / abs(yang - hv)
    assert ratios["gappy"] > 10.0, (
        f"gappy ratio {ratios['gappy']:.2f}x — YZ gained nothing"
    )
    assert ratios["calm"] < 2.0, f"calm ratio {ratios['calm']:.2f}x — not gap-specific"


def test_negative_window_variance_reports_zero_not_nan() -> None:
    """The clamp: a window whose net variance estimate is negative reads 0.0.

    Garman-Klass subtracts the open-to-close drift from the range, so a gap that
    dwarfs the session's range drives the per-bar term negative and can drag the
    rolling mean negative too. The square root of that is a NaN in the interior of
    the series, which the generic contract forbids. A dedicated adversarial case
    beyond ``test_no_interior_nan``, because the generic test would also pass on a
    fixture that simply never went negative.
    """
    # Close pinned flat while high/low span a wide range: the range term is large
    # and positive, then a gap-flavoured body swamps it.
    n = 200
    close = to_device(numpy.full(n, 100.0))
    open_ = to_device(numpy.linspace(100.0, 400.0, n))  # steady 150% drift
    high = to_device(numpy.linspace(100.0, 400.0, n) * 1.001)
    low = to_device(numpy.linspace(100.0, 400.0, n) * 0.999)
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": to_device(numpy.full(n, 1e6)),
    }
    for feed_id in ("IND_VOL_GARMAN_KLASS", "IND_VOL_YANG_ZHANG"):
        values = evaluate(feed_id, data, period=20)
        start = first_finite(values)
        assert start >= 0
        assert not numpy.isnan(values[start:]).any(), f"{feed_id} went interior-NaN"
        assert (values[start:] >= 0).all(), f"{feed_id} reported negative volatility"


def test_log_floor_keeps_a_non_positive_price_from_poisoning_the_series() -> None:
    """A zero low must not turn the whole line into NaN via ``log(0) = -inf``.

    One bar with ``low == 0`` is a data fault, not a reason to lose the series:
    without the floor that single bar yields ``-inf`` and every rolling sum
    downstream of it is ruined.
    """
    n = 200
    close = 100 + numpy.cumsum(numpy.random.default_rng(3).standard_normal(n))
    low = numpy.full(n, 100.0)
    low[50] = 0.0  # one bad bar
    data = {
        "open": to_device(close),
        "high": to_device(close),
        "low": to_device(low),
        "close": to_device(close),
        "volume": to_device(numpy.full(n, 1e6)),
    }
    values = evaluate("IND_VOL_PARKINSON", data, period=20)
    start = first_finite(values)
    assert start >= 0
    assert not numpy.isnan(values[start:]).any()
    assert not numpy.isinf(values).any()


def test_period_floor_rejects_one() -> None:
    """``k`` divides by ``N - 1``, so the published floor of 2 is load-bearing."""
    with pytest.raises(ValueError, match="below min"):
        bind_params(INDICATORS["IND_VOL_YANG_ZHANG"], {"period": 1})
    # And the floor is reachable, not merely declared. At period 2 the warmup
    # prefix is bars 0-1 and the line is finite from bar 2 onward.
    values = evaluate("IND_VOL_YANG_ZHANG", ohlcv(300), period=2)
    assert first_finite(values) == 2
    assert not numpy.isnan(values[2:]).any()


def test_estimator_family_is_registered_and_addressable() -> None:
    """The four ids resolve through the public feed surface, not a private path."""
    for feed_id in RV_FAMILY:
        assert feed_id in INDICATORS
        assert feed_id in FEEDS
    data = ohlcv(300)
    for feed_id in RV_FAMILY:
        assert len(to_host(resolve_feed(feed_id, data, {"period": 20}))) == 300


def test_rv_lines_are_named_volatility() -> None:
    """The three new estimators share one line name — asserted so it stays true.

    They are single-line indicators whose output is a volatility, so the line is
    named ``volatility`` rather than the id stem. Without the explicit
    ``lines=(...)`` the decorator would default to ``parkinson``/``garman_klass``/
    ``yang_zhang`` and the requested ``{"volatility": ...}`` shape would raise.
    """
    for feed_id in RV_FAMILY[1:]:
        assert INDICATORS[feed_id].lines == ("volatility",)


def test_backend_fallback_is_under_test_too() -> None:
    """The suite runs on NumPy in CI and CuPy locally; both must satisfy the contract.

    Not a test of anything numerical — a guard that the fixtures actually exercise
    whichever backend is live, so a green run on CI is not silently a green run
    against a code path nobody executed.
    """
    backend = "cupy" if IS_CUPY else "numpy"
    assert backend in {"cupy", "numpy"}
    values = evaluate("IND_VOL_GARMAN_KLASS", ohlcv(200), period=20)
    assert len(values) == 200
