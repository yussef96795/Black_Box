"""Tests for the ARCH conditional-volatility family.

Two things are being defended here, and they are different in kind.

The first is **arithmetic**: the three recursions and the fitter are checked
against reference implementations written as explicit Python loops over the
model equations. Nothing in the shipped code can be self-consistent and wrong,
because ``lfilter`` solving a difference equation in closed form and a ``for``
loop stepping the same equation are not the same code. Where the reference is
hand-computed from the paper's formula rather than transcribed from the
implementation, that is said in the test name.

The second is **the split between indicator and fitter**. The three indicators
are registered and carry the causality guarantee; ``fit_garch`` is neither, and
the tests that keep it that way are as load-bearing as the recursion ones. An
ARCH family that quietly published a fitted parameter as a feed id would put a
lookahead inside the library's central promise, and nothing else in the suite
would notice.

Note on fixtures: GARCH and GJR-GARCH regress on squared *returns*, not squared
innovations, so the data-generating process here is ``r = sigma * z`` with
``h_t = omega + alpha * r^2_{t-1} + beta * h_{t-1}``. Using ``z^2`` there instead
gives a series whose unconditional variance is ``(omega + alpha)/(1-beta)``
rather than ``omega/(1-alpha-beta)``, and a fit against it is not wrong — it is
answering a different question, which is the more expensive kind of wrong.
"""

from __future__ import annotations

import math

import numpy
import pytest

from black_box.strategylib._math import to_device, to_host
from black_box.strategylib.indicators import FEEDS, INDICATORS
from black_box.strategylib.indicators._registry import bind_params, resolve_feed
from black_box.strategylib.indicators.volatility import (
    GarchFit,
    _ar1,
    _egarch_log_variance,
    _garch_variance,
    _gjr_variance,
    egarch_11,
    fit_garch,
    forecast,
    garch_11,
    gjr_garch_11,
)

STD_ABS = math.sqrt(2.0 / math.pi)
ARCH_IDS = ("IND_GARCH_11", "IND_GJR_GARCH_11", "IND_EGARCH_11")


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------


def simulate(model: str, n: int = 6000, seed: int = 11) -> tuple[numpy.ndarray, dict]:
    """``(returns, truth)`` from a known ARCH model, ``r_t = sigma_t * z_t``."""
    rng = numpy.random.default_rng(seed)
    z = rng.standard_normal(n)
    if model == "garch":
        omega, alpha, beta = 1.0e-5, 0.08, 0.90
        truth = {"omega": omega, "alpha": alpha, "beta": beta}
    elif model == "gjr":
        omega, alpha, beta, gamma = 1.0e-5, 0.05, 0.85, 0.12
        truth = {"omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma}
    else:
        # Inside EGARCH's numerical stability region: |beta| <= 0.90 and
        # gamma <= 0.20 survive twelve seeds, 0.93 and 0.30 do not. See the
        # cliff note on `egarch_11`.
        omega, alpha, gamma, beta = -0.45, 0.12, 0.15, -0.85
        truth = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta}

    variance = numpy.empty(n)
    if model == "egarch":
        log_variance = numpy.empty(n)
        log_variance[0] = omega / (1.0 - beta)
        variance[0] = math.exp(log_variance[0])
        for t in range(1, n):
            log_variance[t] = (
                omega
                + beta * log_variance[t - 1]
                + alpha * z[t - 1]
                + gamma * (abs(z[t - 1]) - STD_ABS)
            )
            variance[t] = math.exp(log_variance[t])
    else:
        persistence = alpha + beta + (0.5 * gamma if model == "gjr" else 0.0)
        variance[0] = omega / (1.0 - persistence)
        for t in range(1, n):
            previous = math.sqrt(variance[t - 1]) * z[t - 1]
            coefficient = alpha + (gamma if previous < 0.0 and model == "gjr" else 0.0)
            variance[t] = omega + coefficient * previous**2 + beta * variance[t - 1]
    return numpy.sqrt(variance) * z, truth


def prices(returns: numpy.ndarray, seed: int = 3) -> dict:
    """A close-only frame on the active backend whose log returns are ``returns``.

    On the backend, because that is what ``as_float`` demands and every other
    indicator in the library expects — a host array here fails deep inside cupy
    with a type error that has nothing to do with the thing under test.
    """
    return {"close": to_device(100.0 * numpy.exp(numpy.cumsum(returns)))}


# ---------------------------------------------------------------------------
# The recursions, against reference loops
# ---------------------------------------------------------------------------


def test_ar1_matches_an_explicit_loop() -> None:
    """``_ar1``'s closed form equals stepping the difference equation by hand.

    ``lfilter`` seeds at zero and the initial value is added back as an explicitly
    decayed term. If the decay exponent were off by one — which is the easy
    mistake here, and looks plausible because it still converges — every value
    would be wrong while the series stayed bounded and smooth.
    """
    rng = numpy.random.default_rng(4)
    shocks, beta, initial = rng.standard_normal(50), 0.93, 2.5
    got = _ar1(shocks, beta, initial)
    want = numpy.empty(len(shocks) + 1)
    want[0] = initial
    for t in range(1, len(want)):
        want[t] = shocks[t - 1] + beta * want[t - 1]
    numpy.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_garch_recursion_matches_an_explicit_loop() -> None:
    """The ``lfilter`` path is the same recursion, not a lookalike of it."""
    returns = numpy.random.default_rng(5).normal(0.0, 0.02, 400)
    omega, alpha, beta, seed = 2e-6, 0.09, 0.88, 1e-4
    got = _garch_variance(returns, omega, alpha, beta, seed)
    want = numpy.empty(len(returns))
    want[0] = seed
    for t in range(1, len(want)):
        want[t] = omega + alpha * returns[t - 1] ** 2 + beta * want[t - 1]
    numpy.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-18)


def test_gjr_recursion_matches_an_explicit_loop() -> None:
    """The switch fires on the *previous* return's sign, not the current one."""
    returns = numpy.random.default_rng(6).normal(0.0, 0.02, 400)
    omega, alpha, beta, gamma, seed = 2e-6, 0.06, 0.85, 0.14, 1e-4
    got = _gjr_variance(returns, omega, alpha, beta, gamma, seed)
    want = numpy.empty(len(returns))
    want[0] = seed
    for t in range(1, len(want)):
        shock = returns[t - 1]
        want[t] = (
            omega
            + (alpha + (gamma if shock < 0.0 else 0.0)) * shock**2
            + beta * want[t - 1]
        )
    numpy.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-18)


def test_egarch_recursion_matches_an_explicit_loop() -> None:
    """``z`` is the innovation the recursion itself produced, one bar behind."""
    returns = numpy.random.default_rng(7).normal(0.0, 0.02, 400)
    omega, alpha, gamma, beta, seed = -0.5, 0.15, 0.2, -0.85, 1e-4
    got = _egarch_log_variance(returns, omega, alpha, gamma, beta, seed)
    want = numpy.empty(len(returns))
    want[0] = math.log(seed)
    for t in range(1, len(want)):
        z = returns[t - 1] / math.sqrt(math.exp(want[t - 1]))
        want[t] = omega + beta * want[t - 1] + alpha * z + gamma * (abs(z) - STD_ABS)
    numpy.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12)


def test_gjr_with_zero_gamma_reduces_to_garch() -> None:
    """The leverage term is an addition, not a change of parameterisation."""
    returns = numpy.random.default_rng(8).normal(0.0, 0.02, 300)
    omega, alpha, beta, seed = 2e-6, 0.07, 0.9, 1e-4
    numpy.testing.assert_allclose(
        _gjr_variance(returns, omega, alpha, beta, 0.0, seed),
        _garch_variance(returns, omega, alpha, beta, seed),
        rtol=1e-12,
        atol=1e-18,
    )


def test_gjr_reacts_to_the_sign_of_a_return_garch_cannot_see() -> None:
    """One negative return of the same size raises the variance more.

    This is the whole point of the model, so it is asserted directly rather than
    inferred from a parameter estimate.
    """
    # The shock goes at index 0, because the recursion at bar t reacts to the
    # return at t-1. Left at index 1, both bars would still be responding to the
    # seed and the comparison would be vacuous — it is one of the few places in
    # this file where an off-by-one silently produces a passing test.
    up, down = 0.02, -0.02
    garch_up = _garch_variance(numpy.array([up, 0.0]), 2e-6, 0.06, 0.85, 1e-4)
    garch_down = _garch_variance(numpy.array([down, 0.0]), 2e-6, 0.06, 0.85, 1e-4)
    assert garch_up == pytest.approx(garch_down), "GARCH must be sign-blind"

    leveraged_up = _gjr_variance(numpy.array([up, 0.0]), 2e-6, 0.06, 0.85, 0.14, 1e-4)[
        1
    ]
    leveraged_down = _gjr_variance(
        numpy.array([down, 0.0]), 2e-6, 0.06, 0.85, 0.14, 1e-4
    )[1]
    assert leveraged_down > leveraged_up
    assert (leveraged_down - leveraged_up) == pytest.approx(0.14 * up**2, rel=1e-9), (
        "the gap should be exactly gamma * eps^2"
    )


@pytest.mark.parametrize("model", ("garch", "gjr", "egarch"))
def test_recursions_emit_one_value_per_return(model: str) -> None:
    """Empty in, empty out. The seed bar is not a free extra bar.

    ``_ar1`` is off by one against this contract on its own — an empty shock list
    still has a seed — so the guard is load-bearing and worth pinning.
    """
    returns = numpy.zeros(0)
    seed = 1e-4
    if model == "garch":
        got = _garch_variance(returns, 2e-6, 0.08, 0.9, seed)
    elif model == "gjr":
        got = _gjr_variance(returns, 2e-6, 0.05, 0.85, 0.12, seed)
    else:
        got = _egarch_log_variance(returns, -0.5, 0.15, 0.2, -0.85, seed)
    assert len(got) == 0


@pytest.mark.parametrize("model", ("garch", "gjr", "egarch"))
def test_recursions_are_one_per_return_at_every_length(model: str) -> None:
    """Output length tracks input length exactly, not approximately."""
    for n in (1, 2, 3, 17, 128):
        returns = numpy.random.default_rng(n).normal(0.0, 0.02, n)
        if model == "garch":
            got = _garch_variance(returns, 2e-6, 0.08, 0.9, 1e-4)
        elif model == "gjr":
            got = _gjr_variance(returns, 2e-6, 0.05, 0.85, 0.12, 1e-4)
        else:
            got = _egarch_log_variance(returns, -0.5, 0.15, 0.2, -0.85, 1e-4)
        assert len(got) == n, f"{model} returned {len(got)} for {n} returns"


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(0.5, 0.6), (0.4, 0.61), (0.9, 0.11)],  # alpha + beta = 1.1, 1.01, 1.01
)
def test_garch_refuses_a_sum_of_alpha_and_beta_at_or_above_one(
    alpha: float, beta: float
) -> None:
    with pytest.raises(ValueError, match="persistence"):
        garch_11(prices(numpy.zeros(900)), omega=2e-6, alpha=alpha, beta=beta)


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma"),
    [
        (0.05, 0.85, 0.4),  # 0.05 + 0.85 + 0.20 = 1.10
        (0.1, 0.85, 0.3),  # 0.10 + 0.85 + 0.15 = 1.10
        (0.3, 0.6, 0.3),  # 0.30 + 0.60 + 0.15 = 1.05
    ],
)
def test_gjr_refuses_a_sum_of_alpha_beta_and_gamma_half_at_or_above_one(
    alpha: float, beta: float, gamma: float
) -> None:
    """The bound is on ``alpha + beta + gamma/2``, and saying so matters.

    Testing ``alpha + beta`` instead is a real and common error: it understates
    GJR persistence by exactly ``gamma/2``, because a symmetric innovation puts
    half its probability on the branch that carries ``gamma``. Each of these three
    cases has ``alpha + beta`` comfortably *below* 1, so a guard written on the
    shorter sum would let every one through. The *message* is asserted as well as
    the raise — a reader hitting the failure is told which condition failed rather
    than being left to guess.
    """
    assert alpha + beta < 1.0, "the case must be invisible to the shorter sum"
    with pytest.raises(ValueError, match="gamma/2"):
        gjr_garch_11(
            prices(numpy.zeros(900)), omega=2e-6, alpha=alpha, beta=beta, gamma=gamma
        )


def test_garch_has_no_gamma_half_term_to_report() -> None:
    """The two models refuse for different reasons, and the messages must not blur.

    Passing a ``gamma`` to ``garch_11`` is a ``TypeError`` from Python, not a
    ``ValueError`` from the guard — the parameter does not exist, and inventing a
    stationarity complaint about a coefficient the model has no use for would be
    actively misleading.
    """
    with pytest.raises(TypeError):
        garch_11(  # type: ignore[call-arg]
            prices(numpy.zeros(900)), omega=2e-6, alpha=0.1, beta=0.8, gamma=0.1
        )


def test_gjr_accepts_the_boundary_of_the_larger_sum() -> None:
    """``alpha + beta + gamma/2 = 0.999`` is fine; exactly 1.0 is not.

    Pins both sides of the inequality, so a guard that silently became inclusive
    or exclusive could not pass.
    """
    gjr_garch_11(
        prices(numpy.zeros(900)), omega=2e-6, alpha=0.0495, beta=0.85, gamma=0.199
    )
    with pytest.raises(ValueError):
        gjr_garch_11(
            prices(numpy.zeros(900)), omega=2e-6, alpha=0.05, beta=0.85, gamma=0.2
        )


def test_egarch_refuses_a_non_negative_beta() -> None:
    """Nelson's condition is ``beta < 0``. There is no ``alpha + beta`` test here."""
    for beta in (0.0, 0.5, 0.98):
        with pytest.raises(ValueError, match="beta < 0"):
            egarch_11(prices(numpy.zeros(900)), beta=beta)


def test_egarch_refuses_a_positive_beta_even_when_alpha_is_negative() -> None:
    """``alpha + beta`` looks admissible at ``alpha = -0.9, beta = 0.5``.

    It is not, and it is the case a habit of testing the GARCH sum would wave
    through, because the sum is negative and therefore comfortably under one.
    """
    with pytest.raises(ValueError, match="beta < 0"):
        egarch_11(prices(numpy.zeros(900)), alpha=-0.9, beta=0.5)


def test_equal_alpha_and_gamma_would_blind_egarch_to_the_size_of_a_down_move() -> None:
    """Why the published ``alpha`` and ``gamma`` are unequal — arithmetic, not taste.

    For a negative shock the whole innovation term collapses to
    ``(gamma - alpha)*|z| - gamma*E|z|``. At ``alpha == gamma`` the ``|z|``
    coefficient is exactly zero, so EGARCH responds to the *sign* of a negative
    return and not at all to its size. That is not a weakly-parameterised EGARCH;
    it is a sign-switching model in EGARCH's clothing, and nothing in the output
    line reveals it.

    This pins the property rather than the default, so a future retune that
    drifted toward ``alpha == gamma`` would fail here even if every robustness
    check still passed — which is exactly what happened while building this.
    """
    alpha = gamma = 0.08
    small = _egarch_log_variance(
        numpy.array([-0.02, 0.0]), -0.5, alpha, gamma, -0.6, 1e-4
    )
    large = _egarch_log_variance(
        numpy.array([-0.20, 0.0]), -0.5, alpha, gamma, -0.6, 1e-4
    )
    assert small[1] == pytest.approx(large[1], rel=1e-12), (
        "equal alpha and gamma cancel the |z| term, so this assertion is the bug"
    )

    # With a real asymmetry the response tracks the size of the move.
    asymmetric = _egarch_log_variance(
        numpy.array([-0.20, 0.0]), -0.5, 0.05, 0.10, -0.6, 1e-4
    )[1]
    assert asymmetric > small[1], "a 10x larger down move must raise the variance"


def test_the_published_egarch_defaults_are_asymmetric() -> None:
    """The shipped indicator, not just the recursion, must not be degenerate."""
    spec = INDICATORS["IND_EGARCH_11"]
    assert spec.params["alpha"]["default"] != spec.params["gamma"]["default"]


@pytest.mark.parametrize("model", ("garch", "gjr", "egarch"))
def test_a_flat_series_never_reaches_a_divergence_check(model: str) -> None:
    """The zero-return case must be boring everywhere, not just in log space.

    A flat series drives the seed variance to exactly 0. EGARCH's state is a log
    of that, so it is the model with a real hazard here; GARCH and GJR have the
    easier version and are *correct* to report a zero variance rather than
    inventing one from a floor.
    """
    spec = {
        "garch": INDICATORS["IND_GARCH_11"],
        "gjr": INDICATORS["IND_GJR_GARCH_11"],
        "egarch": INDICATORS["IND_EGARCH_11"],
    }[model]
    values = to_host(
        spec.fn(prices(numpy.zeros(2000)), **bind_params(spec, {}))[
            "conditional_volatility"
        ]
    )
    finite = values[~numpy.isnan(values)]
    # 2000 prices give 1999 log returns; 500 of them are the burn-in.
    assert len(finite) == 1999 - 500, "the burn-in prefix length must not drift"
    assert numpy.isfinite(finite).all()


def test_egarch_omega_zero_is_a_sentinel_not_a_level() -> None:
    """The default ``omega`` is derived from the data; a real one is used as given.

    EGARCH's ``omega`` is a log-variance intercept, so a published constant is
    calibrated to one instrument's return scale and wrong on another's. Zero is
    reserved as the sentinel because it implies an unconditional variance of
    ``exp(0) = 1`` — a per-bar sigma of 100% — which no real series has, so
    nothing legitimate is claimed by taking it.

    The observable consequence: the same indicator call on two series at very
    different volatility levels reports volatility near each one's own level,
    rather than reporting one level for both.
    """
    spec = INDICATORS["IND_EGARCH_11"]
    ratios = []
    for scale in (0.005, 0.05):
        returns = numpy.random.default_rng(3).normal(0.0, scale, 3000)
        values = to_host(
            spec.fn(prices(returns), **bind_params(spec, {}))["conditional_volatility"]
        )
        published = numpy.nanmean(values[2000:])
        ratios.append(published / (scale * math.sqrt(252)))
    assert ratios[0] == pytest.approx(ratios[1], rel=0.35), (
        "the sentinel should track the data's own volatility level"
    )
    assert 0.2 < ratios[0] < 5.0, f"reported {ratios[0]:.2f}x the realized level"


def test_egarch_omega_bounds_still_contain_the_widest_derivable_sentinel_value() -> (
    None
):
    """A real caller-supplied ``omega`` must survive ``bind_params`` unchanged.

    The bounds exist to catch a typo, not to reject a legitimate log-variance
    intercept: a series at 5% per-bar volatility implies ``omega`` near -9, and
    one at 0.05% implies near -24. Both have to pass.
    """
    spec = INDICATORS["IND_EGARCH_11"]
    for omega in (-8.7, -17.4, -24.0, -40.0):
        assert bind_params(spec, {"omega": omega})["omega"] == omega
    with pytest.raises(ValueError, match="above max"):
        bind_params(spec, {"omega": 1e6})


@pytest.mark.parametrize(
    ("beta", "omega"),
    [(-0.95, -0.45), (-0.97, -0.45), (-0.99, -0.45)],
)
def test_egarch_divergence_is_refused_rather_than_returned_as_a_number(
    beta: float, omega: float
) -> None:
    """``beta < 0`` holds, the variance is stationary, and it diverges anyway.

    Stationarity is necessary and not sufficient. This is the documented cliff: the
    log state feeds back through ``1/sigma``, so the recursion can run away while
    every published condition is met. Returning a large finite volatility here
    would be worse than useless — it would look like a result.

    The series has to be *clustered*, not a plain random walk. Divergence needs a
    bar that moves several sigmas from a level the model has drifted to, and a
    homoskedastic random walk never provides one — which is exactly why this went
    unnoticed until the test was built on EGARCH-simulated data.
    """
    returns, _ = simulate("egarch", n=4000, seed=1)
    with pytest.raises(ValueError, match="diverged"):
        egarch_11(
            prices(returns), omega=omega, alpha=0.15, gamma=0.2, beta=beta, burn_in=500
        )


# ---------------------------------------------------------------------------
# The indicators' contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feed_id", ARCH_IDS)
def test_nan_prefix_is_exactly_the_burn_in(feed_id: str) -> None:
    """The NaN prefix is ``burn_in + 1`` bars: the seed window plus bar 0's
    missing overnight return.

    Not a window and not ``burn_in`` — the seed variance is taken over the first
    ``burn_in`` returns, and those bars are exactly the ones whose value the seed
    is standing in for. So a value that appears before the seed window is done
    would be the recursion reporting on data it was seeded from.
    """
    returns = numpy.random.default_rng(9).normal(0.0, 0.02, 2000)
    spec = INDICATORS[feed_id]
    burn_in = 200
    values = to_host(
        spec.fn(prices(returns), **bind_params(spec, {"burn_in": burn_in}))[
            "conditional_volatility"
        ]
    )
    finite = numpy.where(~numpy.isnan(values))[0]
    assert int(finite[0]) == burn_in + 1
    assert numpy.isnan(values[: burn_in + 1]).all()
    assert not numpy.isnan(values[burn_in + 1 :]).any()


@pytest.mark.parametrize("feed_id", ARCH_IDS)
def test_annualise_is_exactly_a_square_root(feed_id: str) -> None:
    """``annualise`` multiplies the volatility, not the variance.

    Getting this wrong is invisible at ``annualise = 1``, so it is checked at a
    value that would make the two differ visibly.
    """
    returns = numpy.random.default_rng(10).normal(0.0, 0.02, 2000)
    spec = INDICATORS[feed_id]
    base = to_host(
        spec.fn(prices(returns), **bind_params(spec, {"annualise": 1}))[
            "conditional_volatility"
        ]
    )
    scaled = to_host(
        spec.fn(prices(returns), **bind_params(spec, {"annualise": 252}))[
            "conditional_volatility"
        ]
    )
    finite = ~numpy.isnan(base)
    numpy.testing.assert_allclose(
        scaled[finite], base[finite] * math.sqrt(252), rtol=1e-12
    )


@pytest.mark.parametrize("feed_id", ARCH_IDS)
def test_short_input_yields_an_all_nan_line_rather_than_an_error(feed_id: str) -> None:
    """A warmup longer than the input is not an error.

    It is the same thing a ``period=500`` window does on 400 bars, the contract
    permits it, and the line still has to come back the right length. Raising
    here would make these three indicators the only ones in the library that
    cannot be evaluated on a short series.
    """
    spec = INDICATORS[feed_id]
    values = to_host(
        spec.fn(prices(numpy.zeros(400)), **bind_params(spec, {}))[
            "conditional_volatility"
        ]
    )
    assert len(values) == 400
    assert numpy.isnan(values).all()


@pytest.mark.parametrize("feed_id", ARCH_IDS)
def test_a_flat_price_series_does_not_poison_the_recursion(feed_id: str) -> None:
    """Zero returns everywhere: the seed variance is exactly 0.

    EGARCH's state is a log, so ``log(0)`` is the live hazard here. The floor on
    the seed turns it into a very negative state rather than ``-inf``, and the
    recursion climbs back to its own unconditional level instead of emitting NaN
    for the rest of the series.

    GARCH and GJR have the easier version of the same problem and are asserted
    separately, because they are *correct* to report zero: a flat series has no
    return, so a variance model with no innovation to add should say so, and
    clamping to some floor would be inventing volatility that the data does not
    contain. Only the first bar is affected in practice — the published ``omega``
    carries the line off zero from bar two onward.
    """
    spec = INDICATORS[feed_id]
    values = to_host(
        spec.fn(prices(numpy.zeros(2000)), **bind_params(spec, {}))[
            "conditional_volatility"
        ]
    )
    finite = ~numpy.isnan(values)
    assert finite.any(), "flat input produced an entirely NaN line"
    assert numpy.isfinite(values[finite]).all(), (
        "flat input produced a non-finite value"
    )
    if feed_id == "IND_EGARCH_11":
        assert (values[finite] > 0).all(), "log space has no zero to report"


@pytest.mark.parametrize("feed_id", ("IND_GARCH_11", "IND_GJR_GARCH_11"))
def test_garch_reports_a_verbatim_zero_for_a_verbatim_flat_series(feed_id: str) -> None:
    """The contrast case: zero in, zero out, and no floor invented to hide it."""
    spec = INDICATORS[feed_id]
    values = to_host(
        spec.fn(prices(numpy.zeros(2000)), **bind_params(spec, {}))[
            "conditional_volatility"
        ]
    )
    finite = ~numpy.isnan(values)
    assert values[finite][0] == 0.0, "no innovation means no variance on the seed bar"
    assert (values[finite][1:] > 0).all(), "omega should carry the line off zero"


@pytest.mark.parametrize("feed_id", ARCH_IDS)
def test_indicated_volatility_tracks_the_realized_spread_of_the_same_series(
    feed_id: str,
) -> None:
    """A conditional volatility has to react to a volatility cluster.

    Not a pointwise comparison against any realized estimator — those disagree by
    construction — but a ranking: on a series whose volatility drifts up, the
    model's own late section must be the more volatile one.

    The regime change *drifts* over 300 bars rather than stepping, because an
    instantaneous 15x jump is past what any correctly-implemented EGARCH survives
    — see the cliff note on `egarch_11` and the dedicated divergence test below.
    Asserting that case here would be asserting a bug. A drift is also the
    realistic shape: real volatility regimes are entered over weeks, not one bar.

    The ranking is deliberately loose. The magnitude of the response differs a
    great deal between the three — GARCH tracks roughly 2.7x of a 4.7x realized
    shift, EGARCH about 1.2x, because EGARCH's published defaults are held well
    back from its numerical cliff and a caller who needs more response is meant to
    fit. Requiring all three to track proportionally would force the EGARCH
    defaults back into the region that diverges.
    """
    rng = numpy.random.default_rng(12)
    scale = numpy.concatenate(
        [
            numpy.full(1500, 0.002),
            numpy.linspace(0.002, 0.010, 300),
            numpy.full(1200, 0.010),
        ]
    )
    returns = rng.standard_normal(3000) * scale
    spec = INDICATORS[feed_id]
    values = to_host(
        spec.fn(prices(returns), **bind_params(spec, {}))["conditional_volatility"]
    )
    assert numpy.nanmean(values[2400:]) > numpy.nanmean(values[600:1400]), (
        f"{feed_id} did not register the volatility regime change"
    )


# ---------------------------------------------------------------------------
# The fitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "tolerance"),
    [("garch", 0.03), ("gjr", 0.04), ("egarch", 0.05)],
)
def test_fit_recovers_the_parameters_that_generated_the_series(
    model: str, tolerance: float
) -> None:
    """Fit a known model, get that model back.

    Tolerance is loose on purpose. MLE on 6000 draws has real sampling error, and
    a test that demanded the true coefficients to three decimals would be testing
    the luck of the seed. What is being asserted is that the optimiser finds the
    right basin — persistence within a few points of the truth is enough to fail
    loudly if a coefficient is transposed or the omega scaling is wrong.
    """
    returns, truth = simulate(model)
    fit = fit_garch(returns, model, burn_in=500)
    assert fit.converged, fit.message
    for name, value in truth.items():
        assert fit.params[name] == pytest.approx(value, abs=tolerance), (
            f"{model}.{name}: truth {value}, fitted {fit.params[name]}"
        )


@pytest.mark.parametrize("model", ("garch", "gjr", "egarch"))
def test_persistence_is_the_model_specific_number(model: str) -> None:
    """Each model's persistence is its own sum, and GJR's is the larger one.

    Reporting ``alpha + beta`` for GJR understates it by ``gamma/2`` — the same
    error the stationarity guard exists to prevent, so it is worth pinning in the
    reported field too.
    """
    returns, truth = simulate(model)
    fit = fit_garch(returns, model, burn_in=500)
    if model == "garch":
        expected = truth["alpha"] + truth["beta"]
    elif model == "gjr":
        expected = truth["alpha"] + truth["beta"] + 0.5 * truth["gamma"]
    else:
        expected = abs(truth["beta"])
    assert fit.persistence == pytest.approx(expected, abs=0.05)
    if model == "gjr":
        assert fit.persistence > truth["alpha"] + truth["beta"]


def test_fit_reports_the_unconditional_volatility_it_claims() -> None:
    """``unconditional_vol`` must be the model's own long-run variance, annualized.

    A number that is merely plausible would pass a sanity check; this compares it
    against ``sqrt(omega/(1-persistence) * annualise)`` computed from the returned
    parameters, so a wrong formula cannot hide behind a right-looking value.
    """
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500, annualise=252)
    level = math.sqrt(fit.params["omega"] / (1.0 - fit.params["beta"]) * 252)
    assert fit.unconditional_vol == pytest.approx(level, rel=1e-9)


def test_fit_drops_non_finite_observations_rather_than_propagating_them() -> None:
    """One NaN would otherwise poison the entire recursion."""
    returns, _ = simulate("garch")
    dirty = returns.copy()
    dirty[10] = numpy.nan
    dirty[900] = numpy.inf
    fit = fit_garch(dirty, "garch", burn_in=500)
    assert fit.converged, fit.message
    # n_obs is what the likelihood saw, which is the tail after burn_in and after
    # the two dropped observations — not the length of the input.
    assert fit.n_obs == len(returns) - 2 - 500


def test_fit_refuses_a_sample_too_short_to_identify_the_model() -> None:
    """Input error, not a convergence problem, so it raises rather than reporting."""
    with pytest.raises(ValueError, match="at least burn_in \\+ 50"):
        fit_garch(numpy.random.default_rng(13).normal(0.0, 0.02, 400), "garch")


def test_fit_rejects_an_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown ARCH model"):
        fit_garch(numpy.random.default_rng(13).normal(0.0, 0.02, 900), "egarch2")


@pytest.mark.parametrize("model", ("garch", "gjr", "egarch"))
def test_non_convergence_is_reported_and_never_silent(model: str) -> None:
    """``maxiter=1`` cannot converge, and the caller is told rather than misled.

    The distinction that matters: this returns a ``GarchFit`` with
    ``converged=False`` and an explanation. It does not raise, because "the
    optimiser stopped early" is a result to inspect, not an exception to throw at
    a caller who may want the partial answer.
    """
    returns, _ = simulate(model)
    fit = fit_garch(returns, model, burn_in=500, maxiter=1)
    assert isinstance(fit, GarchFit)
    assert not fit.converged
    assert fit.message
    assert len(fit.params) == (3 if model == "garch" else 4)


@pytest.mark.parametrize("field", ("converged", "loglik", "last_variance", "model"))
def test_a_fit_object_is_frozen(field: str) -> None:
    """Every field describes the same optimisation run, and none of them can change.

    If the fields were independently assignable a fit could be made to claim it
    converged after the fact, and every downstream reader would be misled about
    which optimisation produced it. Frozen at the field level, which is what
    ``frozen=True`` actually buys.

    ``params`` is the deliberate exception: it stays a plain mutable ``dict`` so
    that the documented workflow, ``bind_params(spec, {**fit.params})``, needs no
    unwrapping step. Its contents are the same numbers the frozen fields
    describe, so the risk ``frozen=True`` guards against is not reachable through
    it.
    """
    import dataclasses

    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(fit, field, 0.5)


def test_fitted_parameters_are_a_plain_dict_so_they_can_be_spread_into_a_spec() -> None:
    """The one mutability that is allowed, and the reason it is allowed."""
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    spec = INDICATORS["IND_GARCH_11"]
    bound = bind_params(spec, {**fit.params})
    for key in ("omega", "alpha", "beta"):
        assert bound[key] == pytest.approx(fit.params[key])


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def test_garch_one_step_forecast_is_the_hand_computed_value() -> None:
    """``h_{T+1} = omega + alpha * e_T^2 + beta * h_T``, then sqrt and annualize.

    Computed from the formula in the test body rather than by re-running the
    implementation, so a change to either would have to agree by coincidence.
    """
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    omega, alpha, beta = fit.params["omega"], fit.params["alpha"], fit.params["beta"]
    hand = omega + alpha * fit.last_return**2 + beta * fit.last_variance
    path = forecast(fit, 1)
    assert path[1] == pytest.approx(math.sqrt(hand * 252), rel=1e-12)


def test_gjr_one_step_forecast_uses_the_sign_switch() -> None:
    """The coefficient is ``alpha + gamma`` only when the last return was negative."""
    returns, _ = simulate("gjr")
    fit = fit_garch(returns, "gjr", burn_in=500)
    p = fit.params
    coefficient = p["alpha"] + (p["gamma"] if fit.last_return < 0.0 else 0.0)
    hand = p["omega"] + coefficient * fit.last_return**2 + p["beta"] * fit.last_variance
    assert forecast(fit, 1)[1] == pytest.approx(math.sqrt(hand * 252), rel=1e-12)


def test_gjr_forecast_uses_the_larger_persistence_to_decay() -> None:
    """A leveraged model forgets its shock more slowly, not faster.

    ``alpha + beta`` is what a *positive* shock implies; the unconditional rate is
    ``alpha + beta + gamma/2``. Using the smaller one would decay the forecast
    toward the long-run level too quickly.
    """
    returns, _ = simulate("gjr")
    fit = fit_garch(returns, "gjr", burn_in=500)
    p = fit.params
    omega = p["omega"]
    level = omega / (1.0 - p["alpha"] - p["beta"] - 0.5 * p["gamma"])
    first = forecast(fit, 1)[1] ** 2 / 252
    second = forecast(fit, 2)[2] ** 2 / 252
    assert second == pytest.approx(
        level + (first - level) * (p["alpha"] + p["beta"] + 0.5 * p["gamma"])
    )


def test_garch_multi_step_forecast_decays_geometrically_to_the_long_run_level() -> None:
    """Each step multiplies the gap by ``alpha + beta``, approaching but never
    reaching the unconditional variance.

    The level is ``omega/(1 - alpha - beta)``, not ``omega/(1 - beta)``. The
    unconditional variance belongs to the whole model, and dropping the alpha term
    is the same class of slip as testing GJR's stationarity on ``alpha + beta``.
    """
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    p = fit.params
    level = p["omega"] / (1.0 - p["alpha"] - p["beta"])
    gaps = (forecast(fit, 6)[1:] ** 2 / 252) - level
    decay = p["alpha"] + p["beta"]
    for j, gap in enumerate(gaps):
        assert gap == pytest.approx(gaps[0] * decay**j, rel=1e-9)
    assert numpy.sign(gaps[0]) == numpy.sign(gaps[-1]), (
        "decay should not cross the level"
    )
    assert abs(gaps[-1]) < abs(gaps[0])


def test_egarch_forecast_multiplies_the_log_deviation_by_beta_each_step() -> None:
    """EGARCH decays in *log* space, and its deviation changes sign every step.

    Two properties of one formula. The recursion is ``ln h = mu + beta*(ln h - mu)``
    with ``mu = omega/(1-beta)``, so the log-deviation is multiplied by ``beta`` at
    each step — and because stationarity requires ``beta < 0``, the sign flips every
    time. The conditional variance genuinely oscillates around its unconditional
    level on the way to it. That is the model working, not a wobble.

    Applying ``beta**j`` to the *variance* instead — the natural transcription from
    the GARCH branch above — is wrong twice over: the level is wrong, and the
    oscillation never decays.
    """
    returns, _ = simulate("egarch")
    fit = fit_garch(returns, "egarch", burn_in=500)
    p = fit.params
    mu = p["omega"] / (1.0 - p["beta"])
    variances = (forecast(fit, 6) ** 2) / fit.annualise
    deviations = [math.log(v) - mu for v in variances[1:]]
    for j, deviation in enumerate(deviations[1:], start=1):
        assert deviation == pytest.approx(p["beta"] * deviations[j - 1], rel=1e-9)
    assert (variances > 0).all(), "EGARCH can never report a negative variance"


def test_egarch_forecast_amplitude_shrinks_toward_the_unconditional_level() -> None:
    """The oscillation decays by ``|beta|`` per step and never grows."""
    returns, _ = simulate("egarch")
    fit = fit_garch(returns, "egarch", burn_in=500)
    mu = fit.params["omega"] / (1.0 - fit.params["beta"])
    deviations = [
        abs(math.log(v) - mu) for v in (forecast(fit, 8) ** 2) / fit.annualise
    ][1:]
    assert deviations == sorted(deviations, reverse=True)
    assert deviations[-1] < deviations[0]


def test_forecast_reports_the_last_observed_value_first() -> None:
    """Index 0 is ``sigma_T``, so a forecast is readable against its own origin."""
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    path = forecast(fit, 4)
    assert len(path) == 5
    assert path[0] == pytest.approx(math.sqrt(fit.last_variance * 252), rel=1e-12)


def test_forecast_refuses_a_fit_with_no_usable_final_variance() -> None:
    """Raising beats an array of NaNs, which on a chart looks like missing data.

    A NaN forecast and a NaN measurement are visually identical and mean opposite
    things, so this is the case where silence would be actively misleading.
    """
    returns, _ = simulate("egarch")
    fit = fit_garch(returns, "egarch", burn_in=500)
    broken = GarchFit(**{**fit.__dict__, "last_variance": float("nan")})
    with pytest.raises(ValueError, match="no usable final variance"):
        forecast(broken)


# ---------------------------------------------------------------------------
# The indicator/fitter split — the architectural invariant
# ---------------------------------------------------------------------------


def test_fit_garch_is_not_registered_and_publishes_nothing() -> None:
    """The fitter must never reach the feed surface.

    This is the invariant the whole split exists to protect. A full-sample fit
    republished as a feed id would make bar 5's value depend on bar 5000's return
    — a lookahead sitting inside the registry, invisible to the causality test
    (which only walks registered indicators) and lethal in a backtest.
    """
    from black_box.strategylib.indicators import volatility as module

    assert not hasattr(INDICATORS.get("IND_FIT_GARCH"), "fn")
    for name in ("IND_FIT_GARCH", "IND_GARCH_FIT", "IND_GARCH_11_FORECAST"):
        assert name not in INDICATORS
        assert name not in FEEDS
    # Exported from the module — it is public API for callers, just not an indicator.
    assert "fit_garch" in module.__all__
    assert "forecast" in module.__all__


def test_the_indicators_publish_one_line_each() -> None:
    """No ``params`` or ``forecast_next`` line.

    Every key a registered indicator returns is minted as a feed id, and
    ``synthesize.interpret`` requires one ndarray per node — so a dict of fitted
    parameters would become a published feed whose values are not arrays, and the
    forecast would become a line that no strategy could sanely consume. The
    parameters live on ``GarchFit`` instead, which is where nothing can address
    them by accident.
    """
    for feed_id in ARCH_IDS:
        spec = INDICATORS[feed_id]
        assert spec.lines == ("conditional_volatility",)
        assert spec.requires == ("close",), "ARCH derives returns from close alone"
        for forbidden in ("params", "forecast_next", "unconditional_vol"):
            assert forbidden not in spec.lines


def test_the_fitted_parameters_can_be_fed_back_into_the_indicator() -> None:
    """The documented workflow, end to end: fit, then run the recursion on it.

    This is the workaround for the split, so it has to actually work — a fitter
    whose output the indicators reject would leave the family unusable.
    """
    returns, _ = simulate("garch")
    fit = fit_garch(returns, "garch", burn_in=500)
    spec = INDICATORS["IND_GARCH_11"]
    values = to_host(
        spec.fn(prices(returns), **bind_params(spec, {**fit.params, "burn_in": 500}))[
            "conditional_volatility"
        ]
    )
    # The warmup prefix is NaN by design, so the agreement is asserted from the
    # first published bar — and on the last bar, which is the one the fitter saw.
    assert values[-1] == pytest.approx(math.sqrt(fit.last_variance * 252), rel=1e-9)
    assert not numpy.isnan(values[501:]).any()


def test_arch_indicators_resolve_through_the_feed_surface() -> None:
    """They are ordinary registered indicators, addressable by id."""
    data = prices(numpy.random.default_rng(14).normal(0.0, 0.02, 1200))
    for feed_id in ARCH_IDS:
        assert feed_id in FEEDS
        line = to_host(resolve_feed(feed_id, data))
        assert len(line) == 1200
        assert numpy.isfinite(line[600:]).all()
