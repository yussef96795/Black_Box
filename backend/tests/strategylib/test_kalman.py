"""Check IND_KALMAN: matches the textbook recursion, and honours the contract.

The reference below is written straight from the state-space equations rather
than from the implementation, so it fails if the loop transcribes them wrong
rather than merely agreeing with itself.
"""

from __future__ import annotations

import numpy as np

from black_box.strategylib._backend import np as bp
from black_box.strategylib._math import to_device, to_host
from black_box.strategylib.indicators import FEEDS
from black_box.strategylib.indicators.trend import kalman


def _reference(close: np.ndarray, q: float, r: float) -> np.ndarray:
    """The local-level Kalman recursion, one bar at a time, unoptimised."""
    x, p = float(close[0]), 1.0
    out = np.zeros((3, len(close)))
    for t in range(len(close)):
        p_prior = p + q
        k = p_prior / (p_prior + r)
        x += k * (float(close[t]) - x)
        p = (1.0 - k) * p_prior
        out[:, t] = x, p**0.5, k
    return out


def _close(n: int = 400) -> np.ndarray:
    return 100.0 + np.cumsum(np.random.default_rng(11).normal(0, 1, n))


def test_matches_the_state_space_recursion() -> None:
    close = _close()
    out = kalman({"close": to_device(close)})
    expected = _reference(close, 1e-5, 1e-3)
    for row, line in enumerate(("line", "error", "gain")):
        assert np.allclose(to_host(out[line]), expected[row])


def test_seeds_on_the_first_price() -> None:
    """Bar 0 is the state, not a warmup: no NaN prefix, starts on the close."""
    close = _close()
    out = kalman({"close": to_device(close)})
    assert float(to_host(out["line"])[0]) == close[0]


def test_is_causal() -> None:
    """Truncating the future must not move any earlier value."""
    close = _close()
    full = kalman({"close": to_device(close)})
    cut = kalman({"close": to_device(close[:200])})
    for line in ("line", "error", "gain"):
        assert np.allclose(to_host(full[line])[:200], to_host(cut[line]))


def test_contract_shape_and_finiteness() -> None:
    close = _close()
    out = kalman({"close": to_device(close)})
    for line, array in out.items():
        assert isinstance(array, type(bp.zeros(0))), line
        assert len(array) == len(close), line
        assert np.isfinite(to_host(array)).all(), line


def test_empty_input_returns_three_empty_lines() -> None:
    out = kalman({"close": to_device(np.zeros(0))})
    assert set(out) == {"line", "error", "gain"}
    assert all(len(v) == 0 for v in out.values())


def test_flat_input_stays_flat() -> None:
    """A line that never moves has nothing to track; it must not drift."""
    out = kalman({"close": to_device(np.full(120, 42.0))})
    assert float(to_host(out["line"])[-1]) == 42.0
    # K settles once P reaches its steady state. Bar 0 is deliberately not that
    # value — P_0 = 1.0 is pessimistic, so the first bar trusts the measurement.
    gain = to_host(out["gain"])
    assert np.allclose(gain[-1], gain[-2], atol=1e-9)
    assert 0.0 < gain[-1] < 1.0


def test_registry_addresses_every_line() -> None:
    """Multi-line feeds are opt-in by name; a missing one hides a whole line."""
    assert FEEDS["IND_KALMAN"][1] == "line"
    assert FEEDS["IND_KALMAN_ERROR"][1] == "error"
    assert FEEDS["IND_KALMAN_GAIN"][1] == "gain"
