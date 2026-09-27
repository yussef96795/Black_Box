"""Kernel-layer tests — the batched scan and running-extreme primitives.

These are the only hand-written CUDA in the codebase, and the (n, m) time-major
layout is the thing most likely to break silently: a stride mistake shows up as
one series quietly reading its neighbour's values, which still looks like a
plausible indicator. So the load-bearing assertions here are *batch
independence* (column j must be unaffected by its neighbours) and *prefix
invariance* (appending bars must not change the values already computed).

Every numeric assertion is against a plain-Python/NumPy reference, never against
a previous kernel run. `to_np` is the seam: fixtures are built in NumPy, handed
to the backend, and every expected value is computed in NumPy.
"""

from __future__ import annotations

import numpy
import pytest

from black_box.strategylib import _kernels
from black_box.strategylib._backend import IS_CUPY, np

N, M = 200, 6
ALPHAS = [0.0, 1 / 14, 0.1, 0.5, 1.0]


def to_np(x) -> numpy.ndarray:
    """Backend array -> NumPy, for reference math and assertions."""
    return x.get() if IS_CUPY else numpy.asarray(x)


def series(n: int = N, m: int = M) -> np.ndarray:
    """A price-like batch on the backend: (n, m) time-major, strictly positive."""
    rng = numpy.random.default_rng(11)
    return np.asarray(100 + rng.standard_normal((m, n)).cumsum(axis=1) * 0.5).T.copy()


def ref_scan(x: numpy.ndarray, k) -> numpy.ndarray:
    """y[t] = k*x[t] + (1-k)*y[t-1], seeded y[0] = x[0]. Scalar or per-step k."""
    n = len(x)
    out = numpy.empty_like(x)
    out[0] = x[0]
    for t in range(1, n):
        kk = k if isinstance(k, float) else k[t]
        out[t] = kk * x[t] + (1.0 - kk) * out[t - 1]
    return out


# ---------------------------------------------------------------------------
# Fixed-alpha scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_recursive_matches_reference(alpha: float) -> None:
    x = series()
    got = to_np(_kernels.recursive(x, alpha))
    assert got.shape == (N, M)
    numpy.testing.assert_allclose(got, ref_scan(to_np(x), alpha), rtol=0, atol=1e-12)


def test_recursive_batch_independence() -> None:
    """Column j must not read its neighbours — the (n, m) stride trap."""
    x = series()
    batched = to_np(_kernels.recursive(x, 0.3))
    for j in range(M):
        alone = to_np(_kernels.recursive(x[:, j], 0.3))
        numpy.testing.assert_allclose(batched[:, j], alone, rtol=0, atol=1e-12)


def test_recursive_prefix_invariance() -> None:
    """Appending bars must not change values already computed (causality)."""
    x = series(n=200)
    full = to_np(_kernels.recursive(x, 0.25))
    part = to_np(_kernels.recursive(x[:120], 0.25))
    numpy.testing.assert_allclose(part, full[:120], rtol=0, atol=1e-12)


def test_recursive_flat_input_is_flat() -> None:
    """A constant series maps to that constant — the `ema` contract."""
    numpy.testing.assert_allclose(
        to_np(_kernels.recursive(np.full(50, 42.0), 0.2)), 42.0, rtol=0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Per-step scan (KAMA's coefficient is data, not a constant)
# ---------------------------------------------------------------------------


def test_recursive_step_matches_reference() -> None:
    x = series()
    rng = numpy.random.default_rng(5)
    k = np.asarray(rng.uniform(0.0, 0.4, (N, M)))
    numpy.testing.assert_allclose(
        to_np(_kernels.recursive_step(x, k)),
        ref_scan(to_np(x), to_np(k)),
        rtol=0,
        atol=1e-12,
    )


def test_recursive_step_batch_independence() -> None:
    x = series()
    k = np.linspace(0.05, 0.3, N * M).reshape(N, M)
    batched = to_np(_kernels.recursive_step(x, k))
    for j in range(M):
        alone = to_np(_kernels.recursive_step(x[:, j], k[:, j]))
        numpy.testing.assert_allclose(batched[:, j], alone, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Running extremes
# ---------------------------------------------------------------------------


def test_running_max_matches_accumulate() -> None:
    x = series()
    numpy.testing.assert_array_equal(
        to_np(_kernels.running_max(x)),
        numpy.maximum.accumulate(to_np(x), axis=0),
    )


def test_running_min_matches_accumulate() -> None:
    x = series()
    numpy.testing.assert_array_equal(
        to_np(_kernels.running_min(x)),
        numpy.minimum.accumulate(to_np(x), axis=0),
    )


@pytest.mark.parametrize("fn", [_kernels.running_max, _kernels.running_min])
def test_extremes_batch_independence(fn) -> None:
    x = series()
    batched = to_np(fn(x))
    for j in range(M):
        alone = to_np(fn(x[:, j]))
        numpy.testing.assert_array_equal(batched[:, j], alone)


# ---------------------------------------------------------------------------
# Shape and error contract
# ---------------------------------------------------------------------------


def test_one_dimensional_input_preserves_shape() -> None:
    v = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert _kernels.recursive(v, 0.5).shape == (4,)
    assert _kernels.recursive_step(v, np.full(4, 0.5)).shape == (4,)
    assert _kernels.running_max(v).shape == (4,)
    assert _kernels.running_min(v).shape == (4,)


def test_empty_input_is_empty() -> None:
    z = np.zeros(0)
    assert _kernels.recursive(z, 0.5).shape == (0,)
    assert _kernels.recursive_step(z, z).shape == (0,)
    assert _kernels.running_max(z).shape == (0,)


def test_rejects_wrong_rank_and_mismatched_coefficients() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _kernels.recursive(np.zeros((2, 2, 2)), 0.5)
    with pytest.raises(ValueError, match="!= x shape"):
        _kernels.recursive_step(np.zeros((4, 3)), np.zeros((4, 2)))


def test_input_is_not_mutated() -> None:
    x = series()
    before = to_np(x).copy()
    _kernels.recursive(x, 0.5)
    _kernels.running_max(x)
    numpy.testing.assert_array_equal(to_np(x), before)


# ---------------------------------------------------------------------------
# NumPy fallback — the path taken on machines without CUDA
# ---------------------------------------------------------------------------


def test_fallback_matches_gpu_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the no-CUDA deployment: `cp` is NumPy, IS_CUPY is False.

    Guards against the two backends drifting apart, which is invisible on a CUDA
    dev box because only one of them ever runs there. Each side gets its own
    array in its own memory space — CuPy refuses implicit NumPy conversion, so
    the backend copy cannot be reused once `cp` is swapped.
    """
    if not IS_CUPY:  # pragma: no cover — then this *is* the live path
        pytest.skip("already on the NumPy fallback")
    x = series()
    k = np.asarray(numpy.random.default_rng(5).uniform(0.0, 0.4, (N, M)))
    gpu = {
        "recursive": to_np(_kernels.recursive(x, 0.2)),
        "step": to_np(_kernels.recursive_step(x, k)),
        "max": to_np(_kernels.running_max(x)),
        "min": to_np(_kernels.running_min(x)),
    }

    x_cpu = to_np(x).copy()
    k_cpu = to_np(k).copy()
    monkeypatch.setattr(_kernels, "cp", numpy)
    monkeypatch.setattr(_kernels, "IS_CUPY", False)
    cpu = {
        "recursive": _kernels.recursive(x_cpu, 0.2),
        "step": _kernels.recursive_step(x_cpu, k_cpu),
        "max": _kernels.running_max(x_cpu),
        "min": _kernels.running_min(x_cpu),
    }
    for name, value in gpu.items():
        numpy.testing.assert_allclose(cpu[name], value, rtol=1e-9, atol=1e-12)


def test_fallback_refuses_to_compile_a_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The compiled-kernel cache is module-level and would otherwise short-circuit
    # the guard on a box that already ran the GPU path.
    monkeypatch.setattr(_kernels, "cp", numpy)
    monkeypatch.setattr(_kernels, "IS_CUPY", False)
    monkeypatch.setattr(_kernels, "_KERNELS", {})
    with pytest.raises(RuntimeError, match="NumPy fallback"):
        _kernels._kernel("scan_fixed")


__all__: list[str] = []
