"""Batched array kernels — the ONLY module in strategylib that knows a GPU exists.

Every recursive indicator (EMA, Wilder's RMA, DEMA/TEMA, KAMA, McGinley, T3) is
one scan — ``y[t] = k[t]*x[t] + (1-k[t])*y[t-1]`` — differing only in whether
``k`` is a constant or a per-step array (KAMA's efficiency ratio). Running
max/min (Hurst's drawdown) is the other shape strategylib needs and CuPy does not
provide: ``cp.maximum.accumulate`` raises NotImplementedError. So: two kernel
templates, four compilations, and every indicator above inherits them.

Two properties make the hand-written kernel the right call rather than a
premature one:

* **It is the only path on this deployment.** There is no libcublas and no
  libcufft, so ``@``, ``cp.convolve``, ``cp.fft``, ``cp.linalg.*`` and the whole
  ``cupyx.scipy.signal`` package raise ImportError. Rolling work therefore has to
  be closed-form (cumsum / sliding-window reductions) regardless, and recursion
  has to be a kernel.
* **It batches.** ``M`` independent series are scanned by ``M`` threads in one
  launch, which is exactly the shape of a Block C parameter sweep — the same
  indicator at hundreds of parameter points over the same bars. Measured on the
  dev box (RTX 4050): 256 series x 200k bars in 83 ms, max abs error 4.4e-16
  against a CPU loop, versus 5973 ms for ONE 200k-bar series through the
  Python loop this replaces.

Layout is time-major ``(n, M)``, so at step ``t`` the ``M`` threads read
``x[t*m + j]`` for consecutive ``j`` and coalesce. A ``(M, n)`` layout would put
every thread on its own cache line per step.

ponytail: block size is hardcoded at 128 and never tuned — the scan is
bandwidth-bound on a fully coalesced read, so occupancy tuning buys nothing
measurable here. The upgrade path, if a profile ever says otherwise, is a
``_THREADS`` constant read from an env var. Likewise there is no dtype
templating: everything is cast to float64 (the compute contract) rather than
carrying a float32 variant, and the ``.astype`` is free next to the scan.
"""

from __future__ import annotations

from typing import Any

from black_box.strategylib._backend import IS_CUPY, cp

#: Threads per block for the scan/running-extreme kernels. See module ponytail.
_THREADS = 128

# `k` is either a scalar (fixed alpha) or a pointer to a per-step array; the two
# differ only in the declaration and the read, so they are two compilations of
# one template rather than two hand-written kernels.
_SCAN_SRC = r"""
extern "C" __global__ void scan(double* x, double* y,
                                const long n, const long m, %(kdecl)s)
{
    long j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    double prev = x[j];
    y[j] = prev;
    for (long t = 1; t < n; t++) {
        double kk = %(kexpr)s;
        prev = kk * x[t * m + j] + (1.0 - kk) * prev;
        y[t * m + j] = prev;
    }
}
"""

_EXTREME_SRC = r"""
extern "C" __global__ void runext(double* x, double* y,
                                  const long n, const long m)
{
    long j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    double acc = x[j];
    y[j] = acc;
    for (long t = 1; t < n; t++) {
        double v = x[t * m + j];
        acc = %(pick)s;
        y[t * m + j] = acc;
    }
}
"""

_VARIANTS: dict[str, str] = {
    "scan_fixed": _SCAN_SRC % {"kdecl": "const double a", "kexpr": "a"},
    "scan_step": _SCAN_SRC % {"kdecl": "const double* k", "kexpr": "k[t * m + j]"},
    "ext_max": _EXTREME_SRC % {"pick": "(v > acc ? v : acc)"},
    "ext_min": _EXTREME_SRC % {"pick": "(v < acc ? v : acc)"},
}

#: Variant name -> the C entry point it compiles. Not derivable from the name
#: (the running-extreme kernel is called `runext`, not `ext`).
_ENTRY_POINT = {
    "scan_fixed": "scan",
    "scan_step": "scan",
    "ext_max": "runext",
    "ext_min": "runext",
}

#: Compiled lazily and cached — NVRTC compilation is ~100ms and every
#: recursive indicator in a sweep would otherwise pay it again.
_KERNELS: dict[str, Any] = {}


def _kernel(name: str) -> Any:
    if name not in _KERNELS:
        if not IS_CUPY:  # pragma: no cover — guarded by the public wrappers
            raise RuntimeError("RawKernel requested on the NumPy fallback")
        _KERNELS[name] = cp.RawKernel(_VARIANTS[name], _ENTRY_POINT[name])
    return _KERNELS[name]


def _prepare(x: Any) -> tuple[Any, int]:
    """Cast to contiguous float64 time-major ``(n, m)``; return it and input ndim.

    A 1-D input is a single series and becomes ``(n, 1)``; the public wrappers
    restore the caller's shape.
    """
    arr = cp.ascontiguousarray(x, dtype=float)
    ndim = arr.ndim
    if ndim == 1:
        arr = arr.reshape(-1, 1)
    elif ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D (n, m) array, got {ndim}-D")
    return arr, ndim


def _restore(out: Any, ndim: int) -> Any:
    return out[:, 0] if ndim == 1 else out


def _fallback_scan(arr: Any, k: Any) -> Any:
    """NumPy fallback for both scan variants — one loop, scalar or per-step ``k``.

    ponytail: a Python loop, i.e. one device round-trip per bar — the exact cost
    the kernel exists to remove. It is here only because ``_backend`` keeps NumPy
    as an import-time fallback so the package imports and the suite still runs
    without CUDA, and on that path the fixtures are a few hundred bars where a
    loop is irrelevant. The upgrade path, if a no-GPU machine ever needs real
    sweeps, is ``scipy.signal.lfilter`` — a three-line swap for the fixed-alpha
    case, since it solves the identical difference equation in C. Deliberately
    NOT a hard dependency: 30MB to speed up CI is the wrong trade.
    """
    n, m = arr.shape
    out = cp.empty_like(arr)
    if n == 0:
        return out
    fixed = isinstance(k, float)
    for j in range(m):
        prev = arr[0, j]
        out[0, j] = prev
        for t in range(1, n):
            kk = k if fixed else k[t, j]
            prev = kk * arr[t, j] + (1.0 - kk) * prev
            out[t, j] = prev
    return out


def _fallback_extreme(arr: Any, maximum: bool) -> Any:
    """NumPy fallback for running max/min — ``accumulate`` is native here."""
    acc = cp.maximum if maximum else cp.minimum
    return acc.accumulate(arr, axis=0)


# ---------------------------------------------------------------------------
# Public API — backend-agnostic, shape-preserving
# ---------------------------------------------------------------------------


def recursive(x: Any, alpha: float) -> Any:
    """``y[t] = alpha*x[t] + (1-alpha)*y[t-1]``, seeded ``y[0] = x[0]``.

    Accepts ``(n,)`` or time-major ``(n, m)``; returns the same shape.
    """
    arr, ndim = _prepare(x)
    if arr.shape[0] == 0:
        return _restore(arr.copy(), ndim)
    a = float(min(max(float(alpha), 0.0), 1.0))
    if IS_CUPY:
        out = cp.empty_like(arr)
        _launch(
            "scan_fixed",
            arr.shape,
            (arr, out, cp.int64(arr.shape[0]), cp.int64(arr.shape[1]), cp.float64(a)),
        )
    else:
        out = _fallback_scan(arr, a)
    return _restore(out, ndim)


def recursive_step(x: Any, k: Any) -> Any:
    """Scan with a per-step coefficient ``k`` (same shape as ``x``).

    KAMA's smoothing constant is recomputed every bar from the efficiency
    ratio, so the coefficient is data rather than a constant.
    """
    arr, ndim = _prepare(x)
    k_arr, _ = _prepare(k)
    if arr.shape != k_arr.shape:
        raise ValueError(f"k shape {k_arr.shape} != x shape {arr.shape}")
    if arr.shape[0] == 0:
        return _restore(arr.copy(), ndim)
    if IS_CUPY:
        out = cp.empty_like(arr)
        _launch(
            "scan_step",
            arr.shape,
            (arr, out, cp.int64(arr.shape[0]), cp.int64(arr.shape[1]), k_arr),
        )
    else:
        out = _fallback_scan(arr, k_arr)
    return _restore(out, ndim)


def running_max(x: Any) -> Any:
    """Cumulative maximum along the time axis; ``(n,)`` or ``(n, m)`` in/out."""
    return _extreme(x, maximum=True)


def running_min(x: Any) -> Any:
    """Cumulative minimum along the time axis; ``(n,)`` or ``(n, m)`` in/out."""
    return _extreme(x, maximum=False)


def _extreme(x: Any, *, maximum: bool) -> Any:
    arr, ndim = _prepare(x)
    if arr.shape[0] == 0:
        return _restore(arr.copy(), ndim)
    if IS_CUPY:
        out = cp.empty_like(arr)
        _launch(
            "ext_max" if maximum else "ext_min",
            arr.shape,
            (arr, out, cp.int64(arr.shape[0]), cp.int64(arr.shape[1])),
        )
    else:
        out = _fallback_extreme(arr, maximum)
    return _restore(out, ndim)


def _launch(name: str, shape: tuple[int, int], args: tuple[Any, ...]) -> None:
    blocks = (shape[1] + _THREADS - 1) // _THREADS
    _kernel(name)((blocks, 1), (_THREADS,), args)


__all__ = [
    "recursive",
    "recursive_step",
    "running_max",
    "running_min",
]
