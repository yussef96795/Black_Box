"""Array backend for strategylib — CuPy (GPU) primary, NumPy CPU fallback.

Everything in the library (components, math helpers, interpreter, synthesis
self-check, tests) executes on the CuPy GPU backend when a CUDA device is
available. The manifest ``compute`` contract is backend-agnostic:
``dict[str, ndarray] -> ndarray`` of equal length.

ponytail: NumPy is kept ONLY as an import-time fallback so the package still
imports and the test suite still runs on machines without CUDA (CI / dev
laptops). Delete the fallback line to enforce a hard GPU-only dependency.
"""

from __future__ import annotations

try:
    import cupy as cp  # GPU backend (CUDA)

    BACKEND = "cupy"
except ImportError:  # pragma: no cover — no CUDA driver/toolkit
    import numpy as np  # type: ignore[no-redef]

    cp = np  # type: ignore[assignment] — `cp` is the canonical handle, always bound
    BACKEND = "numpy"

#: `np` is the spelling strategylib code uses at call sites (CuPy mirrors the
#: NumPy API, so one name covers both backends); `cp` is the canonical handle and
#: what GPU-only code — `_kernels.py`'s RawKernel — imports to assert it is on
#: CuPy. Both are bound unconditionally: previously `np` existed *only* in the
#: no-CuPy branch, so `import black_box.strategylib` raised ImportError on every
#: machine that actually has a GPU.
#:
#: ponytail: two names for one module, rather than renaming all ~10 call sites
#: from `np` to `cp`. Two spellings is a wart; the one-line alias is the cheap
#: unblock, and the upgrade path is a repo-wide rename to `cp` once the numpy
#: fallback is deleted above (then `np` stops being a lie and only `cp` remains).
np = cp

#: True when the CuPy backend is live. `_kernels.py` dispatches on this to pick
#: the RawKernel path or the NumPy/SciPy fallback.
#:
#: ponytail: `BACKEND` says the *import* succeeded, not that a device exists.
#: A CuPy install with no visible GPU imports fine and then fails at first
#: kernel launch. Probing the device at import time costs ~200ms of CUDA init
#: on every `import strategylib`, so the probe stays lazy — the first
#: `_kernels` call is where a broken install surfaces. Upgrade path if that
#: proves annoying: probe once behind an env flag.
IS_CUPY = BACKEND == "cupy"

__all__ = ["BACKEND", "IS_CUPY", "cp", "np"]
