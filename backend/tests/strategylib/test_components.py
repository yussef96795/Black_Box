"""strategylib tests — registry, numeric component self-checks, synthesis.

Component contract (manifest.json ``contract``): ``compute(data, params)``
returns a same-length ``np.ndarray``; ``data`` keys are
open/high/low/close/volume (equal length). Curated components are checked
numerically here so a silent math drift fails loudly, not in Block C.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from black_box.strategylib import append_manifest, by_pillar, matching, resolve
from black_box.strategylib.components._math import ema
from black_box.strategylib.registry import load_registry
from black_box.strategylib.synthesize import (
    generate_from_primitive,
    interpret,
    synthesize_component,
)

N = 200
EXPECTED_IDS = [
    "entry.ema_cross",
    "entry.vwap_band",
    "exit.atr_stop",
    "exit.time_exit",
    "sizing.fractional",
    "sizing.atr_scaled",
]


@pytest.fixture()
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    close = rng.normal(100, 2, N).cumsum()
    return {
        "open": close - 0.2,
        "high": close + 0.8,
        "low": close - 0.8,
        "close": close,
        "volume": rng.uniform(100, 1000, N),
    }


def flat_data() -> dict[str, np.ndarray]:
    close = np.full(N, 100.0)
    return {
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(N, 500.0),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_six_curated_components() -> None:
    registry = load_registry()
    assert set(registry) == set(EXPECTED_IDS)
    for cid in EXPECTED_IDS:
        assert resolve(cid) is not None
        assert resolve(cid).trust == "curated"  # type: ignore[union-attr]


def test_registry_pillar_and_matching_lookups() -> None:
    assert [c.id for c in by_pillar("entry")] == ["entry.ema_cross", "entry.vwap_band"]
    assert by_pillar("sizing")[0].id == "sizing.atr_scaled"  # id-sorted
    assert resolve("sizing.fractional").params["fraction"]["default"] == 0.01  # type: ignore[union-attr]
    assert {c.id for c in matching("IND_ATR")} == {"exit.atr_stop", "sizing.atr_scaled"}


def test_resolve_registry_miss_returns_none() -> None:
    assert resolve("entry.does_not_exist") is None


# ---------------------------------------------------------------------------
# Component contract — every curated component computes on synthetic data
# ---------------------------------------------------------------------------


def test_compute_output_contract(data: dict[str, np.ndarray]) -> None:
    for cid in EXPECTED_IDS:
        out = resolve(cid).compute(data, {})  # type: ignore[union-attr]
        assert isinstance(out, np.ndarray)
        assert out.shape == (N,), f"{cid}: shape {out.shape} != ({N},)"
        assert np.isfinite(out).all(), f"{cid}: non-finite output"
        assert out.dtype.kind == "f", f"{cid}: expected float dtype"


# ---------------------------------------------------------------------------
# Numeric self-checks per component
# ---------------------------------------------------------------------------


def test_ema_cross_flat_price_no_signals() -> None:
    out = resolve("entry.ema_cross").compute(flat_data(), {"fast": 5, "slow": 20})  # type: ignore[union-attr]
    assert (out == 0).all()


def test_ema_cross_signals_after_price_jump() -> None:
    close = np.full(N, 100.0)
    close[100:] = 200.0
    d = {
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(N, 500.0),
    }
    out = resolve("entry.ema_cross").compute(d, {"fast": 3, "slow": 10})  # type: ignore[union-attr]
    assert (out[:100] == 0).all(), "no cross before the regime change"
    assert (out[100:] >= 0).all(), "fast EMA never crosses below after a surge"
    assert np.any(out[100:] == 1), "a fast-above-slow cross must fire post jump"


def test_vwap_band_breakout_spike() -> None:
    close = np.full(N, 100.0)
    close[100:] = 100.0 + np.arange(N - 100) * 0.5  # steady rise -> vwap lags
    d = {
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(N, 500.0),
    }
    out = resolve("entry.vwap_band").compute(d, {"lookback": 20, "k": 2.0})  # type: ignore[union-attr]
    assert out[-1] == 1.0, "close well above the upper band must trigger +1"
    assert (out[:100] == 0).all(), "no breakout during the flat regime"


def test_atr_stop_flat_is_constant_stop() -> None:
    out = resolve("exit.atr_stop").compute(flat_data(), {"atr_period": 14, "k": 2.0})  # type: ignore[union-attr]
    # high-low spread is constant 2.0 -> ATR = 2.0 -> stop = k * 2.0
    np.testing.assert_allclose(out, 4.0)


def test_time_exit_is_step_at_bars() -> None:
    out = resolve("exit.time_exit").compute(flat_data(), {"bars": 5})  # type: ignore[union-attr]
    assert (out[:4] == 0).all()
    assert (out[4:] == 1).all()


def test_fractional_sizing_is_constant_fraction() -> None:
    out = resolve("sizing.fractional").compute(flat_data(), {"fraction": 0.02})  # type: ignore[union-attr]
    np.testing.assert_allclose(out, 0.02)


def test_atr_scaled_flat_is_risk_pct() -> None:
    out = resolve("sizing.atr_scaled").compute(
        flat_data(), {"risk_pct": 0.01, "atr_period": 14}
    )  # type: ignore[union-attr]
    # Constant spread -> constant ATR -> base == atr -> fraction == risk_pct.
    np.testing.assert_allclose(out, 0.01)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def test_interpret_ema_matches_math_helper(data: dict[str, np.ndarray]) -> None:
    ast = {
        "kind": "binary_op",
        "op": "ema",
        "left": {"kind": "transform", "id": "close"},
        "right": {"kind": "literal", "id": "5"},
    }
    out = interpret(ast, data)
    np.testing.assert_allclose(out, ema(data["close"].astype(float), 5))


def test_interpret_planted_zscore_ast(data: dict[str, np.ndarray]) -> None:
    ast = {
        "kind": "binary_op",
        "op": "zscore",
        "left": {
            "kind": "binary_op",
            "op": "ema",
            "left": {"kind": "transform", "id": "close"},
            "right": {"kind": "literal", "id": "3"},
        },
        "right": {"kind": "literal", "id": "20"},
    }
    out = interpret(ast, data)
    assert out.shape == (N,)
    assert np.isfinite(out).all()
    # zscore of a constant stretch is ~0
    flat = flat_data()
    flat_out = interpret(ast, flat)
    np.testing.assert_allclose(flat_out, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Synthesis (AST -> native module + manifest entry)
# ---------------------------------------------------------------------------


def test_synthesize_component_writes_module_and_entry(tmp_path) -> None:
    ast = {
        "kind": "binary_op",
        "op": "zscore",
        "left": {
            "kind": "binary_op",
            "op": "ema",
            "left": {"kind": "transform", "id": "close"},
            "right": {"kind": "literal", "id": "5"},
        },
        "right": {"kind": "literal", "id": "20"},
    }
    entry = synthesize_component(
        ast,
        component_id="ind.zscore_dev",
        pillar="codebase",
        spec_id="STRAT_SYNTH_T0",
        paper_id="paper-1",
        generated_dir=tmp_path,
    )

    assert entry["id"] == "ind.zscore_dev"
    assert entry["trust"] == "paper_derived"
    assert entry["provenance"]["spec_id"] == "STRAT_SYNTH_T0"
    # Dotted component id -> underscore module path (importable).
    assert entry["impl"].endswith("components.generated.ind_zscore_dev:compute")

    module_path = tmp_path / "ind_zscore_dev.py"
    assert module_path.exists()
    src = module_path.read_text(encoding="utf-8")
    assert "STRAT_SYNTH_T0" in src  # provenance baked in
    assert "interpret(_AST, data, params)" in src

    data = flat_data()
    module = _load_module(module_path)
    np.testing.assert_allclose(module.compute(data, {}), interpret(ast, data))


def test_synthesize_self_check_gate_rejects_discovery_failure(tmp_path) -> None:
    # A literal-only "component" is a plain constant — still compiles and
    # passes (shape + finiteness are the contract). Guard the gate itself:
    # a broken template must raise, so assert synthesize raises when the
    # module cannot be produced (e.g. unreadable ast) via a non-dict ast.
    with pytest.raises((ValueError, TypeError)):
        synthesize_component(
            "not-a-dict",  # type: ignore[arg-type]
            component_id="ind.broken",
            pillar="codebase",
            spec_id="S",
            generated_dir=tmp_path,
        )


def test_append_manifest_idempotent(tmp_path) -> None:
    import json
    import shutil

    from black_box.strategylib.registry import MANIFEST_PATH

    manifest = tmp_path / "manifest.json"
    shutil.copy(MANIFEST_PATH, manifest)
    entry = {
        "id": "ind.zscore_dev",
        "name": "ind.zscore_dev (synthesized)",
        "pillar": "codebase",
        "implements": ["ind.zscore_dev"],
        "params": {},
        "requires": {"data_level": "ohlcv"},
        "impl": "black_box.strategylib.components.generated.ind_zscore_dev:compute",
        "trust": "paper_derived",
        "version": "0.1.0",
    }
    append_manifest(entry, path=manifest)
    append_manifest(entry, path=manifest)  # idempotent
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["components"]) == len(EXPECTED_IDS) + 1


def test_generate_from_primitive_writes_generated_module(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("black_box.strategylib.synthesize.GENERATED_DIR", tmp_path)
    spec_dict = {
        "spec_id": "STRAT_NOVEL_T0",
        "causal_anchor_notes": "paper-9 | section 3",
        "signal_ast": {
            "kind": "binary_op",
            "op": "div",
            "left": {"kind": "transform", "id": "close"},
            "right": {"kind": "transform", "id": "open"},
        },
    }
    entry = generate_from_primitive("IND_NOVEL_SIGNAL", spec_dict, pillar="codebase")
    assert entry["id"] == "ind.novel.signal"
    assert entry["pillar"] == "codebase"
    assert (tmp_path / "ind_novel_signal.py").exists()


def _load_module(path) -> object:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__: list[str] = []
