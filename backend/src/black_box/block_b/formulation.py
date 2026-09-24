"""Block B — strategy formulation engine (deterministic core).

Block B consumes the curated A6 spec (``ExecutableStrategySpec`` from
Block A) and produces backtest-ready *formulations*: complete
entry/exit/sizing triples with a designed parameter sweep grid. Missing
pillars and unbacked primitives surface as structural errors that the HITL
gate resolves via typed cards (see ``hitl.py``).

This module is pure and deterministic — no LLM in the pre-audit path. The
Cynical Auditor (``auditor.py``) is the only other deterministic stage;
LLM-backed variants are the documented upgrade path (ponytail).
"""

from __future__ import annotations

from typing import Any

from black_box.block_a.models import DataGranularity
from black_box.strategylib import by_pillar, resolve
from black_box.strategylib.synthesize import _KNOWN_FEEDS, _KNOWN_TRANSFORMS

MAX_VARIANTS = 4
_TRIGGER_TO_EXIT = {
    "TRIGGER_CROSS_ABOVE": "exit.atr_stop",
    "TRIGGER_CROSS_BELOW": "exit.atr_stop",
    "TRIGGER_BAND_BREAKOUT": "exit.time_exit",
}
_GRANULARITIES = {g.value for g in DataGranularity}


def referenced_primitives(spec: dict[str, Any]) -> set[str]:
    """Every primitive id the spec depends on (triggers, indicators, filters).

    Used by the structural gate to find vocabulary with no executable
    component backing — the trigger for the `missingPrimitive` synthesis
    path (a paper-invented indicator becomes native code).
    """
    prims: set[str] = set()
    for key in ("entry_trigger_primitive", "exit_trigger_primitive"):
        value = spec.get(key)
        if value:
            prims.add(value)
    params = spec.get("parameters") or {}
    for key in ("indicators", "filters", "filter_primitives"):
        for item in params.get(key) or spec.get(key) or []:
            if item:
                prims.add(item)
    if params.get("secondary_signal"):
        prims.add(params["secondary_signal"])
    _walk_ast_ids(spec.get("signal_ast"), prims)
    return prims


def _walk_ast_ids(node: dict[str, Any] | None, acc: set[str]) -> None:
    """Collect operand ids (indicator/transform/literal) from a signal AST."""
    if not isinstance(node, dict):
        return
    if node.get("kind") in ("indicator", "transform") and node.get("id"):
        acc.add(node["id"])
    for child in (node.get("left"), node.get("right")):
        _walk_ast_ids(child, acc)


def unbacked_primitives(spec: dict[str, Any], registry_path=None) -> list[str]:
    """Primitive ids with vocabulary but no executable component behind them."""
    known = set(_KNOWN_FEEDS) | set(_KNOWN_TRANSFORMS) | {"literal"}
    backed = {
        pid
        for comp in by_pillar("entry", registry_path)
        + by_pillar("exit", registry_path)
        + by_pillar("sizing", registry_path)
        for pid in comp.implements
    }
    return sorted(referenced_primitives(spec) - known - backed)


def _pillar_grid(component) -> dict[str, dict[str, float]]:
    """Design the sweep grid for a component from its manifest param schema."""
    grid: dict[str, dict[str, float]] = {}
    for name, spec in (component.params or {}).items():
        default = float(spec.get("default", 0.0))
        low = float(spec.get("min", default))
        high = float(spec.get("max", default))
        step = round((high - low) / 5.0, 6) or 1.0
        grid[name] = {
            "default": default,
            "lo": low,
            "hi": high,
            "step": max(step, 1e-9),
        }
    return grid


def pillar_payload(component, registry_path=None) -> dict:
    """Concrete pillar (component_id + params + sweep grid) for a component.

    Public because the HITL resume path re-derives the full payload on
    `accept` — reusing this keeps the sweep surface consistent wherever it
    is built.
    """
    grid = _pillar_grid(component)
    return {
        "component_id": component.id,
        "params": {name: cfg["default"] for name, cfg in grid.items()},
        "grid": grid,
    }


def _pillar_payload(component_id: str | None, registry_path=None) -> dict | None:
    """Resolve a component id to a concrete pillar payload (or None)."""
    if not component_id:
        return None
    component = resolve(component_id, registry_path)
    return pillar_payload(component, registry_path) if component is not None else None


def _candidates(
    pillar: str, trigger: str, fallback: str, registry_path=None
) -> list[str]:
    """Entry/exit candidates: trigger-aware first, pillar fallback after."""
    matched = [
        c.id for c in by_pillar(pillar, registry_path) if trigger in c.implements
    ]
    fallbacks = [c.id for c in by_pillar(pillar, registry_path)]
    ordered = list(dict.fromkeys(matched + fallbacks))  # dedupe, keep order
    return ordered or ([fallback] if fallback else [])


def hydrate_spec(spec: dict[str, Any], registry_path=None) -> dict[str, Any]:
    """Spec -> formulations (variants) + structural errors.

    Variants are entry/exit/sizing triples; the parameter *surface* that
    Block C sweeps is designed here from the component manifest schemas
    (Block A's `parameters` carry intent/invariants, not bounds).
    """
    errors: list[dict[str, Any]] = []
    params = spec.get("parameters") or {}
    sizing_cfg = params.get("sizing") or {}
    sizing_id = sizing_cfg.get("component_id") if isinstance(sizing_cfg, dict) else None

    entry_trigger = spec.get("entry_trigger_primitive") or ""
    exit_trigger = spec.get("exit_trigger_primitive") or ""
    exit_fallback = _TRIGGER_TO_EXIT.get(exit_trigger, "exit.atr_stop")

    entry_options = _candidates(
        "entry", entry_trigger, "entry.ema_cross", registry_path
    )
    exit_options = _candidates("exit", exit_trigger, exit_fallback, registry_path)

    data = {
        "symbol": spec.get("target_asset", ""),
        "timeframe": spec.get("timeframe", ""),
    }

    variants: list[dict[str, Any]] = []
    for idx in range(min(MAX_VARIANTS, max(len(entry_options), len(exit_options)))):
        entry_cfg = _pillar_payload(
            entry_options[idx] if idx < len(entry_options) else entry_options[0],
            registry_path,
        )
        exit_cfg = _pillar_payload(
            exit_options[idx] if idx < len(exit_options) else exit_options[0],
            registry_path,
        )
        sizing_cfg_p = _pillar_payload(sizing_id, registry_path)
        # A6 never emits sizing — paper-defined sizing arrives via spec
        # params (e.g. {"sizing": {"component_id": ..., "params": ...}});
        # otherwise the missingPillar card proposes a curated default.
        if sizing_cfg_p and isinstance(sizing_cfg, dict):
            sizing_cfg_p["params"].update(sizing_cfg.get("params") or {})
        variants.append(
            {
                "variant_id": f"V{idx}",
                "entry": entry_cfg,
                "exit": exit_cfg,
                "sizing": sizing_cfg_p,
                "data": dict(data),
                "source": "block_a",
            }
        )

    for variant in variants:
        for pillar in ("entry", "exit", "sizing"):
            if variant[pillar] is None:
                errors.append(
                    {
                        "type": "missing_pillar",
                        "pillar": pillar,
                        "variant_id": variant["variant_id"],
                        "message": f"No {pillar} component is defined or resolvable.",
                    }
                )
        grid_err = _grid_bounds_errors(variant)
        errors.extend(grid_err)

    errors.extend(_data_errors(spec))

    return {"variants": variants, "structural_errors": errors}


def _data_errors(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Universe/binding checks — reused by hydration AND the re-audit gate
    (which overwrites hydration errors on every pass, so the gate must own
    the full error set)."""
    errors: list[dict[str, Any]] = []
    symbol = spec.get("target_asset", "")
    timeframe = spec.get("timeframe", "")
    if not symbol or not timeframe:
        errors.append(
            {
                "type": "bad_data",
                "pillar": "universe",
                "message": "Spec has no target_asset/timeframe.",
            }
        )
    elif timeframe not in _GRANULARITIES:
        errors.append(
            {
                "type": "bad_data",
                "pillar": "universe",
                "message": f"timeframe {timeframe!r} not in {sorted(_GRANULARITIES)}",
            }
        )
    return errors


def _grid_bounds_errors(variant: dict[str, Any]) -> list[dict[str, Any]]:
    """Grid sanity: lo < hi, step > 0, default within [lo, hi]."""
    errors: list[dict[str, Any]] = []
    for pillar in ("entry", "exit", "sizing"):
        cfg = variant.get(pillar)
        if not cfg:
            continue
        for name, bound in (cfg.get("grid") or {}).items():
            lo, hi = bound.get("lo", 0.0), bound.get("hi", 0.0)
            if (
                lo >= hi
                or bound.get("step", 0.0) <= 0
                or not (lo <= bound.get("default", lo) <= hi)
            ):
                errors.append(
                    {
                        "type": "grid_bounds",
                        "pillar": pillar,
                        "param": name,
                        "message": f"Invalid sweep grid for {name}: {bound}",
                    }
                )
    return errors


def structural_gate(state: Any, registry_path=None) -> list[dict[str, Any]]:
    """Deterministic well-posedness gate over the current variants.

    Runs at hydration AND on every re-audit after a HITL resume — so an
    approved `missingPrimitive` registration clears the error naturally on
    the next pass.
    """
    errors: list[dict[str, Any]] = []
    spec = state.spec or {}
    errors.extend(_data_errors(spec))
    for pid in unbacked_primitives(spec, registry_path):
        errors.append(
            {
                "type": "unbacked_primitive",
                "pillar": "codebase",
                "primitive": pid,
                "message": f"Primitive {pid} has vocabulary but no executable component behind it.",
                "suggestion": "Synthesize it from the paper AST, or substitute a curated component.",
            }
        )
    for variant in state.variants or []:
        for pillar in ("entry", "exit", "sizing"):
            cfg = variant.get(pillar)
            if cfg is None:
                errors.append(
                    {
                        "type": "missing_pillar",
                        "pillar": pillar,
                        "variant_id": variant["variant_id"],
                        "message": f"Variant has no {pillar} component.",
                    }
                )
                continue
            if resolve(cfg.get("component_id"), registry_path) is None:
                errors.append(
                    {
                        "type": "missing_pillar",
                        "pillar": pillar,
                        "variant_id": variant["variant_id"],
                        "message": f"Component {cfg.get('component_id')!r} is not registered.",
                    }
                )
        errors.extend(_grid_bounds_errors(variant))
    return errors


__all__ = [
    "MAX_VARIANTS",
    "hydrate_spec",
    "referenced_primitives",
    "structural_gate",
    "unbacked_primitives",
]
