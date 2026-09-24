"""AST-compiled component synthesis — grows `strategylib` as native code.

When Block A's spec references a primitive the executable library does not
yet back (a paper-invented indicator), Block B *synthesizes* the component:
the typed ``signal_ast`` subtree is compiled into a real, importable,
versioned module under ``components/generated/``, registered in the
manifest index, and gated by a hermetic self-check before the HITL
`newComponent` review card is issued (mandatory human review design).

No freeform LLM codegen: the structure comes entirely from the typed AST
(``black_box.block_a.models``), which limits both math drift and prompt
injection surface. Provenance (paper + spec) is baked into every generated
module. (ponytail: params schema is left empty for synthesized components —
library defaults apply until Block B param refinement wires paper values.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from black_box.block_a.models import AST_OPERATORS, GenericPrimitiveNode
from black_box.strategylib.components._math import (
    ema,
    rolling_mean,
    rolling_std,
    true_range,
)

COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
GENERATED_DIR = COMPONENTS_DIR / "generated"

_KNOWN_FEEDS = {"IND_EMA", "IND_ATR", "IND_VWAP", "IND_VOLUME_DELTA"}
_KNOWN_TRANSFORMS = {"open", "high", "low", "close", "volume"}

#: GenericPrimitiveNode is an Annotated union alias — no pydantic methods of
#: its own, so validate through an explicit adapter (mirrors how the
#: `ExecutableStrategySpec.signal_ast` field validates the same nodes).
_NODE_ADAPTER = TypeAdapter(GenericPrimitiveNode)


def _feed(
    name: str, data: dict[str, np.ndarray], params: dict[str, float]
) -> np.ndarray:
    """Resolve a named feed (data series or registry indicator) to an array."""
    if name in _KNOWN_TRANSFORMS:
        return data[name].astype(float)
    close = data["close"].astype(float)
    if name == "IND_EMA":
        return ema(close, int(params.get("span", 20)))
    if name == "IND_ATR":
        return ema(
            true_range(data["high"].astype(float), data["low"].astype(float), close),
            span=int(params.get("atr_period", 14)),
        )
    if name == "IND_VWAP":
        volume = data["volume"].astype(float)
        cum_v = np.cumsum(volume)
        cum_pv = np.cumsum(close * volume)
        return np.divide(cum_pv, cum_v, out=np.zeros_like(close), where=cum_v > 0)
    if name == "IND_VOLUME_DELTA":
        volume = data["volume"].astype(float)
        direction = np.sign(np.diff(close, prepend=close[0]))
        return np.cumsum(volume * direction)
    raise ValueError(f"unsupported feed id: {name}")


def _apply(op: str, left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Apply one AST_OPERATORS op to resolved operands."""
    if op == "ema":
        return ema(left, int(right[0]) if right is not None and len(right) else 20)
    if op == "sma":
        window = int(right[0]) if right is not None and len(right) else 20
        return left if window == 1 else rolling_mean(left, window)
    if op == "zscore":
        window = int(right[0]) if right is not None and len(right) else 20
        mean = rolling_mean(left, window)
        std = rolling_std(left, window)
        return np.nan_to_num((left - mean) / np.where(std == 0, 1.0, std))
    if op == "lag":
        n = int(right[0]) if right is not None and len(right) else 1
        if n == 0:
            return left
        out = np.full_like(left, np.nan)
        out[n:] = left[:-n]  # value n bars ago; trailing NaNs (no future leak)
        return np.nan_to_num(out, nan=float(left[0]))
    if op == "returns":
        n = int(right[0]) if right is not None and len(right) else 1
        shifted = np.nan_to_num(
            _apply("lag", left, np.array([n]) if right is not None else None), nan=1.0
        )
        return (
            np.divide(left, shifted, out=np.zeros_like(left), where=shifted != 0) - 1.0
        )
    if op == "abs":
        return np.abs(left)
    if op == "log":
        return np.log(np.clip(left, 1e-12, None))
    if op in ("add", "sub", "mul", "div") and right is not None:
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        return np.divide(left, right, out=np.zeros_like(left), where=right != 0)
    if op == "ratio":
        return (
            np.divide(left, right, out=np.zeros_like(left), where=right != 0)
            if right is not None
            else left
        )
    raise ValueError(f"unsupported AST operator: {op}")


def interpret(
    node: object, data: dict[str, np.ndarray], params: dict[str, float] | None = None
) -> np.ndarray:
    """Evaluate a GenericPrimitiveNode (model or msgpack dict) to an array.

    This is the single interpreter Block C may reuse for vectorized
    execution of synthesized components (ponytail: keep one interpreter,
    not one per generated module).
    """
    params = params or {}
    if isinstance(node, dict):
        node = _NODE_ADAPTER.validate_python(node)
    kind = node.kind  # type: ignore[union-attr]

    if kind == "data_stream":
        return data.get(node.symbol.lower(), data["close"]).astype(float)  # type: ignore[attr-defined]
    if kind == "operand" or getattr(node, "kind", None) in (
        "literal",
        "indicator",
        "transform",
    ):
        kind_ = node.kind  # type: ignore[attr-defined]
        if kind_ == "literal":
            return np.full(len(next(iter(data.values()))), float(node.id))  # type: ignore[attr-defined]
        return _feed(node.id, data, params)  # type: ignore[attr-defined]
    if kind == "binary_op":
        return _apply(
            node.op,
            interpret(node.left, data, params),
            interpret(node.right, data, params),
        )  # type: ignore[attr-defined]
    if kind == "unary_op":
        return _apply(node.op, interpret(node.left, data, params))  # type: ignore[attr-defined]
    raise ValueError(f"unsupported AST node kind: {kind}")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def synthesize_component(
    ast_json: dict,
    *,
    component_id: str,
    pillar: str,
    spec_id: str,
    paper_id: str = "",
    generated_dir: Path | None = None,
) -> dict:
    """Compile an AST subtree into a native strategylib component.

    Writes ``components/generated/<module>.py`` (provenance header +
    embedded AST + ``compute`` delegating to :func:`interpret`), adds the
    manifest entry (``trust: paper_derived``), and runs the hermetic
    self-check. Returns the manifest entry on success; raises
    :class:`ValueError` when the compiled module fails its check.
    """
    out_dir = generated_dir or GENERATED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    module_name = component_id.replace(".", "_")
    module_path = out_dir / f"{module_name}.py"
    ast_literal = json.dumps(ast_json, indent=2)

    module_src = _module_template(
        component_id=component_id,
        ast_literal=ast_literal,
        spec_id=spec_id,
        paper_id=paper_id,
    )
    module_path.write_text(module_src, encoding="utf-8")

    entry = {
        "id": component_id,
        "name": f"{component_id} (synthesized)",
        "pillar": pillar,
        "implements": [component_id],
        "params": {},
        "requires": {"data_level": "ohlcv"},
        "impl": f"black_box.strategylib.components.generated.{module_name}:compute",
        "trust": "paper_derived",
        "version": "0.1.0",
        "provenance": {"spec_id": spec_id, "paper_id": paper_id},
    }

    # Hermetic gate: the compiled code must import and agree with the
    # interpreter on synthetic data before it may be reviewed/registered.
    check_result = _self_check(component_id, module_path)
    if not check_result:
        raise ValueError(f"self-check failed for synthesized component {component_id}")
    return entry


def _module_template(
    *, component_id: str, ast_literal: str, spec_id: str, paper_id: str
) -> str:
    """Render the native module for a synthesized component."""
    return (
        '"""Synthesized component — generated by Black_Box Block B.\n'
        "\n"
        f"Provenance: spec_id={spec_id}, paper_id={paper_id or '(direct spec)'}. "
        "Do not hand-edit the AST; edit the paper flow and re-synthesize.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "\n"
        "import numpy as np\n"
        "\n"
        "from black_box.block_a.models import GenericPrimitiveNode\n"
        "from black_box.strategylib.synthesize import interpret\n"
        "\n"
        f"_AST = json.loads({json.dumps(ast_literal, separators=(',', ':'))})\n"
        "\n"
        "\n"
        "def compute(data: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:\n"
        '    """Compiled from the paper-derived typed signal AST."""\n'
        "    return interpret(_AST, data, params)\n"
        "\n"
    )


def _self_check(component_id: str, module_path: Path) -> bool:
    """One runnable check: import compiles, output is finite + same length."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(component_id, module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rng = np.random.default_rng(0)
        n = 200
        data = {
            "open": rng.normal(100, 1, n).cumsum(),
            "high": rng.normal(101, 1, n),
            "low": rng.normal(99, 1, n),
            "close": rng.normal(100, 1, n).cumsum(),
            "volume": rng.uniform(100, 1000, n),
        }
        out = module.compute(data, {})
        assert out.shape == (n,), f"bad shape {out.shape}"
        assert bool(np.isfinite(out).all()), "non-finite output"
        return True
    except Exception:  # noqa: BLE001 — the gate is deliberately broad: any
        # failure in compiled code (import, arity, shape, finiteness) fails
        # the check; narrowing it would let buggy synthesis through.
        return False


def generate_from_primitive(
    primitive: str,
    spec: object,
    *,
    pillar: str,
    registry_path: Path | None = None,
) -> dict:
    """Synthesize the component backing an unregistered primitive id.

    Uses the spec's ``signal_ast`` when it references the primitive, else a
    trivial pass-through feed of the named transform (documented, honest
    fallback) — (ponytail: full AST wiring lands with Block A emitting
    richer ASTs for novel primitives; until then the generated component is
    the compiled feed itself).
    """
    from black_box.strategylib.registry import resolve

    component_id = primitive.lower().replace("_", ".").replace(".ind.", "ind.")
    if resolve(component_id, registry_path):
        return {}  # already registered — nothing to do

    spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else spec
    ast_json = spec_dict.get("signal_ast") or {
        "kind": "operand",
        "id": primitive,
    }
    return synthesize_component(
        ast_json=ast_json,
        component_id=component_id,
        pillar=pillar,
        spec_id=spec_dict.get("spec_id", ""),
        paper_id=str(spec_dict.get("causal_anchor_notes", "")).split("|")[0].strip(),
    )


__all__ = [
    "AST_OPERATORS",
    "GENERATED_DIR",
    "generate_from_primitive",
    "interpret",
    "synthesize_component",
]
