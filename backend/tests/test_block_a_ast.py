"""Tests for the Block A Stage A6 GenericPrimitiveNode recursive AST.

Covers strict recursive validation (P1 remediation): depth cap, arity rules,
Literal operator enforcement, `extra="forbid"`, registry resolution at compile
time, and the Tier 3 `signal_ast` emission path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from black_box.block_a.catalog import build_catalog
from black_box.block_a.models import (
    MAX_AST_DEPTH,
    DataGranularity,
    DataStreamNode,
    ExecutableStrategySpec,
    GenericPrimitiveNode,
    OperandNode,
    OperatorNode,
    ResourceVerdict,
    StrategyTier,
)
from black_box.block_a.spec_compiler import (
    compile_specs,
    load_registry,
    secondary_signal_ast,
    validate_ast_registry,
)
from tests.block_a_defs import vwap_abstraction, vwap_annotations, vwap_extraction

CATALOG_ROWS = [
    {
        "asset_class": "Crypto",
        "symbol": "BTCUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ETHUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
]


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    return build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)


def _verdicts() -> list[ResourceVerdict]:
    return [
        ResourceVerdict(
            asset_class="Crypto",
            symbol="BTCUSDT",
            granularity=DataGranularity.MINUTE_1,
            status="available",
            note="verified",
        )
    ]


#: GenericPrimitiveNode is an Annotated union alias — validate via TypeAdapter.
ast_adapter = TypeAdapter(GenericPrimitiveNode)


# --- leaf nodes -------------------------------------------------------------


def test_data_stream_node_builds_and_serializes() -> None:
    node = DataStreamNode(symbol="BTCUSDT", granularity="1m")
    data = node.model_dump(mode="json")
    assert data == {
        "kind": "data_stream",
        "symbol": "BTCUSDT",
        "granularity": "1m",
    }
    assert ast_adapter.validate_python(data).kind == "data_stream"


def test_operand_node_kinds() -> None:
    for kind in ("literal", "indicator", "transform"):
        node = ast_adapter.validate_python({"kind": kind, "id": "close"})
        assert node.id == "close"


def test_ast_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        OperandNode.model_validate({"kind": "indicator", "id": "IND_EMA", "boom": 1})


# --- recursion / strictness -------------------------------------------------


def test_recursive_tree_round_trips() -> None:
    tree = OperatorNode.model_validate(
        {
            "kind": "unary_op",
            "op": "zscore",
            "left": {
                "kind": "binary_op",
                "op": "ema",
                "left": {
                    "kind": "data_stream",
                    "symbol": "BTCUSDT",
                    "granularity": "1m",
                },
                "right": {"kind": "literal", "id": "20"},
            },
        }
    )
    round_tripped = OperatorNode.model_validate(
        json.loads(ast_adapter.validate_python(tree).model_dump_json())
    )
    assert round_tripped == tree


def test_ast_depth_cap_enforced() -> None:
    chain: dict = {"kind": "literal", "id": "x"}
    for _ in range(MAX_AST_DEPTH + 2):
        chain = {"kind": "unary_op", "op": "zscore", "left": chain}
    with pytest.raises(ValidationError, match="MAX_AST_DEPTH"):
        OperatorNode.model_validate(chain)


def test_ast_arity_rules() -> None:
    with pytest.raises(ValidationError, match="requires `right`"):
        OperatorNode.model_validate(
            {"kind": "binary_op", "op": "add", "left": {"kind": "literal", "id": "1"}}
        )
    with pytest.raises(ValidationError, match="forbids `right`"):
        OperatorNode.model_validate(
            {
                "kind": "unary_op",
                "op": "zscore",
                "left": {"kind": "literal", "id": "x"},
                "right": {"kind": "literal", "id": "y"},
            }
        )


def test_ast_unknown_operator_rejected() -> None:
    with pytest.raises(ValidationError, match="zscore.*ema"):
        OperatorNode.model_validate(
            {"kind": "unary_op", "op": "sqrt", "left": {"kind": "literal", "id": "1"}}
        )


# --- registry resolution (compile time) -------------------------------------


def test_validate_ast_registry_resolves_and_rejects() -> None:
    reg = load_registry()
    assert validate_ast_registry(secondary_signal_ast("IND_VOLUME_DELTA"), reg) == []
    # OHLCV series and numeric window are allowed escape hatches.
    tree = OperatorNode.model_validate(
        {
            "kind": "unary_op",
            "op": "zscore",
            "left": {
                "kind": "binary_op",
                "op": "ema",
                "left": {
                    "kind": "data_stream",
                    "symbol": "BTCUSDT",
                    "granularity": "1m",
                },
                "right": {"kind": "literal", "id": "20"},
            },
        }
    )
    assert validate_ast_registry(tree, reg) == []
    # Unknown indicator id must be flagged.
    bad = OperatorNode.model_validate(
        {
            "kind": "unary_op",
            "op": "zscore",
            "left": {"kind": "indicator", "id": "IND_UNKNOWN"},
        }
    )
    assert validate_ast_registry(bad, reg) == ["IND_UNKNOWN"]


# --- ExecutableStrategySpec integration -------------------------------------


def test_spec_without_signal_ast_unchanged() -> None:
    spec = ExecutableStrategySpec(
        spec_id="STRAT_CROSS_ABOVE_BTCUSDT_T0",
        tier=StrategyTier.TIER_0_LITERAL,
        target_asset="BTCUSDT",
        timeframe="1m",
        entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
        exit_trigger_primitive="TRIGGER_CROSS_BELOW",
    )
    assert spec.signal_ast is None
    assert (
        ExecutableStrategySpec.model_validate(json.loads(spec.model_dump_json()))
        == spec
    )


def test_spec_with_signal_ast_round_trips() -> None:
    spec = ExecutableStrategySpec(
        spec_id="STRAT_CROSS_ABOVE_BTCUSDT_T3",
        tier=StrategyTier.TIER_3_AUGMENTED,
        target_asset="BTCUSDT",
        timeframe="1m",
        entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
        exit_trigger_primitive="TRIGGER_CROSS_BELOW",
        signal_ast=secondary_signal_ast("IND_VOLUME_DELTA"),
    )
    payload = json.loads(spec.model_dump_json())
    assert payload["signal_ast"]["kind"] == "unary_op"
    assert payload["signal_ast"]["op"] == "zscore"
    revalidated = ExecutableStrategySpec.model_validate(payload)
    assert revalidated.signal_ast is not None
    assert revalidated.signal_ast.kind == "unary_op"


# --- compiler wiring --------------------------------------------------------


def test_tier3_emits_signal_ast(catalog: Path) -> None:
    specs = compile_specs(
        vwap_extraction(),
        vwap_abstraction(),
        vwap_annotations(slippage=True),
        _verdicts(),
        catalog_path=catalog,
    )
    t3 = next(s for s in specs if s.tier == StrategyTier.TIER_3_AUGMENTED)
    assert t3.signal_ast is not None
    assert t3.signal_ast.op == "zscore"
    assert validate_ast_registry(t3.signal_ast, load_registry()) == []
