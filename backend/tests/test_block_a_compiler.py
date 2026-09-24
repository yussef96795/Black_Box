"""Tests for Block A Stage A6 — DSL compiler, tiering, pruning, export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from black_box.block_a.catalog import build_catalog
from black_box.block_a.models import (
    DataGranularity,
    ExecutableStrategySpec,
    OperatorNode,
    ResourceVerdict,
    RiskTag,
    StrategyTier,
)
from black_box.block_a.spec_compiler import (
    _alternate_symbol,
    compile_specs,
    detect_indicators,
    detect_timeframe,
    load_registry,
    map_entry_primitive,
    map_exit_primitive,
    pick_secondary_signal,
    primary_guard_filter,
    risk_union,
    split_entry_exit,
    validate_and_export,
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
    {
        "asset_class": "Crypto",
        "symbol": "ADAUSDT",
        "granularity": "1m",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
]


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    return build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)


def _verdicts(symbol: str = "BTCUSDT") -> list[ResourceVerdict]:
    return [
        ResourceVerdict(
            asset_class="Crypto",
            symbol=symbol,
            granularity=DataGranularity.MINUTE_1,
            status="available",
            note="verified",
        )
    ]


# --- primitive mapping -----------------------------------------------------


def test_entry_primitive_mapping() -> None:
    reg = load_registry()
    assert map_entry_primitive("price crosses above VWAP", reg) == "TRIGGER_CROSS_ABOVE"
    assert (
        map_entry_primitive("buys when price touches the lower band", reg)
        == "TRIGGER_CROSS_BELOW"
    )
    assert (
        map_entry_primitive("band breakout above the range", reg)
        == "TRIGGER_BAND_BREAKOUT"
    )


def test_exit_primitive_mapping() -> None:
    reg = load_registry()
    assert (
        map_exit_primitive("exit when price returns to the middle band", reg)
        == "TRIGGER_CROSS_ABOVE"
    )
    assert (
        map_exit_primitive("exit when price crosses below VWAP", reg)
        == "TRIGGER_CROSS_BELOW"
    )
    assert map_exit_primitive("hard stop-loss at 1.5 ATR", reg) == "TRIGGER_CROSS_BELOW"


def test_split_entry_exit_clause_regions() -> None:
    """The splitter confines each phrase table to its own clause region."""
    entry, exit_ = split_entry_exit(
        "Buy when price falls below the lower band; "
        "exit when price crosses above the anchor."
    )
    assert "falls below the lower band" in entry
    assert "crosses above" in exit_
    assert "crosses above" not in entry  # exit keywords never leak into entry


def test_split_entry_exit_no_exit_marker() -> None:
    """No exit marker → the whole text is the entry clause, exit is empty."""
    entry, exit_ = split_entry_exit("Long when price crosses above VWAP")
    assert entry == "Long when price crosses above VWAP"
    assert exit_ == ""


def test_mixed_clause_does_not_cross_contaminate() -> None:
    """Entry keyword inside the EXIT clause must not flip the entry primitive."""
    reg = load_registry()
    text = (
        "Buy when price falls below the lower band; "
        "exit when price crosses above the anchor."
    )
    entry_clause, exit_clause = split_entry_exit(text)
    # Old behavior mapped both tables over the whole corpus: the exit clause's
    # "crosses above" would have hijacked the entry (CROSS_ABOVE). Per-clause
    # regions keep them apart.
    assert map_entry_primitive(entry_clause, reg) == "TRIGGER_CROSS_BELOW"
    assert map_exit_primitive(exit_clause, reg) == "TRIGGER_CROSS_ABOVE"


def test_indicator_detection_deduped() -> None:
    reg = load_registry()
    inds = detect_indicators(
        "VWAP anchor with an EMA filter and ATR stops and volume delta", reg
    )
    assert inds == ["IND_VWAP", "IND_EMA", "IND_ATR", "IND_VOLUME_DELTA"]


def test_pick_secondary_signal_complements_detected_set() -> None:
    """Augmentation adds a registry indicator the paper does NOT already use."""
    reg = load_registry()
    # VWAP paper already uses VWAP + volume delta → first complement is EMA
    assert pick_secondary_signal(["IND_VWAP", "IND_VOLUME_DELTA"], reg) == "IND_EMA"
    # single detected indicator → complement, not a duplicate
    assert pick_secondary_signal(["IND_EMA"], reg) == "IND_VWAP"
    # every registry indicator already used → no augmentation
    assert pick_secondary_signal([i for i in reg["indicators"]], reg) is None
    # unknown ids are ignored (only registry vocabulary matters)
    assert pick_secondary_signal(["IND_UNKNOWN"], reg) == "IND_VWAP"


def test_timeframe_detection() -> None:
    assert detect_timeframe("on the 15-minute chart", "1m") == "15m"
    assert detect_timeframe("intraday 2h bars", "1m") == "2h"
    assert detect_timeframe("on the 5-day chart", "1m") == "5d"
    assert detect_timeframe("daily data only", "1m") == "1m"  # no digits → fallback


# --- registry extension (phrases / risk_guards / data_series) ---------------


def test_registry_exposes_extension_tables() -> None:
    """P2.1: the JSON registry carries phrase, guard and data-series tables."""
    reg = load_registry()
    assert set(reg["phrases"]) == {"entry", "exit", "indicators"}
    assert reg["phrases"]["entry"][0] == ["band breakout", "TRIGGER_BAND_BREAKOUT"]
    assert [g["tag"] for g in reg["risk_guards"]] == [
        RiskTag.EXECUTION_SLIPPAGE_HEAVY.value,
        RiskTag.LOW_LIQUIDITY_FRAGILITY.value,
        RiskTag.HIGH_SESSION_SENSITIVITY.value,
        RiskTag.PARAMETRIC_OVERFIT_RISK.value,
    ]
    assert reg["data_series"] == ["open", "high", "low", "close", "volume"]


def test_registry_phrases_drive_mapping_minimal_registry() -> None:
    """Phrase tables load from the registry; absent keys fall back intact."""
    reg = load_registry()
    assert map_entry_primitive("price crosses above VWAP", reg) == "TRIGGER_CROSS_ABOVE"
    assert detect_indicators("uses an EMA filter", reg) == ["IND_EMA"]
    # A minimal registry (triggers only, no phrases) still maps via module fallback.
    minimal = {"triggers": ["TRIGGER_CROSS_ABOVE"], "indicators": [], "filters": []}
    assert (
        map_entry_primitive("price crosses above VWAP", minimal)
        == "TRIGGER_CROSS_ABOVE"
    )


def test_registry_risk_guards_resolve_and_fallback() -> None:
    """Tier 1 guard filter resolves from registry risk_guards in priority order."""
    reg = load_registry()
    assert (
        primary_guard_filter(vwap_annotations(slippage=True), reg) == "FLT_VOLUME_RATIO"
    )
    # Module-level default still applies when a registry carries no risk_guards.
    bare = {"triggers": [], "indicators": [], "filters": ["FLT_VOLUME_RATIO"]}
    assert (
        primary_guard_filter(vwap_annotations(slippage=True), bare)
        == "FLT_VOLUME_RATIO"
    )


def test_registry_data_series_fallback_for_ast() -> None:
    """AST validation accepts registry data_series; module set is the fallback."""
    reg = load_registry()
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
    # A registry dict written WITHOUT data_series falls back to AST_DATA_SERIES.
    stripped = {k: v for k, v in reg.items() if k != "data_series"}
    assert validate_ast_registry(tree, stripped) == []


# --- risk guards (Stage A5 → Tier 1) ---------------------------------------


def test_risk_union_priority_order() -> None:
    annotations = vwap_annotations(slippage=True)
    tags = risk_union(annotations)
    # EXECUTION_SLIPPAGE_HEAVY outranks LOW_LIQUIDITY_FRAGILITY
    assert tags.index(RiskTag.EXECUTION_SLIPPAGE_HEAVY) < tags.index(
        RiskTag.LOW_LIQUIDITY_FRAGILITY
    )
    assert primary_guard_filter(annotations) == "FLT_VOLUME_RATIO"


def test_guard_filter_none_without_tags() -> None:
    annotations = [a for a in vwap_annotations() if not a.risk_tags]
    assert primary_guard_filter(annotations) is None


# --- compile / tier generation ---------------------------------------------


def test_compile_full_matrix(catalog: Path) -> None:
    specs = compile_specs(
        vwap_extraction(),
        vwap_abstraction(),
        vwap_annotations(slippage=True),
        _verdicts(),
        catalog_path=catalog,
    )
    assert 3 <= len(specs) <= 5
    canonical = [
        StrategyTier.TIER_0_LITERAL,
        StrategyTier.TIER_1_PARAMETRIC,
        StrategyTier.TIER_2_GENERALIZED,
        StrategyTier.TIER_3_AUGMENTED,
    ]
    tiers = [s.tier for s in specs]
    assert tiers == sorted(tiers, key=canonical.index)

    t0 = next(s for s in specs if s.tier == StrategyTier.TIER_0_LITERAL)
    assert t0.target_asset == "BTCUSDT"
    assert t0.filter_primitives == []
    assert t0.entry_trigger_primitive == "TRIGGER_CROSS_ABOVE"
    assert t0.exit_trigger_primitive == "TRIGGER_CROSS_BELOW"
    assert set(t0.parameters["indicators"]) == {"IND_VWAP", "IND_VOLUME_DELTA"}
    assert "VWAP" in t0.causal_anchor_notes

    t1 = next(s for s in specs if s.tier == StrategyTier.TIER_1_PARAMETRIC)
    assert t1.filter_primitives == ["FLT_VOLUME_RATIO"]
    assert t1.risk_annotations
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY in t1.risk_annotations

    t2 = next(s for s in specs if s.tier == StrategyTier.TIER_2_GENERALIZED)
    # catalog drives availability (1m rows exist for ETH/ADA — no L2 filter)
    assert t2.target_asset in {"ETHUSDT", "ADAUSDT"}
    assert t2.parameters["mode"] == "dynamic"
    assert "generalized_from:BTCUSDT" in t2.parameters["origin"]

    t3 = next(s for s in specs if s.tier == StrategyTier.TIER_3_AUGMENTED)
    assert (
        t3.parameters["secondary_signal"] == "IND_EMA"
    )  # IND_VOLUME_DELTA already present


def test_spec_ids_and_dedup(catalog: Path) -> None:
    specs = compile_specs(
        vwap_extraction(),
        vwap_abstraction(),
        vwap_annotations(slippage=True),
        _verdicts(),
        catalog_path=catalog,
    )
    ids = [s.spec_id for s in specs]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("STRAT_CROSS_ABOVE_") for i in ids)
    keys = {(s.target_asset, s.timeframe, s.tier) for s in specs}
    assert len(keys) == len(specs)


def test_no_drop_annotations_never_untestable(catalog: Path) -> None:
    """A5 soft-failure annotations must never flip is_testable (No-Drop)."""
    specs = compile_specs(
        vwap_extraction(),
        vwap_abstraction(),
        vwap_annotations(slippage=True),
        _verdicts(),
        catalog_path=catalog,
    )
    assert specs, "expected compiled specs for a verified paper"
    assert all(s.is_testable for s in specs)


def test_no_anchor_dataset_emits_nothing(catalog: Path) -> None:
    extraction = vwap_extraction()
    extraction.datasets_used = []
    specs = compile_specs(
        extraction, vwap_abstraction(), [], _verdicts(), catalog_path=catalog
    )
    assert specs == []


def test_missing_verdict_still_not_emitted(catalog: Path) -> None:
    """Compiler never emits for a failed resource check — upstream hard stop."""
    extraction = vwap_extraction()
    missing = [
        ResourceVerdict(
            asset_class="Equities",
            symbol="QQQ",
            granularity=DataGranularity.MINUTE_1,
            status="missing",
            note="absent from catalog",
        )
    ]
    specs = compile_specs(
        extraction, vwap_abstraction(), [], missing, catalog_path=catalog
    )
    assert specs == []


def test_max_specs_cap(tmp_path: Path) -> None:
    paths = build_catalog(tmp_path / "big.parquet", rows=CATALOG_ROWS)
    specs = compile_specs(
        vwap_extraction(),
        vwap_abstraction(),
        vwap_annotations(slippage=True),
        _verdicts(),
        catalog_path=paths,
        max_specs=2,
    )
    assert len(specs) == 2


def test_alternate_symbol_prefers_target_granularity(tmp_path: Path) -> None:
    """Tier 2 target = symbol with a row at the paper's timeframe (no L2 gate)."""
    rows = [
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
        {
            "asset_class": "Crypto",
            "symbol": "ADAUSDT",
            "granularity": "1m",
            "has_l2_book": False,
            "has_order_flow": False,
            "start_year": 2021,
        },
    ]
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=rows)
    # 1m rows exist for ETH and ADA → same class, symbol-ascending: ADA first
    assert _alternate_symbol(catalog, "BTCUSDT", "1m") == "ADAUSDT"
    # no 15m rows at all → fall back to any alternate (still deterministic)
    assert _alternate_symbol(catalog, "BTCUSDT", "15m") == "ADAUSDT"
    # anchor excluded, never itself
    assert _alternate_symbol(catalog, "ADAUSDT", "1m") == "BTCUSDT"


def test_alternate_symbol_no_other_symbols(tmp_path: Path) -> None:
    rows = [
        {
            "asset_class": "Crypto",
            "symbol": "BTCUSDT",
            "granularity": "1m",
            "has_l2_book": True,
            "has_order_flow": True,
            "start_year": 2020,
        },
    ]
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=rows)
    assert _alternate_symbol(catalog, "BTCUSDT", "1m") is None


# --- export -----------------------------------------------------------------


def test_validate_and_export_roundtrip(tmp_path: Path) -> None:
    spec = ExecutableStrategySpec(
        spec_id="STRAT_VWAP_BTCUSDT_T0",
        tier=StrategyTier.TIER_0_LITERAL,
        target_asset="BTCUSDT",
        timeframe="1m",
        entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
        exit_trigger_primitive="TRIGGER_CROSS_BELOW",
        parameters={"indicators": ["IND_VWAP"], "mode": "static"},
        risk_annotations=[RiskTag.EXECUTION_SLIPPAGE_HEAVY],
        causal_anchor_notes="anchor notes",
    )
    out = validate_and_export([spec], tmp_path / "out" / "block_a_specs.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    restored = [ExecutableStrategySpec.model_validate(o) for o in payload]
    assert restored == [spec]


def test_schema_rejects_invalid_spec_payload() -> None:
    """The export read-back contract: malformed JSON items must not validate."""
    with pytest.raises(ValidationError):
        ExecutableStrategySpec.model_validate({"spec_id": 123})
