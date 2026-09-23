"""Schema contract tests — Block A Pydantic models (spec §3/§4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from black_box.block_a.models import (
    BlockAState,
    CausalAbstractionSchema,
    DataGranularity,
    DatasetRequirement,
    ExecutableStrategySpec,
    OperatorAnnotations,
    PaperExtractionSchema,
    ResourceCheckResult,
    ResourceVerdict,
    RiskTag,
    StrategyAnnotation,
    StrategyTier,
)


def test_data_granularity_enum_values() -> None:
    assert DataGranularity.MINUTE_1.value == "1m"
    assert DataGranularity.TICK.value == "tick"
    assert {g.value for g in DataGranularity} == {"tick", "1s", "1m", "1h", "1d"}


def test_dataset_requirement_defaults() -> None:
    req = DatasetRequirement(
        asset_class="Crypto",
        symbol="BTCUSDT",
        required_granularity=DataGranularity.MINUTE_1,
    )
    assert req.requires_l2_book is False
    assert req.requires_order_flow is False
    assert req.start_year is None


def test_strategy_annotation_operator_name_literal() -> None:
    StrategyAnnotation(
        operator_name="Adversarial",
        observation="spoofing can fake delta",
        proposed_modification="add filter",
    )
    with pytest.raises(ValidationError):
        StrategyAnnotation(
            operator_name="DELETE",  # not a quant operator
            observation="x",
            proposed_modification="y",
        )


def test_no_drop_is_structural() -> None:
    """Stage A5 annotations carry NO is_testable field (No-Drop rule)."""
    with pytest.raises(ValidationError):
        StrategyAnnotation(
            operator_name="Restrict",
            observation="x",
            proposed_modification="y",
            is_testable=False,  # must be rejected — schema-level guarantee
        )


def test_operator_annotations_enforces_full_sweep() -> None:
    one = StrategyAnnotation(
        operator_name="Restrict", observation="x", proposed_modification="y"
    )
    with pytest.raises(ValidationError):
        OperatorAnnotations(annotations=[one])  # needs exactly 8


def test_spec_defaults_is_testable_true() -> None:
    spec = ExecutableStrategySpec(
        spec_id="STRAT_VWAP_BTCUSDT_T0",
        tier=StrategyTier.TIER_0_LITERAL,
        target_asset="BTCUSDT",
        timeframe="1m",
        entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
        exit_trigger_primitive="TRIGGER_CROSS_BELOW",
    )
    assert spec.is_testable is True
    assert spec.filter_primitives == []
    assert spec.parameters == {}
    assert spec.risk_annotations == []


def test_spec_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExecutableStrategySpec(
            spec_id="x",
            tier=StrategyTier.TIER_0_LITERAL,
            target_asset="BTCUSDT",
            timeframe="1m",
            entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
            exit_trigger_primitive="TRIGGER_CROSS_BELOW",
            delete_me=True,
        )


def test_resource_check_all_available() -> None:
    ok = ResourceCheckResult(
        verdicts=[
            ResourceVerdict(
                asset_class="Crypto",
                symbol="BTCUSDT",
                granularity=DataGranularity.MINUTE_1,
                status="available",
            )
        ],
        all_available=True,
    )
    assert ok.all_available is True


def test_extraction_roundtrip_json() -> None:
    extraction = PaperExtractionSchema(
        paper_title="P",
        primary_hypothesis="H",
        datasets_used=[
            DatasetRequirement(
                asset_class="Crypto",
                symbol="BTCUSDT",
                required_granularity=DataGranularity.MINUTE_1,
            )
        ],
        core_mechanism="mechanism",
    )
    restored = PaperExtractionSchema.model_validate_json(extraction.model_dump_json())
    assert restored == extraction


def test_block_a_state_msgpack_safe() -> None:
    """State carries primitives only — checkpoints cleanly with MemorySaver."""
    state = BlockAState(
        paper_id="p1",
        chunks=[{"id": "0", "text": "chunk"}],
        status="running",
    )
    data = state.model_dump()
    assert isinstance(data["chunks"], list)
    assert isinstance(data["chunks"][0]["text"], str)


def test_causal_abstraction_roundtrip() -> None:
    a = CausalAbstractionSchema(
        abstract_hypothesis="h",
        causal_anchor="c",
        non_negotiable_invariants=["i1"],
    )
    b = CausalAbstractionSchema.model_validate(a.model_dump())
    assert b.non_negotiable_invariants == ["i1"]


def test_risk_tag_enum_members() -> None:
    assert {t.value for t in RiskTag} == {
        "HIGH_SESSION_SENSITIVITY",
        "LOW_LIQUIDITY_FRAGILITY",
        "PARAMETRIC_OVERFIT_RISK",
        "EXECUTION_SLIPPAGE_HEAVY",
    }
